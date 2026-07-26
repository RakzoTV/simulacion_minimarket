"""Motor de eventos discretos del API Gateway con reserva de capacidad.

Implementa las reglas fijadas en la sección 1 de SPEC_implementacion_simulacion.md:
pool compartido de N hilos, R reservados para reabastecimiento, rechazo inmediato
sin colas.
"""

import itertools
import random
from dataclasses import dataclass, field

import simpy


@dataclass
class Solicitud:
    id: int
    tipo: str            # "consulta" | "reabastecimiento"
    sucursal_id: int      # entero 1..S
    momento_llegada: float
    momento_fin: float | None = None
    estado: str = "pendiente"   # "atendida" | "rechazada"


class PoolDeHilos:
    """Pool compartido de N hilos con R reservados para reabastecimiento.

    intentar_aceptar/liberar son operaciones instantáneas (sin yield): la
    regla de reserva se evalúa en un único paso atómico dentro del hilo de
    ejecución de SimPy, que es cooperativo y no hay concurrencia real.
    """

    def __init__(self, env, N, R):
        self.env = env
        self.N = N
        self.R = R
        self.c = 0  # hilos ocupados por consultas
        self.r = 0  # hilos ocupados por reabastecimiento

    def intentar_aceptar(self, tipo):
        """Devuelve True/False de inmediato. No bloquea, no encola."""
        if tipo == "consulta":
            if self.c + self.r < self.N - self.R:
                self.c += 1
                return True
            return False
        else:  # reabastecimiento
            if self.c + self.r < self.N:
                self.r += 1
                return True
            return False

    def liberar(self, tipo):
        if tipo == "consulta":
            self.c -= 1
        else:
            self.r -= 1

    def libres(self):
        return self.N - self.c - self.r


def proceso_solicitud(env, pool, solicitud, tiempo_servicio, registro):
    aceptada = pool.intentar_aceptar(solicitud.tipo)
    if not aceptada:
        solicitud.estado = "rechazada"
        registro.append(solicitud)
        return
    yield env.timeout(tiempo_servicio)
    pool.liberar(solicitud.tipo)
    solicitud.estado = "atendida"
    solicitud.momento_fin = env.now
    registro.append(solicitud)


def generador_llegadas(env, pool, tipo, tasa_lambda, distribucion_servicio, S, registro, id_counter):
    while True:
        yield env.timeout(random.expovariate(tasa_lambda))
        sid = next(id_counter)
        sucursal = random.randint(1, S)
        tiempo_servicio = distribucion_servicio()
        sol = Solicitud(id=sid, tipo=tipo, sucursal_id=sucursal, momento_llegada=env.now)
        env.process(proceso_solicitud(env, pool, sol, tiempo_servicio, registro))


# --- Distribuciones de servicio (sección 2.4) ---

def distribuciones_servicio(modo, mu_C=None, mu_R=None):
    """Devuelve (SC, SR): callables sin argumentos que generan un tiempo de servicio.

    modo == "empirica": lognormal para consultas, Weibull para reabastecimiento
                         (parámetros fijos de la conceptualización del proyecto).
    modo == "markoviana": exponenciales con las medias 1/mu_C y 1/mu_R (default 0.5s y 3.0s).
    """
    if modo == "empirica":
        SC = lambda: random.lognormvariate(-0.7731, 0.40)
        SR = lambda: random.weibullvariate(alpha=3.3735, beta=1.8)
        return SC, SR
    elif modo == "markoviana":
        media_C = 1 / mu_C if mu_C else 0.5
        media_R = 1 / mu_R if mu_R else 3.0
        SC = lambda: random.expovariate(1 / media_C)
        SR = lambda: random.expovariate(1 / media_R)
        return SC, SR
    else:
        raise ValueError(f"modo de distribución desconocido: {modo!r}")


@dataclass
class ResultadoReplica:
    registro: list = field(default_factory=list)
    N: int = 0
    R: int = 0
    horizonte: float = 0.0
    calentamiento: float = 0.0
    trayectoria: list = field(default_factory=list)   # [(t, c, r)] muestreada, opcional


def correr_replica(
    N,
    R,
    lambda_C,
    lambda_R,
    modo_servicio,
    horizonte,
    calentamiento=0.0,
    S=20,
    mu_C=None,
    mu_R=None,
    seed=None,
    franjas=None,
    registrar_trayectoria=False,
):
    """Corre una única réplica de la simulación DES.

    franjas: lista opcional de (t_inicio, t_fin, multiplicador) para modelar
    picos/valles (sección 2.5). Si es None, lambda_C/lambda_R son constantes.
    """
    if seed is not None:
        random.seed(seed)

    env = simpy.Environment()
    pool = PoolDeHilos(env, N, R)
    registro = []
    trayectoria = []
    id_counter = itertools.count(1)

    SC, SR = distribuciones_servicio(modo_servicio, mu_C=mu_C, mu_R=mu_R)

    if franjas is None:
        env.process(generador_llegadas(env, pool, "consulta", lambda_C, SC, S, registro, id_counter))
        env.process(generador_llegadas(env, pool, "reabastecimiento", lambda_R, SR, S, registro, id_counter))
    else:
        env.process(
            _generador_llegadas_franjas(
                env, pool, "consulta", lambda_C, SC, S, registro, id_counter, franjas
            )
        )
        env.process(
            _generador_llegadas_franjas(
                env, pool, "reabastecimiento", lambda_R, SR, S, registro, id_counter, franjas
            )
        )

    if registrar_trayectoria:
        env.process(_muestreo_trayectoria(env, pool, trayectoria, paso=1.0))

    env.run(until=horizonte)

    return ResultadoReplica(
        registro=registro,
        N=N,
        R=R,
        horizonte=horizonte,
        calentamiento=calentamiento,
        trayectoria=trayectoria,
    )


def _generador_llegadas_franjas(env, pool, tipo, lambda_base, distribucion_servicio, S, registro, id_counter, franjas):
    """Variante de generador_llegadas con lambda variable por franja horaria (sección 2.5).

    Aproximación por tramos constantes: la tasa usada para muestrear el próximo
    entre-llegada es la vigente en el instante de la llegada anterior. No es un
    proceso de Poisson no homogéneo exacto (thinning), pero es razonable cuando
    las franjas son largas frente al tiempo medio entre llegadas.
    """
    while True:
        mult = _multiplicador_vigente(env.now, franjas)
        yield env.timeout(random.expovariate(lambda_base * mult))
        sid = next(id_counter)
        sucursal = random.randint(1, S)
        tiempo_servicio = distribucion_servicio()
        sol = Solicitud(id=sid, tipo=tipo, sucursal_id=sucursal, momento_llegada=env.now)
        env.process(proceso_solicitud(env, pool, sol, tiempo_servicio, registro))


def _multiplicador_vigente(t, franjas):
    for t_inicio, t_fin, mult in franjas:
        if t_inicio <= t < t_fin:
            return mult
    return 1.0


def _muestreo_trayectoria(env, pool, trayectoria, paso=1.0):
    while True:
        trayectoria.append((env.now, pool.c, pool.r))
        yield env.timeout(paso)
