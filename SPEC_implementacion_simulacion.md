# Especificación de implementación — Simulación API Gateway con Reserva de Capacidad

## 0. Contexto y objetivo

Se debe implementar, en Python, el modelo de simulación conceptualizado en el informe del proyecto:
un **API Gateway** que recibe dos tipos de solicitudes (**consultas de stock** y **solicitudes de
reabastecimiento**) desde una red de sucursales de un Centro de Distribución (CD), y las asigna a un
**pool compartido de N hilos de procesamiento**, sin colas de espera (rechazo inmediato si no hay
capacidad), aplicando una **regla de reserva de capacidad** que prioriza al reabastecimiento.

El objetivo final es generar los datos, gráficos y evidencias de verificación/validación que
alimentarán las secciones 15 a 19 del informe (Construcción del modelo, Escenarios, Resultados,
Análisis, Verificación). La sección de Validación (20) y las conclusiones se completan después,
en base a estos resultados.

**Stack requerido:** Python 3.11+, `simpy` (motor de eventos discretos), `numpy`, `scipy`
(para resolver el sistema lineal de la cadena de Markov y `scipy.stats` para intervalos de
confianza), `pandas` (tablas de resultados), `plotly` o `matplotlib` (gráficos), `streamlit`
(interfaz).

**No usar colas de simpy (`simpy.Store`, etc.) para las solicitudes rechazadas.** El rechazo es
instantáneo y no debe encolarse — ver sección 2.2.

---

## 1. Reglas del modelo (obligatorias, no negociables)

Estas reglas están fijadas por la conceptualización formal del informe (Red de Petri Coloreada,
sección 9) y no deben modificarse sin avisar:

1. Hay `N` hilos de procesamiento en total (capacidad del pool), compartidos entre ambos tipos
   de solicitud.
2. De esos `N`, una cantidad `R` está **reservada exclusivamente** para solicitudes de
   reabastecimiento.
3. Sea `c` = hilos actualmente ocupados por consultas, `r` = hilos actualmente ocupados por
   reabastecimiento. El estado del sistema es el par `(c, r)`.
4. **Regla de aceptación — consulta de stock:** se acepta si y solo si `c + r < N - R`.
5. **Regla de aceptación — reabastecimiento:** se acepta si y solo si `c + r < N`.
6. Si la regla correspondiente no se cumple, la solicitud se **rechaza de inmediato** (no espera,
   no hay cola, no hay reintento automático).
7. Al finalizar el procesamiento de una solicitud, el hilo se libera inmediatamente
   (`c -= 1` o `r -= 1` según corresponda).
8. **Invariante que debe cumplirse siempre:** si `n` = hilos libres, entonces `n + c + r == N` en
   todo momento de la simulación. Esto se usa como test de verificación (sección 5).

---

## 2. Motor de eventos discretos (`model/simulation.py`)

### 2.1. Entidades

```python
@dataclass
class Solicitud:
    id: int
    tipo: str          # "consulta" | "reabastecimiento"
    sucursal_id: int    # entero 1..S, ver sección 2.4
    momento_llegada: float
    momento_fin: float | None = None
    estado: str = "pendiente"   # "atendida" | "rechazada"
```

### 2.2. Lógica de aceptación/rechazo (NO usar `simpy.Resource` directo)

`simpy.Resource` no soporta capacidad diferenciada por tipo de cliente de forma nativa. Implementar
un objeto propio que envuelva un `simpy.Container` o simplemente contadores enteros (`self.c`,
`self.r`) protegidos por la lógica de la regla de reserva:

```python
class PoolDeHilos:
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
```

El proceso de cada solicitud en SimPy es entonces:

```python
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
```

### 2.3. Generadores de llegadas (procesos de Poisson)

Dos procesos independientes, uno por tipo de solicitud, con tiempos entre llegadas exponenciales:

```python
def generador_llegadas(env, pool, tipo, tasa_lambda, distribucion_servicio, S, registro, id_counter):
    while True:
        yield env.timeout(random.expovariate(tasa_lambda))
        sid = next(id_counter)
        sucursal = random.randint(1, S)
        tiempo_servicio = distribucion_servicio()
        sol = Solicitud(id=sid, tipo=tipo, sucursal_id=sucursal, momento_llegada=env.now)
        env.process(proceso_solicitud(env, pool, sol, tiempo_servicio, registro))
```

### 2.4. Variables aleatorias — dos configuraciones intercambiables

El simulador debe poder correr en dos "modos" de distribución de servicio, seleccionables por
parámetro (ver tabla en sección 4). **No hardcodear una sola.**

**Modo `empirica` (config principal, responde la pregunta del proyecto):**
```python
SC = lambda: random.lognormvariate(mu=-0.7731, sigma=0.40)   # segundos
SR = lambda: random.weibullvariate(alpha=3.3735, beta=1.8)   # ojo: random.weibullvariate(alpha=escala, beta=forma)
```
> Nota de implementación: `random.weibullvariate(alpha, beta)` en la librería estándar de Python
> usa `alpha` como parámetro de **escala** y `beta` como parámetro de **forma**. Confirmar que
> escala=3.3735 y forma (k)=1.8 coincidan con la convención `scipy.stats.weibull_min(c=k, scale=λ)`
> antes de dar por buena la implementación — si hay dudas, usar `scipy.stats.weibull_min.rvs`
> explícitamente en vez de la librería estándar, para evitar errores de convención.

**Modo `markoviana` (config de contraste, para verificación/validación):**
```python
SC = lambda: random.expovariate(1/0.5)   # media 0.5s
SR = lambda: random.expovariate(1/3.0)   # media 3.0s
```

### 2.5. Franja horaria (solo para el experimento secundario de jornada completa — prioridad baja)

Si hay tiempo, implementar el experimento de 12h con variable de franja horaria (pico/valle) que
multiplica λC y λR por 1.0 (pico) o 0.35 (valle). **Esto es de menor prioridad — priorizar primero
que el experimento principal (franja pico, 2h) funcione y esté verificado.**

---

## 3. Módulo de Cadenas de Markov (`model/markov.py`)

Sirve para verificación y validación (secciones 19 y 20), no para responder la pregunta principal.

### 3.1. Construcción del generador Q

Estados: todos los pares `(c, r)` con `0 <= c <= N-R`, `0 <= r <= N`, `c + r <= N`.

Tasas de transición:
- `(c,r) -> (c+1,r)` con tasa `λC`, solo si `c + r < N - R`
- `(c,r) -> (c,r+1)` con tasa `λR`, solo si `c + r < N`
- `(c,r) -> (c-1,r)` con tasa `c * μC`, solo si `c >= 1`
- `(c,r) -> (c,r-1)` con tasa `r * μR`, solo si `r >= 1`

Donde `μC = 1/E[SC]` y `μR = 1/E[SR]` (usar las medias, no importa la distribución exacta para
esta parte analítica).

### 3.2. Resolver distribución estacionaria

Resolver `π · Q = 0` sujeto a `sum(π) = 1` (sistema lineal, usar `numpy.linalg.solve` o
`scipy.linalg` reemplazando una fila por la condición de normalización — es el procedimiento
estándar para cadenas de Markov de tiempo continuo).

A partir de π, calcular:
- `BC` = suma de π sobre estados con `c + r >= N - R` (probabilidad de bloqueo de consulta)
- `BR` = suma de π sobre estados con `c + r == N` (probabilidad de bloqueo de reabastecimiento)
- `U` = `(E[c] + E[r]) / N` (utilización esperada del pool)

### 3.3. Fórmula B de Erlang (caso de verificación con R=0)

```python
def erlang_b(N, a):
    """a = carga ofrecida total en Erlangs (lambda_total * E[servicio])"""
    inv_b = 1.0
    for k in range(1, N + 1):
        inv_b = 1.0 + (k / a) * inv_b
    return 1.0 / inv_b
```
(Usar la recursión de Erlang, no la fórmula directa con factoriales — la directa desborda
numéricamente para N grande.)

**Importante:** cuando `R = 0`, por el teorema de Erlang-Sevastyanov (sistemas de pérdida con
compartición completa son insensibles a la distribución del tiempo de servicio y dependen solo de
la media), la probabilidad de bloqueo teórica para **ambos** tipos de solicitud converge a
`erlang_b(N, a_total)` donde `a_total = λC·E[SC] + λR·E[SR]`, sin importar si μC ≠ μR ni la forma
de las distribuciones. Este es el caso de prueba central de la sección 19 (Verificación).

---

## 3A. Módulo de Monte Carlo (`model/monte_carlo.py`)

**Nota:** esto NO son las 30 réplicas del DES para intervalos de confianza (eso es diseño
experimental estándar, sección 6.3 / 7). Esta es una **tercera estrategia independiente**, de
naturaleza estática y combinatoria, sin avance de reloj — al estilo del caso *Thundering Herd*
de la guía metodológica del curso. Si más adelante se decide reencuadrar o quitar esta parte,
avisar para actualizar este documento.

### 3A.1. Qué calcula

La probabilidad de que, ante una **ráfaga instantánea** de solicitudes llegando en una ventana de
tiempo muy corta `Δt` (por ejemplo, 100ms), la demanda combinada sature el pool de hilos **antes
de que ningún hilo llegue a liberarse**. Es un análisis de "peor caso" puntual, no una trayectoria
temporal — por eso no usa SimPy ni la FEL.

### 3A.2. Implementación

```python
import numpy as np

def probabilidad_saturacion_rafaga(lambda_C, lambda_R, N, R, delta_t=0.1, n_trials=100_000, seed=None):
    """
    Calcula, mediante muestreo Monte Carlo estático, la probabilidad de que una ráfaga de
    solicitudes en una ventana delta_t sature el pool de hilos, asumiendo que ningún hilo se
    libera durante esa ventana (cota superior conservadora del riesgo de saturación instantánea).
    """
    rng = np.random.default_rng(seed)

    # Número de llegadas en la ventana, para cada tipo (proceso de Poisson -> conteo Poisson)
    llegadas_consulta = rng.poisson(lam=lambda_C * delta_t, size=n_trials)
    llegadas_reab = rng.poisson(lam=lambda_R * delta_t, size=n_trials)

    # Simulación combinatoria de aceptación secuencial dentro de la ráfaga, respetando la
    # regla de reserva (orden de llegada aleatorio dentro de la ventana)
    saturo_consulta = np.zeros(n_trials, dtype=bool)
    saturo_reab = np.zeros(n_trials, dtype=bool)

    for i in range(n_trials):
        c, r = 0, 0
        eventos = (["C"] * llegadas_consulta[i]) + (["R"] * llegadas_reab[i])
        rng.shuffle(eventos)
        for ev in eventos:
            if ev == "C":
                if c + r < N - R:
                    c += 1
                else:
                    saturo_consulta[i] = True
            else:
                if c + r < N:
                    r += 1
                else:
                    saturo_reab[i] = True

    return {
        "prob_rechazo_consulta_en_rafaga": saturo_consulta.mean(),
        "prob_rechazo_reab_en_rafaga": saturo_reab.mean(),
        "prob_alguna_saturacion": (saturo_consulta | saturo_reab).mean(),
    }
```

> Nota de rendimiento: el bucle `for i in range(n_trials)` es lento en Python puro para
> `n_trials=100_000`. Si el tiempo de ejecución es un problema, vectorizar con `numpy` (ordenar
> las llegadas por tipo y aplicar la regla acumulativamente con `cumsum`) o, si no alcanza el
> tiempo del proyecto, reducir `n_trials` a 10_000–20_000 y dejarlo documentado como limitación.

### 3A.3. Para qué se usa en el informe

Esta pieza responde una pregunta **distinta** a la del DES: no "¿cuál es la tasa de rechazo en
régimen estacionario?", sino "¿qué tan riesgoso es un pico súbito de tráfico (ej. una promoción,
un reinicio simultáneo de varias sucursales) incluso si en promedio el sistema no está saturado?".
Es información complementaria para la sección 18 (Análisis) y para las recomendaciones (22),
por ejemplo si conviene un mecanismo de *backpressure* fuera del alcance actual del modelo.

Correrlo para Escenario A y B, con `Δt` variable (ej. 50ms, 100ms, 500ms) para mostrar cómo el
riesgo de saturación instantánea cae a medida que la ventana de la ráfaga se agranda (porque hay
más tiempo para que se liberen hilos en la práctica — aunque este modelo estático no lo capture,
conviene aclararlo como supuesto conservador en el informe).

---

## 4. Parámetros y escenarios (`model/parametros.py`)

Centralizar todos los parámetros en un único lugar (diccionario o dataclass), nunca hardcodeados
dentro del motor de simulación, para que Streamlit pueda modificarlos:

```python
ESCENARIOS = {
    "A": {"servidores": 2, "hilos_por_servidor": 4, "N": 8, "R": 2},
    "B": {"servidores": 4, "hilos_por_servidor": 4, "N": 16, "R": 4},
}

PARAMETROS_ESTOCASTICOS = {
    "lambda_C": 8.0,      # solicitudes/s
    "lambda_R": 1.0,      # solicitudes/s
    "mu_C": 2.0,           # 1 / 0.5s
    "mu_R": 1/3,           # 1 / 3.0s
}

PARAMETROS_EJECUCION = {
    "horizonte_principal": 7200,     # segundos (2h)
    "horizonte_jornada": 43200,      # segundos (12h) - experimento secundario
    "calentamiento": 300,            # segundos
    "n_replicas": 30,
    "S_sucursales": 20,
}
```

Todos estos valores deben ser editables desde la interfaz Streamlit (sección 6), con los valores
de arriba como default.

---

## 5. Verificación (soporte de código para la sección 19 del informe)

Escribir tests (`pytest`, carpeta `tests/`) que verifiquen:

1. **Invariante de conservación:** en cada evento de la simulación, `n + c + r == N` (agregar un
   contador `n` derivado y chequear en cada paso, o verificarlo post-hoc sobre el log de eventos).
2. **Caso extremo — capacidad 1:** con `N=1, R=0`, solo una solicitud puede procesarse a la vez;
   confirmar que la segunda solicitud simultánea se rechaza.
3. **Caso extremo — probabilidad de fallo / saturación total:** con `λC, λR` muy altos respecto a
   `μC, μR`, la tasa de rechazo debe aproximarse a 1 (casi todo se rechaza).
4. **Convergencia a Erlang B (el más importante):** correr el DES con `R=0` y la configuración
   markoviana, muchas réplicas, horizonte largo, y verificar que la tasa de rechazo simulada
   (tanto de consultas como de reabastecimiento) converge al valor de `erlang_b(N, a_total)`
   calculado analíticamente, dentro de un margen razonable (ej. el valor teórico debe caer dentro
   del intervalo de confianza del 95% de las réplicas simuladas).
5. **Cola vacía / sistema recién iniciado:** verificar que en `t=0` el sistema arranca en estado
   `(c=0, r=0)` y que la primera solicitud que llega siempre se acepta (si `N > 0`).

Guardar los resultados de estos tests de forma que puedan copiarse directamente a la tabla de
verificación del informe (columna: caso, entrada, resultado esperado, resultado obtenido, estado).

---

## 6. Interfaz Streamlit (`app.py`)

### 6.1. Panel de parámetros (sidebar)
- Selector de escenario (A / B / personalizado)
- Sliders/inputs para N, R, λC, λR, medias de servicio
- Selector de configuración de servicio: "Empírica" / "Markoviana"
- Horizonte de simulación, calentamiento, número de réplicas
- Botón "Ejecutar simulación"

### 6.2. Panel de resultados
- Tabla resumen: tasa de rechazo por tipo (con IC 95%), utilización, pico de ocupación
- Gráfico de barras comparando Escenario A vs B
- Gráfico de evolución temporal de `c(t)` y `r(t)` para una réplica de ejemplo (útil para
  mostrar visualmente la "región de bloqueo parcial" del informe, sección 9.3.5)
- Comparación DES vs valor exacto de Markov (cuando R=0) — panel de verificación visual
- Botón para exportar resultados a CSV (para que estos datos alimenten después el informe)

### 6.3. Reproducibilidad
Cada réplica debe usar una semilla explícita (`random.seed(i)` o mejor, `numpy.random.default_rng(i)`
para no depender del estado global de `random`). Guardar las semillas usadas junto con los
resultados exportados, para que el informe pueda documentar la técnica de "números aleatorios
comunes" ya mencionada en la sección 16.3 del informe (mismas semillas entre Escenario A y B para
reducir varianza en la comparación).

---

## 7. Salidas esperadas (para la fase de redacción del informe)

Al finalizar, el código debe poder generar, por cada combinación de (escenario, configuración de
servicio):

1. Un **CSV por réplica** o consolidado, con al menos: escenario, configuración, réplica, tasa de
   rechazo consulta, tasa de rechazo reabastecimiento, utilización, pico de ocupación.
2. Un **resumen estadístico** (media, desviación estándar, IC 95%) por combinación.
3. Los **valores exactos de la cadena de Markov** (BC, BR, U) por escenario, para comparar contra
   el punto 2.
4. Los **resultados de Monte Carlo** (probabilidad de saturación en ráfaga) por escenario y por
   valor de `Δt` probado.
5. La **tabla de verificación** de la sección 5 de este documento, con resultado obtenido vs
   esperado.
6. Gráficos exportados (PNG o HTML de Plotly) listos para insertar en el informe.

No hace falta redactar texto explicativo — eso se hace después, en el informe, con los datos que
este código produzca.

---

## 8. Orden de implementación sugerido

1. `PoolDeHilos` + lógica de aceptación/rechazo (sin SimPy todavía, probar la lógica pura)
2. Motor DES completo en SimPy, una sola réplica, escenario A, config markoviana
3. Verificar el invariante `n+c+r=N` manualmente sobre esa primera corrida
4. Módulo de Markov (`markov.py`) + Erlang B
5. Test de convergencia (punto 4 de la sección 5) — **si esto no converge, hay un error en el
   motor DES y no hay que seguir hasta resolverlo**
6. Agregar configuración empírica (Lognormal/Weibull)
7. Módulo de réplicas + estadística (IC 95%)
8. Escenario B, comparación A vs B
9. Módulo de Monte Carlo (sección 3A) — independiente del resto, se puede hacer en paralelo
10. Interfaz Streamlit
11. Experimento secundario de jornada completa (si alcanza el tiempo)
