"""Tests de verificación del modelo.

Cada test agrega una fila a outputs/tabla_verificacion.csv con:
caso, entrada, resultado esperado, resultado obtenido, estado.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from model.markov import erlang_b
from model.simulation import PoolDeHilos, correr_replica
from model.replicas import correr_replicas

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
TABLA_PATH = OUTPUTS_DIR / "tabla_verificacion.csv"

_FILAS_VERIFICACION = []


def _registrar(caso, entrada, esperado, obtenido, estado):
    _FILAS_VERIFICACION.append({
        "caso": caso,
        "entrada": entrada,
        "resultado_esperado": esperado,
        "resultado_obtenido": obtenido,
        "estado": estado,
    })


@pytest.fixture(scope="session", autouse=True)
def _volcar_tabla_al_final():
    yield
    OUTPUTS_DIR.mkdir(exist_ok=True)
    if _FILAS_VERIFICACION:
        with open(TABLA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["caso", "entrada", "resultado_esperado", "resultado_obtenido", "estado"])
            writer.writeheader()
            writer.writerows(_FILAS_VERIFICACION)


# --- 1. Invariante de conservación: n + c + r == N en todo momento ---

def test_invariante_conservacion():
    N, R = 8, 2
    res = correr_replica(
        N=N, R=R, lambda_C=8.0, lambda_R=1.0,
        modo_servicio="markoviana", horizonte=7200, S=20, seed=1,
    )

    eventos = []
    for s in res.registro:
        if s.estado == "atendida":
            eventos.append((s.momento_llegada, "llega", s.tipo))
            eventos.append((s.momento_fin, "sale", s.tipo))
    eventos.sort(key=lambda e: e[0])

    c = r = 0
    violacion = False
    for t, ev, tipo in eventos:
        if ev == "llega":
            c += 1 if tipo == "consulta" else 0
            r += 1 if tipo == "reabastecimiento" else 0
        else:
            c -= 1 if tipo == "consulta" else 0
            r -= 1 if tipo == "reabastecimiento" else 0
        n = N - c - r
        if n + c + r != N or n < 0 or c < 0 or r < 0:
            violacion = True
            break

    estado = "OK" if not violacion else "FALLO"
    _registrar(
        caso="Invariante de conservación n+c+r=N",
        entrada=f"N={N}, R={R}, horizonte=7200s, seed=1",
        esperado="n+c+r=N en todo momento, sin violaciones",
        obtenido=f"violacion={violacion}, c_final={c}, r_final={r}",
        estado=estado,
    )
    assert not violacion


# --- 2. Caso extremo: capacidad 1 ---

def test_capacidad_uno_rechaza_segunda_solicitud_simultanea():
    pool = PoolDeHilos(env=None, N=1, R=0)

    primera = pool.intentar_aceptar("consulta")
    segunda = pool.intentar_aceptar("reabastecimiento")

    estado = "OK" if (primera and not segunda) else "FALLO"
    _registrar(
        caso="Capacidad N=1: segunda solicitud simultánea rechazada",
        entrada="N=1, R=0; 1ra=consulta, 2da=reabastecimiento (ambas antes de liberar)",
        esperado="1ra aceptada=True, 2da aceptada=False",
        obtenido=f"1ra={primera}, 2da={segunda}",
        estado=estado,
    )
    assert primera is True
    assert segunda is False


# --- 3. Saturación total: lambda >> mu implica tasa de rechazo -> 1 ---

def test_saturacion_total_tasa_rechazo_cercana_a_uno():
    N, R = 8, 2
    res = correr_replica(
        N=N, R=R, lambda_C=1000.0, lambda_R=1000.0,
        modo_servicio="markoviana", horizonte=600, S=20,
        mu_C=2.0, mu_R=1 / 3, seed=42,
    )
    rechazadas = sum(1 for s in res.registro if s.estado == "rechazada")
    tasa = rechazadas / len(res.registro)

    estado = "OK" if tasa > 0.95 else "FALLO"
    _registrar(
        caso="Saturación total (lambda >> mu)",
        entrada="N=8, R=2, lambda_C=lambda_R=1000/s, mu_C=2, mu_R=1/3, horizonte=600s",
        esperado="tasa de rechazo global > 0.95",
        obtenido=f"tasa={tasa:.5f}",
        estado=estado,
    )
    assert tasa > 0.95


# --- 4. Convergencia a Erlang B (el más importante) ---

def test_convergencia_erlang_b_con_r_cero():
    N, R = 8, 0
    lambda_C, lambda_R = 8.0, 1.0
    mu_C, mu_R = 2.0, 1 / 3
    horizonte = 7200
    n_replicas = 30

    a_total = lambda_C / mu_C + lambda_R / mu_R
    eb = erlang_b(N, a_total)

    df = correr_replicas(
        N=N, R=R, lambda_C=lambda_C, lambda_R=lambda_R,
        modo_servicio="markoviana", horizonte=horizonte,
        n_replicas=n_replicas, S=20, mu_C=mu_C, mu_R=mu_R,
        escenario="verificacion_erlang_b", seed_base=0,
        registrar_trayectoria=False,
    )

    def ic95(x):
        x = np.asarray(x)
        m = x.mean()
        se = x.std(ddof=1) / np.sqrt(len(x))
        h = se * stats.t.ppf(0.975, len(x) - 1)
        return m, m - h, m + h

    mc, lo_c, hi_c = ic95(df["tasa_rechazo_consulta"])
    mr, lo_r, hi_r = ic95(df["tasa_rechazo_reab"])

    contiene_c = lo_c <= eb <= hi_c
    contiene_r = lo_r <= eb <= hi_r
    estado = "OK" if (contiene_c and contiene_r) else "FALLO"

    _registrar(
        caso="Convergencia DES -> Erlang B (R=0, config markoviana)",
        entrada=f"N={N}, R={R}, lambda_C={lambda_C}, lambda_R={lambda_R}, "
                f"mu_C={mu_C}, mu_R={mu_R:.4f}, horizonte={horizonte}s, {n_replicas} réplicas",
        esperado=f"Erlang B teórico ({eb:.5f}) dentro del IC95% de ambas tasas simuladas",
        obtenido=f"consulta: media={mc:.5f} IC95=[{lo_c:.5f},{hi_c:.5f}]; "
                 f"reab: media={mr:.5f} IC95=[{lo_r:.5f},{hi_r:.5f}]",
        estado=estado,
    )
    assert contiene_c, f"Erlang B {eb} fuera del IC95 de consulta [{lo_c},{hi_c}]"
    assert contiene_r, f"Erlang B {eb} fuera del IC95 de reabastecimiento [{lo_r},{hi_r}]"


# --- 5. Sistema recién iniciado: t=0 en (0,0), primera solicitud siempre aceptada ---

def test_sistema_arranca_vacio_y_primera_solicitud_aceptada():
    pool = PoolDeHilos(env=None, N=8, R=2)
    estado_inicial_ok = (pool.c == 0 and pool.r == 0)

    primera_aceptada = pool.intentar_aceptar("consulta")

    estado = "OK" if (estado_inicial_ok and primera_aceptada) else "FALLO"
    _registrar(
        caso="Sistema recién iniciado: (c=0,r=0) y primera solicitud aceptada",
        entrada="N=8, R=2, t=0 (sin solicitudes previas)",
        esperado="c=0, r=0 en t=0; primera solicitud (N>0) aceptada=True",
        obtenido=f"c={pool.c - 1 if primera_aceptada else pool.c}, r={pool.r}, primera_aceptada={primera_aceptada}",
        estado=estado,
    )
    assert estado_inicial_ok
    assert primera_aceptada is True
