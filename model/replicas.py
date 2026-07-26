"""Ejecución de múltiples réplicas del DES con estadística (IC 95%).

Usa semillas explícitas por réplica (sección 6.3) para permitir la técnica de
números aleatorios comunes: la réplica i del Escenario A usa la misma semilla
que la réplica i del Escenario B, reduciendo varianza en la comparación.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from model.simulation import correr_replica


@dataclass
class MetricasReplica:
    escenario: str
    configuracion: str
    replica: int
    seed: int
    tasa_rechazo_consulta: float
    tasa_rechazo_reab: float
    utilizacion: float
    pico_ocupacion: int
    n_consultas: int
    n_reab: int


def _metricas_de_replica(res, escenario, configuracion, replica, seed):
    calentamiento = res.calentamiento
    duracion_medicion = res.horizonte - calentamiento

    # Se descartan las solicitudes llegadas durante el calentamiento: solo
    # interesa el comportamiento en régimen estacionario.
    registro = [s for s in res.registro if s.momento_llegada >= calentamiento]
    cons = [s for s in registro if s.tipo == "consulta"]
    reab = [s for s in registro if s.tipo == "reabastecimiento"]

    tasa_c = sum(1 for s in cons if s.estado == "rechazada") / len(cons) if cons else float("nan")
    tasa_r = sum(1 for s in reab if s.estado == "rechazada") / len(reab) if reab else float("nan")

    atendidas = [s for s in registro if s.estado == "atendida"]
    tiempo_ocupado = sum(
        min(s.momento_fin, res.horizonte) - s.momento_llegada for s in atendidas
    )
    utilizacion = (
        tiempo_ocupado / (res.N * duracion_medicion) if duracion_medicion > 0 else float("nan")
    )

    pico = 0
    if res.trayectoria:
        pico = max(
            (c + r for t, c, r in res.trayectoria if t >= calentamiento),
            default=0,
        )

    return MetricasReplica(
        escenario=escenario,
        configuracion=configuracion,
        replica=replica,
        seed=seed,
        tasa_rechazo_consulta=tasa_c,
        tasa_rechazo_reab=tasa_r,
        utilizacion=utilizacion,
        pico_ocupacion=pico,
        n_consultas=len(cons),
        n_reab=len(reab),
    )


def correr_replicas(
    N,
    R,
    lambda_C,
    lambda_R,
    modo_servicio,
    horizonte,
    n_replicas=30,
    S=20,
    mu_C=None,
    mu_R=None,
    escenario="custom",
    seed_base=0,
    registrar_trayectoria=True,
    calentamiento=0.0,
):
    """Corre n_replicas réplicas independientes con semillas seed_base..seed_base+n-1.

    Las métricas de cada réplica excluyen las solicitudes llegadas antes de
    `calentamiento` segundos (régimen transitorio).

    Devuelve un DataFrame con una fila por réplica (ver MetricasReplica).
    """
    filas = []
    for i in range(n_replicas):
        seed = seed_base + i
        res = correr_replica(
            N=N, R=R, lambda_C=lambda_C, lambda_R=lambda_R,
            modo_servicio=modo_servicio, horizonte=horizonte, S=S,
            mu_C=mu_C, mu_R=mu_R, seed=seed,
            registrar_trayectoria=registrar_trayectoria,
            calentamiento=calentamiento,
        )
        filas.append(_metricas_de_replica(res, escenario, modo_servicio, i, seed))

    df = pd.DataFrame([vars(f) for f in filas])
    return df


def resumen_ic95(df, columnas=("tasa_rechazo_consulta", "tasa_rechazo_reab", "utilizacion", "pico_ocupacion")):
    """Media, desviación estándar e IC 95% (t de Student) por columna, agrupado
    por escenario y configuración."""
    filas = []
    for (escenario, config), grupo in df.groupby(["escenario", "configuracion"]):
        fila = {"escenario": escenario, "configuracion": config, "n_replicas": len(grupo)}
        for col in columnas:
            x = grupo[col].dropna().to_numpy()
            m = x.mean()
            sd = x.std(ddof=1) if len(x) > 1 else 0.0
            se = sd / np.sqrt(len(x)) if len(x) > 1 else 0.0
            h = se * stats.t.ppf(0.975, len(x) - 1) if len(x) > 1 else 0.0
            fila[f"{col}_media"] = m
            fila[f"{col}_sd"] = sd
            fila[f"{col}_ic95_lo"] = m - h
            fila[f"{col}_ic95_hi"] = m + h
        filas.append(fila)
    return pd.DataFrame(filas)
