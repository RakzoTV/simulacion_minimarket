"""Estrategia Monte Carlo estática: probabilidad de saturación del
pool ante una ráfaga instantánea de solicitudes en una ventana delta_t, asumiendo
que ningún hilo se libera durante la ventana (cota superior conservadora).

Independiente del DES y de la cadena de Markov: no avanza el reloj, es un
muestreo combinatorio puntual (peor caso), al estilo Thundering Herd.
"""

import numpy as np


def probabilidad_saturacion_rafaga(lambda_C, lambda_R, N, R, delta_t=0.1, n_trials=100_000, seed=None):
    """
    Calcula, mediante muestreo Monte Carlo estático, la probabilidad de que una ráfaga de
    solicitudes en una ventana delta_t sature el pool de hilos, asumiendo que ningún hilo se
    libera durante esa ventana (cota superior conservadora del riesgo de saturación instantánea).
    """
    rng = np.random.default_rng(seed)

    llegadas_consulta = rng.poisson(lam=lambda_C * delta_t, size=n_trials)
    llegadas_reab = rng.poisson(lam=lambda_R * delta_t, size=n_trials)

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


def barrido_delta_t(lambda_C, lambda_R, N, R, deltas_t=(0.05, 0.1, 0.5), n_trials=20_000, seed=None):
    """Corre probabilidad_saturacion_rafaga para varios valores de delta_t y
    devuelve una lista de dicts (uno por delta_t), lista para volcar a DataFrame."""
    filas = []
    for dt in deltas_t:
        r = probabilidad_saturacion_rafaga(
            lambda_C, lambda_R, N, R, delta_t=dt, n_trials=n_trials, seed=seed
        )
        r["delta_t"] = dt
        filas.append(r)
    return filas
