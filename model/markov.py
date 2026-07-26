"""Cadena de Markov de tiempo continuo para el pool de hilos con reserva de capacidad.

Usado para verificación y validación (secciones 19 y 20 del informe), no para
responder la pregunta principal del proyecto.
"""

import numpy as np


def estados(N, R):
    """Todos los pares (c, r) con 0<=c<=N-R, 0<=r<=N, c+r<=N."""
    return [
        (c, r)
        for c in range(N - R + 1)
        for r in range(N + 1)
        if c + r <= N
    ]


def construir_generador(N, R, lambda_C, lambda_R, mu_C, mu_R):
    """Construye la matriz generadora Q (tasas de transición) de la CTMC.

    Devuelve (lista_estados, Q) donde Q[i,j] es la tasa de i a j (i!=j) y
    Q[i,i] = -suma de tasas salientes de i.
    """
    ests = estados(N, R)
    idx = {e: k for k, e in enumerate(ests)}
    n = len(ests)
    Q = np.zeros((n, n))

    for (c, r) in ests:
        i = idx[(c, r)]
        if c + r < N - R:
            j = idx[(c + 1, r)]
            Q[i, j] += lambda_C
        if c + r < N:
            j = idx[(c, r + 1)]
            Q[i, j] += lambda_R
        if c >= 1:
            j = idx[(c - 1, r)]
            Q[i, j] += c * mu_C
        if r >= 1:
            j = idx[(c, r - 1)]
            Q[i, j] += r * mu_R
        Q[i, i] = -Q[i].sum()

    return ests, Q


def distribucion_estacionaria(ests, Q):
    """Resuelve pi * Q = 0 sujeto a sum(pi) = 1, reemplazando una columna por
    la condición de normalización (procedimiento estándar para CTMC)."""
    n = len(ests)
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return pi


def metricas(ests, pi, N, R):
    """A partir de pi, calcula BC (bloqueo consulta), BR (bloqueo reabastecimiento)
    y U (utilización esperada del pool)."""
    BC = sum(p for (c, r), p in zip(ests, pi) if c + r >= N - R)
    BR = sum(p for (c, r), p in zip(ests, pi) if c + r == N)
    E_c = sum(c * p for (c, r), p in zip(ests, pi))
    E_r = sum(r * p for (c, r), p in zip(ests, pi))
    U = (E_c + E_r) / N
    return {"BC": BC, "BR": BR, "U": U, "E_c": E_c, "E_r": E_r}


def resolver_markov(N, R, lambda_C, lambda_R, mu_C, mu_R):
    """Atajo: construye Q, resuelve pi y devuelve las métricas."""
    ests, Q = construir_generador(N, R, lambda_C, lambda_R, mu_C, mu_R)
    pi = distribucion_estacionaria(ests, Q)
    return metricas(ests, pi, N, R)


def erlang_b(N, a):
    """Fórmula B de Erlang vía recursión (evita desbordar con factoriales para N grande).

    a = carga ofrecida total en Erlangs (lambda_total * E[servicio])
    """
    inv_b = 1.0
    for k in range(1, N + 1):
        inv_b = 1.0 + (k / a) * inv_b
    return 1.0 / inv_b
