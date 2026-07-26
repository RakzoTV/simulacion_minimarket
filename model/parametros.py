"""Parámetros centralizados del modelo. Nunca hardcodear estos valores en otros módulos."""

ESCENARIOS = {
    "A": {"servidores": 2, "hilos_por_servidor": 4, "N": 8, "R": 2},
    "B": {"servidores": 4, "hilos_por_servidor": 4, "N": 16, "R": 4},
}

PARAMETROS_ESTOCASTICOS = {
    "lambda_C": 8.0,   # solicitudes/s
    "lambda_R": 1.0,   # solicitudes/s
    "mu_C": 2.0,       # 1 / 0.5s
    "mu_R": 1 / 3,     # 1 / 3.0s
}

PARAMETROS_EJECUCION = {
    "horizonte_principal": 7200,   # segundos (2h)
    "horizonte_jornada": 43200,    # segundos (12h) - experimento secundario
    "calentamiento": 300,          # segundos
    "n_replicas": 30,
    "S_sucursales": 20,
}

# Franja horaria (sección 2.5): multiplicadores de lambda_C y lambda_R
FRANJAS_HORARIAS = {
    "pico": 1.0,
    "valle": 0.35,
}

# Medias de las distribuciones empíricas (para consistencia con el modo markoviano
# y para el cálculo analítico de Markov/Erlang B, que solo necesita la media)
MEDIAS_EMPIRICAS = {
    "SC": 0.5,   # segundos, E[SC] lognormal(mu=-0.7731, sigma=0.40)
    "SR": 3.0,   # segundos, E[SR] weibull(escala=3.3735, forma=1.8)
}
