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

# Cronograma de la jornada completa de 12h (sección 12.3.5 del informe):
# "la jornada comercial... comprende doce horas, dentro de las cuales se
# identifican cinco bloques horarios de alta actividad -apertura, media
# mañana y mediodía- frente a siete bloques de menor actividad."
# Una tupla (hora_inicio, hora_fin, "pico"|"valle") por cada una de las 12
# horas de la jornada. Apertura = hora 0; media mañana = horas 2-3;
# mediodía = horas 5-6 -> 5 bloques "pico" agrupados, 7 "valle".
TRAMOS_JORNADA_COMPLETA = [
    (0, 1, "pico"),    # apertura
    (1, 2, "valle"),
    (2, 3, "pico"),    # media mañana
    (3, 4, "pico"),    # media mañana
    (4, 5, "valle"),
    (5, 6, "pico"),    # mediodía
    (6, 7, "pico"),    # mediodía
    (7, 8, "valle"),
    (8, 9, "valle"),
    (9, 10, "valle"),
    (10, 11, "valle"),
    (11, 12, "valle"),
]


def franjas_jornada_completa():
    """Convierte TRAMOS_JORNADA_COMPLETA (horas, etiqueta) al formato que espera
    `correr_replica(..., franjas=...)`: [(t_inicio_seg, t_fin_seg, multiplicador), ...]
    """
    return [
        (h_inicio * 3600, h_fin * 3600, FRANJAS_HORARIAS[etiqueta])
        for h_inicio, h_fin, etiqueta in TRAMOS_JORNADA_COMPLETA
    ]


def etiqueta_tramo_jornada(t_segundos):
    """Clasifica un instante (en segundos, dentro de [0, 43200)) como 'pico' o
    'valle' según TRAMOS_JORNADA_COMPLETA. Usado para desglosar métricas por
    franja horaria en el experimento de jornada completa."""
    for h_inicio, h_fin, etiqueta in TRAMOS_JORNADA_COMPLETA:
        if h_inicio * 3600 <= t_segundos < h_fin * 3600:
            return etiqueta
    return "valle"

# Medias de las distribuciones empíricas (para consistencia con el modo markoviano
# y para el cálculo analítico de Markov/Erlang B, que solo necesita la media)
MEDIAS_EMPIRICAS = {
    "SC": 0.5,   # segundos, E[SC] lognormal(mu=-0.7731, sigma=0.40)
    "SR": 3.0,   # segundos, E[SR] weibull(escala=3.3735, forma=1.8)
}
