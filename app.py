"""Interfaz Streamlit del simulador API Gateway con reserva de capacidad.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.markov import resolver_markov
from model.monte_carlo import barrido_delta_t
from model.parametros import (
    ESCENARIOS,
    PARAMETROS_ESTOCASTICOS,
    PARAMETROS_EJECUCION,
    TRAMOS_JORNADA_COMPLETA,
)
from model.replicas import correr_replicas, correr_replicas_jornada, resumen_ic95, COLUMNAS_JORNADA
from model.simulation import correr_replica

# --- Paleta (validada para daltonismo/contraste, ver skill dataviz) ---
COLOR_CONSULTA = "#2a78d6"        # azul — slot categórico 1
COLOR_REAB = "#eb6834"            # naranja — slot categórico 2
COLOR_TOTAL = "#898781"           # gris muted — serie derivada (c+r)
COLOR_MARKOV = "#4a3aa7"          # violeta — referencia analítica
COLOR_BUENO = "#0ca30c"
COLOR_ALERTA = "#d03b3b"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"

st.set_page_config(page_title="Simulación API Gateway", layout="wide")

st.title("Simulación API Gateway — Reserva de Capacidad")
st.caption(
    "Pool compartido de **N** hilos, con **R** reservados en exclusiva para "
    "reabastecimiento. Sin colas: toda solicitud que no encuentra hilo libre "
    "se **rechaza de inmediato**."
)

# =============================================================================
# 6.1 — Panel de parámetros (sidebar)
# =============================================================================

with st.sidebar:
    st.header("Parámetros")

    opcion_escenario = st.selectbox(
        "Escenario", ["A", "B", "Personalizado"],
        help="A y B son las configuraciones fijas del informe. 'Personalizado' "
             "permite explorar cualquier combinación de N y R.",
    )

    if opcion_escenario == "Personalizado":
        N = st.number_input("N — hilos totales", min_value=1, value=8, step=1)
        R = st.number_input("R — hilos reservados p/ reabastecimiento", min_value=0, max_value=int(N), value=2, step=1)
    else:
        esc = ESCENARIOS[opcion_escenario]
        N = st.number_input("N — hilos totales", min_value=1, value=esc["N"], step=1)
        R = st.number_input("R — hilos reservados p/ reabastecimiento", min_value=0, max_value=int(N), value=esc["R"], step=1)

    st.caption(f"Capacidad compartida (consulta): **{N - R}** de {N} hilos · Reservado: **{R}**")

    st.divider()
    st.subheader("Tráfico entrante")
    lambda_C = st.number_input("λC — consultas/s", min_value=0.01, value=PARAMETROS_ESTOCASTICOS["lambda_C"])
    lambda_R = st.number_input("λR — reabastecimiento/s", min_value=0.01, value=PARAMETROS_ESTOCASTICOS["lambda_R"])

    st.divider()
    st.subheader("Tiempo de servicio")
    modo_servicio = st.radio(
        "Configuración", ["Empírica", "Markoviana"], horizontal=True,
        help="Empírica: Lognormal (consulta) + Weibull (reabastecimiento), del "
             "relevamiento real. Markoviana: exponenciales — usada para "
             "verificar el motor contra la fórmula de Erlang B.",
    )
    modo_servicio_key = "empirica" if modo_servicio == "Empírica" else "markoviana"

    mu_C = st.number_input("μC — 1/E[SC] (consulta)", min_value=0.001, value=PARAMETROS_ESTOCASTICOS["mu_C"])
    mu_R = st.number_input("μR — 1/E[SR] (reabastecimiento)", min_value=0.001, value=PARAMETROS_ESTOCASTICOS["mu_R"])

    st.divider()
    st.subheader("Ejecución")
    horizonte = st.number_input("Horizonte (s)", min_value=1, value=PARAMETROS_EJECUCION["horizonte_principal"])
    calentamiento = st.number_input(
        "Calentamiento (s)", min_value=0, value=PARAMETROS_EJECUCION["calentamiento"],
        help="Las solicitudes llegadas antes de este instante se descartan de "
             "las métricas, para medir solo el régimen estacionario.",
    )
    n_replicas = st.number_input("Número de réplicas", min_value=1, value=PARAMETROS_EJECUCION["n_replicas"])
    S_sucursales = st.number_input("Sucursales (S)", min_value=1, value=PARAMETROS_EJECUCION["S_sucursales"])
    seed_base = st.number_input(
        "Semilla base", min_value=0, value=0, step=1,
        help="Misma semilla base para A y B → números aleatorios comunes, "
             "reduce la varianza al comparar escenarios.",
    )

    st.divider()
    st.subheader("Experimento secundario")
    correr_jornada = st.checkbox(
        "Correr también jornada completa (12h)",
        help="Corre 30 réplicas adicionales de 12h (43200s) para el escenario "
             "activo, con tasas de llegada moduladas por franjas horarias "
             "pico/valle (apertura, media mañana y mediodía como pico; "
             "5 de 12 horas) en vez de tasa constante. Siempre en configuración "
             "empírica, según la sección 13.4 del informe. Ver tab "
             "'Jornada completa (12h)'.",
    )

    st.divider()
    ejecutar = st.button("Ejecutar simulación", type="primary", use_container_width=True)

if "resultados" not in st.session_state:
    st.session_state["resultados"] = None

if ejecutar:
    with st.spinner("Corriendo réplicas del DES..."):
        df_A = correr_replicas(
            N=ESCENARIOS["A"]["N"], R=ESCENARIOS["A"]["R"],
            lambda_C=lambda_C, lambda_R=lambda_R,
            modo_servicio=modo_servicio_key, horizonte=horizonte,
            n_replicas=n_replicas, S=S_sucursales,
            mu_C=mu_C, mu_R=mu_R, escenario="A", seed_base=seed_base,
            calentamiento=calentamiento,
        )
        df_B = correr_replicas(
            N=ESCENARIOS["B"]["N"], R=ESCENARIOS["B"]["R"],
            lambda_C=lambda_C, lambda_R=lambda_R,
            modo_servicio=modo_servicio_key, horizonte=horizonte,
            n_replicas=n_replicas, S=S_sucursales,
            mu_C=mu_C, mu_R=mu_R, escenario="B", seed_base=seed_base,
            calentamiento=calentamiento,
        )
        df_custom = None
        if opcion_escenario == "Personalizado":
            df_custom = correr_replicas(
                N=N, R=R, lambda_C=lambda_C, lambda_R=lambda_R,
                modo_servicio=modo_servicio_key, horizonte=horizonte,
                n_replicas=n_replicas, S=S_sucursales,
                mu_C=mu_C, mu_R=mu_R, escenario="Personalizado", seed_base=seed_base,
                calentamiento=calentamiento,
            )

        df_todo = pd.concat([d for d in [df_A, df_B, df_custom] if d is not None], ignore_index=True)

        esc_activo = ESCENARIOS.get(opcion_escenario, {"N": N, "R": R})
        res_trayectoria = correr_replica(
            N=esc_activo["N"], R=esc_activo["R"],
            lambda_C=lambda_C, lambda_R=lambda_R,
            modo_servicio=modo_servicio_key, horizonte=min(horizonte, 600),
            S=S_sucursales, mu_C=mu_C, mu_R=mu_R, seed=seed_base,
            registrar_trayectoria=True,
        )

        markov_A = resolver_markov(ESCENARIOS["A"]["N"], ESCENARIOS["A"]["R"], lambda_C, lambda_R, mu_C, mu_R)
        markov_B = resolver_markov(ESCENARIOS["B"]["N"], ESCENARIOS["B"]["R"], lambda_C, lambda_R, mu_C, mu_R)

        # Los resultados de jornada completa se acumulan por escenario en
        # session_state para que correr primero A y después B (o viceversa)
        # no se pisen — cada corrida solo actualiza la entrada de su propio
        # escenario, dejando las demás disponibles para comparar.
        jornadas_previas = (st.session_state.get("resultados") or {}).get("jornadas", {})
        if correr_jornada:
            with st.spinner("Corriendo 30 réplicas de jornada completa (12h)..."):
                df_jornada, trayectoria_jornada = correr_replicas_jornada(
                    N=esc_activo["N"], R=esc_activo["R"],
                    lambda_C=lambda_C, lambda_R=lambda_R,
                    horizonte=PARAMETROS_EJECUCION["horizonte_jornada"],
                    n_replicas=n_replicas, S=S_sucursales,
                    mu_C=mu_C, mu_R=mu_R,
                    escenario=opcion_escenario, seed_base=seed_base,
                    calentamiento=calentamiento,
                )
                jornadas_previas[opcion_escenario] = {
                    "df": df_jornada,
                    "resumen": resumen_ic95(df_jornada, columnas=COLUMNAS_JORNADA),
                    "trayectoria": trayectoria_jornada,
                    "N": esc_activo["N"],
                    "R": esc_activo["R"],
                    "seeds": list(range(seed_base, seed_base + n_replicas)),
                }

        st.session_state["resultados"] = {
            "df_todo": df_todo,
            "resumen": resumen_ic95(df_todo),
            "trayectoria": res_trayectoria,
            "jornadas": jornadas_previas,
            "markov_A": markov_A,
            "markov_B": markov_B,
            "modo_servicio": modo_servicio_key,
            "seeds": list(range(seed_base, seed_base + n_replicas)),
            "escenario_activo": opcion_escenario,
            "N_activo": esc_activo["N"],
            "R_activo": esc_activo["R"],
        }

resultados = st.session_state["resultados"]

# =============================================================================
# Estado vacío
# =============================================================================

if resultados is None:
    st.info("Configura los parámetros en la barra lateral y presiona **Ejecutar simulación** para ver resultados.")
    st.stop()

resumen = resultados["resumen"]


def fila(esc):
    return resumen[resumen["escenario"] == esc].iloc[0] if (resumen["escenario"] == esc).any() else None


# =============================================================================
# 6.2 — KPIs de un vistazo (escenario activo)
# =============================================================================

st.subheader(f"Resumen — Escenario {resultados['escenario_activo']}")

fila_activa = fila(resultados["escenario_activo"]) if resultados["escenario_activo"] != "Personalizado" else fila("Personalizado")

if fila_activa is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Rechazo · Consulta",
        f"{fila_activa['tasa_rechazo_consulta_media']:.1%}",
        help=f"IC 95%: [{fila_activa['tasa_rechazo_consulta_ic95_lo']:.1%}, {fila_activa['tasa_rechazo_consulta_ic95_hi']:.1%}]",
    )
    c2.metric(
        "Rechazo · Reabastecimiento",
        f"{fila_activa['tasa_rechazo_reab_media']:.1%}",
        help=f"IC 95%: [{fila_activa['tasa_rechazo_reab_ic95_lo']:.1%}, {fila_activa['tasa_rechazo_reab_ic95_hi']:.1%}]",
    )
    c3.metric(
        "Utilización del pool",
        f"{fila_activa['utilizacion_media']:.1%}",
        help="Fracción promedio de los N hilos ocupada durante la ventana de medición.",
    )
    c4.metric(
        "Pico de ocupación",
        f"{fila_activa['pico_ocupacion_media']:.0f} / {resultados['N_activo']}",
        help="Máximo de c(t)+r(t) observado en la réplica de ejemplo.",
    )

st.caption(
    f"{int(resultados['df_todo']['replica'].max()) + 1} réplicas · "
    f"semillas {resultados['seeds'][0]}–{resultados['seeds'][-1]} (números aleatorios comunes) · "
    f"servicio **{resultados['modo_servicio']}**"
)

st.divider()

# =============================================================================
# Tabs de contenido
# =============================================================================

tab_comparacion, tab_trayectoria, tab_jornada, tab_verificacion, tab_rafagas, tab_datos = st.tabs(
    ["Escenario A vs B", "Trayectoria c(t) / r(t)", "Jornada completa (12h)", "DES vs Markov", "Ráfagas (Monte Carlo)", "Datos y exportar"]
)

# --- Tab: comparación A vs B ---
with tab_comparacion:
    st.markdown(
        "Comparación de la **tasa de rechazo** entre escenarios, con barras de "
        "error correspondientes al **intervalo de confianza del 95%** sobre las réplicas."
    )

    fig_barras = go.Figure()
    for col, nombre, color in [
        ("tasa_rechazo_consulta_media", "Consulta", COLOR_CONSULTA),
        ("tasa_rechazo_reab_media", "Reabastecimiento", COLOR_REAB),
    ]:
        fig_barras.add_trace(go.Bar(
            x=resumen["escenario"], y=resumen[col], name=nombre,
            marker_color=color,
            error_y=dict(
                type="data", color="rgba(0,0,0,0.35)", thickness=1.5, width=4,
                array=resumen[col.replace("_media", "_ic95_hi")] - resumen[col],
                arrayminus=resumen[col] - resumen[col.replace("_media", "_ic95_lo")],
            ),
            hovertemplate="%{x} · " + nombre + "<br>Tasa de rechazo: %{y:.2%}<extra></extra>",
        ))
    fig_barras.update_layout(
        barmode="group",
        yaxis=dict(title="Tasa de rechazo", tickformat=".0%", gridcolor=COLOR_GRID, zerolinecolor=COLOR_AXIS),
        xaxis=dict(title="Escenario", gridcolor=COLOR_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig_barras, use_container_width=True)

    col_a, col_b = st.columns(2)
    for col, nombre in [(col_a, "A"), (col_b, "B")]:
        f = fila(nombre)
        if f is not None:
            with col:
                st.markdown(f"**Escenario {nombre}** — N={ESCENARIOS[nombre]['N']}, R={ESCENARIOS[nombre]['R']}")
                st.markdown(
                    f"- Rechazo consulta: **{f['tasa_rechazo_consulta_media']:.2%}** "
                    f"[{f['tasa_rechazo_consulta_ic95_lo']:.2%}, {f['tasa_rechazo_consulta_ic95_hi']:.2%}]\n"
                    f"- Rechazo reabastecimiento: **{f['tasa_rechazo_reab_media']:.2%}** "
                    f"[{f['tasa_rechazo_reab_ic95_lo']:.2%}, {f['tasa_rechazo_reab_ic95_hi']:.2%}]\n"
                    f"- Utilización: **{f['utilizacion_media']:.2%}**"
                )

# --- Tab: trayectoria temporal ---
with tab_trayectoria:
    st.markdown(
        "Ocupación del pool a lo largo del tiempo para **una réplica de ejemplo** "
        f"del escenario activo (N={resultados['N_activo']}, R={resultados['R_activo']}). "
        "La línea punteada gris marca el límite de capacidad total; la banda "
        "sombreada es la **región reservada** — solo accesible a reabastecimiento."
    )
    trayectoria = resultados["trayectoria"].trayectoria
    if trayectoria:
        t = [p[0] for p in trayectoria]
        c = np.array([p[1] for p in trayectoria])
        r = np.array([p[2] for p in trayectoria])
        N_act, R_act = resultados["N_activo"], resultados["R_activo"]

        fig_tray = go.Figure()
        fig_tray.add_hrect(
            y0=N_act - R_act, y1=N_act, fillcolor=COLOR_REAB, opacity=0.08, line_width=0,
            annotation_text="región reservada (R)", annotation_position="top left",
            annotation_font_color=COLOR_REAB, annotation_font_size=11,
        )
        fig_tray.add_trace(go.Scatter(
            x=t, y=c, name="c(t) — consultas ocupadas", mode="lines",
            line=dict(color=COLOR_CONSULTA, width=2),
        ))
        fig_tray.add_trace(go.Scatter(
            x=t, y=r, name="r(t) — reabastecimiento ocupado", mode="lines",
            line=dict(color=COLOR_REAB, width=2),
        ))
        fig_tray.add_trace(go.Scatter(
            x=t, y=c + r, name="c(t) + r(t) — total ocupado", mode="lines",
            line=dict(color=COLOR_TOTAL, width=1.5, dash="dot"),
        ))
        fig_tray.add_hline(y=N_act, line_dash="dash", line_color=COLOR_AXIS, line_width=1,
                            annotation_text=f"N = {N_act}", annotation_font_size=11)
        fig_tray.update_layout(
            xaxis=dict(title="tiempo (s)", gridcolor=COLOR_GRID),
            yaxis=dict(title="hilos ocupados", gridcolor=COLOR_GRID, zerolinecolor=COLOR_AXIS,
                       range=[0, N_act + 0.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig_tray, use_container_width=True)
    else:
        st.warning("No se registró trayectoria para esta corrida.")

# --- Tab: jornada completa (12h, franjas pico/valle) ---
with tab_jornada:
    st.markdown(
        "Experimento secundario (sección 2.5 de la especificación, sección 13.4 "
        "del informe): jornada comercial completa de 12h, con la tasa de llegada "
        "modulada por franjas horarias en vez de ser constante. Apertura, media "
        "mañana y mediodía son bloques de **pico** (multiplicador ×1.0); el "
        "resto son bloques de **valle** (×0.35) — 5 de las 12 horas en pico, "
        "7 en valle. Corre siempre en configuración **empírica**, con 30 "
        "réplicas y las mismas semillas comunes que el experimento principal."
    )

    jornadas = resultados.get("jornadas", {})
    if not jornadas:
        st.info(
            "Marcá **'Correr también jornada completa (12h)'** en la barra "
            "lateral y volvé a ejecutar la simulación para ver estos resultados."
        )
    else:
        escenarios_jornada = list(jornadas.keys())
        esc_jornada_sel = (
            st.radio("Escenario", escenarios_jornada, horizontal=True)
            if len(escenarios_jornada) > 1 else escenarios_jornada[0]
        )
        datos_j = jornadas[esc_jornada_sel]
        resumen_j = datos_j["resumen"].iloc[0]
        N_act, R_act = datos_j["N"], datos_j["R"]

        st.markdown(f"**Escenario {esc_jornada_sel}** — N={N_act}, R={R_act} · "
                     f"{len(datos_j['df'])} réplicas · semillas "
                     f"{datos_j['seeds'][0]}–{datos_j['seeds'][-1]}")

        k1, k2 = st.columns(2)
        k1.metric(
            "Rechazo consulta — horas pico",
            f"{resumen_j['tasa_rechazo_consulta_pico_media']:.1%}",
            help=f"IC 95%: [{resumen_j['tasa_rechazo_consulta_pico_ic95_lo']:.1%}, "
                 f"{resumen_j['tasa_rechazo_consulta_pico_ic95_hi']:.1%}]",
        )
        k2.metric(
            "Rechazo consulta — horas valle",
            f"{resumen_j['tasa_rechazo_consulta_valle_media']:.1%}",
            help=f"IC 95%: [{resumen_j['tasa_rechazo_consulta_valle_ic95_lo']:.1%}, "
                 f"{resumen_j['tasa_rechazo_consulta_valle_ic95_hi']:.1%}]",
        )
        k3, k4 = st.columns(2)
        k3.metric(
            "Rechazo reabastecimiento — horas pico",
            f"{resumen_j['tasa_rechazo_reab_pico_media']:.1%}",
            help=f"IC 95%: [{resumen_j['tasa_rechazo_reab_pico_ic95_lo']:.1%}, "
                 f"{resumen_j['tasa_rechazo_reab_pico_ic95_hi']:.1%}]",
        )
        k4.metric(
            "Rechazo reabastecimiento — horas valle",
            f"{resumen_j['tasa_rechazo_reab_valle_media']:.1%}",
            help=f"IC 95%: [{resumen_j['tasa_rechazo_reab_valle_ic95_lo']:.1%}, "
                 f"{resumen_j['tasa_rechazo_reab_valle_ic95_hi']:.1%}]",
        )
        st.caption(
            f"Agregado de las 12h — Rechazo consulta: "
            f"**{resumen_j['tasa_rechazo_consulta_total_media']:.1%}** · "
            f"Rechazo reabastecimiento: "
            f"**{resumen_j['tasa_rechazo_reab_total_media']:.1%}**"
        )

        trayectoria_j = datos_j["trayectoria"]
        if trayectoria_j:
            t_j = [p[0] for p in trayectoria_j]
            c_j = np.array([p[1] for p in trayectoria_j])
            r_j = np.array([p[2] for p in trayectoria_j])

            fig_jornada = go.Figure()
            for h_inicio, h_fin, etiqueta in TRAMOS_JORNADA_COMPLETA:
                if etiqueta == "pico":
                    fig_jornada.add_vrect(
                        x0=h_inicio * 3600, x1=h_fin * 3600,
                        fillcolor=COLOR_REAB, opacity=0.06, line_width=0,
                    )
            fig_jornada.add_trace(go.Scatter(
                x=t_j, y=c_j, name="c(t) — consultas ocupadas", mode="lines",
                line=dict(color=COLOR_CONSULTA, width=1.5),
            ))
            fig_jornada.add_trace(go.Scatter(
                x=t_j, y=r_j, name="r(t) — reabastecimiento ocupado", mode="lines",
                line=dict(color=COLOR_REAB, width=1.5),
            ))
            fig_jornada.add_hline(y=N_act, line_dash="dash", line_color=COLOR_AXIS, line_width=1,
                                   annotation_text=f"N = {N_act}", annotation_font_size=11)
            fig_jornada.update_layout(
                xaxis=dict(
                    title="tiempo (h)", gridcolor=COLOR_GRID,
                    tickvals=[h * 3600 for h in range(0, 13)],
                    ticktext=[str(h) for h in range(0, 13)],
                ),
                yaxis=dict(title="hilos ocupados", gridcolor=COLOR_GRID, zerolinecolor=COLOR_AXIS,
                           range=[0, N_act + 0.5]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=60, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_jornada, use_container_width=True)
            st.caption(
                "Trayectoria de la primera réplica, como ejemplo. Las bandas "
                "sombreadas en naranja marcan los bloques de tráfico pico."
            )
        else:
            st.warning("No se registró trayectoria para la corrida de jornada completa.")

        st.divider()
        st.markdown("**Detalle por réplica** (desglose pico/valle):")
        st.dataframe(datos_j["df"], use_container_width=True, hide_index=True)

        col_j1, col_j2 = st.columns(2)
        with col_j1:
            st.download_button(
                f"Descargar réplicas jornada — Escenario {esc_jornada_sel} (CSV)",
                datos_j["df"].to_csv(index=False).encode("utf-8"),
                file_name=f"jornada_replicas_{esc_jornada_sel}.csv", mime="text/csv",
                use_container_width=True,
            )
        with col_j2:
            st.download_button(
                f"Descargar resumen IC 95% jornada — Escenario {esc_jornada_sel} (CSV)",
                datos_j["resumen"].to_csv(index=False).encode("utf-8"),
                file_name=f"jornada_resumen_ic95_{esc_jornada_sel}.csv", mime="text/csv",
                use_container_width=True,
            )

# --- Tab: verificación DES vs Markov ---
with tab_verificacion:
    st.markdown(
        "Contraste entre la simulación (DES) y la solución **analítica exacta** "
        "de la cadena de Markov de tiempo continuo — mismo modelo, sin ruido de "
        "muestreo. Valores cercanos son evidencia de que el motor DES está "
        "correctamente implementado."
    )
    filas_markov = []
    for nombre, m in [("A", resultados["markov_A"]), ("B", resultados["markov_B"])]:
        f = fila(nombre)
        if f is not None:
            filas_markov.append({
                "Escenario": nombre,
                "BC (Markov)": m["BC"], "BC (DES)": f["tasa_rechazo_consulta_media"],
                "Δ BC": abs(m["BC"] - f["tasa_rechazo_consulta_media"]),
                "BR (Markov)": m["BR"], "BR (DES)": f["tasa_rechazo_reab_media"],
                "Δ BR": abs(m["BR"] - f["tasa_rechazo_reab_media"]),
                "U (Markov)": m["U"], "U (DES)": f["utilizacion_media"],
            })
    if filas_markov:
        df_v = pd.DataFrame(filas_markov)
        st.dataframe(
            df_v.style.format({c: "{:.2%}" if c != "Escenario" else "{}" for c in df_v.columns})
            .background_gradient(subset=["Δ BC", "Δ BR"], cmap="RdYlGn_r", vmin=0, vmax=0.05),
            use_container_width=True, hide_index=True,
        )

    if ESCENARIOS["A"]["R"] == 0 or ESCENARIOS["B"]["R"] == 0:
        st.info(
            "ℹ️ Con **R = 0**, por el teorema de Erlang-Sevastyanov, BC y BR deberían "
            "converger al **mismo** valor teórico de Erlang B, sin importar la forma "
            "de las distribuciones de servicio."
        )

# --- Tab: Monte Carlo ---
with tab_rafagas:
    st.markdown(
        "**Cota superior conservadora** de saturación instantánea: probabilidad de "
        "que una ráfaga de solicitudes en una ventana Δt sature el pool, asumiendo "
        "que **ningún hilo se libera** durante esa ventana. Es un análisis de peor "
        "caso puntual (ej. una promoción o reinicio simultáneo de sucursales), no "
        "una trayectoria temporal."
    )
    esc_mc = ESCENARIOS.get(opcion_escenario, {"N": N, "R": R}) if resultados["escenario_activo"] == "Personalizado" else ESCENARIOS.get(resultados["escenario_activo"], {"N": resultados["N_activo"], "R": resultados["R_activo"]})
    filas_mc = barrido_delta_t(lambda_C, lambda_R, esc_mc["N"], esc_mc["R"], deltas_t=(0.05, 0.1, 0.5), n_trials=20_000, seed=seed_base)
    df_mc = pd.DataFrame(filas_mc)[["delta_t", "prob_rechazo_consulta_en_rafaga", "prob_rechazo_reab_en_rafaga", "prob_alguna_saturacion"]]
    df_mc.columns = ["Δt (s)", "P(rechazo consulta)", "P(rechazo reabastecimiento)", "P(alguna saturación)"]

    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=df_mc["Δt (s)"], y=df_mc["P(rechazo consulta)"], name="Consulta",
        mode="lines+markers", line=dict(color=COLOR_CONSULTA, width=2), marker=dict(size=9),
    ))
    fig_mc.add_trace(go.Scatter(
        x=df_mc["Δt (s)"], y=df_mc["P(rechazo reabastecimiento)"], name="Reabastecimiento",
        mode="lines+markers", line=dict(color=COLOR_REAB, width=2), marker=dict(size=9),
    ))
    fig_mc.update_layout(
        xaxis=dict(title="Δt — ventana de ráfaga (s)", gridcolor=COLOR_GRID),
        yaxis=dict(title="Probabilidad de saturación", tickformat=".0%", gridcolor=COLOR_GRID, zerolinecolor=COLOR_AXIS),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mc, use_container_width=True)
    st.dataframe(
        df_mc.style.format({c: "{:.2%}" for c in df_mc.columns if c != "Δt (s)"}),
        use_container_width=True, hide_index=True,
    )
    st.caption("Cuanto mayor Δt, más tiempo conservador se asume sin liberación de hilos — supuesto de peor caso, aclarar en el informe.")

# --- Tab: datos crudos y export ---
with tab_datos:
    st.markdown("**Tabla resumen** (media, desvío estándar e IC 95%) por escenario:")
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.markdown("**Detalle por réplica:**")
    st.dataframe(resultados["df_todo"], use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Descargar réplicas (CSV)",
            resultados["df_todo"].to_csv(index=False).encode("utf-8"),
            file_name="replicas.csv", mime="text/csv", use_container_width=True,
        )
    with col2:
        st.download_button(
            "Descargar resumen IC 95% (CSV)",
            resumen.to_csv(index=False).encode("utf-8"),
            file_name="resumen_ic95.csv", mime="text/csv", use_container_width=True,
        )
