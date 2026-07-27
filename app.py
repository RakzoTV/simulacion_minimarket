"""Interfaz Streamlit del simulador API Gateway con reserva de capacidad.

Ver sección 6 de SPEC_implementacion_simulacion.md.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.markov import resolver_markov
from model.monte_carlo import barrido_delta_t
from model.parametros import ESCENARIOS, PARAMETROS_ESTOCASTICOS, PARAMETROS_EJECUCION
from model.replicas import correr_replicas, resumen_ic95
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

        st.session_state["resultados"] = {
            "df_todo": df_todo,
            "resumen": resumen_ic95(df_todo),
            "trayectoria": res_trayectoria,
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

tab_comparacion, tab_trayectoria, tab_verificacion, tab_rafagas, tab_datos = st.tabs(
    ["Escenario A vs B", "Trayectoria c(t) / r(t)", "DES vs Markov", "Ráfagas (Monte Carlo)", "Datos y exportar"]
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
