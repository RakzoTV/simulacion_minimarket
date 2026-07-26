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

st.set_page_config(page_title="Simulación API Gateway", layout="wide")
st.title("Simulación API Gateway — Reserva de Capacidad")

# --- 6.1 Panel de parámetros (sidebar) ---

st.sidebar.header("Parámetros")

opcion_escenario = st.sidebar.selectbox("Escenario", ["A", "B", "Personalizado"])

if opcion_escenario == "Personalizado":
    N = st.sidebar.number_input("N (hilos totales)", min_value=1, value=8, step=1)
    R = st.sidebar.number_input("R (hilos reservados)", min_value=0, max_value=int(N), value=2, step=1)
else:
    esc = ESCENARIOS[opcion_escenario]
    N = st.sidebar.number_input("N (hilos totales)", min_value=1, value=esc["N"], step=1)
    R = st.sidebar.number_input("R (hilos reservados)", min_value=0, max_value=int(N), value=esc["R"], step=1)

lambda_C = st.sidebar.number_input("λC (consultas/s)", min_value=0.01, value=PARAMETROS_ESTOCASTICOS["lambda_C"])
lambda_R = st.sidebar.number_input("λR (reabastecimiento/s)", min_value=0.01, value=PARAMETROS_ESTOCASTICOS["lambda_R"])

modo_servicio = st.sidebar.selectbox("Configuración de servicio", ["Empírica", "Markoviana"])
modo_servicio_key = "empirica" if modo_servicio == "Empírica" else "markoviana"

mu_C = st.sidebar.number_input("μC (1/E[SC])", min_value=0.001, value=PARAMETROS_ESTOCASTICOS["mu_C"])
mu_R = st.sidebar.number_input("μR (1/E[SR])", min_value=0.001, value=PARAMETROS_ESTOCASTICOS["mu_R"])

horizonte = st.sidebar.number_input("Horizonte (s)", min_value=1, value=PARAMETROS_EJECUCION["horizonte_principal"])
calentamiento = st.sidebar.number_input("Calentamiento (s)", min_value=0, value=PARAMETROS_EJECUCION["calentamiento"])
n_replicas = st.sidebar.number_input("Número de réplicas", min_value=1, value=PARAMETROS_EJECUCION["n_replicas"])
S_sucursales = st.sidebar.number_input("Sucursales (S)", min_value=1, value=PARAMETROS_EJECUCION["S_sucursales"])
seed_base = st.sidebar.number_input("Semilla base", min_value=0, value=0, step=1)

ejecutar = st.sidebar.button("Ejecutar simulación", type="primary")

if "resultados" not in st.session_state:
    st.session_state["resultados"] = None

if ejecutar:
    with st.spinner("Corriendo réplicas..."):
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

        # réplica de ejemplo con trayectoria, para el escenario seleccionado
        esc_activo = ESCENARIOS.get(opcion_escenario, {"N": N, "R": R})
        res_trayectoria = correr_replica(
            N=esc_activo["N"], R=esc_activo["R"],
            lambda_C=lambda_C, lambda_R=lambda_R,
            modo_servicio=modo_servicio_key, horizonte=min(horizonte, 600),
            S=S_sucursales, mu_C=mu_C, mu_R=mu_R, seed=seed_base,
            registrar_trayectoria=True,
        )

        # Markov exacto (solo tiene sentido comparar en la config markoviana)
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
        }

resultados = st.session_state["resultados"]

if resultados is None:
    st.info("Configurá los parámetros en la barra lateral y presioná 'Ejecutar simulación'.")
else:
    resumen = resultados["resumen"]

    st.subheader("Tabla resumen (IC 95%)")
    st.dataframe(resumen, use_container_width=True)

    st.subheader("Escenario A vs B — tasa de rechazo")
    fig_barras = go.Figure()
    for col, nombre in [("tasa_rechazo_consulta_media", "Consulta"), ("tasa_rechazo_reab_media", "Reabastecimiento")]:
        fig_barras.add_trace(go.Bar(
            x=resumen["escenario"], y=resumen[col], name=nombre,
            error_y=dict(
                type="data",
                array=resumen[col.replace("_media", "_ic95_hi")] - resumen[col],
                arrayminus=resumen[col] - resumen[col.replace("_media", "_ic95_lo")],
            ),
        ))
    fig_barras.update_layout(barmode="group", yaxis_title="Tasa de rechazo", xaxis_title="Escenario")
    st.plotly_chart(fig_barras, use_container_width=True)

    st.subheader("Evolución temporal de c(t) y r(t) — réplica de ejemplo")
    trayectoria = resultados["trayectoria"].trayectoria
    if trayectoria:
        t = [p[0] for p in trayectoria]
        c = [p[1] for p in trayectoria]
        r = [p[2] for p in trayectoria]
        fig_tray = go.Figure()
        fig_tray.add_trace(go.Scatter(x=t, y=c, name="c(t) — consultas ocupadas", mode="lines"))
        fig_tray.add_trace(go.Scatter(x=t, y=r, name="r(t) — reabastecimiento ocupado", mode="lines"))
        fig_tray.add_trace(go.Scatter(x=t, y=np.array(c) + np.array(r), name="c(t)+r(t)", mode="lines", line=dict(dash="dot")))
        fig_tray.update_layout(xaxis_title="tiempo (s)", yaxis_title="hilos ocupados")
        st.plotly_chart(fig_tray, use_container_width=True)
    else:
        st.warning("No se registró trayectoria para esta corrida.")

    st.subheader("Comparación DES vs Markov exacto")
    filas_markov = []
    for nombre, m in [("A", resultados["markov_A"]), ("B", resultados["markov_B"])]:
        fila_des = resumen[resumen["escenario"] == nombre]
        if not fila_des.empty:
            filas_markov.append({
                "escenario": nombre,
                "BC_markov": m["BC"], "BC_des": fila_des["tasa_rechazo_consulta_media"].values[0],
                "BR_markov": m["BR"], "BR_des": fila_des["tasa_rechazo_reab_media"].values[0],
                "U_markov": m["U"], "U_des": fila_des["utilizacion_media"].values[0],
            })
    if filas_markov:
        st.dataframe(pd.DataFrame(filas_markov), use_container_width=True)

    if ESCENARIOS["A"]["R"] == 0 or ESCENARIOS["B"]["R"] == 0:
        st.caption("Con R=0, BC y BR deberían converger al mismo valor de Erlang B (Erlang-Sevastyanov).")

    st.subheader("Monte Carlo — saturación en ráfaga")
    esc_mc = ESCENARIOS.get(opcion_escenario, {"N": N, "R": R})
    filas_mc = barrido_delta_t(lambda_C, lambda_R, esc_mc["N"], esc_mc["R"], deltas_t=(0.05, 0.1, 0.5), n_trials=20_000, seed=seed_base)
    df_mc = pd.DataFrame(filas_mc)
    st.dataframe(df_mc, use_container_width=True)

    st.subheader("Exportar")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Descargar réplicas (CSV)",
            resultados["df_todo"].to_csv(index=False).encode("utf-8"),
            file_name="replicas.csv", mime="text/csv",
        )
    with col2:
        st.download_button(
            "Descargar resumen (CSV)",
            resumen.to_csv(index=False).encode("utf-8"),
            file_name="resumen_ic95.csv", mime="text/csv",
        )
    st.caption(f"Semillas usadas: {resultados['seeds'][0]}..{resultados['seeds'][-1]} (números aleatorios comunes entre escenarios)")
