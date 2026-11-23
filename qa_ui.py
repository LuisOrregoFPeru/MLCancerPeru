"""
Interfaz Streamlit de la sección "💬 Pregunta a los datos": controles en la
barra lateral y resultado en el panel principal.

La lógica de interpretación y cálculo vive en `qa_engine.py`; aquí solo
se dibujan los controles, se gestiona el estado y se muestra la respuesta.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qa_engine import QAAnswer, QAContext, answer_question, ask_ai, build_ai_context

DEFAULT_AI_MODEL = "claude-sonnet-5-5"

EXAMPLES = [
    "¿Cuántos casos de cáncer de mama hubo en 2024?",
    "Top 10 localizaciones en Lima en 2024",
    "Evolución anual del cáncer de cuello uterino en Perú",
    "¿Qué departamentos tienen más casos de estómago en 2023?",
    "¿Hay tendencia significativa en cáncer de próstata desde 2010?",
    "Variación de casos de pulmón entre 2019 y 2024",
    "Evolución del devengado del INEN",
]

_QA_CSS = """
<style>
.st-key-qa_examples button {
    padding: 0.05rem 0 !important; min-height: 0 !important; justify-content: flex-start;
}
.st-key-qa_examples button p {
    font-style: italic; color: #2f6fa8; font-size: 0.8rem; text-align: left; line-height: 1.25;
}
.st-key-qa_examples button:hover p { text-decoration: underline; }
.qa-answer {
    background: #f7f7fb; border: 1px solid #e6e6ef; border-left: 4px solid #8b1a4f;
    border-radius: 8px; padding: 0.45rem 0.7rem; margin: 0.4rem 0;
}
.qa-answer div { color: #1f2d5c; font-size: 0.85rem; font-weight: 600; }
.st-key-qa_answer_box {
    background: #f4f6fb; border-left: 4px solid #1f2d5c; border-radius: 4px;
    padding: 0.55rem 0.9rem;
}
.st-key-qa_answer_box p { font-size: 0.92rem; margin-bottom: 0.2rem; color: #1f2d5c; }
.qa-main-title { font-size: 1.15rem; font-weight: 700; color: #1f2d5c; margin: 0.15rem 0 0 0; }
.qa-separator { border: none; border-top: 2px solid #d9dbe6; margin: 1.1rem 0 1.2rem 0; }
</style>
"""

_PALETTE = ["#8b1a4f", "#2f6fa8", "#e07a2c", "#3ba776", "#8850c4", "#c94141", "#4aa8c9"]


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001  (sin secrets.toml)
        return None


def _set_question(q: str) -> None:
    st.session_state["qa_question"] = q
    st.session_state["qa_run"] = True


def _clear() -> None:
    st.session_state["qa_question"] = ""
    st.session_state.pop("qa_result", None)
    st.session_state["qa_run"] = False


def _chart(ans: QAAnswer) -> go.Figure | None:
    ch = ans.chart
    if not ch:
        return None
    fig = go.Figure()
    if ch["kind"] == "barh" and ans.table is not None:
        t = ans.table.iloc[::-1]
        fig.add_bar(x=t[ch["x"]], y=t[ch["y"]].astype(str), orientation="h", marker_color="#8b1a4f",
                    hovertemplate="%{y}: %{x:,.0f}<extra></extra>")
        height = 120 + 22 * len(t)
    elif ch["kind"] == "line_wide" and ch.get("data") is not None:
        data: pd.DataFrame = ch["data"]
        for i, col in enumerate(data.columns):
            s = data[col].dropna()
            fig.add_scatter(x=s.index, y=s.values, mode="lines+markers", name=str(col),
                            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2), marker=dict(size=4),
                            hovertemplate="%{x}: %{y:,.0f}<extra>" + str(col) + "</extra>")
        height = 260
    else:
        return None
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=8, b=4), showlegend=len(fig.data) > 1,
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        plot_bgcolor="white", font=dict(size=10),
        xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
    )
    return fig


def render_qa_section(df: pd.DataFrame, ctx: QAContext, ctx_no_filters: QAContext) -> None:
    """Dibuja el expander completo. `ctx` refleja los filtros del panel;
    `ctx_no_filters` usa los valores neutros (Perú, todas, todo el rango)."""
    st.markdown(_QA_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("qa_question", "")

    with st.expander("💬 Pregunta a los datos", expanded=True):
        use_filters = st.toggle(
            "Aplicar los filtros del panel", value=True, key="qa_use_filters",
            help=(
                "Si está activo, lo que la pregunta no especifique (localización, departamento, "
                "años, exclusiones del total nacional) se toma del panel de control. "
                "Lo que sí mencione la pregunta siempre tiene prioridad."
            ),
        )
        st.text_area(
            "Pregunta", key="qa_question", label_visibility="collapsed", height=80,
            placeholder="Ej.: ¿cuántos casos de cáncer de mama hubo en Lima en 2024?",
        )
        c1, c2 = st.columns(2)
        run = c1.button("Consultar", type="primary", use_container_width=True, key="qa_btn_run")
        c2.button("Limpiar", use_container_width=True, key="qa_btn_clear", on_click=_clear)

        if st.session_state.get("qa_run"):
            run = True
            st.session_state["qa_run"] = False

        # --- Modo IA (popover) -----------------------------------------
        server_key = _secret("ANTHROPIC_API_KEY")
        model = _secret("ANTHROPIC_MODEL") or DEFAULT_AI_MODEL
        ai_on = bool(st.session_state.get("qa_ai_on"))
        user_key = st.session_state.get("qa_api_key") or ""
        api_key = server_key or user_key

        if run and st.session_state["qa_question"].strip():
            q = st.session_state["qa_question"].strip()
            active_ctx = ctx if use_filters else ctx_no_filters
            with st.spinner("Calculando..."):
                ans = answer_question(q, df, active_ctx)
            ai_text, ai_err = None, None
            if ai_on and api_key:
                try:
                    with st.spinner("Consultando al modelo..."):
                        ai_text = ask_ai(q, build_ai_context(df, active_ctx, ans), api_key, model)
                except Exception as exc:  # noqa: BLE001
                    ai_err = str(exc)
            st.session_state["qa_result"] = {
                "q": q, "ans": ans, "ai": ai_text, "ai_err": ai_err,
                "filters": use_filters, "model": model if ai_text else None,
            }
        elif run:
            st.warning("Escribe una pregunta o elige un ejemplo.")

        if st.session_state.get("qa_result"):
            st.caption("✅ Resultado mostrado en el panel principal.")

        st.caption("Ejemplos de preguntas (tócalos para consultar)")
        with st.container(key="qa_examples"):
            for i, ex in enumerate(EXAMPLES):
                st.button(ex, key=f"qa_ex_{i}", type="tertiary", on_click=_set_question, args=(ex,))

        if ai_on and api_key:
            st.caption(
                f"✨ Modo IA activo ({model}). Las cifras salen de agregaciones sobre los datos del "
                "dashboard; al modelo solo se envían tablas resumidas, sin identificadores."
            )
        else:
            st.caption(
                "🌙 Modo básico por reglas. Las cifras salen de agregaciones sobre los datos del "
                "dashboard; al modelo solo se envían tablas resumidas, sin identificadores."
            )

        with st.popover("Activar modo IA", use_container_width=True):
            st.toggle("Usar modo IA", key="qa_ai_on",
                      help="Redacta la respuesta con un modelo de Anthropic a partir de tablas agregadas.")
            if server_key:
                st.caption("🔑 Clave de API configurada en el servidor.")
            else:
                st.text_input(
                    "Clave de API de Anthropic", type="password", key="qa_api_key",
                    help="Solo se usa en esta sesión y no se guarda. En Streamlit Cloud puede "
                         "configurarse como ANTHROPIC_API_KEY en Secrets.",
                )
                if st.session_state.get("qa_ai_on") and not st.session_state.get("qa_api_key"):
                    st.info("Ingresa una clave para activar el modo IA.")
            st.caption(
                "Se envían: totales anuales nacionales, top 15 localizaciones, casos por departamento, "
                "la serie seleccionada en el panel y el resultado calculado. Nunca registros individuales."
            )


_INTENT_LABELS = {
    "total": "Total de casos",
    "ranking_localizaciones": "Ranking de localizaciones",
    "ranking_departamentos": "Ranking de departamentos",
    "evolucion": "Evolución anual",
    "variacion": "Variación entre años",
    "pico": "Año pico / mínimo",
    "proporcion": "Proporción del total",
    "tendencia": "Tendencia (Mann-Kendall + APC)",
    "presupuesto_evolucion": "Evolución presupuestal",
    "presupuesto_anio": "Ejecución presupuestal de un año",
    "presupuesto_fuente": "Presupuesto por fuente de financiamiento",
}


def _close_result() -> None:
    st.session_state.pop("qa_result", None)


def render_qa_result_main() -> None:
    """Muestra el resultado de la última consulta en el panel principal,
    en un recuadro con borde y una línea de separación debajo."""
    res = st.session_state.get("qa_result")
    if not res:
        return
    ans: QAAnswer = res["ans"]

    with st.container(border=True):
        h1, h2 = st.columns([6, 1], vertical_alignment="center")
        h1.markdown(f'<p class="qa-main-title">💬 {res["q"]}</p>', unsafe_allow_html=True)
        h2.button("✕ Cerrar", key="qa_close", on_click=_close_result, use_container_width=True)

        mode = f"✨ Modo IA ({res['model']})" if res.get("ai") else "🌙 Modo por reglas"
        filt = "con filtros del panel" if res.get("filters") else "sin filtros del panel"
        meta = " · ".join(x for x in [mode, filt, ans.interpretation] if x)
        st.caption(meta)

        with st.container(key="qa_answer_box"):
            st.markdown(f"**{ans.title}**")
            st.markdown(res.get("ai") or ans.text)
        if res.get("ai_err"):
            st.warning(f"No se pudo usar el modo IA ({res['ai_err'][:200]}). Se muestra la respuesta por reglas.")

        fig = _chart(ans) if ans.ok else None
        if fig is not None:
            n = len(ans.table) if ans.table is not None else 10
            fig.update_layout(
                height=(160 + 30 * n) if ans.chart and ans.chart.get("kind") == "barh" else 400,
                margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
            )
            st.plotly_chart(fig, use_container_width=True, key="qa_chart_main",
                            config={"displaylogo": False})

        if ans.table is not None and ans.ok:
            with st.expander("Ver tabla del resultado", expanded=False):
                st.dataframe(ans.table, use_container_width=True)
                st.download_button(
                    "⬇️ Descargar tabla (CSV)",
                    ans.table.to_csv(index=False).encode("utf-8-sig"),
                    file_name="respuesta_pregunta_datos.csv", mime="text/csv", key="qa_dl_main",
                )

        st.caption(
            "ℹ️ Las cifras corresponden a casos nuevos registrados en el INEN por departamento de "
            "residencia, no a incidencia poblacional."
        )
        if ans.table is not None and ans.ok and ans.intent.startswith("ranking"):
            st.caption(f"ℹ️ Se muestran {len(ans.table)} categorías (mayor número de casos primero).")

        with st.expander("¿Cómo se interpretó la pregunta?", expanded=False):
            st.markdown(
                f"- **Pregunta:** {res['q']}\n"
                f"- **Tipo de consulta:** {_INTENT_LABELS.get(ans.intent, ans.intent or 'sin clasificar')}\n"
                f"- **Parámetros:** {ans.interpretation or 'no aplica'}\n"
                f"- **Filtros del panel:** {'aplicados a lo que la pregunta no especifica' if res.get('filters') else 'no aplicados (Perú, todas las localizaciones, todo el periodo)'}\n"
                f"- **Motor:** {mode}"
            )
            st.caption("Lo que menciona la pregunta (localización, departamento, años) siempre tiene prioridad sobre el panel.")

    st.markdown('<hr class="qa-separator">', unsafe_allow_html=True)
