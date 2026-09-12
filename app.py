"""
Dashboard "Cáncer en el Tiempo — Perú (INEN)"
==============================================
Inspirado en el visor "Cancer Over Time" de GLOBOCAN/IARC, construido
sobre los datos del INEN (Instituto Nacional de Enfermedades
Neoplásicas) para los 25 departamentos del Perú + total nacional.

Ejecutar localmente:
    streamlit run app.py

Estructura del proyecto:
    app.py                -> esta app (interfaz)
    data_processing.py    -> carga, limpieza y actualización de datos
    data/                 -> Excel original + caché limpia
    requirements.txt      -> dependencias
    .streamlit/config.toml-> tema visual
"""

from __future__ import annotations

import colorsys
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_processing import (
    ALL_SITES_LABEL,
    PERU_LABEL,
    append_new_data,
    departamentos,
    load_data,
    localizaciones,
)
from stats_analysis import (
    chow_test_arbitrary_break,
    detect_breakpoint,
    linear_trend,
    mann_kendall_trend,
    project_series,
    significance_label,
    smooth_series,
)

# Generación de GIF para el ranking animado: dependencias opcionales — si no
# están disponibles, la animación interactiva de Plotly sigue funcionando y
# solo se oculta el botón de descarga de GIF.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    _GIF_EXPORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GIF_EXPORT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cáncer en el Tiempo | INEN Perú",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#8b1a4f"      # línea principal (similar al magenta de GLOBOCAN)
SECONDARY_COLOR = "#2f6fa8"    # línea secundaria (azul)
ACCENT_COLOR = "#1f2d5c"       # navy institucional

CUSTOM_CSS = f"""
<style>
    .main .block-container {{ padding-top: 1.3rem; max-width: 1300px; }}
    h1, h2, h3 {{ color: {ACCENT_COLOR}; }}
    .app-header {{
        background: linear-gradient(90deg, {ACCENT_COLOR} 0%, {PRIMARY_COLOR} 100%);
        padding: 1.1rem 1.6rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.2rem;
    }}
    .app-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .app-header p {{ color: #e8e6f0; margin: 0.2rem 0 0 0; font-size: 0.95rem; }}
    div[data-testid="stMetric"] {{
        background: #f7f7fb;
        border: 1px solid #e6e6ef;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
    }}
    .source-note {{ font-size: 0.8rem; color: #666; margin-top: 0.4rem; }}
    .citation-box {{
        background: #f7f7fb;
        border: 1px solid #e6e6ef;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-top: 0.8rem;
    }}
    .citation-box h4 {{
        color: {ACCENT_COLOR};
        font-size: 0.8rem;
        letter-spacing: 0.03em;
        margin: 0.7rem 0 0.3rem 0;
        text-transform: uppercase;
    }}
    .citation-box h4:first-child {{ margin-top: 0; }}
    .citation-box p {{
        font-size: 0.88rem;
        color: #333;
        margin: 0 0 0.4rem 0;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Cargando datos del INEN...")
def get_data() -> pd.DataFrame:
    return load_data()


df = get_data()

st.markdown(
    f"""
    <div class="app-header">
        <h1>🎗️ Cáncer en el Tiempo — Perú</h1>
        <p>Casos nuevos de cáncer registrados por el INEN ({int(df['Anio'].min())}–{int(df['Anio'].max())})</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PALETTE = [
    PRIMARY_COLOR, SECONDARY_COLOR, "#e07a2c", "#3ba776", "#8850c4",
    "#c94141", "#4aa8c9", "#c9a13b", "#5c6ac4", "#7d7d7d",
]


def format_region_list(regions: list[str], max_show: int = 3) -> str:
    """Muestra los nombres de las regiones seleccionadas (no solo el
    conteo); si hay muchas, trunca y agrega '+N más' para no romper
    el layout de la tarjeta."""
    if not regions:
        return "—"
    if len(regions) <= max_show:
        return ", ".join(regions)
    return ", ".join(regions[:max_show]) + f" +{len(regions) - max_show} más"


def shades_of(hex_color: str, n: int, l_min: float = 0.30, l_max: float = 0.80) -> list[str]:
    """Genera `n` matices del mismo color (variando solo la luminosidad),
    del más oscuro (primer puesto en el ranking) al más claro. Se usa en
    el gráfico de ranking para que el orden se perciba visualmente sin
    necesitar colores distintos por barra."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, _l, s = colorsys.rgb_to_hls(r, g, b)
    lights = [l_min] if n <= 1 else np.linspace(l_min, l_max, n)
    colors = []
    for lig in lights:
        rr, gg, bb = colorsys.hls_to_rgb(h, float(lig), s)
        colors.append(
            "#%02x%02x%02x" % (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)))
        )
    return colors


def dept_casos_series(dept: str, site_val: str, year_lo: int, year_hi: int) -> pd.DataFrame:
    """Devuelve las filas Anio/Casos para un departamento y una
    localización del tumor primario, en el rango de años dado.

    Si `dept` es "Perú (Total nacional)" y el usuario configuró
    exclusiones en el panel lateral (variable global `peru_exclude`,
    p. ej. para no contar 'Extranjero'), esos casos se restan del total
    nacional tal como lo reporta el INEN. Si el departamento excluido no
    tiene dato para un año dado, no se resta nada ese año (se asume 0,
    no una ausencia de dato)."""
    base = df[
        df["Localizacion"].eq(site_val)
        & df["Departamento"].eq(dept)
        & df["Anio"].between(year_lo, year_hi)
    ][["Anio", "Casos"]].copy()

    if dept == PERU_LABEL and peru_exclude:
        excl = (
            df[
                df["Localizacion"].eq(site_val)
                & df["Departamento"].isin(peru_exclude)
                & df["Anio"].between(year_lo, year_hi)
            ]
            .groupby("Anio")["Casos"]
            .sum(min_count=1)
        )

        def _subtract(row):
            if pd.isna(row["Casos"]):
                return row["Casos"]
            excl_val = excl.get(row["Anio"], 0)
            # Si el/los departamento(s) excluidos no reportaron dato ese
            # año (NaN, no cero), se asume que no hay nada que restar en
            # vez de propagar el NaN y "borrar" el total nacional de ese año.
            if pd.isna(excl_val):
                excl_val = 0
            return row["Casos"] - excl_val

        base["Casos"] = base.apply(_subtract, axis=1)
    return base


_RANK_MEDALS = ["🥇", "🥈", "🥉"]


def get_year_ranking(
    dept: str,
    year: int,
    exclude: list[str] | None = None,
    top_n: int | None = None,
    peru_excl_override: list[str] | None = None,
) -> pd.DataFrame:
    """Devuelve el ranking (descendente por N° de casos) de
    localizaciones del tumor primario para un departamento y año dados.

    Aplica el ajuste de exclusión del total nacional (cuando `dept` es
    Perú) usando `peru_excl_override` si se especifica, o si no la
    variable global `peru_exclude` configurada en el panel lateral.
    Opcionalmente excluye localizaciones específicas o recorta a las
    primeras `top_n`. Se usa tanto en la pestaña "Ranking por año" como
    en "Ranking animado", para no duplicar la lógica."""
    exclude = exclude or []
    effective_peru_exclude = (
        peru_excl_override if peru_excl_override is not None else peru_exclude
    )
    data = (
        df[
            (df["Anio"] == year)
            & (df["Departamento"] == dept)
            & (df["Localizacion"] != ALL_SITES_LABEL)
            & (~df["Localizacion"].isin(exclude))
        ][["Localizacion", "Casos"]]
        .dropna(subset=["Casos"])
        .copy()
    )

    if dept == PERU_LABEL and effective_peru_exclude:
        excl = (
            df[
                (df["Anio"] == year)
                & (df["Departamento"].isin(effective_peru_exclude))
                & (df["Localizacion"] != ALL_SITES_LABEL)
            ]
            .groupby("Localizacion")["Casos"]
            .sum(min_count=1)
        )

        def _sub(row):
            e = excl.get(row["Localizacion"], 0)
            if pd.isna(e):
                e = 0
            return row["Casos"] - e

        data["Casos"] = data.apply(_sub, axis=1)

    data = data.sort_values("Casos", ascending=False).reset_index(drop=True)
    if top_n is not None:
        data = data.head(top_n)
    return data


def top3_localizaciones(dept: str, year: int) -> str:
    """Texto (HTML, para hover de Plotly) con el top 3 de localizaciones
    del tumor primario con más casos para un departamento y año dados.

    Si `dept` es Perú y hay exclusiones configuradas (variable global
    `peru_exclude`), se aplican de la misma forma que en el resto del
    dashboard antes de calcular el ranking."""
    sub = (
        df[
            (df["Anio"] == year)
            & (df["Departamento"] == dept)
            & (df["Localizacion"] != ALL_SITES_LABEL)
        ][["Localizacion", "Casos"]]
        .dropna(subset=["Casos"])
        .copy()
    )

    if dept == PERU_LABEL and peru_exclude:
        excl = (
            df[
                (df["Anio"] == year)
                & (df["Departamento"].isin(peru_exclude))
                & (df["Localizacion"] != ALL_SITES_LABEL)
            ]
            .groupby("Localizacion")["Casos"]
            .sum(min_count=1)
        )

        def _sub(row):
            e = excl.get(row["Localizacion"], 0)
            if pd.isna(e):
                e = 0
            return row["Casos"] - e

        sub["Casos"] = sub.apply(_sub, axis=1)

    sub = sub.sort_values("Casos", ascending=False).head(3)
    if sub.empty:
        return "Sin datos por localización"
    return "<br>".join(
        f"{_RANK_MEDALS[i]} {row.Localizacion}: {row.Casos:,.0f}"
        for i, row in enumerate(sub.itertuples())
    )

# ---------------------------------------------------------------------------
# Barra lateral — controles (equivalente al panel "Display" de GLOBOCAN)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Panel de control")

    with st.expander("📥 Incorporar datos recientes", expanded=False):
        st.caption(
            "Sube un archivo .xlsx o .csv con columnas "
            "`Departamento`, `localizacion` y una columna por cada año "
            "nuevo (ej. `2024`) con el N° de casos. Se combinará con el "
            "histórico; si un dato ya existía, el nuevo lo reemplaza."
        )
        new_file = st.file_uploader(
            "Archivo con datos nuevos", type=["xlsx", "xls", "csv"], key="uploader"
        )
        if new_file is not None:
            tmp_path = f"/tmp/{new_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(new_file.getbuffer())
            try:
                updated = append_new_data(tmp_path, save=True)
                st.success(
                    f"Datos actualizados: {len(updated):,} filas totales. "
                    "Recarga la página para verlos reflejados en los gráficos."
                )
                get_data.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo procesar el archivo: {exc}")

    with st.expander("📊 Base de datos adicional (serie temporal)", expanded=False):
        st.caption(
            "Sube cualquier serie temporal adicional (CSV o Excel) con al menos "
            "una columna de fecha/año y una columna numérica. Se analizará de "
            "forma exploratoria en la pestaña «🔍 Exploración (base adicional)» "
            "y se comparará con los casos de cáncer en «🔀 Análisis cruzado»."
        )
        _sec_upload = st.file_uploader(
            "Archivo de serie adicional", type=["xlsx", "xls", "csv"], key="upload_sec"
        )
        if _sec_upload is not None:
            try:
                _is_csv = _sec_upload.name.lower().endswith(".csv")
                if _is_csv:
                    _sec_raw = pd.read_csv(_sec_upload)
                else:
                    _xls = pd.ExcelFile(_sec_upload)
                    _sheet_names = _xls.sheet_names
                    if len(_sheet_names) > 1:
                        _sec_sheet = st.selectbox(
                            "Hoja a utilizar", _sheet_names, key="sec_sheet",
                            help="El archivo tiene varias hojas — elige cuál contiene la serie temporal.",
                        )
                    else:
                        _sec_sheet = _sheet_names[0]
                        st.caption(f"Hoja: **{_sec_sheet}**")
                    _sec_raw = _xls.parse(_sec_sheet)
                _sec_cols = _sec_raw.columns.tolist()
                st.caption(f"Columnas detectadas: {', '.join(str(c) for c in _sec_cols)}")
                _sec_date_col = st.selectbox("Columna de fecha o año", _sec_cols, key="sec_date_col")
                _num_cols = [c for c in _sec_cols if c != _sec_date_col]
                _sec_val_cols = st.multiselect(
                    "Columna(s) de valor (numérico)",
                    _num_cols,
                    default=[_num_cols[0]] if _num_cols else [],
                    key="sec_val_cols",
                    help="Selecciona una o más columnas. Si eliges varias, puedes combinarlas o mantenerlas por separado.",
                )
                _sec_label = st.text_input("Etiqueta para esta serie", "Serie adicional", key="sec_label")
                _sec_units = st.text_input(
                    "Unidades del eje Y", "Valor", key="sec_units",
                    help="Ej: 'Soles (S/)', 'Población', 'Presupuesto (S/ mill.)'",
                )
                _multi_cols = len(_sec_val_cols) > 1
                if _multi_cols:
                    _sec_combine = st.checkbox(
                        "Combinar columnas en una sola serie",
                        value=False, key="sec_combine",
                        help="Si está marcado, las columnas se suman o promedian fila a fila. Sin marcar, cada columna conserva su propia serie.",
                    )
                else:
                    _sec_combine = True
                if _sec_combine:
                    _sec_agg = st.radio(
                        "Agregar por", ["Suma", "Promedio"], horizontal=True, key="sec_agg",
                    )
                else:
                    _sec_agg = "Suma"
                if not _sec_val_cols:
                    st.warning("Selecciona al menos una columna de valor.")
                elif st.button("Cargar serie adicional", key="btn_load_sec"):
                    try:
                        _sdf = _sec_raw[[_sec_date_col] + _sec_val_cols].copy()
                        _sdf = _sdf.rename(columns={_sec_date_col: "tiempo"})
                        if not _sec_combine or len(_sec_val_cols) == 1:
                            _sdf["valor"] = pd.to_numeric(_sdf[_sec_val_cols[0]], errors="coerce")
                        else:
                            _num_df = _sdf[_sec_val_cols].apply(pd.to_numeric, errors="coerce")
                            _sdf["valor"] = _num_df.sum(axis=1) if _sec_agg == "Suma" else _num_df.mean(axis=1)
                        _sdf = _sdf[["tiempo", "valor"]]
                        # Detectar si la columna de fecha son solo años (enteros 1900-2100)
                        _as_num = pd.to_numeric(_sdf["tiempo"], errors="coerce")
                        _is_year_col = _as_num.notna().all() and _as_num.between(1900, 2100).all()
                        if _is_year_col:
                            _sdf["Anio"] = _as_num.astype(int)
                        else:
                            _parsed = pd.to_datetime(_sdf["tiempo"], dayfirst=True, errors="coerce")
                            _sdf["Anio"] = _parsed.dt.year
                        _sdf = _sdf.dropna(subset=["Anio"])
                        _sdf["Anio"] = _sdf["Anio"].astype(int)
                        _agg_fn = "sum" if _sec_agg == "Suma" else "mean"
                        # La serie principal de este dashboard es ANUAL: si la
                        # fuente adicional viene con más de una fila por año
                        # (ej. datos mensuales), se agrega a nivel de año.
                        _s = _sdf.groupby("Anio")["valor"].agg(_agg_fn)
                        _s = _s[_s.index.notna()]
                        if _s.empty:
                            st.error(
                                "No se encontraron años válidos en la columna "
                                "seleccionada. Verifica el formato (ej. 2021, "
                                "2021-01-01, 01/01/2021, enero 2021)."
                            )
                        else:
                            _sec_cols_dict: dict[str, pd.Series] = {}
                            _raw_num = _sec_raw[[_sec_date_col] + _sec_val_cols].copy()
                            _raw_num = _raw_num.rename(columns={_sec_date_col: "tiempo"})
                            _raw_num["Anio"] = _sdf["Anio"].reindex(_raw_num.index)
                            for _col in _sec_val_cols:
                                _cv = pd.to_numeric(_raw_num[_col], errors="coerce")
                                _raw_num["_v"] = _cv
                                _sc = _raw_num.dropna(subset=["Anio"]).groupby("Anio")["_v"].agg(_agg_fn)
                                _sec_cols_dict[_col] = _sc
                            st.session_state["secondary_serie"] = _s
                            st.session_state["secondary_label"] = _sec_label
                            st.session_state["secondary_units"] = _sec_units
                            st.session_state["secondary_cols_dict"] = _sec_cols_dict
                            st.session_state["secondary_val_cols"] = list(_sec_val_cols)
                            st.success(
                                f"✓ Serie cargada: {len(_s)} año(s) "
                                f"({int(_s.index.min())} a {int(_s.index.max())})"
                            )
                    except Exception as _e:  # noqa: BLE001
                        st.error(f"Error al procesar el archivo: {_e}")
            except Exception as _e:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo: {_e}")

        if "secondary_serie" in st.session_state:
            _lbl = st.session_state.get("secondary_label", "Serie adicional")
            _ss = st.session_state["secondary_serie"]
            st.info(f"**'{_lbl}'** cargada — {len(_ss)} año(s) ({int(_ss.index.min())} a {int(_ss.index.max())})")
            if st.button("🗑️ Quitar serie adicional", key="btn_rm_sec"):
                for _k in [
                    "secondary_serie", "secondary_label", "secondary_units",
                    "secondary_cols_dict", "secondary_val_cols",
                ]:
                    st.session_state.pop(_k, None)
                st.rerun()

    st.subheader("Localización del tumor primario")
    site = st.selectbox(
        "Selecciona una localización del tumor primario",
        options=localizaciones(df),
        index=0,
    )

    show_extra_sites = st.checkbox(
        "Comparar varias localizaciones (gráficos adicionales)",
        value=False,
        help=(
            "Agrega un gráfico adicional por cada localización del tumor "
            "primario que selecciones, para evaluarlas simultáneamente sin "
            "perder la selección principal de arriba."
        ),
    )
    extra_sites: list[str] = []
    if show_extra_sites:
        extra_sites = st.multiselect(
            "Localizaciones adicionales a mostrar",
            options=localizaciones(df),
            default=[],
            max_selections=6,
            key="extra_sites",
        )

    st.subheader("Departamento de residencia a comparar")
    depts_available = departamentos(df)
    default_depts = [PERU_LABEL]
    depts_selected = st.multiselect(
        "Selecciona uno o más departamentos de residencia",
        options=depts_available,
        default=default_depts,
        max_selections=10,
    )
    if not depts_selected:
        depts_selected = default_depts

    peru_exclude: list[str] = []
    if PERU_LABEL in depts_selected:
        peru_exclude = st.multiselect(
            "Excluir del total nacional (Perú)",
            options=[d for d in depts_available if d != PERU_LABEL],
            default=[],
            key="peru_exclude",
            help=(
                "Ej.: si no quieres que los casos de 'Extranjero' cuenten "
                "dentro del total de Perú, selecciónalo aquí. Se resta del "
                "total nacional (reportado por el INEN) para la localización "
                "y los años seleccionados, y afecta también al ranking, "
                "Mann-Kendall, quiebres y proyección cuando el departamento "
                "analizado es Perú."
            ),
        )

    show_dept_split = st.checkbox(
        "Comparar varios departamentos (gráficos adicionales)",
        value=False,
        help=(
            "Para la localización principal elegida arriba, muestra un "
            "gráfico independiente por cada departamento que elijas abajo "
            "(en vez de mezclarlos como líneas en un solo gráfico) — útil "
            "cuando un departamento grande como Lima o Perú aplasta "
            "visualmente a uno más pequeño en la escala combinada."
        ),
    )
    dept_split_list: list[str] = []
    if show_dept_split:
        dept_split_list = st.multiselect(
            "Departamentos adicionales a mostrar",
            options=depts_available,
            default=[],
            max_selections=6,
            key="dept_split_list",
        )

    st.subheader("Periodo")
    year_min, year_max = int(df["Anio"].min()), int(df["Anio"].max())
    year_range = st.slider(
        "Rango de años",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    st.subheader("Visualización")
    chart_type = st.radio(
        "Tipo de gráfico", ["Línea", "Barras"], horizontal=True
    )
    show_markers = st.checkbox("Mostrar marcadores", value=True)
    show_labels = st.checkbox("Mostrar etiquetas en cada punto", value=False)
    show_rank_labels = st.checkbox(
        "Mostrar etiquetas con ranking",
        value=False,
        help=(
            "Al pasar el cursor sobre un punto, muestra además el top 3 "
            "de localizaciones del tumor primario con más casos ese año "
            "para el departamento correspondiente (🥇🥈🥉)."
        ),
    )
    log_scale = st.checkbox("Escala logarítmica (eje Y)", value=False)
    show_smooth = st.checkbox(
        "Mostrar línea temporal suavizada (LOWESS)",
        value=False,
        help=(
            "Suaviza la serie con regresión local ponderada (LOWESS), "
            "sin imponer una forma lineal, para ver la tendencia general "
            "sin el ruido año a año."
        ),
    )
    smooth_frac = 0.3
    if show_smooth:
        smooth_frac = st.slider(
            "Nivel de suavizado", 0.1, 0.9, 0.3, 0.05,
            help="Valores bajos siguen más de cerca los datos; valores altos generan una curva más suave.",
        )

    st.subheader("Análisis estadístico")
    show_trend = st.checkbox("Línea de tendencia (regresión lineal)", value=False)
    show_breakpoint = st.checkbox(
        "Detectar punto de quiebre (cambio estructural)", value=False,
        help=(
            "Ajusta una regresión segmentada de dos tramos y aplica un "
            "test de Chow para hallar el año en el que la tendencia "
            "cambia de forma estadísticamente significativa."
        ),
    )
    breakpoint_target = None
    if show_breakpoint and depts_selected:
        breakpoint_target = st.selectbox(
            "Departamento a analizar para el quiebre",
            options=depts_selected,
            index=0,
            help="El análisis de quiebre se calcula para una sola serie a la vez.",
        )

    show_mk = st.checkbox(
        "Test de tendencia Mann-Kendall (robusto a autocorrelación)",
        value=False,
        help=(
            "A diferencia de la regresión lineal (que puede mostrar "
            "significancia artificial cuando los residuos están "
            "autocorrelacionados), este test no paramétrico corrige la "
            "varianza por autocorrelación (modificación de Hamed-Rao) "
            "y responde de forma más confiable si existe o no una "
            "tendencia monótona. Incluye la pendiente de Sen, robusta "
            "a valores atípicos."
        ),
    )
    mk_target = None
    mk_year_range = (year_min, year_max)
    if show_mk and depts_selected:
        mk_target = st.selectbox(
            "Departamento a analizar (Mann-Kendall)",
            options=depts_selected,
            index=0,
            key="mk_target_select",
        )
        mk_year_range = st.slider(
            "Rango de años a evaluar (Mann-Kendall)",
            min_value=year_min,
            max_value=year_max,
            value=year_range,
            step=1,
            key="mk_year_range",
            help=(
                "Acota el test a un sub-periodo específico (por ejemplo, "
                "antes o después de un cambio de política), independiente "
                "del rango general del gráfico."
            ),
        )

    show_arbitrary_break = st.checkbox(
        "Evaluar un quiebre en un año específico (evento)",
        value=False,
        help=(
            "A diferencia de la detección automática, aquí indicas tú "
            "el año de un evento (p. ej. una ley aprobada o la "
            "inauguración de un hospital oncológico) y el test de Chow "
            "evalúa si la tendencia cambió realmente a partir de ese "
            "punto. Puedes excluir años de 'implementación' donde el "
            "efecto todavía no se consolida."
        ),
    )
    arb_target = None
    arb_break_year = None
    arb_lag = 0
    if show_arbitrary_break and depts_selected:
        arb_target = st.selectbox(
            "Departamento a analizar (quiebre por evento)",
            options=depts_selected,
            index=0,
            key="arb_target_select",
        )
        arb_break_year = st.number_input(
            "Año del evento",
            min_value=year_min,
            max_value=year_max,
            value=min(max(year_min + 1, (year_min + year_max) // 2), year_max),
            step=1,
            key="arb_break_year",
            help="Ej.: el año en que se aprobó una ley o se inauguró un hospital oncológico.",
        )
        arb_lag = st.slider(
            "Años de implementación a excluir tras el evento",
            0, 5, 0,
            key="arb_lag",
            help=(
                "Si el efecto tarda en consolidarse (p. ej. una ley "
                "aprobada en 2015 que recién se aplica plenamente desde "
                "2017), usa 2 para excluir 2015-2016 del análisis y no "
                "diluir la comparación antes/después."
            ),
        )

    st.subheader("Proyección de casos")
    show_projection = st.checkbox(
        "Proyectar casos a futuro (con análisis de sensibilidad)",
        value=False,
        help=(
            "Muestra 3 escenarios (conservador, recomendado y lineal) "
            "en vez de un único número, para no sobreestimar. El "
            "escenario recomendado usa suavizado exponencial con "
            "tendencia amortiguada (Holt damped trend)."
        ),
    )
    proj_target = None
    proj_horizon = 5
    if show_projection and depts_selected:
        proj_target = st.selectbox(
            "Departamento a proyectar",
            options=depts_selected,
            index=0,
            key="proj_target_select",
        )
        proj_horizon = st.slider("Horizonte de proyección (años)", 1, 10, 5)

    with st.expander("🎨 Personalizar colores"):
        custom_colors = st.checkbox("Elegir color por departamento", value=False)
        color_overrides: dict[str, str] = {}
        if custom_colors:
            for i, d in enumerate(depts_selected):
                color_overrides[d] = st.color_picker(
                    f"Color · {d}", value=DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
                )

    st.divider()
    st.caption(
        "Fuente: Instituto Nacional de Enfermedades Neoplásicas (INEN), Perú. "
        "Panel construido con Streamlit + Plotly."
    )

# ---------------------------------------------------------------------------
# Filtrado
# ---------------------------------------------------------------------------

# Se construye departamento por departamento (en vez de un filtro simple)
# para poder aplicar el ajuste de exclusión del total nacional (Perú menos
# los departamentos marcados en `peru_exclude`, p. ej. "Extranjero").
_filtered_frames = []
for _dept in depts_selected:
    _sub = dept_casos_series(_dept, site, year_range[0], year_range[1]).copy()
    _sub["Departamento"] = _dept
    _sub["Localizacion"] = site
    _filtered_frames.append(_sub)

if _filtered_frames:
    filtered = pd.concat(_filtered_frames, ignore_index=True)[
        ["Departamento", "Localizacion", "Anio", "Casos"]
    ].sort_values(["Departamento", "Anio"])
else:
    filtered = df.iloc[0:0][["Departamento", "Localizacion", "Anio", "Casos"]]

# ---------------------------------------------------------------------------
# Resumen general + KPIs
# ---------------------------------------------------------------------------

available_years_kpi = (
    sorted(filtered["Anio"].unique().tolist()) if not filtered.empty else [year_min, year_max]
)
default_year_a = available_years_kpi[0]
default_year_b = available_years_kpi[-1]

# Se leen los años de comparación desde session_state (si el usuario ya
# interactuó con los selectores más abajo) para poder mostrar la tarjeta
# "Casos en <año>" en la primera fila, antes de dibujar los selectores.
year_a = st.session_state.get("kpi_year_a", default_year_a)
year_b = st.session_state.get("kpi_year_b", default_year_b)
if year_a not in available_years_kpi:
    year_a = default_year_a
if year_b not in available_years_kpi:
    year_b = default_year_b

total_year_b = filtered.loc[filtered["Anio"] == year_b, "Casos"].sum(min_count=1)
total_year_a = filtered.loc[filtered["Anio"] == year_a, "Casos"].sum(min_count=1)
delta_pct = (
    ((total_year_b - total_year_a) / total_year_a * 100) if total_year_a else float("nan")
)

row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric("Localización del tumor primario", site)
row1_col2.metric("Departamento(s) de residencia", format_region_list(depts_selected))
row1_col3.metric(
    f"Casos en {year_b}",
    f"{int(total_year_b):,}" if pd.notna(total_year_b) else "s/d",
)

st.markdown(
    """
    <div style="background:#1f2d5c; color:white; padding:0.45rem 1rem;
                border-radius:8px; margin:0.9rem 0 0.7rem 0; font-weight:600;
                font-size:0.95rem; letter-spacing:0.03em;">
        📊 KPI
    </div>
    """,
    unsafe_allow_html=True,
)

cmp1, cmp2 = st.columns(2)
with cmp1:
    year_a = st.selectbox(
        "Año base (comparar desde)",
        options=available_years_kpi,
        index=available_years_kpi.index(year_a),
        key="kpi_year_a",
    )
with cmp2:
    year_b = st.selectbox(
        "Año a comparar (comparar hasta)",
        options=available_years_kpi,
        index=available_years_kpi.index(year_b),
        key="kpi_year_b",
    )

# Recalcular con los valores confirmados por los selectores (mismos que
# arriba salvo que el usuario los acabe de cambiar en este rerun)
total_year_b = filtered.loc[filtered["Anio"] == year_b, "Casos"].sum(min_count=1)
total_year_a = filtered.loc[filtered["Anio"] == year_a, "Casos"].sum(min_count=1)
delta_pct = (
    ((total_year_b - total_year_a) / total_year_a * 100) if total_year_a else float("nan")
)
delta_abs = (
    (total_year_b - total_year_a)
    if pd.notna(total_year_a) and pd.notna(total_year_b)
    else float("nan")
)
n_years_cmp = year_b - year_a
if n_years_cmp != 0 and pd.notna(total_year_a) and total_year_a > 0 and pd.notna(total_year_b) and total_year_b > 0:
    cagr = (total_year_b / total_year_a) ** (1 / n_years_cmp) - 1
else:
    cagr = float("nan")

# Serie agregada (suma de todos los departamentos seleccionados) para
# los KPIs que miran todo el periodo filtrado, no solo los 2 años
# comparados: año pico y tendencia resumida.
year_series = filtered.groupby("Anio")["Casos"].sum(min_count=1).dropna()
if not year_series.empty:
    peak_year = int(year_series.idxmax())
    peak_val = float(year_series.max())
else:
    peak_year, peak_val = None, float("nan")

mk_kpi = mann_kendall_trend(year_series.index.to_numpy(), year_series.to_numpy())

kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
kcol1.metric(
    f"Variación {year_a} → {year_b}",
    f"{delta_pct:+.1f}%" if pd.notna(delta_pct) else "s/d",
)
kcol2.metric(
    "Cambio absoluto",
    f"{delta_abs:+,.0f} casos" if pd.notna(delta_abs) else "s/d",
)
kcol3.metric(
    "CAGR (crec. anual compuesto)",
    f"{cagr * 100:+.1f}%/año" if pd.notna(cagr) else "s/d",
)
kcol4.metric(
    "Año pico",
    f"{peak_year} · {peak_val:,.0f}" if peak_year is not None else "s/d",
)
if mk_kpi is not None:
    trend_icon = {
        "increasing": "📈 Creciente",
        "decreasing": "📉 Decreciente",
        "no trend": "➖ Sin tendencia",
    }.get(mk_kpi.trend, mk_kpi.trend)
    kcol5.metric(
        "Tendencia (Mann-Kendall)",
        trend_icon,
        delta=f"p = {mk_kpi.p_value:.3f}",
        delta_color="off",
    )
else:
    kcol5.metric("Tendencia (Mann-Kendall)", "s/d")

# ---------------------------------------------------------------------------
# Gráfico principal (equivalente al "Graphic" tab de GLOBOCAN)
# ---------------------------------------------------------------------------

(
    tab_graph, tab_table, tab_ranking, tab_ranking_anim, tab_projection,
    tab_downloads, tab_explore_sec, tab_cross,
) = st.tabs(
    [
        "📈 Gráfico", "📋 Tabla", "🏆 Ranking por año", "🎬 Ranking animado",
        "🔮 Proyección", "⬇️ Descargas",
        "🔍 Exploración (base adicional)", "🔀 Análisis cruzado",
    ]
)


def render_site_analysis_chart(
    site_val: str,
    filtered_local: pd.DataFrame,
    compact: bool = False,
    key_suffix: str = "",
    depts_override: list[str] | None = None,
    local_event_year: int | None = None,
    local_event_lag: int = 0,
    local_breakpoint: bool = False,
) -> None:
    """Construye y renderiza el gráfico de una localización del tumor
    primario, con línea de tendencia, suavizado LOWESS, quiebre
    automático, Mann-Kendall y quiebre por evento —  todo según los
    controles del panel lateral. La usan tanto el gráfico principal
    como los gráficos adicionales de otras localizaciones y de otros
    departamentos, así todos comparten exactamente el mismo análisis
    estadístico y la misma personalización de colores.

    `compact=True` reduce tamaños de fuente/altura y agrupa los
    resultados estadísticos en un expander, para el layout en columnas
    de los gráficos adicionales.

    `depts_override`, si se especifica, reemplaza la lista global de
    departamentos seleccionados (usado para el modo "un gráfico por
    departamento" — cada llamada solo dibuja un departamento a la vez).

    `local_event_year`, si se especifica, evalúa el quiebre por evento
    de forma independiente para ESTE gráfico (con su propio año y años
    de implementación), en vez de usar la configuración global del
    panel lateral — pensado para los gráficos adicionales, donde cada
    uno puede corresponder a un evento distinto (p. ej. una norma que
    aplicó en años diferentes por departamento).

    `local_breakpoint`, si es True, activa la detección automática de
    punto de quiebre para ESTE gráfico independientemente del switch
    global del panel lateral (que sigue aplicando igual si está
    activado; ambos se combinan con "o", no se excluyen).
    """
    depts_to_plot = depts_override if depts_override is not None else depts_selected

    if filtered_local.empty or filtered_local["Casos"].dropna().empty:
        st.info(f"No hay datos para **{site_val}** con la selección actual.")
        return

    fig = go.Figure()
    for i, dept in enumerate(depts_to_plot):
        sub = filtered_local[filtered_local["Departamento"] == dept]
        color = (
            color_overrides.get(dept, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
            if custom_colors
            else DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
        )
        mode = "lines+markers" if show_markers else "lines"
        if show_labels:
            mode += "+text"
        raw_opacity = 0.35 if show_smooth else 1.0

        # "Mostrar etiquetas con ranking" es la única opción que NO se
        # aplica a los gráficos compactos de "Comparar con otras
        # localizaciones" — solo al gráfico principal.
        if show_rank_labels and not compact:
            rank_customdata = [top3_localizaciones(dept, y) for y in sub["Anio"]]
            hover_tpl = (
                "%{x}: %{y:,.0f} casos<br><br><b>Top 3 en " + dept + "</b><br>"
                "%{customdata}<extra></extra>"
            )
        else:
            rank_customdata = None
            hover_tpl = "%{x}: %{y:,.0f} casos<extra>" + dept + "</extra>"

        if chart_type == "Línea":
            fig.add_trace(
                go.Scatter(
                    x=sub["Anio"],
                    y=sub["Casos"],
                    name=dept,
                    mode=mode,
                    line=dict(width=2.0 if compact else 2.5, color=color),
                    marker=dict(size=4 if compact else 5),
                    opacity=raw_opacity,
                    text=sub["Casos"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else ""),
                    textposition="top center",
                    textfont=dict(size=8 if compact else 10, color=color),
                    connectgaps=True,
                    customdata=rank_customdata,
                    hovertemplate=hover_tpl,
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=sub["Anio"],
                    y=sub["Casos"],
                    name=dept,
                    marker_color=color,
                    opacity=raw_opacity,
                    text=sub["Casos"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
                    if show_labels
                    else None,
                    textposition="outside",
                    customdata=rank_customdata,
                    hovertemplate=hover_tpl,
                )
            )

        if show_trend:
            trend = linear_trend(sub["Anio"].to_numpy(), sub["Casos"].to_numpy())
            if trend is not None:
                fig.add_trace(
                    go.Scatter(
                        x=trend.x,
                        y=trend.y_pred,
                        name=f"Tendencia · {dept} (R²={trend.r2:.2f})",
                        mode="lines",
                        line=dict(width=1.6, color=color, dash="dot"),
                        hoverinfo="skip",
                    )
                )

        if show_smooth:
            sm = smooth_series(sub["Anio"].to_numpy(), sub["Casos"].to_numpy(), frac=smooth_frac)
            if sm is not None:
                sm_x, sm_y = sm
                fig.add_trace(
                    go.Scatter(
                        x=sm_x,
                        y=sm_y,
                        name=f"Suavizado (LOWESS) · {dept}",
                        mode="lines",
                        line=dict(width=2.5, color=color, shape="spline"),
                        hoverinfo="skip",
                    )
                )

    bp_summary = None
    bp_dept = None
    if local_breakpoint or show_breakpoint:
        if local_breakpoint and len(depts_to_plot) == 1:
            # Activado localmente en este gráfico específico: siempre
            # usa el único departamento que se está mostrando aquí.
            bp_dept = depts_to_plot[0]
        elif breakpoint_target and breakpoint_target in depts_to_plot:
            bp_dept = breakpoint_target
        elif len(depts_to_plot) == 1:
            # Un solo departamento en este gráfico: se usa como objetivo
            # del análisis automáticamente, sin depender de que coincida
            # con el selector de la barra lateral (así el análisis
            # también aplica a los gráficos adicionales por departamento).
            bp_dept = depts_to_plot[0]
    if bp_dept:
        sub_bp = filtered_local[filtered_local["Departamento"] == bp_dept]
        bp = detect_breakpoint(sub_bp["Anio"].to_numpy(), sub_bp["Casos"].to_numpy())
        if bp is None:
            st.info(
                f"No hay suficientes años con datos en **{bp_dept}** "
                f"({site_val}) para estimar un punto de quiebre (se requieren al menos 6)."
            )
        else:
            bp_summary = bp
            fig.add_trace(
                go.Scatter(
                    x=bp.x_before, y=bp.y_pred_before, mode="lines",
                    line=dict(width=2, color="black", dash="dash"),
                    name=f"Tramo antes de {bp.year}", hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=bp.x_after, y=bp.y_pred_after, mode="lines",
                    line=dict(width=2, color="black", dash="dash"),
                    name=f"Tramo desde {bp.year}", hoverinfo="skip",
                )
            )
            fig.add_vline(
                x=bp.year, line_width=1.5, line_dash="dash", line_color="black",
                annotation_text=f"Quiebre: {bp.year}", annotation_position="top",
            )

    mk_summary = None
    mk_dept = None
    if show_mk:
        if mk_target and mk_target in depts_to_plot:
            mk_dept = mk_target
        elif len(depts_to_plot) == 1:
            mk_dept = depts_to_plot[0]
    if mk_dept:
        sub_mk = dept_casos_series(mk_dept, site_val, mk_year_range[0], mk_year_range[1])
        mkr = mann_kendall_trend(sub_mk["Anio"].to_numpy(), sub_mk["Casos"].to_numpy())
        if mkr is None:
            st.info(
                f"No hay suficientes años con datos en **{mk_dept}** ({site_val}) "
                f"entre {mk_year_range[0]} y {mk_year_range[1]} para el test de "
                "Mann-Kendall (se requieren al menos 6)."
            )
        else:
            mk_summary = mkr
            fig.add_trace(
                go.Scatter(
                    x=mkr.x, y=mkr.y_sen, mode="lines",
                    line=dict(width=2, color="#0f9b8e", dash="dashdot"),
                    name=f"Pendiente de Sen · {mk_dept}", hoverinfo="skip",
                )
            )

    arb_summary = None
    arb_dept = None
    # Si se especifica un evento local (independiente por gráfico), tiene
    # prioridad sobre la configuración global del panel lateral y siempre
    # se evalúa para el único departamento de este gráfico.
    effective_break_year = local_event_year if local_event_year is not None else arb_break_year
    effective_lag = local_event_lag if local_event_year is not None else arb_lag
    use_arb = (local_event_year is not None) or show_arbitrary_break
    if use_arb and effective_break_year is not None:
        if local_event_year is not None:
            arb_dept = depts_to_plot[0] if len(depts_to_plot) == 1 else None
        elif arb_target and arb_target in depts_to_plot:
            arb_dept = arb_target
        elif len(depts_to_plot) == 1:
            arb_dept = depts_to_plot[0]
    if arb_dept:
        sub_arb = filtered_local[filtered_local["Departamento"] == arb_dept]
        arb = chow_test_arbitrary_break(
            sub_arb["Anio"].to_numpy(),
            sub_arb["Casos"].to_numpy(),
            break_year=int(effective_break_year),
            implementation_lag=int(effective_lag),
        )
        if arb is None:
            st.info(
                f"No hay suficientes años antes/después de {int(effective_break_year)} "
                f"(considerando {int(effective_lag)} año(s) de implementación) en "
                f"**{arb_dept}** ({site_val}) para aplicar el test (se requieren al "
                "menos 3 años a cada lado)."
            )
        else:
            arb_summary = arb
            fig.add_trace(
                go.Scatter(
                    x=arb.x_before, y=arb.y_pred_before, mode="lines",
                    line=dict(width=2, color="#e07a2c", dash="longdash"),
                    name=f"Antes del evento ({arb.break_year})", hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=arb.x_after, y=arb.y_pred_after, mode="lines",
                    line=dict(width=2, color="#e07a2c", dash="longdash"),
                    name="Después del evento", hoverinfo="skip",
                )
            )
            fig.add_vline(
                x=arb.break_year, line_width=1.5, line_dash="dot", line_color="#e07a2c",
                annotation_text=f"Evento: {arb.break_year}", annotation_position="bottom",
            )
            if arb.implementation_lag:
                fig.add_vrect(
                    x0=arb.break_year, x1=arb.break_year + arb.implementation_lag,
                    fillcolor="#e07a2c", opacity=0.10, line_width=0,
                )

    # Cuando el gráfico muestra un único departamento (ya sea porque solo
    # hay uno seleccionado en el panel lateral, o por el modo "Comparar
    # departamentos por separado"), se incluye su nombre en el título —
    # la leyenda de Plotly solo aparece con 2+ líneas, así que sin esto
    # un gráfico de un solo departamento no indica cuál es en ningún lado.
    if len(depts_to_plot) == 1:
        chart_title = f"Casos de cáncer — {site_val} · {depts_to_plot[0]}"
    else:
        chart_title = f"Casos de cáncer — {site_val}"

    fig.update_layout(
        title=chart_title,
        xaxis_title="Año",
        yaxis_title="N° de casos nuevos",
        yaxis_type="log" if log_scale else "linear",
        barmode="group",
        height=320 if compact else 580,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=10 if compact else 12),
        ),
        hovermode="x unified",
        margin=dict(t=60 if compact else 70, b=40),
        font=dict(size=11 if compact else 13),
    )
    all_years_local = sorted(filtered_local["Anio"].unique())
    fig.update_xaxes(
        tickmode="array", tickvals=all_years_local, tickangle=45,
        tickfont=dict(size=9 if compact else 11),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"site_chart_{key_suffix}")

    if PERU_LABEL in depts_to_plot and peru_exclude:
        st.caption(
            f"ℹ️ El total de **{PERU_LABEL}** mostrado excluye: "
            f"{', '.join(peru_exclude)} (configurado en el panel lateral)."
        )

    summary_blocks = []
    if bp_summary is not None:
        sig_txt = (
            "**estadísticamente significativo**"
            if bp_summary.significant
            else "no alcanza significancia estadística"
        )
        summary_blocks.append(
            f"**📐 Punto de quiebre estimado para {bp_dept} ({site_val}): "
            f"año {bp_summary.year}** — el cambio de tendencia es {sig_txt} "
            f"(test de Chow: F = {bp_summary.f_stat:.2f}, p = {bp_summary.p_value:.4f}). "
            f"Pendiente antes: {bp_summary.slope_before:+.1f} casos/año · "
            f"pendiente después: {bp_summary.slope_after:+.1f} casos/año."
        )
    if mk_summary is not None:
        trend_es = {
            "increasing": "creciente 📈",
            "decreasing": "decreciente 📉",
            "no trend": "sin tendencia clara ➖",
        }.get(mk_summary.trend, mk_summary.trend)
        sig_txt = (
            "**estadísticamente significativa**"
            if mk_summary.significant
            else "no alcanza significancia estadística (α=0.05)"
        )
        summary_blocks.append(
            f"**🧪 Mann-Kendall para {mk_dept} ({site_val}, "
            f"{mk_year_range[0]}–{mk_year_range[1]}): tendencia {trend_es}** — {sig_txt} "
            f"(z = {mk_summary.z_stat:.2f}, p = {mk_summary.p_value:.4f}, "
            f"τ de Kendall = {mk_summary.tau:.2f}). "
            f"Pendiente de Sen: {mk_summary.sen_slope:+.1f} casos/año "
            f"(n = {mk_summary.n_obs} años; robusta a outliers, a diferencia de la "
            f"pendiente de mínimos cuadrados). Método: {mk_summary.method}."
        )
    if arb_summary is not None:
        sig_txt = (
            "**estadísticamente significativo**"
            if arb_summary.significant
            else "no alcanza significancia estadística"
        )
        lag_txt = (
            f" (excluyendo {arb_summary.implementation_lag} año(s) de "
            "implementación tras el evento)"
            if arb_summary.implementation_lag
            else ""
        )
        summary_blocks.append(
            f"**🏛️ Quiebre por evento en {arb_dept} ({site_val}): "
            f"año {arb_summary.break_year}**{lag_txt} — el cambio de tendencia es "
            f"{sig_txt} (test de Chow: F = {arb_summary.f_stat:.2f}, "
            f"p = {arb_summary.p_value:.4f}). "
            f"Pendiente antes: {arb_summary.slope_before:+.1f} casos/año · "
            f"pendiente después: {arb_summary.slope_after:+.1f} casos/año "
            f"(n antes = {arb_summary.n_before}, n después = {arb_summary.n_after})."
        )

    if summary_blocks:
        if compact:
            with st.expander("📊 Detalles del análisis estadístico"):
                for block in summary_blocks:
                    st.markdown(block)
        else:
            for block in summary_blocks:
                st.markdown(block)

    if not compact:
        st.markdown(
            '<p class="source-note">Nota: los valores corresponden a casos '
            "nuevos registrados por el INEN, no a tasas ajustadas por edad "
            "ni a la incidencia nacional total (el INEN es un centro de "
            "referencia, no cubre el 100% de los casos del país). "
            "Para tasas estandarizadas se requiere población por departamento y año.</p>",
            unsafe_allow_html=True,
        )


with tab_graph:
    _local_bp_main = st.checkbox(
        "🔍 Detectar punto de quiebre (cambio estructural) en este gráfico",
        value=False, key="local_bp_main",
        help=(
            "Además del switch del panel lateral, actívalo aquí para forzar "
            "la detección automática de quiebre en el gráfico principal sin "
            "tener que ir a la barra lateral."
        ),
    )
    render_site_analysis_chart(
        site, filtered, compact=False, key_suffix="main",
        local_breakpoint=_local_bp_main,
    )

    # -----------------------------------------------------------------
    # Gráficos adicionales: comparar simultáneamente otras localizaciones
    # del tumor primario, cada una en su propio gráfico más compacto (2
    # por fila), con el mismo análisis estadístico y personalización de
    # colores que el gráfico principal.
    # -----------------------------------------------------------------
    def _local_event_controls(widget_key: str) -> tuple[int | None, int]:
        """Control de 'quiebre por evento' independiente para un gráfico
        adicional específico — año y años de implementación propios,
        sin depender de la configuración global del panel lateral."""
        with st.expander("🏛️ Quiebre por evento en este gráfico", expanded=False):
            _on = st.checkbox(
                "Evaluar en este gráfico", value=False, key=f"local_evt_on_{widget_key}",
            )
            _yr, _lag = None, 0
            if _on:
                _yr = st.number_input(
                    "Año del evento", min_value=year_min, max_value=year_max,
                    value=min(max(year_min + 1, (year_min + year_max) // 2), year_max),
                    step=1, key=f"local_evt_year_{widget_key}",
                )
                _lag = st.slider(
                    "Años de implementación a excluir", 0, 5, 0,
                    key=f"local_evt_lag_{widget_key}",
                )
        return (int(_yr) if _on else None), int(_lag)

    def _local_breakpoint_control(widget_key: str) -> bool:
        """Checkbox de detección automática de quiebre, independiente
        para un gráfico adicional específico."""
        return st.checkbox(
            "🔍 Detectar punto de quiebre (cambio estructural) en este gráfico",
            value=False, key=f"local_bp_{widget_key}",
        )

    extra_sites_to_plot = [s for s in extra_sites if s != site]
    if show_extra_sites and extra_sites_to_plot:
        st.markdown("---")
        st.markdown("#### Comparar con otras localizaciones")
        for row_start in range(0, len(extra_sites_to_plot), 2):
            row_sites = extra_sites_to_plot[row_start : row_start + 2]
            cols = st.columns(len(row_sites))
            for col, extra_site in zip(cols, row_sites):
                with col:
                    extra_frames = []
                    for dept in depts_selected:
                        sub_extra = dept_casos_series(
                            dept, extra_site, year_range[0], year_range[1]
                        ).copy()
                        sub_extra["Departamento"] = dept
                        extra_frames.append(sub_extra)
                    extra_filtered = (
                        pd.concat(extra_frames, ignore_index=True)
                        if extra_frames
                        else pd.DataFrame(columns=["Anio", "Casos", "Departamento"])
                    )
                    _local_bp_site = _local_breakpoint_control(f"site_{extra_site}")
                    _le_year, _le_lag = _local_event_controls(f"site_{extra_site}")
                    render_site_analysis_chart(
                        extra_site, extra_filtered, compact=True,
                        key_suffix=f"extra_{extra_site}",
                        local_event_year=_le_year, local_event_lag=_le_lag,
                        local_breakpoint=_local_bp_site,
                    )

    # -----------------------------------------------------------------
    # Gráficos adicionales: comparar simultáneamente varios departamentos,
    # cada uno en su propio gráfico (en vez de mezclados como líneas en
    # el gráfico principal), para la misma localización elegida arriba.
    # -----------------------------------------------------------------
    if show_dept_split and dept_split_list:
        st.markdown("---")
        st.markdown("#### Comparar departamentos por separado")
        for row_start in range(0, len(dept_split_list), 2):
            row_depts = dept_split_list[row_start : row_start + 2]
            cols = st.columns(len(row_depts))
            for col, one_dept in zip(cols, row_depts):
                with col:
                    dept_series = dept_casos_series(
                        one_dept, site, year_range[0], year_range[1]
                    ).copy()
                    dept_series["Departamento"] = one_dept
                    _local_bp_dept = _local_breakpoint_control(f"dept_{one_dept}")
                    _le_year, _le_lag = _local_event_controls(f"dept_{one_dept}")
                    render_site_analysis_chart(
                        site, dept_series, compact=True,
                        key_suffix=f"dept_{one_dept}",
                        depts_override=[one_dept],
                        local_event_year=_le_year, local_event_lag=_le_lag,
                        local_breakpoint=_local_bp_dept,
                    )
    elif show_dept_split and not dept_split_list:
        st.info(
            "Selecciona uno o más departamentos en **'Departamentos "
            "adicionales a mostrar'** (panel lateral) para ver esta comparación."
        )

with tab_table:
    pivot = filtered.pivot_table(
        index="Departamento", columns="Anio", values="Casos", aggfunc="sum"
    )
    st.dataframe(pivot.style.format("{:,.0f}", na_rep="—"), use_container_width=True)

with tab_ranking:
    st.markdown("#### Ranking de localizaciones de cáncer por año y departamento")
    st.caption(
        "Compara qué tipos de cáncer concentran más casos en un año y "
        "departamento específicos (similar al ranking por país de GLOBOCAN, "
        "aquí aplicado a localizaciones del tumor primario dentro de un departamento)."
    )
    years_all_desc = sorted(df["Anio"].unique().tolist(), reverse=True)
    depts_all = departamentos(df)
    all_sites = localizaciones(df)[1:]  # sin "Todas las localizaciones"

    rc1, rc2, rc3 = st.columns([1, 1.4, 1.2])
    with rc1:
        rank_year = st.selectbox("Año", options=years_all_desc, index=0, key="rank_year")
    with rc2:
        rank_dept = st.selectbox(
            "Departamento",
            options=depts_all,
            index=depts_all.index(PERU_LABEL),
            key="rank_dept",
        )
    with rc3:
        rank_top_n = st.slider("Top N localizaciones", 5, 38, 15, key="rank_top_n")

    rank_exclude = st.multiselect(
        "Excluir localización(es) del ranking",
        options=all_sites,
        default=[],
        key="rank_exclude",
        help=(
            "Útil para quitar categorías genéricas como 'Otros' o "
            "'Primario Desconocido', que suelen concentrar muchos casos "
            "sin ser clínicamente específicas, y así ver mejor el "
            "ranking de los tipos de cáncer concretos."
        ),
    )

    rank_data = get_year_ranking(rank_dept, rank_year, exclude=rank_exclude, top_n=rank_top_n)

    if rank_dept == PERU_LABEL and peru_exclude:
        st.caption(
            f"ℹ️ El total de Perú excluye: {', '.join(peru_exclude)} "
            "(configurado en el panel lateral)."
        )

    if rank_data.empty:
        st.warning(
            f"No hay datos registrados para **{rank_dept}** en **{rank_year}**"
            + (" con las localizaciones excluidas actuales." if rank_exclude else ".")
        )
    else:
        n_bars = len(rank_data)
        rank_base_color = (
            color_overrides.get(rank_dept, PRIMARY_COLOR) if custom_colors else PRIMARY_COLOR
        )
        bar_colors = shades_of(rank_base_color, n_bars)
        ranks = list(range(1, n_bars + 1))
        bar_text = [f"{r}° · {c:,.0f}" for r, c in zip(ranks, rank_data["Casos"])]

        fig_rank = go.Figure(
            go.Bar(
                x=rank_data["Casos"],
                y=rank_data["Localizacion"],
                orientation="h",
                marker_color=bar_colors,
                text=bar_text,
                textposition="outside",
                textfont=dict(size=15),
                hovertemplate="%{y}: %{x:,.0f} casos<extra></extra>",
            )
        )
        fig_rank.update_layout(
            title=f"Ranking de cánceres — {rank_dept}, {rank_year}",
            xaxis_title="N° de casos nuevos",
            yaxis_title="",
            yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
            height=max(420, 26 * len(rank_data)),
            template="plotly_white",
            margin=dict(l=10, r=80, t=60, b=40),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        rank_table = rank_data[["Localizacion", "Casos"]].reset_index(drop=True)
        rank_table.insert(0, "Puesto", ranks)
        rank_table = rank_table.rename(
            columns={"Localizacion": "Localización del tumor primario"}
        )
        st.dataframe(
            rank_table.style.format({"Casos": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

        rank_csv = rank_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar ranking (CSV)",
            data=rank_csv,
            file_name=f"ranking_{rank_dept.replace(' ', '_')}_{rank_year}.csv",
            mime="text/csv",
        )

with tab_ranking_anim:
    st.markdown("#### Ranking animado — evolución de los cánceres en el tiempo")
    st.caption(
        "Muestra cómo cambia el ranking de localizaciones del tumor primario "
        "año a año ('bar chart race'). Reproduce la animación directamente "
        "aquí, o descárgala como GIF."
    )

    years_all_asc = sorted(df["Anio"].unique().tolist())

    ac1, ac2 = st.columns([1.6, 1])
    with ac1:
        anim_dept = st.selectbox(
            "Departamento",
            options=depts_all,
            index=depts_all.index(PERU_LABEL),
            key="anim_dept",
        )
    with ac2:
        anim_top_n = st.slider("Top N localizaciones", 5, 15, 8, key="anim_top_n")

    anim_year_range = st.select_slider(
        "Rango de años a animar",
        options=years_all_asc,
        value=(years_all_asc[0], years_all_asc[-1]),
        key="anim_year_range",
    )

    anim_exclude = st.multiselect(
        "Excluir localización(es) de la animación",
        options=all_sites,
        default=[],
        key="anim_exclude",
    )

    anim_dept_exclude: list[str] = []
    if anim_dept == PERU_LABEL:
        anim_dept_exclude = st.multiselect(
            "Excluir departamento(s) del total nacional (ej. Extranjero)",
            options=[d for d in depts_all if d != PERU_LABEL],
            default=peru_exclude,
            key="anim_dept_exclude",
            help=(
                "Igual que en el panel lateral, pero específico para esta "
                "animación. Por defecto trae lo configurado en el panel "
                "lateral; puedes cambiarlo aquí sin afectar el gráfico principal."
            ),
        )

    anim_seconds = st.slider(
        "Velocidad: segundos por año",
        min_value=0.2,
        max_value=2.0,
        value=0.7,
        step=0.1,
        key="anim_seconds",
        help="Valores bajos = animación más rápida. Aplica tanto a la vista interactiva como al GIF descargable.",
    )

    if anim_dept == PERU_LABEL and anim_dept_exclude:
        st.caption(f"ℹ️ El total de Perú excluye: {', '.join(anim_dept_exclude)}.")

    generate_anim = st.button("🎬 Generar ranking animado", key="anim_generate")

    if generate_anim:
        anim_years = list(range(anim_year_range[0], anim_year_range[1] + 1))
        if len(anim_years) < 2:
            st.warning("Selecciona un rango de al menos 2 años para animar.")
        else:
            with st.spinner("Generando animación..."):
                year_rankings = {}
                universe: set[str] = set()
                for y in anim_years:
                    r = get_year_ranking(
                        anim_dept, y, exclude=anim_exclude, peru_excl_override=anim_dept_exclude
                    )
                    year_rankings[y] = r.set_index("Localizacion")["Casos"]
                    universe.update(r.head(anim_top_n)["Localizacion"].tolist())
                universe = sorted(universe)

                if not universe:
                    st.warning(
                        f"No hay datos para **{anim_dept}** en el rango "
                        f"{anim_year_range[0]}–{anim_year_range[1]} con los "
                        "filtros actuales."
                    )
                else:
                    global_max = 0.0
                    for y in anim_years:
                        v = (
                            year_rankings[y]
                            .reindex(universe)
                            .fillna(0)
                            .sort_values(ascending=False)
                            .head(anim_top_n)
                        )
                        if len(v):
                            global_max = max(global_max, float(v.max()))
                    if global_max <= 0:
                        global_max = 1.0

                    # Duración de cada cuadro vs. duración de la transición:
                    # una transición que ocupa la mayor parte del intervalo
                    # (con easing) es lo que hace que el movimiento se vea
                    # fluido en vez de "saltar" de un año a otro.
                    frame_duration_ms = int(anim_seconds * 1000)
                    transition_ms = max(50, int(frame_duration_ms * 0.85))

                    year_annotation = dict(
                        text=str(anim_years[0]),
                        xref="x domain",
                        yref="y domain",
                        x=0.99,
                        y=0.03,
                        xanchor="right",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=58, color="black", family="Arial Black, Arial"),
                    )

                    anim_base_color = (
                        color_overrides.get(anim_dept, PRIMARY_COLOR)
                        if custom_colors
                        else PRIMARY_COLOR
                    )

                    frames = []
                    for y in anim_years:
                        vals = year_rankings[y].reindex(universe).fillna(0)
                        ordered = vals.sort_values(ascending=False).head(anim_top_n)
                        names = ordered.index.tolist()
                        values = ordered.values.tolist()
                        n = len(names)
                        y_pos = list(range(n, 0, -1))
                        colors = shades_of(anim_base_color, n)
                        text = [
                            f"{r}° {name}: {val:,.0f}"
                            for r, (name, val) in enumerate(zip(names, values), start=1)
                        ]
                        frame_annotation = dict(year_annotation, text=str(y))
                        frames.append(
                            go.Frame(
                                name=str(y),
                                data=[
                                    go.Bar(
                                        x=values,
                                        y=y_pos,
                                        orientation="h",
                                        marker_color=colors,
                                        text=text,
                                        textposition="outside",
                                        textfont=dict(size=14),
                                        hovertemplate="%{text}<extra></extra>",
                                    )
                                ],
                                layout=go.Layout(annotations=[frame_annotation]),
                            )
                        )

                    fig_anim = go.Figure(
                        data=frames[0].data,
                        layout=go.Layout(
                            title=f"Ranking de cánceres — {anim_dept}",
                            annotations=[year_annotation],
                            xaxis=dict(
                                title="N° de casos nuevos", range=[0, global_max * 1.2]
                            ),
                            yaxis=dict(
                                showticklabels=False,
                                range=[0.3, anim_top_n + 0.9],
                            ),
                            template="plotly_white",
                            height=max(420, 34 * anim_top_n) + 140,
                            margin=dict(l=20, r=160, t=70, b=170),
                            updatemenus=[
                                dict(
                                    type="buttons",
                                    direction="right",
                                    showactive=False,
                                    x=0.5,
                                    xanchor="center",
                                    y=-0.16,
                                    yanchor="top",
                                    pad=dict(t=10, r=10),
                                    buttons=[
                                        dict(
                                            label="▶ Reproducir",
                                            method="animate",
                                            args=[
                                                None,
                                                {
                                                    "frame": {
                                                        "duration": frame_duration_ms,
                                                        "redraw": True,
                                                    },
                                                    "fromcurrent": True,
                                                    "transition": {
                                                        "duration": transition_ms,
                                                        "easing": "cubic-in-out",
                                                    },
                                                },
                                            ],
                                        ),
                                        dict(
                                            label="⏸ Pausar",
                                            method="animate",
                                            args=[
                                                [None],
                                                {
                                                    "frame": {"duration": 0, "redraw": False},
                                                    "mode": "immediate",
                                                },
                                            ],
                                        ),
                                    ],
                                )
                            ],
                            sliders=[
                                dict(
                                    active=0,
                                    x=0.0,
                                    y=-0.28,
                                    len=1.0,
                                    currentvalue=dict(prefix="Año: ", font=dict(size=13)),
                                    steps=[
                                        dict(
                                            method="animate",
                                            label=str(y),
                                            args=[
                                                [str(y)],
                                                {
                                                    "frame": {
                                                        "duration": frame_duration_ms,
                                                        "redraw": True,
                                                    },
                                                    "mode": "immediate",
                                                    "transition": {
                                                        "duration": transition_ms,
                                                        "easing": "cubic-in-out",
                                                    },
                                                },
                                            ],
                                        )
                                        for y in anim_years
                                    ],
                                )
                            ],
                        ),
                        frames=frames,
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)

                    if _GIF_EXPORT_AVAILABLE:
                        with st.spinner("Generando GIF descargable..."):
                            images = []
                            for y in anim_years:
                                vals = year_rankings[y].reindex(universe).fillna(0)
                                ordered = vals.sort_values(ascending=False).head(anim_top_n)
                                names = ordered.index.tolist()
                                values = ordered.values.tolist()
                                n = len(names)
                                colors = shades_of(anim_base_color, n)

                                mpl_fig, ax = plt.subplots(
                                    figsize=(9, 0.5 * anim_top_n + 1.4), dpi=110
                                )
                                ax.barh(range(n), values, color=colors)
                                ax.set_yticks(range(n))
                                ax.set_yticklabels(
                                    [f"{r}° {name}" for r, name in enumerate(names, start=1)],
                                    fontsize=12,
                                )
                                ax.set_xlim(0, global_max * 1.2)
                                for i, val in enumerate(values):
                                    ax.text(
                                        val + global_max * 0.01, i, f"{val:,.0f}",
                                        va="center", fontsize=11,
                                    )
                                ax.set_title(anim_dept, fontsize=13, loc="left", color="#555")
                                ax.text(
                                    0.98, 0.04, str(y),
                                    transform=ax.transAxes,
                                    fontsize=46, fontweight="black",
                                    color="black", ha="right", va="bottom",
                                )
                                ax.set_xlabel("N° de casos nuevos")
                                ax.invert_yaxis()
                                ax.spines[["top", "right"]].set_visible(False)
                                mpl_fig.tight_layout()

                                buf = io.BytesIO()
                                mpl_fig.savefig(buf, format="png")
                                plt.close(mpl_fig)
                                buf.seek(0)
                                images.append(imageio.imread(buf))

                            gif_buf = io.BytesIO()
                            imageio.mimsave(
                                gif_buf, images, format="GIF", duration=anim_seconds
                            )
                            gif_buf.seek(0)
                            gif_bytes = gif_buf.getvalue()

                        st.download_button(
                            "Descargar animación (GIF)",
                            data=gif_bytes,
                            file_name=f"ranking_animado_{anim_dept.replace(' ', '_')}_{anim_years[0]}_{anim_years[-1]}.gif",
                            mime="image/gif",
                        )
                    else:
                        st.info(
                            "La descarga en GIF no está disponible en este entorno "
                            "(faltan las librerías matplotlib/imageio), pero la "
                            "animación interactiva de arriba funciona igual."
                        )

def render_projection_chart(site_val: str, compact: bool = False, key_suffix: str = "") -> None:
    """Construye y renderiza la proyección de casos (3 escenarios +
    banda de incertidumbre) para una localización del tumor primario
    dada. La usan tanto la proyección principal como las proyecciones
    adicionales de otras localizaciones."""
    sub_proj = dept_casos_series(proj_target, site_val, year_range[0], year_range[1])
    proj = project_series(
        sub_proj["Anio"].to_numpy(), sub_proj["Casos"].to_numpy(), horizon=proj_horizon
    )
    if proj is None:
        st.warning(
            f"No hay suficientes años con datos en **{proj_target}** ({site_val}) "
            "para proyectar (se requieren al menos 6)."
        )
        return

    st.markdown(
        f"#### Proyección de casos — {site_val} · {proj_target} "
        f"({int(proj.years_future[0])}–{int(proj.years_future[-1])})"
    )
    if not compact:
        st.caption(proj.method_note)

    fig_proj = go.Figure()
    fig_proj.add_trace(
        go.Scatter(
            x=proj.years_hist, y=proj.values_hist, name="Histórico",
            mode="lines+markers", line=dict(width=2.0 if compact else 2.5, color=ACCENT_COLOR),
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=np.concatenate([proj.years_future, proj.years_future[::-1]]),
            y=np.concatenate([proj.recommended_upper, proj.recommended_lower[::-1]]),
            fill="toself", fillcolor="rgba(47,111,168,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Banda de incertidumbre (90%)", hoverinfo="skip",
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=proj.years_future, y=proj.recommended,
            name="Recomendado (Holt amortiguado)",
            mode="lines+markers", line=dict(width=2.0 if compact else 2.5, color=SECONDARY_COLOR),
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=proj.years_future, y=proj.conservative,
            name="Conservador (promedio reciente, plano)",
            mode="lines", line=dict(width=2, color="#7d7d7d", dash="dot"),
        )
    )
    fig_proj.add_trace(
        go.Scatter(
            x=proj.years_future, y=proj.optimistic,
            name="Lineal (pendiente de Sen, sin amortiguar)",
            mode="lines", line=dict(width=2, color="#c94141", dash="dash"),
        )
    )
    fig_proj.update_layout(
        xaxis_title="Año", yaxis_title="N° de casos (proyectado)",
        height=320 if compact else 520,
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=10 if compact else 12),
        ),
        hovermode="x unified", margin=dict(t=40, b=40),
        font=dict(size=11 if compact else 13),
    )
    all_proj_years = list(proj.years_hist) + list(proj.years_future)
    fig_proj.update_xaxes(
        tickmode="array", tickvals=all_proj_years, tickangle=45,
        tickfont=dict(size=9 if compact else 11),
    )
    st.plotly_chart(fig_proj, use_container_width=True, key=f"proj_chart_{key_suffix}")

    table_proj = pd.DataFrame(
        {
            "Año": proj.years_future.astype(int),
            "Conservador": proj.conservative.round(0),
            "Recomendado": proj.recommended.round(0),
            "Recomendado (banda 90%)": [
                f"{lo:,.0f} – {hi:,.0f}"
                for lo, hi in zip(proj.recommended_lower, proj.recommended_upper)
            ],
            "Lineal (Sen)": proj.optimistic.round(0),
        }
    ).set_index("Año")

    if compact:
        with st.expander("📋 Tabla de escenarios"):
            st.dataframe(table_proj, use_container_width=True)
    else:
        st.markdown("##### Tabla de escenarios")
        st.dataframe(table_proj, use_container_width=True)

    proj_csv = table_proj.to_csv().encode("utf-8")
    st.download_button(
        "Descargar proyección (CSV)",
        data=proj_csv,
        file_name=f"proyeccion_{site_val.replace(' ', '_')}_{proj_target.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"proj_dl_{key_suffix}",
    )

    if not compact:
        st.markdown(
            '<p class="source-note">⚠️ Toda proyección es una extrapolación '
            "estadística del comportamiento histórico y no reemplaza la "
            "planificación epidemiológica basada en programas de tamizaje, "
            "cambios demográficos u otros factores no capturados por el "
            "modelo. Úsala como referencia de escenarios, no como cifra única.</p>",
            unsafe_allow_html=True,
        )


with tab_projection:
    if not show_projection or not proj_target:
        st.info(
            "Activa **'Proyectar casos a futuro'** en el panel lateral para ver "
            "esta sección. Se recomienda elegir una localización del tumor primario con serie "
            "relativamente completa (pocos años sin datos)."
        )
    else:
        render_projection_chart(site, compact=False, key_suffix="main")

        extra_sites_to_plot = [s for s in extra_sites if s != site]
        if show_extra_sites and extra_sites_to_plot:
            st.markdown("---")
            st.markdown("#### Proyección para otras localizaciones")
            for row_start in range(0, len(extra_sites_to_plot), 2):
                row_sites = extra_sites_to_plot[row_start : row_start + 2]
                cols = st.columns(len(row_sites))
                for col, extra_site in zip(cols, row_sites):
                    with col:
                        render_projection_chart(
                            extra_site, compact=True, key_suffix=f"extra_{extra_site}"
                        )

with tab_downloads:
    st.write("Descarga los datos filtrados actualmente en el panel:")
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV (formato largo)",
        data=csv_bytes,
        file_name=f"cancer_inen_{site.replace(' ', '_')}.csv",
        mime="text/csv",
    )
    pivot_csv = pivot.to_csv().encode("utf-8")
    st.download_button(
        "Descargar CSV (tabla dinámica: departamento x año)",
        data=pivot_csv,
        file_name=f"cancer_inen_{site.replace(' ', '_')}_tabla.csv",
        mime="text/csv",
    )
    st.write("Descarga el dataset completo (todas las localizaciones del tumor primario y departamentos):")
    full_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar dataset completo (CSV)",
        data=full_csv,
        file_name="cancer_inen_dataset_completo.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Exploración (base adicional) y Análisis cruzado
# ---------------------------------------------------------------------------
# Adaptados del dashboard hermano "Observatorio del Financiamiento
# Oncológico INEN" a la granularidad ANUAL de este proyecto (aquí no hay
# concepto de "normas"/puntos de corte regulatorios, así que esa parte del
# dashboard hermano no aplica; en su lugar se reutilizan las mismas
# herramientas estadísticas ya presentes en este dashboard — tendencia
# lineal, quiebre automático, Mann-Kendall, LOWESS).

OK_COLOR = "#3ba776"
WARN_COLOR = "#c9a13b"
BAD_COLOR = "#c94141"

with tab_explore_sec:
    st.markdown("#### Exploración de la base de datos adicional")

    if "secondary_serie" not in st.session_state:
        st.info(
            "Aún no se ha cargado ninguna base de datos adicional. "
            "Usa el panel lateral → **📊 Base de datos adicional** para subir un archivo CSV o Excel."
        )
    else:
        _sec_s: pd.Series = st.session_state["secondary_serie"]
        _sec_lbl: str = st.session_state.get("secondary_label", "Serie adicional")
        _sec_units_lbl: str = st.session_state.get("secondary_units", "Valor")
        _sec_cols_dict_kpi: dict = st.session_state.get("secondary_cols_dict", {})
        _sec_val_cols_kpi: list = st.session_state.get("secondary_val_cols", [])

        st.caption(
            f"**{_sec_lbl}** · {len(_sec_s)} año(s) · "
            f"{int(_sec_s.index.min())} a {int(_sec_s.index.max())}"
        )

        def _cagr_sec(s: pd.Series) -> str:
            _snn = s[s > 0].dropna()
            if len(_snn) < 2:
                return "s/d"
            _n = _snn.index[-1] - _snn.index[0]
            if _n <= 0 or _snn.iloc[0] <= 0:
                return "s/d"
            return f"{((_snn.iloc[-1] / _snn.iloc[0]) ** (1 / _n) - 1) * 100:+.1f}%/año"

        def _render_kpi_row_sec(sf: pd.Series, label: str):
            _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
            _k1.metric(f"Total — {label}", f"{sf.sum():,.2f}")
            _k2.metric("Promedio anual", f"{sf.mean():,.2f}")
            _var = (
                f"{(sf.iloc[-1] / sf.iloc[0] - 1) * 100:+.1f}%"
                if len(sf) >= 2 and sf.iloc[0] != 0 else "s/d"
            )
            _k3.metric(
                f"Variación {int(sf.index[0])} → {int(sf.index[-1])}" if len(sf) >= 2 else "Variación",
                _var,
            )
            _k4.metric("CAGR (crec. anual compuesto)", _cagr_sec(sf))
            _peak = sf.idxmax() if len(sf) else None
            _k5.metric(
                "Año pico",
                f"{int(_peak)}" if _peak is not None else "s/d",
                f"{sf.max():,.2f}" if len(sf) else "",
            )
            _mk_r = mann_kendall_trend(sf.index.to_numpy(), sf.to_numpy()) if len(sf) >= 6 else None
            _k6.metric(
                "Tendencia (Mann-Kendall)",
                _mk_r.trend if _mk_r else "s/d",
                f"p={_mk_r.p_value:.3f}" if _mk_r else "",
            )
            return _mk_r

        _mk_by_col: dict = {}
        if len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi:
            for _kpi_col in _sec_val_cols_kpi:
                _ks = _sec_cols_dict_kpi.get(_kpi_col)
                if _ks is None:
                    continue
                st.caption(f"**{_kpi_col}**")
                _excl_col = st.multiselect(
                    "Excluir años", options=sorted(_ks.index.tolist()), default=[],
                    key=f"xp_excl_{_kpi_col}", label_visibility="collapsed",
                    placeholder="Excluir años del análisis (opcional)…",
                )
                _ks_f = _ks[~_ks.index.isin(_excl_col)] if _excl_col else _ks
                _mk_by_col[_kpi_col] = _render_kpi_row_sec(_ks_f, _kpi_col)
            _kpi_lbl = _sec_val_cols_kpi[0]
            _mk_sec = _mk_by_col.get(_kpi_lbl)
        else:
            _excl_single = st.multiselect(
                "Excluir años del análisis KPI / estadísticas",
                options=sorted(_sec_s.index.tolist()), default=[], key="xp_excl_single",
                placeholder="Excluir años del análisis (opcional)…",
            )
            _ks_single = _sec_s[~_sec_s.index.isin(_excl_single)] if _excl_single else _sec_s
            _kpi_lbl = _sec_lbl
            _mk_sec = _render_kpi_row_sec(_ks_single, _sec_lbl)
            _mk_by_col[_sec_lbl] = _mk_sec

        st.divider()

        _xp_c1, _xp_c2, _xp_c3 = st.columns(3)
        _xp_chart = _xp_c1.radio("Tipo de gráfico", ["Línea", "Barras"], horizontal=True, key="xp_chart")
        _xp_markers = _xp_c2.checkbox("Marcadores", value=True, key="xp_markers")
        _xp_lowess = _xp_c3.checkbox("LOWESS (suavizado)", value=False, key="xp_lowess")
        _xp_log = _xp_c1.checkbox("Escala logarítmica", value=False, key="xp_log")

        _xp_lo, _xp_hi = int(_sec_s.index.min()), int(_sec_s.index.max())
        if _xp_lo < _xp_hi:
            _xp_range = st.slider(
                "Rango de años (exploración)", min_value=_xp_lo, max_value=_xp_hi,
                value=(_xp_lo, _xp_hi), key="xp_range",
            )
            _sec_s_filt = _sec_s[(_sec_s.index >= _xp_range[0]) & (_sec_s.index <= _xp_range[1])]
        else:
            st.caption(f"Serie de un único año: {_xp_lo}")
            _xp_range = (_xp_lo, _xp_hi)
            _sec_s_filt = _sec_s.copy()

        def _filtered_for_stats_sec(col_name: str, raw_s: pd.Series) -> pd.Series:
            _s2 = raw_s[(raw_s.index >= _xp_range[0]) & (raw_s.index <= _xp_range[1])]
            _excl_key = f"xp_excl_{col_name}" if len(_sec_val_cols_kpi) > 1 else "xp_excl_single"
            _excl_yrs = st.session_state.get(_excl_key, [])
            if _excl_yrs:
                _s2 = _s2[~_s2.index.isin(_excl_yrs)]
            return _s2

        _xp_show_cols: list[str] = []
        if len(_sec_val_cols_kpi) > 1:
            _xp_show_cols = st.multiselect(
                "Columnas a graficar individualmente", options=_sec_val_cols_kpi,
                default=_sec_val_cols_kpi, key="xp_show_cols",
            )

        _desc_col, _dist_col = st.columns(2)
        _rename_map = {
            "count": "N obs.", "mean": "Media", "std": "Desv. estándar",
            "min": "Mínimo", "25%": "Percentil 25", "50%": "Mediana",
            "75%": "Percentil 75", "max": "Máximo",
        }
        with _desc_col:
            st.markdown("**Estadísticas descriptivas**")
            if len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi:
                _desc_frames = {}
                for _dcol in _sec_val_cols_kpi:
                    _ds_raw = _sec_cols_dict_kpi.get(_dcol)
                    if _ds_raw is None:
                        continue
                    _desc_frames[_dcol] = _filtered_for_stats_sec(_dcol, _ds_raw).describe().rename(_rename_map)
                if _desc_frames:
                    st.dataframe(pd.DataFrame(_desc_frames).style.format("{:,.4f}"), use_container_width=True)
                for _mkcol, _mkres in _mk_by_col.items():
                    if _mkres:
                        st.caption(
                            f"Mann-Kendall **{_mkcol}**: tendencia **{_mkres.trend}** "
                            f"(p={_mkres.p_value:.4f}, método: {_mkres.method}). "
                            f"Pendiente de Sen: **{_mkres.sen_slope:,.4f}** u/año."
                        )
            else:
                _desc = _filtered_for_stats_sec(_sec_lbl, _sec_s).describe().rename(_rename_map)
                st.dataframe(_desc.to_frame(name=_sec_lbl).style.format("{:,.4f}"), use_container_width=True)
                if _mk_sec:
                    st.caption(
                        f"Mann-Kendall ({_kpi_lbl}): tendencia **{_mk_sec.trend}** "
                        f"(p={_mk_sec.p_value:.4f}, método: {_mk_sec.method}). "
                        f"Pendiente de Sen: **{_mk_sec.sen_slope:,.4f}** u/año."
                    )

        with _dist_col:
            st.markdown("**Distribución**")
            _hist_cols = (
                list(_sec_cols_dict_kpi.items())
                if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi)
                else [(_sec_lbl, _sec_s)]
            )
            _hist_pairs = [_hist_cols[i:i + 2] for i in range(0, len(_hist_cols), 2)]
            for _hrow in _hist_pairs:
                _hcols_ui = st.columns(len(_hrow))
                for _hci, (_hcol, _hs) in enumerate(_hrow):
                    _hs_filt = _filtered_for_stats_sec(_hcol, _hs)
                    _hvnz = _hs_filt[_hs_filt > 0]
                    _hcolor = DEFAULT_PALETTE[list(dict(_hist_cols).keys()).index(_hcol) % len(DEFAULT_PALETTE)]
                    _fig_h = go.Figure()
                    if len(_hvnz) > 0:
                        _fig_h.add_trace(go.Histogram(x=_hvnz.values, nbinsx=12, name=_hcol, marker_color=_hcolor, opacity=0.85))
                        _fig_h.add_vline(x=float(_hvnz.mean()), line_dash="dash", line_color=SECONDARY_COLOR, annotation_text="Media")
                        _fig_h.add_vline(x=float(_hvnz.median()), line_dash="dot", line_color=OK_COLOR, annotation_text="Mediana")
                    _fig_h.update_layout(
                        title=dict(text=_hcol, font=dict(size=12)), height=260,
                        xaxis_title=_sec_units_lbl, yaxis_title="Frecuencia",
                        margin=dict(t=35, b=30), showlegend=False,
                    )
                    _hcols_ui[_hci].plotly_chart(_fig_h, use_container_width=True)

        st.divider()

        _fig_sec = go.Figure()
        _mode_sec = "lines+markers" if _xp_markers else "lines"
        if _xp_show_cols and _sec_cols_dict_kpi:
            for _ci, _col_name in enumerate(_xp_show_cols):
                _col_s = _sec_cols_dict_kpi.get(_col_name)
                if _col_s is None:
                    continue
                _col_s_filt = _col_s[(_col_s.index >= _xp_range[0]) & (_col_s.index <= _xp_range[1])]
                _col_color = DEFAULT_PALETTE[_ci % len(DEFAULT_PALETTE)]
                if _xp_chart == "Barras":
                    _fig_sec.add_trace(go.Bar(x=_col_s_filt.index, y=_col_s_filt.values, name=_col_name, marker_color=_col_color))
                else:
                    _fig_sec.add_trace(go.Scatter(x=_col_s_filt.index, y=_col_s_filt.values, mode=_mode_sec, name=_col_name, line=dict(width=2, color=_col_color), marker=dict(size=5)))
        else:
            if _xp_chart == "Barras":
                _fig_sec.add_trace(go.Bar(x=_sec_s_filt.index, y=_sec_s_filt.values, name=_sec_lbl, marker_color=PRIMARY_COLOR))
            else:
                _fig_sec.add_trace(go.Scatter(x=_sec_s_filt.index, y=_sec_s_filt.values, mode=_mode_sec, name=_sec_lbl, line=dict(width=2, color=PRIMARY_COLOR), marker=dict(size=5)))

        if _xp_lowess and len(_sec_s_filt) >= 4:
            _sm_sec = smooth_series(_sec_s_filt.index.to_numpy(), _sec_s_filt.to_numpy())
            if _sm_sec is not None:
                _fig_sec.add_trace(go.Scatter(x=_sm_sec[0], y=_sm_sec[1], mode="lines", name="LOWESS", line=dict(width=2, color=OK_COLOR)))

        _fig_sec.update_layout(
            height=420, hovermode="x unified", yaxis_title=_sec_units_lbl,
            yaxis_type="log" if _xp_log else "linear",
            xaxis=dict(dtick=1, tickangle=-45),
            legend=dict(orientation="h", y=-0.25), margin=dict(t=30), barmode="group",
        )
        st.plotly_chart(_fig_sec, use_container_width=True)

        st.divider()

        st.markdown("#### 📈 Análisis de tendencia detallado")
        _trend_cols_list = (
            list(_sec_cols_dict_kpi.items())
            if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi)
            else [(_sec_lbl, _sec_s)]
        )
        for _tcol, _ts_raw in _trend_cols_list:
            _ts = _filtered_for_stats_sec(_tcol, _ts_raw)
            with st.expander(f"🔎 Tendencia — {_tcol}", expanded=False):
                if len(_ts) < 6:
                    st.info("Se necesitan al menos 6 años para el análisis de tendencia (Mann-Kendall).")
                    continue
                _mk_t = mann_kendall_trend(_ts.index.to_numpy(), _ts.to_numpy())
                _t1, _t2, _t3 = st.columns(3)
                _t1.metric("Tendencia (Mann-Kendall)", _mk_t.trend if _mk_t else "s/d")
                _t2.metric("p-valor", f"{_mk_t.p_value:.4f}" if _mk_t else "s/d", significance_label(_mk_t.p_value) if _mk_t else "")
                _t3.metric("Pendiente de Sen (u/año)", f"{_mk_t.sen_slope:,.4f}" if _mk_t else "s/d")

                _trend_lin = linear_trend(_ts.index.to_numpy(), _ts.to_numpy())
                _fig_t = go.Figure()
                _fig_t.add_trace(go.Scatter(x=_ts.index, y=_ts.values, mode="lines+markers", name=_tcol, line=dict(width=2, color=PRIMARY_COLOR), marker=dict(size=5)))
                if _trend_lin is not None:
                    _fig_t.add_trace(go.Scatter(x=_trend_lin.x, y=_trend_lin.y_pred, mode="lines", name=f"Tendencia lineal (R²={_trend_lin.r2:.2f})", line=dict(width=2, color=SECONDARY_COLOR, dash="dash")))
                _fig_t.update_layout(height=300, hovermode="x unified", yaxis_title=_sec_units_lbl, xaxis=dict(dtick=1, tickangle=-45), legend=dict(orientation="h", y=-0.3), margin=dict(t=20))
                st.plotly_chart(_fig_t, use_container_width=True)
                if _mk_t:
                    sig_word = significance_label(_mk_t.p_value)
                    st.caption(
                        f"Tendencia **{_mk_t.trend}** (p={_mk_t.p_value:.4f}, método: {_mk_t.method}). "
                        f"Pendiente de Sen: **{_mk_t.sen_slope:,.4f}** unidades/año — {sig_word.lower()}."
                    )

        st.divider()

        with st.expander("🔍 Detección automática de punto de quiebre"):
            _bp_col_options = (
                list(_sec_cols_dict_kpi.keys()) if (len(_sec_val_cols_kpi) > 1 and _sec_cols_dict_kpi) else [_sec_lbl]
            )
            if len(_bp_col_options) > 1:
                _bp_sel_col = st.selectbox("Variable a analizar", options=_bp_col_options, key="bp_sel_col")
                _bp_raw = _sec_cols_dict_kpi.get(_bp_sel_col, _sec_s)
            else:
                _bp_sel_col = _sec_lbl
                _bp_raw = _sec_s
            _bp_series = _bp_raw[(_bp_raw.index >= _xp_range[0]) & (_bp_raw.index <= _xp_range[1])]
            _auto_sec = detect_breakpoint(_bp_series.index.to_numpy(), _bp_series.to_numpy()) if len(_bp_series) >= 6 else None
            if _auto_sec is None:
                st.info("Necesitas al menos 6 años en el rango seleccionado.")
            else:
                _ac1, _ac2, _ac3 = st.columns(3)
                _ac1.metric("Año de quiebre detectado", f"{_auto_sec.year}")
                _ac2.metric("p-valor (test de Chow)", f"{_auto_sec.p_value:.4f}", significance_label(_auto_sec.p_value))
                _ac3.metric("Pendiente antes → después", f"{_auto_sec.slope_before:,.4f} → {_auto_sec.slope_after:,.4f}")
                _fig_ab = go.Figure()
                _fig_ab.add_trace(go.Scatter(x=_bp_series.index, y=_bp_series.values, mode="markers", name="Observado", marker=dict(color=PRIMARY_COLOR, size=6)))
                _fig_ab.add_trace(go.Scatter(x=_auto_sec.x_before, y=_auto_sec.y_pred_before, mode="lines", name="Ajuste antes", line=dict(color=SECONDARY_COLOR, width=2)))
                _fig_ab.add_trace(go.Scatter(x=_auto_sec.x_after, y=_auto_sec.y_pred_after, mode="lines", name="Ajuste después", line=dict(color=OK_COLOR, width=2)))
                _fig_ab.add_vline(x=_auto_sec.year, line_dash="dot", line_color="gray")
                _fig_ab.update_layout(height=320, hovermode="x unified", yaxis_title=_sec_units_lbl, margin=dict(t=20), xaxis=dict(dtick=1, tickangle=-45), legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(_fig_ab, use_container_width=True)

        st.download_button(
            "⬇️ Descargar serie adicional (CSV)",
            _sec_s_filt.reset_index().rename(columns={"index": "Anio", 0: _sec_lbl}).to_csv(index=False).encode("utf-8"),
            file_name="serie_adicional.csv", mime="text/csv", key="dl_sec_csv",
        )

with tab_cross:
    st.markdown("#### Análisis cruzado: casos de cáncer vs. base adicional")

    if "secondary_serie" not in st.session_state:
        st.info(
            "Carga una base de datos adicional desde el panel lateral "
            "(📊 **Base de datos adicional**) para habilitar este análisis."
        )
    else:
        _sec_s_x: pd.Series = st.session_state["secondary_serie"]
        _sec_lbl_x: str = st.session_state.get("secondary_label", "Serie adicional")
        _sec_units_x: str = st.session_state.get("secondary_units", "Valor")

        # Serie principal: casos totales por año para la selección actual
        # (localización + departamentos + exclusiones del panel lateral).
        _prim_s = year_series  # ya calculada más arriba (Anio -> Casos)
        _idx_common = _prim_s.index.intersection(_sec_s_x.index)

        if len(_idx_common) < 4:
            st.warning(
                f"Las dos series solo se solapan en {len(_idx_common)} año(s). "
                "Ajusta el rango de años en el panel lateral o sube una base "
                "adicional con más años en común con los casos de cáncer."
            )
        else:
            _p = _prim_s.reindex(_idx_common)
            _s = _sec_s_x.reindex(_idx_common)

            st.caption(
                f"Período de solapamiento: {int(_idx_common.min())} a "
                f"{int(_idx_common.max())} · {len(_idx_common)} año(s) en común"
            )

            st.markdown("#### Gráfico cruzado: casos de cáncer + base adicional")
            _cx0_c1, _cx0_c2, _cx0_c3 = st.columns(3)
            _cx0_markers = _cx0_c1.checkbox("Marcadores", value=True, key="cx0_markers")
            _cx0_log_l = _cx0_c2.checkbox("Log eje izq.", value=False, key="cx0_log_l")
            _cx0_log_r = _cx0_c3.checkbox("Log eje dcho.", value=False, key="cx0_log_r")

            _cx0_cols_dict = st.session_state.get("secondary_cols_dict", {})
            _cx0_val_cols = st.session_state.get("secondary_val_cols", [])
            _cx0_sel_cols: list[str] = []
            if len(_cx0_val_cols) > 1:
                _cx0_sel_cols = st.multiselect(
                    "Columnas adicionales a superponer", options=_cx0_val_cols,
                    default=_cx0_val_cols, key="cx0_sel_cols",
                )

            _cx0_mode = "lines+markers" if _cx0_markers else "lines"
            _fig_cx0 = go.Figure()
            _fig_cx0.add_trace(go.Scatter(
                x=_p.index, y=_p.values, mode=_cx0_mode, name=f"Casos — {site}",
                line=dict(color=PRIMARY_COLOR, width=2), marker=dict(size=5), yaxis="y1",
            ))
            _cx0_palette = [SECONDARY_COLOR, OK_COLOR, WARN_COLOR, "#8850c4", "#5c6ac4"]
            if _cx0_sel_cols and _cx0_cols_dict:
                for _ci, _col_name in enumerate(_cx0_sel_cols):
                    _col_s_raw = _cx0_cols_dict.get(_col_name)
                    if _col_s_raw is None:
                        continue
                    _col_aligned = _col_s_raw.reindex(_idx_common)
                    _fig_cx0.add_trace(go.Scatter(
                        x=_col_aligned.index, y=_col_aligned.values, mode=_cx0_mode, name=_col_name,
                        line=dict(color=_cx0_palette[_ci % len(_cx0_palette)], width=2), marker=dict(size=5), yaxis="y2",
                    ))
            else:
                _fig_cx0.add_trace(go.Scatter(
                    x=_s.index, y=_s.values, mode=_cx0_mode, name=_sec_lbl_x,
                    line=dict(color=SECONDARY_COLOR, width=2), marker=dict(size=5), yaxis="y2",
                ))
            _fig_cx0.update_layout(
                height=440, hovermode="x unified",
                yaxis=dict(title=dict(text=f"Casos — {site}", font=dict(color=PRIMARY_COLOR)), type="log" if _cx0_log_l else "linear", tickfont=dict(color=PRIMARY_COLOR)),
                yaxis2=dict(title=dict(text=f"{_sec_lbl_x} ({_sec_units_x})", font=dict(color=SECONDARY_COLOR)), type="log" if _cx0_log_r else "linear", overlaying="y", side="right", tickfont=dict(color=SECONDARY_COLOR)),
                legend=dict(orientation="h", y=-0.28), margin=dict(t=35),
                xaxis=dict(dtick=1, tickangle=-45),
            )
            st.plotly_chart(_fig_cx0, use_container_width=True)

            st.divider()

            st.markdown("#### 1. Comparación visual (series normalizadas, base = 100 en el primer año)")
            _cx_show_raw = st.checkbox("Mostrar también valores absolutos (doble eje)", value=False, key="cx_raw")

            _p_base = _p.iloc[0] if _p.iloc[0] != 0 else 1
            _s_base = _s.iloc[0] if _s.iloc[0] != 0 else 1
            _fig_cmp = go.Figure()
            _fig_cmp.add_trace(go.Scatter(x=_idx_common, y=_p.values / _p_base * 100, mode="lines", name=f"Casos — {site} (índice)", line=dict(color=PRIMARY_COLOR, width=2)))
            _fig_cmp.add_trace(go.Scatter(x=_idx_common, y=_s.values / _s_base * 100, mode="lines", name=f"{_sec_lbl_x} (índice)", line=dict(color=SECONDARY_COLOR, width=2)))
            if _cx_show_raw:
                _fig_cmp.add_trace(go.Scatter(x=_idx_common, y=_p.values, mode="lines", name="Casos (eje dcho.)", yaxis="y2", line=dict(color=PRIMARY_COLOR, width=1, dash="dot")))
                _fig_cmp.add_trace(go.Scatter(x=_idx_common, y=_s.values, mode="lines", name=f"{_sec_lbl_x} (eje dcho.)", yaxis="y2", line=dict(color=SECONDARY_COLOR, width=1, dash="dot")))
                _fig_cmp.update_layout(yaxis2=dict(title="Valores absolutos", overlaying="y", side="right"))
            _fig_cmp.update_layout(height=400, hovermode="x unified", yaxis_title="Índice (primer año = 100)", legend=dict(orientation="h", y=-0.25), margin=dict(t=30), xaxis=dict(dtick=1, tickangle=-45))
            st.plotly_chart(_fig_cmp, use_container_width=True)

            st.divider()

            st.markdown("#### 2. Correlación contemporánea")
            from scipy.stats import pearsonr, spearmanr

            _mask_valid = (_p > 0) & (_s > 0)
            _p_v = _p[_mask_valid].values
            _s_v = _s[_mask_valid].values

            if len(_p_v) >= 4:
                _r_pear, _p_pear = pearsonr(_p_v, _s_v)
                _r_spear, _p_spear = spearmanr(_p_v, _s_v)
                _cc1, _cc2, _cc3, _cc4 = st.columns(4)
                _cc1.metric("Pearson r", f"{_r_pear:.4f}")
                _cc2.metric("p-valor (Pearson)", f"{_p_pear:.4f}", significance_label(_p_pear))
                _cc3.metric("Spearman ρ", f"{_r_spear:.4f}")
                _cc4.metric("p-valor (Spearman)", f"{_p_spear:.4f}", significance_label(_p_spear))

                _fig_sc = go.Figure()
                _fig_sc.add_trace(go.Scatter(
                    x=_p_v, y=_s_v, mode="markers",
                    marker=dict(color=np.arange(len(_p_v)), colorscale="Blues", showscale=True, size=8, colorbar=dict(title="Año")),
                    text=[str(int(y)) for y in _idx_common[_mask_valid]],
                    hovertemplate="<b>%{text}</b><br>Casos: %{x:,.0f}<br>" + f"{_sec_lbl_x}: " + "%{y:,.2f}<extra></extra>",
                    name="Años",
                ))
                if len(_p_v) >= 3:
                    _z = np.polyfit(_p_v, _s_v, 1)
                    _x_trend = np.linspace(_p_v.min(), _p_v.max(), 100)
                    _fig_sc.add_trace(go.Scatter(x=_x_trend, y=np.polyval(_z, _x_trend), mode="lines", name=f"Tendencia (r={_r_pear:.3f})", line=dict(color=SECONDARY_COLOR, width=2, dash="dash")))
                _fig_sc.update_layout(height=380, xaxis_title=f"Casos — {site}", yaxis_title=f"{_sec_lbl_x} ({_sec_units_x})", legend=dict(orientation="h", y=-0.25), margin=dict(t=20))
                st.plotly_chart(_fig_sc, use_container_width=True)

                interp_r = (
                    "correlación positiva fuerte" if _r_pear >= 0.7
                    else "correlación positiva moderada" if _r_pear >= 0.4
                    else "correlación positiva débil" if _r_pear >= 0.1
                    else "correlación negativa fuerte" if _r_pear <= -0.7
                    else "correlación negativa moderada" if _r_pear <= -0.4
                    else "correlación negativa débil" if _r_pear <= -0.1
                    else "sin correlación lineal apreciable"
                )
                st.caption(
                    f"Las dos series muestran **{interp_r}** (Pearson r={_r_pear:.3f}, p={_p_pear:.4f}; "
                    f"Spearman ρ={_r_spear:.3f}, p={_p_spear:.4f}). El scatter está coloreado "
                    "cronológicamente — el patrón de dispersión en la dirección del tiempo puede "
                    "revelar si la asociación cambió en distintos subperíodos. Correlación no implica causalidad."
                )
            else:
                st.info("Muy pocos años con valores positivos en ambas series para calcular correlación.")

            st.divider()

            st.markdown("#### 3. Correlación cruzada con rezago (lead-lag)")
            st.caption(
                "Muestra si la serie adicional ANTICIPA o SIGUE a los casos de cáncer. "
                "Un rezago negativo (lag<0) significativo indica que la variable adicional "
                "predice los casos con ese número de años de anticipación."
            )
            _max_lag = st.slider("Rezagos máximos a explorar (años)", 1, max(1, min(6, len(_idx_common) // 3)), min(3, max(1, len(_idx_common) // 3)), key="cx_lag")

            _pv = _p.values.astype(float)
            _sv = _s.values.astype(float)
            _pv_std = (_pv - _pv.mean()) / (_pv.std() + 1e-12)
            _sv_std = (_sv - _sv.mean()) / (_sv.std() + 1e-12)
            _lags = np.arange(-_max_lag, _max_lag + 1)
            _ccf_vals = []
            for _lag in _lags:
                if _lag < 0:
                    _ccf_vals.append(float(np.corrcoef(_pv_std[-_lag:], _sv_std[:_lag])[0, 1]))
                elif _lag > 0:
                    _ccf_vals.append(float(np.corrcoef(_pv_std[:-_lag], _sv_std[_lag:])[0, 1]))
                else:
                    _ccf_vals.append(float(np.corrcoef(_pv_std, _sv_std)[0, 1]))
            _sig_band = 1.96 / np.sqrt(len(_idx_common))
            _bar_colors = [OK_COLOR if abs(v) >= _sig_band else "#AAAAAA" for v in _ccf_vals]
            _fig_ccf = go.Figure()
            _fig_ccf.add_trace(go.Bar(x=_lags, y=_ccf_vals, marker_color=_bar_colors, name="CCF"))
            _fig_ccf.add_hline(y=_sig_band, line_dash="dot", line_color=WARN_COLOR, annotation_text=f"±{_sig_band:.2f} (banda 95%)")
            _fig_ccf.add_hline(y=-_sig_band, line_dash="dot", line_color=WARN_COLOR)
            _fig_ccf.update_layout(height=320, xaxis_title="Rezago (años; negativo = serie adicional lidera)", yaxis_title="Correlación cruzada", legend=dict(orientation="h", y=-0.3), margin=dict(t=20), xaxis=dict(tickmode="linear", dtick=1))
            st.plotly_chart(_fig_ccf, use_container_width=True)

            _best_lag_idx = int(np.argmax(np.abs(_ccf_vals)))
            _best_lag = int(_lags[_best_lag_idx])
            _best_ccf = _ccf_vals[_best_lag_idx]
            if _best_lag < 0:
                _lag_interp = f"la serie adicional lidera a los casos de cáncer en {abs(_best_lag)} año(s)"
            elif _best_lag > 0:
                _lag_interp = f"los casos de cáncer lideran a la serie adicional en {_best_lag} año(s)"
            else:
                _lag_interp = "las dos series están más correlacionadas sin rezago (contemporáneamente)"
            st.caption(
                f"El rezago de máxima correlación es **{_best_lag:+d} año(s)** (r={_best_ccf:.3f}), "
                f"lo que sugiere que {_lag_interp}. Interpreta con cautela si hay pocos años en común, "
                "ya que la banda de significancia se estrecha con más observaciones."
            )

            st.divider()

            _joint_df = pd.DataFrame({"Anio": _idx_common, f"Casos_{site}": _p.values, _sec_lbl_x: _s.values})
            st.download_button(
                "⬇️ Descargar tabla cruzada alineada (CSV)",
                _joint_df.to_csv(index=False).encode("utf-8"),
                file_name="analisis_cruzado.csv", mime="text/csv", key="dl_cross_csv",
            )

# ---------------------------------------------------------------------------
# Autoría
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    <p style="text-align:center; color:#666; font-size:0.85rem;">
    © Luis A. Orrego Ferreyros, DDS, Econ., MCE, MMD, PhD(c), CQRM ·
    Epidemiólogo y Economista de la Salud ·
    Dirección de Servicios de Apoyo al Diagnóstico y Tratamiento — INEN
    </p>
    """,
    unsafe_allow_html=True,
)

_cite_year = pd.Timestamp.today().year
_cite_url = "https://mlcancerperu.streamlit.app"

st.markdown(
    f"""<div class="citation-box">
<h4>REFERENCIA WEB (formato AMA)</h4>
<p>Orrego-Ferreyros LA. Cáncer en el Tiempo — Perú: Casos Registrados por el INEN.
Instituto Nacional de Enfermedades Neoplásicas. Published {_cite_year}. Accessed [Mes Día, Año]. {_cite_url}</p>
<h4>WEB REFERENCE (AMA format)</h4>
<p>Orrego-Ferreyros LA. Cancer Over Time — Peru: Cases Registered by INEN. National Institute
of Neoplastic Diseases. Published {_cite_year}. Accessed [Month Day, Year]. {_cite_url}</p>
</div>""",
    unsafe_allow_html=True,
)

_ris_content = (
    "TY  - ELEC\n"
    "AU  - Orrego-Ferreyros, Luis Alexander\n"
    "TI  - Cáncer en el Tiempo — Perú: Casos Registrados por el INEN / "
    "Cancer Over Time — Peru: Cases Registered by INEN\n"
    f"PY  - {_cite_year}\n"
    "PB  - Instituto Nacional de Enfermedades Neoplásicas\n"
    f"UR  - {_cite_url}\n"
    "ER  - \n"
)
_bib_content = (
    f"@misc{{orregoferreyros{_cite_year}cancerentiempo,\n"
    "  author = {Orrego-Ferreyros, Luis Alexander},\n"
    "  title = {C{\\'a}ncer en el Tiempo --- Per{\\'u}: Casos Registrados por el INEN "
    "/ Cancer Over Time --- Peru: Cases Registered by INEN},\n"
    f"  year = {{{_cite_year}}},\n"
    "  publisher = {Instituto Nacional de Enfermedades Neopl{\\'a}sicas},\n"
    f"  url = {{{_cite_url}}}\n"
    "}\n"
)
_endnote_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<xml><records><record>
<ref-type name="Web Page">12</ref-type>
<contributors><authors><author>Orrego-Ferreyros, Luis Alexander</author></authors></contributors>
<titles><title>C\u00e1ncer en el Tiempo \u2014 Per\u00fa: Casos Registrados por el INEN / Cancer Over Time \u2014 Peru: Cases Registered by INEN</title></titles>
<dates><year>{_cite_year}</year></dates>
<publisher>Instituto Nacional de Enfermedades Neopl\u00e1sicas</publisher>
<urls><related-urls><url>{_cite_url}</url></related-urls></urls>
</record></records></xml>
"""
_dl1, _dl2, _dl3 = st.columns(3)
_dl1.download_button("⬇️ BibTeX (*.bib)", _bib_content, file_name="cancer_en_tiempo_inen.bib", mime="text/plain")
_dl2.download_button("⬇️ EndNote XML (*.xml)", _endnote_xml, file_name="cancer_en_tiempo_inen.xml", mime="application/xml")
_dl3.download_button("⬇️ RIS (*.ris)", _ris_content, file_name="cancer_en_tiempo_inen.ris", mime="application/x-research-info-systems")
