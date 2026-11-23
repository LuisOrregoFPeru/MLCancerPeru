"""
Motor de "Pregunta a los datos" para el dashboard Cáncer en el Tiempo.

Dos modos:

* Básico por reglas (por defecto): interpreta la pregunta en español
  (localización, departamento, años, "top N", intención) y responde con
  agregaciones calculadas directamente sobre los datos del dashboard.
* IA (opcional): toma la respuesta por reglas más un conjunto de tablas
  resumidas (agregadas, sin identificadores) y le pide a un modelo de
  Anthropic que redacte la respuesta. El modelo nunca ve microdatos.

El módulo no depende de Streamlit, para poder probarlo de forma aislada.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_processing import ALL_SITES_LABEL, PERU_LABEL
from stats_analysis import format_p, log_linear_apc, mann_kendall_trend

# ---------------------------------------------------------------------------
# Normalización y diccionarios de sinónimos
# ---------------------------------------------------------------------------


def norm(text: str) -> str:
    """Minúsculas, sin tildes pero conservando la ñ (para no confundir
    'año' con la localización 'Ano')."""
    text = str(text).lower().replace("ñ", "\x00")
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", text.replace("\x00", "ñ")).strip()


# Sinónimos adicionales a los propios nombres de cada localización.
SITE_SYNONYMS: dict[str, list[str]] = {
    "Ano": [r"cancer de ano", r"\banal\b", r"canal anal"],
    "Cavidad Oral": [r"\boral\b", r"\bboca\b", r"labio", r"lengua"],
    "Colon": [r"\bcolon\b"],
    "Cuello Uterino": [r"cuello uterino", r"cervix", r"cervical", r"cervicouterino"],
    "Cuerpo Uterino": [r"cuerpo uterino", r"endometri", r"\butero\b"],
    "Estómago": [r"estomago", r"gastric"],
    "Esófago": [r"esofag"],
    "Hígado": [r"higado", r"hepat"],
    "Leucemia": [r"leucemia"],
    "Linfoma No Hodgkin": [r"no hodgkin", r"\blnh\b"],
    "Linfoma de Hodgkin": [r"(?<!no )hodgkin"],
    "Mama": [r"\bmama\b", r"\bseno\b", r"\bsenos\b(?! paranasal)", r"mamario"],
    "Melanoma de Piel": [r"melanoma"],
    "Piel No Melanoma": [r"piel no melanoma", r"\bpiel\b(?! no)"],
    "Próstata": [r"prostat"],
    "Pulmón": [r"pulmon", r"pulmonar", r"bronquio"],
    "Páncreas": [r"pancrea"],
    "Riñón": [r"riñon", r"renal"],
    "Sist. Nervioso Central": [r"sistema nervioso", r"\bsnc\b", r"cerebr", r"sist\.? nervioso"],
    "Tiroides": [r"tiroid"],
    "Vejiga": [r"vejiga"],
    "Vesícula Biliar": [r"vesicula"],
    "Testículo": [r"testicul"],
    "Ovario": [r"ovari"],
    "Mieloma": [r"mieloma"],
    "Primario Desconocido": [r"primario desconocido", r"origen desconocido"],
    ALL_SITES_LABEL: [r"todas las localizaciones", r"todos los (tipos de )?cancer(es)?", r"total de casos"],
}

DEPT_SYNONYMS: dict[str, list[str]] = {
    PERU_LABEL: [r"\bperu\b", r"nivel nacional", r"total nacional", r"\bpais\b", r"\bnacional\b"],
    "Cusco": [r"\bcusco\b", r"\bcuzco\b"],
    "Ancash": [r"\bancash\b"],
    "Lima": [r"\blima\b(?! metropolitana y callao)"],
}


@dataclass
class QAContext:
    """Valores por defecto tomados del panel de control."""

    default_depts: list[str]
    default_site: str
    year_range: tuple[int, int]
    peru_exclude: list[str] = field(default_factory=list)
    budget_anual: pd.DataFrame | None = None
    budget_detalle: pd.DataFrame | None = None


@dataclass
class QAAnswer:
    title: str
    text: str
    table: pd.DataFrame | None = None
    chart: dict | None = None  # {"kind": "line"|"bar"|"barh", "x":..., "y":..., "series":...}
    intent: str = ""
    interpretation: str = ""
    ok: bool = True


# ---------------------------------------------------------------------------
# Extracción de entidades
# ---------------------------------------------------------------------------


def _find_sites(q: str, sites: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for s in sites:
        pats = SITE_SYNONYMS.get(s, [])
        # El propio nombre (salvo "Ano", que chocaría con palabras comunes)
        if s != "Ano":
            pats = pats + [r"\b" + re.escape(norm(s)) + r"\b"]
        for p in pats:
            m = re.search(p, q)
            if m:
                found.append((m.start(), s))
                break
    # Si se detectó "piel no melanoma", no contar además "melanoma de piel"
    names = [s for _, s in sorted(found)]
    if "Piel No Melanoma" in names and "Melanoma de Piel" in names and "no melanoma" in q and q.count("melanoma") == 1:
        names.remove("Melanoma de Piel")
    if "Linfoma No Hodgkin" in names and "Linfoma de Hodgkin" in names and q.count("hodgkin") == 1:
        names.remove("Linfoma de Hodgkin")
    return list(dict.fromkeys(names))


def _find_depts(q: str, depts: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for d in depts:
        pats = DEPT_SYNONYMS.get(d, [r"\b" + re.escape(norm(d)) + r"\b"])
        for p in pats:
            m = re.search(p, q)
            if m:
                found.append((m.start(), d))
                break
    return list(dict.fromkeys(d for _, d in sorted(found)))


def _find_years(q: str, y_min: int, y_max: int) -> list[int]:
    ys = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    return [y for y in ys if y_min - 50 <= y <= y_max + 50]


def _find_top_n(q: str) -> int | None:
    m = re.search(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s+(?:principales|primer[oa]s|mayores|localizaciones|departamentos|tipos|canceres)", q)
    if m:
        return int(m.group(1) or m.group(2))
    words = {"tres": 3, "cinco": 5, "diez": 10, "quince": 15, "veinte": 20}
    m = re.search(r"\b(tres|cinco|diez|quince|veinte)\s+(?:principales|primer|mayores|localizaciones|departamentos|tipos)", q)
    if m:
        return words[m.group(1)]
    return None


def _has(q: str, *patterns: str) -> bool:
    return any(re.search(p, q) for p in patterns)


# ---------------------------------------------------------------------------
# Agregaciones
# ---------------------------------------------------------------------------


def series_for(df: pd.DataFrame, dept: str, site: str, y0: int, y1: int, peru_exclude: list[str]) -> pd.Series:
    """Serie anual de casos para un departamento y localización, aplicando
    la exclusión del total nacional igual que en el resto del dashboard."""
    base = df[df["Localizacion"].eq(site) & df["Departamento"].eq(dept) & df["Anio"].between(y0, y1)]
    s = base.groupby("Anio")["Casos"].sum(min_count=1)
    if dept == PERU_LABEL and peru_exclude:
        ex = (
            df[df["Localizacion"].eq(site) & df["Departamento"].isin(peru_exclude) & df["Anio"].between(y0, y1)]
            .groupby("Anio")["Casos"].sum(min_count=1)
            .reindex(s.index)
            .fillna(0)
        )
        s = s - ex
    return s.dropna()


def site_ranking(df: pd.DataFrame, dept: str, year: int, peru_exclude: list[str]) -> pd.DataFrame:
    sub = df[(df["Anio"] == year) & (df["Departamento"] == dept) & (df["Localizacion"] != ALL_SITES_LABEL)]
    r = sub.groupby("Localizacion")["Casos"].sum(min_count=1).dropna()
    if dept == PERU_LABEL and peru_exclude:
        ex = (
            df[(df["Anio"] == year) & df["Departamento"].isin(peru_exclude) & (df["Localizacion"] != ALL_SITES_LABEL)]
            .groupby("Localizacion")["Casos"].sum(min_count=1)
            .reindex(r.index)
            .fillna(0)
        )
        r = r - ex
    r = r.sort_values(ascending=False)
    total = r.sum()
    out = r.reset_index().rename(columns={"Casos": "Casos"})
    out["% del total"] = (out["Casos"] / total * 100).round(1) if total else np.nan
    out.index = np.arange(1, len(out) + 1)
    out.index.name = "Puesto"
    return out


def dept_ranking(df: pd.DataFrame, site: str, year: int, exclude: list[str]) -> pd.DataFrame:
    sub = df[
        (df["Anio"] == year)
        & (df["Localizacion"] == site)
        & (df["Departamento"] != PERU_LABEL)
        & (~df["Departamento"].isin(exclude))
    ]
    r = sub.groupby("Departamento")["Casos"].sum(min_count=1).dropna().sort_values(ascending=False)
    total = r.sum()
    out = r.reset_index()
    out["% del total"] = (out["Casos"] / total * 100).round(1) if total else np.nan
    out.index = np.arange(1, len(out) + 1)
    out.index.name = "Puesto"
    return out


def _fmt(v: float) -> str:
    return "s/d" if v is None or pd.isna(v) else f"{v:,.0f}"


def _site_label(site: str) -> str:
    return "todas las localizaciones" if site == ALL_SITES_LABEL else site


def _dept_label(dept: str) -> str:
    return "Perú (total nacional)" if dept == PERU_LABEL else dept


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------


def answer_question(question: str, df: pd.DataFrame, ctx: QAContext) -> QAAnswer:
    q = norm(question)
    if not q:
        return QAAnswer("Pregunta vacía", "Escribe una pregunta o elige un ejemplo.", ok=False)

    all_sites = sorted(df["Localizacion"].unique().tolist())
    all_depts = sorted(df["Departamento"].unique().tolist())
    y_min, y_max = int(df["Anio"].min()), int(df["Anio"].max())

    sites = _find_sites(q, all_sites)
    depts = _find_depts(q, all_depts)
    years = _find_years(q, y_min, y_max)
    top_n = _find_top_n(q)

    # --- Rango de años --------------------------------------------------
    y0, y1 = ctx.year_range
    if len(years) >= 2:
        y0, y1 = min(years), max(years)
    elif len(years) == 1 and _has(q, r"\bdesde\b", r"\ba partir\b"):
        y0 = years[0]
    elif len(years) == 1 and _has(q, r"\bhasta\b"):
        y1 = years[0]
    y0, y1 = max(y0, y_min), min(y1, y_max)
    out_of_range = [y for y in years if y < y_min or y > y_max]
    single_year = years[-1] if years else y1
    if single_year > y_max or single_year < y_min:
        single_year = y_max

    site = sites[0] if sites else ctx.default_site
    dept_list = depts or list(ctx.default_depts) or [PERU_LABEL]
    dept = dept_list[0]
    pexcl = ctx.peru_exclude

    interp_parts = [
        f"Localización: {_site_label(site)}" + ("" if sites else " (del panel)"),
        f"Departamento(s): {', '.join(_dept_label(d) for d in dept_list)}" + ("" if depts else " (del panel)"),
    ]
    note_oor = (
        f"\n\n⚠️ Los datos disponibles van de {y_min} a {y_max}; se ignoró {', '.join(map(str, out_of_range))}."
        if out_of_range
        else ""
    )

    # --- 1. Presupuesto -------------------------------------------------
    if _has(q, r"presupuest", r"devengad", r"\bpim\b", r"\bpia\b", r"girado", r"ejecucion", r"gasto", r"financiamiento"):
        return _budget_answer(q, ctx, years, y0, y1, single_year, note_oor)

    # --- 2. Ranking de departamentos -----------------------------------
    wants_rank = _has(
        q, r"\btop\b", r"ranking", r"principales", r"mas frecuentes", r"\bmayor(es)?\b", r"mas casos",
        r"cuales", r"que (departamentos|regiones|localizaciones|tipos|canceres)", r"menos casos", r"\bmenor(es)?\b",
    )
    asks_depts = _has(q, r"departamento", r"regiones", r"\bregion\b", r"de donde", r"procedencia", r"residencia")
    if asks_depts and wants_rank and len(depts) <= 1 or (asks_depts and not depts and not _has(q, r"evolucion", r"tendencia")):
        excl = list(pexcl)
        if _has(q, r"sin extranjero", r"excluy\w* extranjero"):
            excl.append("Extranjero")
        tab = dept_ranking(df, site, single_year, excl)
        asc = _has(q, r"menos casos", r"\bmenor(es)?\b")
        if asc:
            tab = tab.sort_values("Casos").reset_index(drop=True)
            tab.index = np.arange(1, len(tab) + 1)
            tab.index.name = "Puesto"
        n = top_n or (10 if len(tab) > 10 else len(tab))
        tab = tab.head(n)
        if tab.empty:
            return QAAnswer("Sin datos", f"No hay casos registrados de {_site_label(site)} en {single_year}.", ok=False)
        lead = tab.iloc[0]
        text = (
            f"En **{single_year}**, para **{_site_label(site)}**, el departamento con "
            f"{'menos' if asc else 'más'} casos registrados en el INEN fue **{lead['Departamento']}** "
            f"con **{_fmt(lead['Casos'])}** casos ({lead['% del total']:.1f}% del total departamental)."
        ) + note_oor
        return QAAnswer(
            f"Top {len(tab)} departamentos · {_site_label(site)} · {single_year}", text, tab,
            {"kind": "barh", "x": "Casos", "y": "Departamento"}, "ranking_departamentos",
            " · ".join([f"Localización: {_site_label(site)}", f"Año: {single_year}"]),
        )

    # --- 3. Ranking de localizaciones ----------------------------------
    asks_sites = _has(q, r"localizacion", r"tipos? de cancer", r"canceres", r"neoplasias", r"\btop\b", r"ranking", r"principales", r"mas frecuentes")
    if wants_rank and asks_sites and (not sites or site == ALL_SITES_LABEL):
        tab = site_ranking(df, dept, single_year, pexcl)
        n = top_n or 10
        tab = tab.head(n)
        if tab.empty:
            return QAAnswer("Sin datos", f"No hay datos por localización para {_dept_label(dept)} en {single_year}.", ok=False)
        top3 = ", ".join(f"{r.Localizacion} ({_fmt(r.Casos)})" for r in tab.head(3).itertuples())
        text = (
            f"En **{_dept_label(dept)}**, año **{single_year}**, las localizaciones con más casos fueron: {top3}. "
            f"Las {len(tab)} primeras concentran el **{tab['% del total'].sum():.1f}%** de los casos con localización registrada."
        ) + note_oor
        return QAAnswer(
            f"Top {len(tab)} localizaciones · {_dept_label(dept)} · {single_year}", text, tab,
            {"kind": "barh", "x": "Casos", "y": "Localizacion"}, "ranking_localizaciones",
            f"Departamento: {_dept_label(dept)} · Año: {single_year}",
        )

    # --- 4. Proporción / participación ---------------------------------
    if _has(q, r"proporcion", r"porcentaje", r"participacion", r"\bpeso\b", r"que parte", r"%"):
        rows = []
        for d in dept_list:
            tot = series_for(df, d, ALL_SITES_LABEL, single_year, single_year, pexcl).sum(min_count=1)
            val = series_for(df, d, site, single_year, single_year, pexcl).sum(min_count=1)
            rows.append({"Departamento": _dept_label(d), "Casos": val, "Total (todas)": tot,
                         "% del total": round(val / tot * 100, 2) if tot else np.nan})
        tab = pd.DataFrame(rows)
        r0 = tab.iloc[0]
        text = (
            f"En **{single_year}**, **{_site_label(site)}** representó el **{r0['% del total']:.1f}%** "
            f"de todos los casos de {r0['Departamento']} ({_fmt(r0['Casos'])} de {_fmt(r0['Total (todas)'])})."
        ) + note_oor
        return QAAnswer(f"Participación de {_site_label(site)} · {single_year}", text, tab, None, "proporcion", " · ".join(interp_parts))

    # --- 5. Tendencia estadística --------------------------------------
    if _has(q, r"tendencia", r"mann", r"kendall", r"significativ", r"\bapc\b", r"esta (creciendo|aumentando|disminuyendo|bajando)", r"crece", r"decrece"):
        rows, sentences = [], []
        for d in dept_list:
            s = series_for(df, d, site, y0, y1, pexcl)
            mk = mann_kendall_trend(s.index.to_numpy(), s.to_numpy())
            apc = log_linear_apc(s.index.to_numpy(), s.to_numpy())
            lbl = {"increasing": "Creciente", "decreasing": "Decreciente", "no trend": "Sin tendencia"}
            rows.append({
                "Departamento": _dept_label(d),
                "Tendencia (MK)": lbl.get(mk.trend, mk.trend) if mk else "s/d",
                "p (MK)": format_p(mk.p_value) if mk else "s/d",
                "Pendiente de Sen (casos/año)": round(mk.sen_slope, 1) if mk else np.nan,
                "APC (%/año)": round(apc.apc, 2) if apc else np.nan,
                "IC95% APC": f"{apc.ci_low:+.1f} a {apc.ci_high:+.1f}" if apc else "s/d",
                "Años": f"{int(s.index.min())}–{int(s.index.max())}" if len(s) else "s/d",
            })
            if mk:
                verdict = (
                    "una tendencia **{}** estadísticamente significativa".format(
                        "creciente" if mk.trend == "increasing" else "decreciente")
                    if mk.significant else "**ninguna tendencia monótona significativa**"
                )
                sentences.append(
                    f"{_dept_label(d)}: Mann-Kendall muestra {verdict} ({format_p(mk.p_value)}), "
                    f"pendiente de Sen {mk.sen_slope:+,.1f} casos/año"
                    + (f", APC {apc.apc:+.1f}%/año" if apc else "") + "."
                )
        text = (f"**{_site_label(site)}**, {y0}–{y1}. " + " ".join(sentences)) if sentences else "Serie demasiado corta para evaluar tendencia (mín. 6 años)."
        return QAAnswer(
            f"Tendencia · {_site_label(site)} · {y0}–{y1}", text + note_oor, pd.DataFrame(rows),
            _line_chart(df, dept_list, site, y0, y1, pexcl), "tendencia", " · ".join(interp_parts + [f"Años: {y0}–{y1}"]),
        )

    # --- 6. Variación entre dos años -----------------------------------
    if len(years) >= 2 and _has(q, r"variacion", r"cambio", r"crecimiento", r"aumento", r"diferencia", r"compar", r"entre", r"crecio", r"disminu", r"cayo", r"subio"):
        ya, yb = y0, y1
        rows = []
        for d in dept_list:
            s = series_for(df, d, site, ya, yb, pexcl)
            a, b = s.get(ya, np.nan), s.get(yb, np.nan)
            rows.append({"Departamento": _dept_label(d), f"Casos {ya}": a, f"Casos {yb}": b,
                         "Cambio absoluto": b - a, "Variación %": round((b - a) / a * 100, 1) if a else np.nan})
        tab = pd.DataFrame(rows)
        r0 = tab.iloc[0]
        text = (
            f"Para **{_site_label(site)}** en **{r0['Departamento']}**, los casos pasaron de **{_fmt(r0[f'Casos {ya}'])}** "
            f"en {ya} a **{_fmt(r0[f'Casos {yb}'])}** en {yb}, una variación de **{r0['Variación %']:+.1f}%** "
            f"({r0['Cambio absoluto']:+,.0f} casos)."
        ) + note_oor
        return QAAnswer(f"Variación {ya} → {yb} · {_site_label(site)}", text, tab, None, "variacion", " · ".join(interp_parts))

    # --- 7. Año pico ----------------------------------------------------
    if _has(q, r"\bpico\b", r"maximo", r"record", r"en que año", r"que año", r"cuando hubo mas", r"minimo"):
        want_min = _has(q, r"minimo", r"menos casos")
        rows = []
        for d in dept_list:
            s = series_for(df, d, site, y0, y1, pexcl)
            if s.empty:
                continue
            yy = int(s.idxmin() if want_min else s.idxmax())
            rows.append({"Departamento": _dept_label(d), "Año": yy, "Casos": s[yy]})
        if not rows:
            return QAAnswer("Sin datos", "No hay datos para ese filtro.", ok=False)
        tab = pd.DataFrame(rows)
        r0 = tab.iloc[0]
        text = (
            f"Entre {y0} y {y1}, el año con {'menos' if want_min else 'más'} casos de **{_site_label(site)}** "
            f"en {r0['Departamento']} fue **{r0['Año']}**, con **{_fmt(r0['Casos'])}** casos."
        ) + note_oor
        return QAAnswer(f"Año {'mínimo' if want_min else 'pico'} · {_site_label(site)}", text, tab,
                        _line_chart(df, dept_list, site, y0, y1, pexcl), "pico", " · ".join(interp_parts))

    # --- 8. Evolución anual / comparación de series --------------------
    if _has(q, r"evolucion", r"\bserie\b", r"por año", r"\banual", r"a lo largo", r"historic", r"en el tiempo", r"comportamiento", r"compar") or (len(years) != 1 and (len(depts) > 1 or len(sites) > 1)):
        cols = {}
        if len(sites) > 1:
            for s_ in sites:
                cols[_site_label(s_)] = series_for(df, dept, s_, y0, y1, pexcl)
            subj = f"{', '.join(_site_label(s) for s in sites)} en {_dept_label(dept)}"
        else:
            for d in dept_list:
                cols[_dept_label(d)] = series_for(df, d, site, y0, y1, pexcl)
            subj = f"{_site_label(site)} en {', '.join(_dept_label(d) for d in dept_list)}"
        tab = pd.DataFrame(cols)
        tab.index.name = "Año"
        chart = {"kind": "line_wide", "data": tab.copy()}
        tab = tab.reset_index()
        first = tab.columns[1]
        s0 = tab.set_index("Año")[first].dropna()
        if s0.empty:
            return QAAnswer("Sin datos", f"No hay datos para {subj} entre {y0} y {y1}.", ok=False)
        text = (
            f"Evolución de **{subj}**, {y0}–{y1}. En {first}: de **{_fmt(s0.iloc[0])}** casos en {int(s0.index[0])} "
            f"a **{_fmt(s0.iloc[-1])}** en {int(s0.index[-1])}; máximo de **{_fmt(s0.max())}** en {int(s0.idxmax())}."
        ) + note_oor
        return QAAnswer(f"Evolución anual · {subj}", text, tab, chart, "evolucion",
                        " · ".join(interp_parts + [f"Años: {y0}–{y1}"]))

    # --- 9. Por defecto: total de casos ---------------------------------
    rows = []
    for d in dept_list:
        s = series_for(df, d, site, y0 if len(years) != 1 else single_year, y1 if len(years) != 1 else single_year, pexcl)
        rows.append({"Departamento": _dept_label(d), "Casos": s.sum(min_count=1)})
    tab = pd.DataFrame(rows)
    period = f"en {single_year}" if len(years) == 1 else (f"en {y1}" if not years and not _has(q, r"total", r"acumulad", r"periodo") else f"entre {y0} y {y1}")
    if not years and not _has(q, r"total", r"acumulad", r"periodo"):
        # Sin años explícitos: se reporta el último año del rango.
        rows = []
        for d in dept_list:
            s = series_for(df, d, site, y1, y1, pexcl)
            rows.append({"Departamento": _dept_label(d), "Casos": s.sum(min_count=1)})
        tab = pd.DataFrame(rows)
        if tab["Casos"].isna().all():
            # Sin dato en el último año: se usa el último año con dato.
            last = series_for(df, dept_list[0], site, ctx.year_range[0], y1, pexcl)
            if not last.empty:
                ly = int(last.index.max())
                period = f"en {ly} (último año con dato)"
                tab = pd.DataFrame([
                    {"Departamento": _dept_label(d), "Casos": series_for(df, d, site, ly, ly, pexcl).sum(min_count=1)}
                    for d in dept_list
                ])
    total = tab["Casos"].sum(min_count=1)
    if pd.isna(total):
        return QAAnswer("Sin datos", f"No hay datos de {_site_label(site)} para ese filtro.", ok=False)
    detail = "" if len(tab) == 1 else " Desglose por departamento en la tabla."
    text = (
        f"Se registraron **{_fmt(total)}** casos de **{_site_label(site)}** {period} en "
        f"{', '.join(_dept_label(d) for d in dept_list)}.{detail}"
    ) + note_oor
    return QAAnswer(f"Casos · {_site_label(site)} · {period}", text, tab if len(tab) > 1 else None, None, "total",
                    " · ".join(interp_parts))


def _line_chart(df, dept_list, site, y0, y1, pexcl) -> dict:
    wide = pd.DataFrame({_dept_label(d): series_for(df, d, site, y0, y1, pexcl) for d in dept_list})
    wide.index.name = "Año"
    return {"kind": "line_wide", "data": wide}


def _budget_answer(q, ctx: QAContext, years, y0, y1, single_year, note_oor) -> QAAnswer:
    ba = ctx.budget_anual
    if ba is None or ba.empty:
        return QAAnswer(
            "Base presupuestal no disponible",
            "No se pudo leer la base de ejecución presupuestal del INEN.", ok=False,
        )
    metric = "Devengado"
    for m, pat in [("PIA", r"\bpia\b"), ("PIM", r"\bpim\b"), ("Girado", r"girado"), ("Compromiso", r"compromis"), ("Devengado", r"devengad")]:
        if re.search(pat, q):
            metric = m
            break
    b_min, b_max = int(ba["Anio"].min()), int(ba["Anio"].max())
    lo, hi = (min(years), max(years)) if len(years) >= 2 else (b_min, b_max)
    if _has(q, r"fuente"):
        bd = ctx.budget_detalle
        yr = single_year if years else b_max
        if bd is None or yr not in set(bd["Anio"]):
            yr = int(bd["Anio"].max()) if bd is not None else b_max
        tab = bd[bd["Anio"] == yr][["Fuente", metric]].sort_values(metric, ascending=False).reset_index(drop=True)
        tot = tab[metric].sum()
        tab["% del total"] = (tab[metric] / tot * 100).round(1)
        r0 = tab.iloc[0]
        return QAAnswer(
            f"{metric} por fuente de financiamiento · {yr}",
            f"En **{yr}**, el {metric} total del INEN fue **S/ {tot:,.0f}**; la principal fuente fue "
            f"**{r0['Fuente']}** ({r0['% del total']:.1f}%).",
            tab, {"kind": "barh", "x": metric, "y": "Fuente"}, "presupuesto_fuente", f"Métrica: {metric} · Año: {yr}",
        )
    sub = ba[ba["Anio"].between(lo, hi)][["Anio", "PIA", "PIM", "Devengado", "Girado"]].reset_index(drop=True)
    if len(years) == 1 and not _has(q, r"evolucion", r"desde", r"hasta"):
        row = ba[ba["Anio"] == years[0]]
        if row.empty:
            return QAAnswer("Sin datos", f"No hay ejecución presupuestal para {years[0]}.", ok=False)
        v = row.iloc[0]
        avance = v["Devengado"] / v["PIM"] * 100 if v["PIM"] else np.nan
        return QAAnswer(
            f"Ejecución presupuestal INEN · {years[0]}",
            f"En **{years[0]}** el INEN tuvo un PIM de **S/ {v['PIM']:,.0f}** y un devengado de **S/ {v['Devengado']:,.0f}**"
            + (f" (avance **{avance:.1f}%**)." if pd.notna(avance) else "."),
            row[["Anio", "PIA", "PIM", "Devengado", "Girado"]].reset_index(drop=True), None, "presupuesto_anio", f"Año: {years[0]}",
        )
    s = sub.set_index("Anio")[metric]
    return QAAnswer(
        f"Evolución del {metric} INEN · {lo}–{hi}",
        f"El {metric} del INEN pasó de **S/ {s.iloc[0]:,.0f}** en {int(s.index[0])} a **S/ {s.iloc[-1]:,.0f}** "
        f"en {int(s.index[-1])}; máximo de S/ {s.max():,.0f} en {int(s.idxmax())}." + note_oor,
        sub, {"kind": "line_wide", "data": sub.set_index("Anio")[[metric]]}, "presupuesto_evolucion", f"Métrica: {metric} · Años: {lo}–{hi}",
    )


# ---------------------------------------------------------------------------
# Modo IA: contexto resumido + llamada al modelo
# ---------------------------------------------------------------------------


def build_ai_context(df: pd.DataFrame, ctx: QAContext, rule_answer: QAAnswer | None) -> str:
    """Tablas agregadas (sin identificadores) que se envían al modelo."""
    y_max = int(df["Anio"].max())
    parts: list[str] = []

    nat = series_for(df, PERU_LABEL, ALL_SITES_LABEL, int(df["Anio"].min()), y_max, ctx.peru_exclude)
    parts.append("## Casos totales (todas las localizaciones), Perú total nacional, por año\n" + nat.to_csv(header=["Casos"]))

    rk = site_ranking(df, PERU_LABEL, y_max, ctx.peru_exclude).head(15)
    parts.append(f"## Top 15 localizaciones, Perú, {y_max}\n" + rk.to_csv())

    dr = dept_ranking(df, ALL_SITES_LABEL, y_max, ctx.peru_exclude)
    parts.append(f"## Casos por departamento de residencia (todas las localizaciones), {y_max}\n" + dr.to_csv())

    y0, y1 = ctx.year_range
    sel = pd.DataFrame(
        {_dept_label(d): series_for(df, d, ctx.default_site, y0, y1, ctx.peru_exclude) for d in ctx.default_depts}
    )
    parts.append(
        f"## Selección actual del panel: {_site_label(ctx.default_site)}, {y0}–{y1}\n" + sel.to_csv(index_label="Anio")
    )

    if rule_answer is not None and rule_answer.ok and rule_answer.table is not None:
        parts.append(
            f"## Resultado calculado por reglas para la pregunta ({rule_answer.title})\n"
            + rule_answer.table.head(40).to_csv()
        )
    if ctx.budget_anual is not None:
        parts.append("## Ejecución presupuestal INEN por año (S/)\n" + ctx.budget_anual[["Anio", "PIA", "PIM", "Devengado", "Girado"]].to_csv(index=False))
    return "\n\n".join(parts)


AI_SYSTEM_PROMPT = (
    "Eres un analista de datos del Instituto Nacional de Enfermedades Neoplásicas (INEN, Perú). "
    "Respondes en español, de forma breve y precisa, preguntas sobre casos nuevos de cáncer registrados "
    "en el INEN por departamento de residencia, localización del tumor primario y año, y sobre la "
    "ejecución presupuestal del INEN. Usa EXCLUSIVAMENTE las tablas proporcionadas; si la información "
    "no está en ellas, dilo con claridad y sugiere cómo reformular la pregunta. No inventes cifras. "
    "Recuerda que son casos atendidos en el INEN, no incidencia poblacional. Escribe en prosa fluida, "
    "sin rayas largas, con cifras con separador de miles."
)


def ask_ai(question: str, context_text: str, api_key: str, model: str) -> str:
    import anthropic  # importación diferida: dependencia opcional

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=900,
        system=AI_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"<datos>\n{context_text}\n</datos>\n\nPregunta: {question}",
        }],
    )
    return "".join(getattr(b, "text", "") for b in msg.content).strip()
