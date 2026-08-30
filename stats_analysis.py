"""
stats_analysis.py
------------------
Utilidades de análisis estadístico para las series temporales del
dashboard:

  - linear_trend()      -> línea de tendencia simple (mínimos cuadrados)
  - detect_breakpoint()  -> punto de quiebre estructural (regresión
                             segmentada de 2 tramos + test de Chow)
  - mann_kendall_trend() -> test de tendencia robusto a autocorrelación
                             (Hamed & Rao 1998, con respaldo automático
                             en Yue & Wang 2004 si la corrección de
                             varianza de Hamed-Rao es inestable) +
                             pendiente de Sen
  - project_series()     -> proyección de casos a futuro con análisis
                             de sensibilidad (3 escenarios) usando
                             suavizado exponencial con tendencia
                             amortiguada (Holt damped trend) como
                             método recomendado, más bandas de
                             incertidumbre por bootstrap de residuos
  - chow_test_arbitrary_break() -> test de Chow en un año de quiebre
                             ELEGIDO por el usuario (p. ej. una ley o
                             la apertura de un hospital oncológico),
                             con opción de excluir años de
                             "implementación" tras el evento
  - smooth_series()       -> suavizado LOWESS de la serie temporal,
                             para ver la tendencia sin el ruido
                             año a año

Por qué no basta con la regresión lineal simple
------------------------------------------------
`linear_trend()` (mínimos cuadrados ordinarios) asume residuos
independientes. En conteos anuales de casos de cáncer los residuos
suelen estar autocorrelacionados (un año alto tiende a seguir a otro
año alto), lo que **subestima el error estándar de la pendiente e
infla artificialmente la significancia** — puede "detectar" tendencia
donde solo hay ruido correlacionado. `mann_kendall_trend()` corrige
esto: es no paramétrico (no asume forma funcional ni normalidad) y su
varianza se ajusta explícitamente por la autocorrelación de la serie
(modificación de Hamed-Rao), por lo que es la prueba recomendada para
responder "¿hay o no tendencia?". La pendiente de Sen que la acompaña
es la mediana de todas las pendientes posibles entre pares de puntos:
un estimador robusto a outliers, a diferencia de la pendiente OLS.

Por qué la proyección no es una simple extrapolación lineal
--------------------------------------------------------------
Extrapolar una recta (u otra curva sin amortiguar) hacia el futuro
asume que el ritmo de cambio se mantendrá constante indefinidamente,
lo que casi nunca ocurre en series de salud (la cobertura de registro,
la capacidad diagnóstica, etc. tienden a saturarse) y típicamente
**sobreestima** a mediano plazo. `project_series()` usa como método
central el suavizado exponencial de Holt con tendencia amortiguada
("damped trend"), que reduce gradualmente la pendiente proyectada a
medida que el horizonte crece — es el método estándar en la
literatura de pronósticos (Gardner & McKenzie) para evitar ese sesgo.
Se complementa con un análisis de sensibilidad de 3 escenarios y una
banda de incertidumbre por bootstrap.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pymannkendall as mk
except ImportError:  # pragma: no cover
    mk = None

try:
    from statsmodels.tsa.holtwinters import Holt
except ImportError:  # pragma: no cover
    Holt = None

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except ImportError:  # pragma: no cover
    _lowess = None


@dataclass
class TrendResult:
    slope: float
    intercept: float
    r2: float
    x: np.ndarray
    y_pred: np.ndarray


@dataclass
class BreakpointResult:
    year: int
    slope_before: float
    slope_after: float
    intercept_before: float
    intercept_after: float
    f_stat: float
    p_value: float
    significant: bool
    x_before: np.ndarray
    y_pred_before: np.ndarray
    x_after: np.ndarray
    y_pred_after: np.ndarray


@dataclass
class ArbitraryBreakResult:
    """Resultado del test de Chow aplicado a un año de quiebre elegido
    por el usuario (a diferencia de BreakpointResult, que busca el año
    óptimo automáticamente)."""
    break_year: int
    implementation_lag: int
    slope_before: float
    slope_after: float
    intercept_before: float
    intercept_after: float
    f_stat: float
    p_value: float
    significant: bool
    x_before: np.ndarray
    y_pred_before: np.ndarray
    x_after: np.ndarray
    y_pred_after: np.ndarray
    excluded_years: np.ndarray
    n_before: int
    n_after: int


@dataclass
class MannKendallResult:
    trend: str          # "increasing" | "decreasing" | "no trend"
    p_value: float
    significant: bool
    z_stat: float
    tau: float           # correlación de Kendall (fuerza de la tendencia, -1 a 1)
    sen_slope: float      # casos/año, robusto a outliers
    sen_intercept: float
    x: np.ndarray
    y_sen: np.ndarray     # línea de la pendiente de Sen, para graficar
    n_obs: int
    method: str           # qué variante del test se usó realmente


@dataclass
class ProjectionResult:
    years_hist: np.ndarray
    values_hist: np.ndarray
    years_future: np.ndarray
    conservative: np.ndarray
    recommended: np.ndarray
    recommended_lower: np.ndarray
    recommended_upper: np.ndarray
    optimistic: np.ndarray
    method_note: str


def linear_trend(years: np.ndarray, values: np.ndarray) -> TrendResult | None:
    """Ajusta una recta de tendencia simple (mínimos cuadrados)."""
    mask = ~np.isnan(values)
    x, y = years[mask].astype(float), values[mask].astype(float)
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return TrendResult(slope=slope, intercept=intercept, r2=r2, x=x, y_pred=y_pred)


def detect_breakpoint(
    years: np.ndarray, values: np.ndarray, min_segment: int = 3
) -> BreakpointResult | None:
    """Detecta el año de quiebre estructural más probable en la serie
    usando regresión segmentada de dos tramos + test de Chow.

    Devuelve None si la serie es demasiado corta (menos de
    2 * min_segment puntos válidos) para dividirla en dos tramos
    confiables.
    """
    mask = ~np.isnan(values)
    x_all = years[mask].astype(float)
    y_all = values[mask].astype(float)
    order = np.argsort(x_all)
    x_all, y_all = x_all[order], y_all[order]
    n = len(x_all)

    if n < 2 * min_segment:
        return None

    # Modelo único (pooled) para toda la serie, referencia del test de Chow
    c_pooled = np.polyfit(x_all, y_all, 1)
    rss_pooled = np.sum((y_all - np.polyval(c_pooled, x_all)) ** 2)

    best = None
    for i in range(min_segment, n - min_segment + 1):
        x1, y1 = x_all[:i], y_all[:i]
        x2, y2 = x_all[i:], y_all[i:]
        c1 = np.polyfit(x1, y1, 1)
        c2 = np.polyfit(x2, y2, 1)
        rss1 = np.sum((y1 - np.polyval(c1, x1)) ** 2)
        rss2 = np.sum((y2 - np.polyval(c2, x2)) ** 2)
        rss_split = rss1 + rss2
        if best is None or rss_split < best[0]:
            best = (rss_split, i, c1, c2, x1, y1, x2, y2)

    rss_split, split_idx, c1, c2, x1, y1, x2, y2 = best

    # Test de Chow: compara el modelo único vs. el modelo segmentado
    k = 2  # parámetros por tramo (pendiente + intercepto)
    df1 = k
    df2 = n - 2 * k
    if df2 <= 0 or rss_split <= 0:
        f_stat, p_value = float("nan"), float("nan")
    else:
        f_stat = ((rss_pooled - rss_split) / df1) / (rss_split / df2)
        p_value = float(stats.f.sf(f_stat, df1, df2))

    breakpoint_year = int(x2[0])  # primer año del segundo tramo

    return BreakpointResult(
        year=breakpoint_year,
        slope_before=c1[0],
        slope_after=c2[0],
        intercept_before=c1[1],
        intercept_after=c2[1],
        f_stat=f_stat,
        p_value=p_value,
        significant=bool(p_value < 0.05) if p_value == p_value else False,  # NaN check
        x_before=x1,
        y_pred_before=np.polyval(c1, x1),
        x_after=x2,
        y_pred_after=np.polyval(c2, x2),
    )


def _regularize(years: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reindexa a una malla anual contigua (recorta huecos al inicio y
    al final; interpola linealmente los huecos internos), porque tanto
    Mann-Kendall como Holt asumen observaciones regularmente espaciadas."""
    s = pd.Series(values, index=years.astype(int)).sort_index()
    s = s.dropna()
    if s.empty:
        return np.array([]), np.array([])
    full_index = np.arange(s.index.min(), s.index.max() + 1)
    s = s.reindex(full_index)
    s = s.interpolate(limit_direction="both")
    return s.index.to_numpy(), s.to_numpy()


def mann_kendall_trend(
    years: np.ndarray, values: np.ndarray, min_obs: int = 6
) -> MannKendallResult | None:
    """Test de tendencia Mann-Kendall corregido por autocorrelación,
    acompañado de la pendiente de Sen (mediana de pendientes entre
    todos los pares de puntos, robusta a outliers).

    Método principal: modificación de Hamed & Rao (1998). En series
    con autocorrelación muy fuerte, la corrección de varianza de
    Hamed-Rao puede volverse numéricamente inestable (varianza
    negativa); en ese caso se usa automáticamente como respaldo la
    modificación de Yue & Wang (2004), otra corrección por
    autocorrelación bien establecida en la literatura. El campo
    `method` del resultado indica cuál se usó realmente, para que
    quede claro en la interfaz.

    Devuelve None si hay muy pocas observaciones (< min_obs) o si la
    librería pymannkendall no está disponible.
    """
    if mk is None:
        return None
    x, y = _regularize(years, values)
    if len(x) < min_obs:
        return None

    method = "Hamed-Rao (1998)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mk.hamed_rao_modification_test(y)
        if not np.isfinite(result.p) or not np.isfinite(result.z):
            # Respaldo: la corrección de Hamed-Rao no fue estable para esta
            # serie (frecuente con autocorrelación muy fuerte); se usa
            # Yue-Wang, otra corrección robusta a autocorrelación.
            method = "Yue-Wang (2004), respaldo tras inestabilidad de Hamed-Rao"
            result = mk.yue_wang_modification_test(y)

    y_sen = result.slope * (x - x[0]) + y[0]

    return MannKendallResult(
        trend=result.trend,
        p_value=float(result.p),
        significant=bool(result.h),
        z_stat=float(result.z),
        tau=float(result.Tau),
        sen_slope=float(result.slope),
        sen_intercept=float(y[0] - result.slope * x[0]),
        x=x,
        y_sen=y_sen,
        n_obs=len(x),
        method=method,
    )


def project_series(
    years: np.ndarray,
    values: np.ndarray,
    horizon: int = 5,
    n_boot: int = 300,
    ci: float = 0.90,
    min_obs: int = 6,
    random_state: int = 42,
) -> ProjectionResult | None:
    """Proyecta la serie `horizon` años hacia adelante con 3 escenarios:

      - conservative: promedio de los últimos 3 años, sin tendencia
        (cota inferior; asume que el crecimiento se detiene).
      - recommended: Holt con tendencia amortiguada (damped trend) —
        el método recomendado, porque atenúa la pendiente a medida que
        el horizonte crece en vez de mantenerla constante.
      - optimistic: extrapolación lineal de la pendiente de Sen sin
        amortiguar (cota superior; probablemente sobreestima en
        horizontes largos).

    La banda de incertidumbre del escenario recomendado se calcula por
    bootstrap de los residuos del modelo Holt (remuestreo con
    reemplazo, re-simulación de la proyección), para no depender de
    una fórmula analítica que también podría subestimar el error por
    autocorrelación.
    """
    if Holt is None:
        return None
    x, y = _regularize(years, values)
    if len(x) < min_obs:
        return None

    future_years = np.arange(x[-1] + 1, x[-1] + horizon + 1)

    # --- Escenario conservador: promedio de los últimos 3 años, plano ---
    last_avg = float(np.mean(y[-3:]))
    conservative = np.full(horizon, last_avg)

    # --- Escenario recomendado: Holt con tendencia amortiguada ---
    model = Holt(y, damped_trend=True, initialization_method="estimated")
    fit = model.fit(optimized=True)
    recommended = np.asarray(fit.forecast(horizon))

    # Bootstrap de residuos para la banda de incertidumbre
    resid = np.asarray(fit.resid)
    resid = resid[~np.isnan(resid)]
    rng = np.random.default_rng(random_state)
    boot_forecasts = np.empty((n_boot, horizon))
    if len(resid) >= 3:
        for b in range(n_boot):
            sim_resid = rng.choice(resid, size=len(y), replace=True)
            y_sim = np.asarray(fit.fittedvalues) + sim_resid
            try:
                m_b = Holt(y_sim, damped_trend=True, initialization_method="estimated")
                f_b = m_b.fit(optimized=True)
                boot_forecasts[b] = f_b.forecast(horizon)
            except Exception:  # noqa: BLE001 - robustez ante fallos numéricos puntuales
                boot_forecasts[b] = recommended
        alpha = 1 - ci
        lower = np.nanpercentile(boot_forecasts, 100 * alpha / 2, axis=0)
        upper = np.nanpercentile(boot_forecasts, 100 * (1 - alpha / 2), axis=0)
    else:
        lower = recommended.copy()
        upper = recommended.copy()

    # --- Escenario optimista/lineal: pendiente de Sen sin amortiguar ---
    mkr = mann_kendall_trend(x, y, min_obs=min_obs)
    if mkr is not None:
        last_fit = mkr.sen_intercept + mkr.sen_slope * x[-1]
        optimistic = last_fit + mkr.sen_slope * (future_years - x[-1])
    else:
        trend = linear_trend(x, y)
        optimistic = (
            trend.slope * future_years + trend.intercept
            if trend is not None
            else recommended.copy()
        )

    # Ningún escenario debería proyectar casos negativos
    conservative = np.clip(conservative, 0, None)
    recommended = np.clip(recommended, 0, None)
    lower = np.clip(lower, 0, None)
    upper = np.clip(upper, 0, None)
    optimistic = np.clip(optimistic, 0, None)

    note = (
        "Método recomendado: suavizado exponencial de Holt con tendencia "
        "amortiguada (damped trend), que reduce gradualmente la pendiente "
        "proyectada para no sobreestimar a mediano plazo. El escenario "
        "'lineal (Sen)' extrapola la pendiente robusta de todo el "
        "histórico sin amortiguar — no siempre es el más alto: si la "
        "tendencia reciente es más pronunciada que el promedio histórico, "
        "el método amortiguado puede superarlo en el corto plazo. Compara "
        "los tres para dimensionar la incertidumbre, no asumas que uno es "
        "siempre el techo o el piso."
    )

    return ProjectionResult(
        years_hist=x,
        values_hist=y,
        years_future=future_years,
        conservative=conservative,
        recommended=recommended,
        recommended_lower=lower,
        recommended_upper=upper,
        optimistic=optimistic,
        method_note=note,
    )


def chow_test_arbitrary_break(
    years: np.ndarray,
    values: np.ndarray,
    break_year: int,
    implementation_lag: int = 0,
    min_segment: int = 3,
) -> ArbitraryBreakResult | None:
    """Aplica el test de Chow en un punto de quiebre ELEGIDO por el
    usuario (p. ej. el año de una ley aprobada o la inauguración de un
    hospital oncológico), en vez de buscar automáticamente el año que
    mejor ajusta (eso lo hace `detect_breakpoint`).

    Permite excluir una ventana de "implementación" (`implementation_lag`
    años) inmediatamente después del evento, durante la cual el efecto
    aún no se consolida y por lo tanto podría diluir la comparación
    estadística entre el "antes" y el "después":

      - Tramo "antes":    años < break_year
      - Ventana excluida: [break_year, break_year + implementation_lag)
      - Tramo "después":  años >= break_year + implementation_lag

    El modelo único de referencia del test de Chow se ajusta solo con
    los puntos efectivamente usados en los dos tramos (excluyendo
    también la ventana de implementación), para que la comparación sea
    justa.

    Devuelve None si alguno de los dos tramos queda con menos de
    `min_segment` observaciones válidas.
    """
    mask = ~np.isnan(values)
    x_all = years[mask].astype(float)
    y_all = values[mask].astype(float)
    order = np.argsort(x_all)
    x_all, y_all = x_all[order], y_all[order]

    after_start = break_year + implementation_lag
    before_mask = x_all < break_year
    after_mask = x_all >= after_start
    excluded_mask = (~before_mask) & (~after_mask)

    x1, y1 = x_all[before_mask], y_all[before_mask]
    x2, y2 = x_all[after_mask], y_all[after_mask]

    if len(x1) < min_segment or len(x2) < min_segment:
        return None

    c1 = np.polyfit(x1, y1, 1)
    c2 = np.polyfit(x2, y2, 1)
    rss1 = np.sum((y1 - np.polyval(c1, x1)) ** 2)
    rss2 = np.sum((y2 - np.polyval(c2, x2)) ** 2)
    rss_split = rss1 + rss2

    # Modelo único de referencia: se excluye también la ventana de
    # implementación, para comparar contra los mismos puntos.
    x_pooled = np.concatenate([x1, x2])
    y_pooled = np.concatenate([y1, y2])
    c_pooled = np.polyfit(x_pooled, y_pooled, 1)
    rss_pooled = np.sum((y_pooled - np.polyval(c_pooled, x_pooled)) ** 2)

    n = len(x_pooled)
    k = 2
    df1 = k
    df2 = n - 2 * k
    if df2 <= 0 or rss_split <= 0:
        f_stat, p_value = float("nan"), float("nan")
    else:
        f_stat = ((rss_pooled - rss_split) / df1) / (rss_split / df2)
        p_value = float(stats.f.sf(f_stat, df1, df2))

    return ArbitraryBreakResult(
        break_year=int(break_year),
        implementation_lag=int(implementation_lag),
        slope_before=c1[0],
        slope_after=c2[0],
        intercept_before=c1[1],
        intercept_after=c2[1],
        f_stat=f_stat,
        p_value=p_value,
        significant=bool(p_value < 0.05) if p_value == p_value else False,
        x_before=x1,
        y_pred_before=np.polyval(c1, x1),
        x_after=x2,
        y_pred_after=np.polyval(c2, x2),
        excluded_years=x_all[excluded_mask],
        n_before=len(x1),
        n_after=len(x2),
    )


def smooth_series(
    years: np.ndarray, values: np.ndarray, frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray] | None:
    """Suaviza la serie temporal con LOWESS (regresión local ponderada),
    para visualizar la tendencia general sin el ruido año a año, sin
    imponer una forma funcional (a diferencia de la línea de tendencia
    lineal). `frac` controla qué fracción de los datos se usa en cada
    ajuste local: valores bajos siguen más de cerca los datos, valores
    altos generan una curva más suave.

    Devuelve None si hay muy pocos puntos (< 4) o si statsmodels no
    está disponible.
    """
    if _lowess is None:
        return None
    mask = ~np.isnan(values)
    x, y = years[mask].astype(float), values[mask].astype(float)
    if len(x) < 4:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    smoothed = _lowess(y, x, frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]
