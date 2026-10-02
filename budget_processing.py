"""
budget_processing.py
---------------------
Carga y limpieza de la base NATIVA de ejecución presupuestal del INEN
(PIA, PIM, Compromiso, Devengado, Girado) por año y fuente de
financiamiento, 2000-2026.

El archivo fuente es un Excel con dos hojas ya pre-formateadas por el
usuario para este tipo de análisis:

  - "2Detalle por Fuente": una fila por año x fuente de financiamiento
    (Recursos Ordinarios, Recursos Directamente Recaudados, Donaciones
    y Transferencias, Recursos por Operaciones Oficiales de Crédito),
    con PIA, PIM, Compromiso, Devengado, Girado y dos columnas de
    avance % ya calculadas.
  - "2Resumen Anual (Serie Temporal)": una fila por año con los mismos
    montos agregados a nivel nacional (todas las fuentes sumadas) más
    el número de fuentes activas ese año.

Este módulo es independiente de Streamlit (puede probarse con
`python budget_processing.py`) y expone `load_budget_data()`, que
entrega ambas tablas ya limpias y tipadas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BUDGET_FILE = DATA_DIR / "INEN_Serie_Temporal_Ejecucion_Presupuestal_v2_para_carga.xlsx"

SHEET_DETALLE = "2Detalle por Fuente"
SHEET_RESUMEN = "2Resumen Anual (Serie Tempor"

MONTO_COLS = ["PIA", "PIM", "Compromiso", "Devengado", "Girado"]

# Etiquetas legibles para los montos (usadas en gráficos y selectores)
MONTO_LABELS = {
    "PIA": "PIA — Presupuesto Institucional de Apertura",
    "PIM": "PIM — Presupuesto Institucional Modificado",
    "Compromiso": "Compromiso",
    "Devengado": "Devengado (ejecución real)",
    "Girado": "Girado",
}


def load_budget_data(path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Lee el workbook de ejecución presupuestal INEN y devuelve:

      - "anual": serie temporal agregada a nivel nacional, una fila por
        año (Anio, PIA, PIM, Compromiso, Devengado, Girado,
        Avance_pct [fracción 0-1], N_fuentes).
      - "detalle": desagregado por fuente de financiamiento, una fila
        por año x fuente (Anio, Fuente, PIA, PIM, Compromiso,
        Devengado, Girado, Avance_pct_pim, Avance_pct_mef).

    Lanza una excepción clara si el archivo no existe o el formato de
    las hojas cambió.
    """
    fp = Path(path) if path else BUDGET_FILE
    if not fp.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de ejecución presupuestal en: {fp}"
        )

    detalle = pd.read_excel(fp, sheet_name=SHEET_DETALLE)
    detalle = detalle.iloc[:, :9]
    detalle.columns = [
        "Anio", "Fuente", "PIA", "PIM", "Compromiso", "Devengado", "Girado",
        "Avance_pct_pim", "Avance_pct_mef",
    ]
    detalle["Anio"] = pd.to_numeric(detalle["Anio"], errors="coerce")
    detalle = detalle.dropna(subset=["Anio", "Fuente"]).copy()
    detalle["Anio"] = detalle["Anio"].astype(int)
    detalle["Fuente"] = detalle["Fuente"].astype(str).str.strip()
    for c in MONTO_COLS:
        detalle[c] = pd.to_numeric(detalle[c], errors="coerce").fillna(0.0)
    detalle["Avance_pct_pim"] = pd.to_numeric(detalle["Avance_pct_pim"], errors="coerce")
    detalle["Avance_pct_mef"] = pd.to_numeric(detalle["Avance_pct_mef"], errors="coerce")
    detalle = detalle.sort_values(["Anio", "Fuente"]).reset_index(drop=True)

    anual = pd.read_excel(fp, sheet_name=SHEET_RESUMEN)
    anual = anual.iloc[:, :8]
    anual.columns = [
        "Anio", "PIA", "PIM", "Compromiso", "Devengado", "Girado",
        "Avance_pct", "N_fuentes",
    ]
    anual["Anio"] = pd.to_numeric(anual["Anio"], errors="coerce")
    anual = anual.dropna(subset=["Anio"]).copy()
    anual["Anio"] = anual["Anio"].astype(int)
    for c in MONTO_COLS:
        anual[c] = pd.to_numeric(anual[c], errors="coerce")
    anual["Avance_pct"] = pd.to_numeric(anual["Avance_pct"], errors="coerce")
    anual["N_fuentes"] = pd.to_numeric(anual["N_fuentes"], errors="coerce")
    anual = anual.sort_values("Anio").reset_index(drop=True)

    return {"anual": anual, "detalle": detalle}


if __name__ == "__main__":
    _data = load_budget_data()
    print(_data["anual"].tail(10))
    print(_data["detalle"].tail(10))
