"""
data_processing.py
-------------------
Carga, limpieza, normalización y actualización de los datos de casos de
cáncer del INEN (Perú) usados por el dashboard.

El archivo fuente (Excel) tiene un formato "ancho": una fila por
Departamento x Localización, y una columna por año con el número de
casos registrados. Este módulo lo transforma a formato "largo"
(Departamento, Localizacion, Anio, Casos), normaliza los nombres de
localización (que en el archivo original vienen con mayúsculas,
minúsculas, tildes inconsistentes, asteriscos y espacios extra) y
expone funciones para:

  - cargar los datos limpios (load_data)
  - agregar/actualizar años recientes desde un archivo nuevo (append_new_data)
  - listar catálogos (departamentos, localizaciones)

Todo el módulo es independiente de Streamlit: puede probarse con
`python data_processing.py` o importarse desde app.py.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
RAW_FILE = DATA_DIR / "Datos_Cancer_INEN_con_Peru_Total.xlsx"
CLEAN_CACHE = DATA_DIR / "cancer_data_long.csv"

PERU_LABEL = "Perú (Total nacional)"
ALL_SITES_LABEL = "Todas las localizaciones"

# Mapa de nombre canónico (bonito, con tildes) -> lista de alias que
# aparecen en el archivo crudo, ya normalizados (sin tildes, en
# mayúsculas, sin asteriscos ni espacios extra). Si en el futuro llegan
# nuevas variantes de escritura, basta con agregarlas aquí.
CANONICAL_SITES: dict[str, list[str]] = {
    "Ano": ["ANO"],
    "Cavidad Oral": ["CAVIDAD ORAL"],
    "Colon": ["COLON"],
    "Coriocarcinoma": ["CORIOCARCINOMA"],
    "Cuello Uterino": ["CUELLO UTERINO"],
    "Cuerpo Uterino": ["CUERPO UTERINO"],
    "Esófago": ["ESOFAGO"],
    "Estómago": ["ESTOMAGO"],
    "Fosa Nasal": ["FOSA NASAL"],
    "Hígado": ["HIGADO"],
    "Huesos y Cartílago": ["HUESOS Y CARTILAGO"],
    "Laringe": ["LARINGE"],
    "Leucemia": ["LEUCEMIA"],
    "Linfoma de Hodgkin": ["LINFOMA DE HODGKIN", "LINFOMA DE HODKIN"],
    "Linfoma No Hodgkin": ["LINFOMA NO HODGKIN", "LINFOMA NO HODKIN"],
    "Mama": ["MAMA"],
    "Melanoma de Piel": ["MELANOMA DE PIEL"],
    "Mieloma": ["MIELOMA", "MIELOMA Y TUMORES MALIGNOS CELULAS PLASMATICAS"],
    "Ojo": ["OJO"],
    "Otros": ["OTROS"],
    "Ovario": ["OVARIO"],
    "Páncreas": ["PANCREAS"],
    "Pene": ["PENE"],
    "Piel No Melanoma": ["PIEL NO MELANOMA"],
    "Primario Desconocido": ["PRIMARIO DESCONOCIDO"],
    "Próstata": ["PROSTATA"],
    "Pulmón": ["PULMON"],
    "Recto": ["RECTO"],
    "Riñón": ["RINON"],
    "Senos Paranasales": ["SENOS PARANASALES"],
    "Sist. Nervioso Central": ["SIST. NERVIOSO CENTRAL", "SNC"],
    "Tejidos Blandos y Peritoneo": [
        "TEJIDOS BLANDOS Y PERITONEO",
        "TEJIDOS BLANDOS Y PERIOTONEO",
    ],
    "Testículo": ["TESTICULO"],
    "Tiroides": ["TIROIDES"],
    "Vejiga": ["VEJIGA"],
    "Vesícula Biliar": ["VESICULA BILIAR"],
    "Vías Biliares": ["VIAS BILIARES"],
    "Vulva": ["VULVA"],
    ALL_SITES_LABEL: ["TODAS LAS LOCALIZACIONES"],
}

# Índice inverso: alias normalizado -> nombre canónico
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canon for canon, aliases in CANONICAL_SITES.items() for alias in aliases
}


def _unaccent(text: str) -> str:
    """Quita tildes/diacríticos de un texto."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _normalize_key(raw: str) -> str:
    """Normaliza un nombre de localización crudo a una llave comparable:
    sin espacios extra, sin asteriscos finales, sin tildes, en mayúsculas."""
    text = str(raw).strip().rstrip("*").strip()
    text = re.sub(r"\s+", " ", text)
    text = _unaccent(text).upper()
    return text


def canonicalize_site(raw: str) -> str:
    """Devuelve el nombre canónico y legible de una localización.

    Si no se reconoce el alias (p. ej. porque llegó un archivo nuevo con
    una localización que no estaba mapeada), se devuelve el texto
    original en "Title Case" en vez de fallar, para que los datos no se
    pierdan silenciosamente.
    """
    key = _normalize_key(raw)
    if key in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[key]
    return str(raw).strip().rstrip("*").strip().title()


def _year_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if str(c).strip().isdigit()]


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte el formato ancho (una columna por año) a formato largo."""
    year_cols = _year_columns(df)
    long_df = df.melt(
        id_vars=["Departamento", "localizacion"],
        value_vars=year_cols,
        var_name="Anio",
        value_name="Casos",
    )
    long_df["Anio"] = long_df["Anio"].astype(int)
    long_df["Casos"] = pd.to_numeric(long_df["Casos"], errors="coerce")
    long_df["Localizacion"] = long_df["localizacion"].apply(canonicalize_site)
    long_df["Departamento"] = long_df["Departamento"].astype(str).str.strip()
    long_df.loc[long_df["Departamento"] == "Perú", "Departamento"] = PERU_LABEL

    long_df = long_df.drop(columns=["localizacion"])
    # Si una misma combinación Departamento/Localizacion/Anio aparece más
    # de una vez tras la normalización (por variantes que colapsan a un
    # mismo nombre), se suman los casos.
    long_df = (
        long_df.groupby(["Departamento", "Localizacion", "Anio"], as_index=False)[
            "Casos"
        ].sum(min_count=1)
    )
    return long_df


def build_clean_dataset(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    """Lee el Excel crudo del INEN y devuelve el dataset limpio en
    formato largo: columnas Departamento, Localizacion, Anio, Casos."""
    raw = pd.read_excel(raw_path)
    return _wide_to_long(raw)


def load_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Carga el dataset limpio, usando una caché en parquet para que el
    dashboard arranque rápido. Si no existe la caché (o se agregaron
    datos nuevos con append_new_data), se reconstruye desde el Excel."""
    if CLEAN_CACHE.exists() and not force_rebuild:
        return pd.read_csv(CLEAN_CACHE)
    df = build_clean_dataset()
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(CLEAN_CACHE, index=False)
    return df


def append_new_data(new_file_path: str | Path, save: bool = True) -> pd.DataFrame:
    """Incorpora datos recientes (por ejemplo, el año 2024 o 2025) a
    partir de un archivo nuevo subido por el usuario.

    El archivo nuevo puede ser .xlsx o .csv y debe tener columnas
    equivalentes a las del archivo original: 'Departamento',
    'localizacion' (o 'Localizacion') y una o más columnas de año
    (ej. 2024, 2025) con el número de casos.

    Devuelve el dataset combinado (histórico + nuevo) y, si save=True,
    sobrescribe la caché parquet para que el dashboard use los datos
    actualizados en el siguiente reinicio.
    """
    new_file_path = Path(new_file_path)
    if new_file_path.suffix.lower() in (".xlsx", ".xls"):
        new_raw = pd.read_excel(new_file_path)
    else:
        new_raw = pd.read_csv(new_file_path)

    # Aceptar tanto 'localizacion' como 'Localizacion' como nombre de columna
    cols = {c.lower(): c for c in new_raw.columns}
    if "localizacion" not in [c.lower() for c in new_raw.columns]:
        raise ValueError(
            "El archivo nuevo debe tener una columna 'localizacion' "
            "(o 'Localizacion')."
        )
    rename_map = {}
    if "departamento" in cols:
        rename_map[cols["departamento"]] = "Departamento"
    if "localizacion" in cols:
        rename_map[cols["localizacion"]] = "localizacion"
    new_raw = new_raw.rename(columns=rename_map)

    new_long = _wide_to_long(new_raw)

    base = load_data()
    combined = pd.concat([base, new_long], ignore_index=True)
    # Si el año/departamento/localización ya existía, el dato nuevo
    # reemplaza al anterior (permite corregir datos preliminares).
    combined = combined.drop_duplicates(
        subset=["Departamento", "Localizacion", "Anio"], keep="last"
    )
    combined = combined.sort_values(["Departamento", "Localizacion", "Anio"])

    if save:
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        combined.to_csv(CLEAN_CACHE, index=False)

    return combined


def departamentos(df: pd.DataFrame) -> list:
    depts = sorted(d for d in df["Departamento"].unique() if d != PERU_LABEL)
    return [PERU_LABEL] + depts


def localizaciones(df: pd.DataFrame) -> list:
    sites = sorted(s for s in df["Localizacion"].unique() if s != ALL_SITES_LABEL)
    return [ALL_SITES_LABEL] + sites


if __name__ == "__main__":
    data = load_data(force_rebuild=True)
    print(f"Filas: {len(data):,}")
    print(f"Departamentos: {len(departamentos(data))}")
    print(f"Localizaciones: {len(localizaciones(data))}")
    print(f"Años: {data['Anio'].min()}–{data['Anio'].max()}")
    print(data.head())
