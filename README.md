# 🎗️ Cáncer en el Tiempo — Perú (INEN)

Dashboard interactivo en **Streamlit**, inspirado en el visor
[*Cancer Over Time*](https://gco.iarc.who.int) de GLOBOCAN/IARC, construido
sobre los datos de casos registrados por el **INEN** (Instituto Nacional de
Enfermedades Neoplásicas) para los 25 departamentos del Perú + el total nacional,
2000–2023.

## ✨ Funcionalidades

- Selección de **localización del tumor primario** y **departamento(s) de residencia** a comparar (hasta 10 a la vez) — la tarjeta de resumen muestra los nombres de los departamentos seleccionados, no solo la cantidad.
- **Comparación de periodos personalizable**: elige tú mismo el año base y el año de comparación para la tarjeta de variación (%).
- Rango de años ajustable con slider.
- Gráfico de **líneas o barras** (Plotly, interactivo: zoom, hover, exportar como PNG).
- Eje X mostrando **todos los años** disponibles (no solo algunos ticks).
- **Etiquetas de datos** activables en cada punto/barra.
- **Línea de tendencia** (regresión lineal simple, con R²) por región.
- **Detección de punto de quiebre estructural**: regresión segmentada de dos
  tramos + test de Chow, para identificar el año en que la tendencia cambia
  de forma estadísticamente significativa (ver `stats_analysis.py`).
- **Test de tendencia Mann-Kendall** (Hamed-Rao, con respaldo automático en
  Yue-Wang si la corrección de autocorrelación es numéricamente inestable) +
  **pendiente de Sen**: responde de forma robusta "¿hay o no tendencia?" sin
  el riesgo de falsos positivos de la regresión lineal simple cuando los
  datos están autocorrelacionados.
- **Proyección de casos a futuro** con análisis de sensibilidad (3
  escenarios: conservador, recomendado con Holt amortiguado, y lineal) más
  banda de incertidumbre por bootstrap, para no sobreestimar.
- **Ranking por año**: pestaña dedicada para ver qué localizaciones de
  cáncer concentran más casos en un año y departamento específicos
  (gráfico de barras horizontales ordenado, con matices del mismo color
  y el puesto de cada una — 1°, 2°, 3°... — al estilo del ranking por
  país de GLOBOCAN).
- **Panel de KPI**: además de la variación % entre dos años elegibles por
  el usuario, muestra cambio absoluto, CAGR (crecimiento anual compuesto),
  año pico y un resumen de tendencia (Mann-Kendall) de un vistazo.
- **Quiebre en un año específico (evento)**: a diferencia de la detección
  automática, permite indicar el año exacto de un evento (una ley, la
  apertura de un hospital oncológico, etc.) y aplica el test de Chow ahí
  mismo, con opción de excluir años de "implementación" tras el evento.
- **Línea temporal suavizada (LOWESS)**: para ver la tendencia general sin
  el ruido año a año, sin imponer una forma lineal.
- Escala logarítmica opcional.
- **Paleta de colores personalizable** por región.
- Tabla dinámica (región × año) y **descarga en CSV** (filtrado, tabla dinámica o dataset completo).
- **Carga de datos recientes**: sube un `.xlsx`/`.csv` con años nuevos (ej. 2024, 2025) directamente desde la barra lateral, sin tocar el código.
- Normalización automática de nombres de localización (el archivo original del INEN tiene mayúsculas/minúsculas y tildes inconsistentes; `data_processing.py` los estandariza).

## 📁 Estructura del proyecto

```
inen-cancer-dashboard/
├── app.py                     # Interfaz Streamlit (todo el dashboard)
├── data_processing.py         # Carga, limpieza y actualización de datos
├── stats_analysis.py          # Tendencia lineal + detección de quiebre (test de Chow)
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Tema de colores de la app
├── data/
│   ├── Datos_Cancer_INEN_con_Peru_Total.xlsx   # Fuente original
│   └── plantilla_datos_nuevos.csv              # Ejemplo para incorporar datos nuevos
└── README.md
```

## 🚀 Ejecutar en local

```bash
git clone <URL_DE_TU_REPO>
cd inen-cancer-dashboard
python -m venv venv && source venv/bin/activate   # opcional pero recomendado
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá en `http://localhost:8501`.

## ☁️ Desplegar gratis en Streamlit Community Cloud (vía GitHub)

1. Crea un repositorio en GitHub y sube **todo** el contenido de esta carpeta
   (incluyendo la subcarpeta `data/` con el Excel original):
   ```bash
   git init
   git add .
   git commit -m "Dashboard cáncer INEN"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
   git push -u origin main
   ```
2. Entra a **https://share.streamlit.io** con tu cuenta de GitHub.
3. Clic en **"New app"** → selecciona el repositorio, la rama (`main`) y el
   archivo principal (`app.py`).
4. Clic en **"Deploy"**. En 1–2 minutos tendrás una URL pública
   (`https://<tu-app>.streamlit.app`).

Cada vez que hagas `git push` con cambios, la app se re-despliega sola.

## 🔄 Incorporar datos recientes

Hay dos formas:

**A) Desde la interfaz (recomendado, sin tocar código)**
En la barra lateral, abre "📥 Incorporar datos recientes" y sube un archivo
con estas columnas:

| Departamento | localizacion | 2024 |
|---|---|---|
| Lima | MAMA | 850 |
| Arequipa | MAMA | 90 |
| Perú | Todas las localizaciones | 15000 |

(ver `data/plantilla_datos_nuevos.csv` como ejemplo). El archivo se combina
con el histórico; si una combinación región/localización/año ya existía, el
valor nuevo la reemplaza (útil para corregir cifras preliminares).

> ⚠️ En Streamlit Community Cloud el sistema de archivos **no es
> persistente**: los datos subidos así se reflejan mientras la app esté
> activa, pero se pierden si el contenedor se reinicia. Para que un año
> nuevo quede permanente, usa la opción B.

**B) De forma permanente (edita el repositorio)**
Reemplaza o amplía `data/Datos_Cancer_INEN_con_Peru_Total.xlsx` agregando
columnas de año nuevas, borra `data/cancer_data_long.csv` si existe (caché),
haz `git commit` + `git push`. La app reconstruirá la caché automáticamente.

## 🎨 Personalización

- **Colores del tema**: edita `.streamlit/config.toml`.
- **Colores por región en el gráfico**: activa "Elegir color por región" en el panel lateral.
- **Paleta por defecto / logo / textos**: variables al inicio de `app.py` (`PRIMARY_COLOR`, `SECONDARY_COLOR`, `ACCENT_COLOR`, el bloque `CUSTOM_CSS`, el encabezado `app-header`).
- **Agregar una pestaña de tasas ajustadas (ASR)**: el archivo del INEN trae
  *casos*, no tasas. Si consigues población por región/año (ej. INEI), puedes
  añadir una función en `data_processing.py` que una ambas tablas y calcule
  `Casos / Población * 100 000`, y un radio adicional "Casos" vs "Tasa" en
  `app.py`.
- **Nuevas fuentes de datos** (otros países, otras enfermedades): basta con
  que el archivo tenga columnas `Departamento`/`localizacion`/años; el mismo
  pipeline de limpieza se reutiliza.

## 📊 Fuente y notas metodológicas

- Fuente: Instituto Nacional de Enfermedades Neoplásicas (INEN), Perú.
- Los valores son **casos nuevos registrados por el INEN**, no la incidencia
  nacional total (el INEN es un centro de referencia oncológica, no cubre el
  100% de los casos diagnosticados en el país) ni tasas estandarizadas por
  edad — a diferencia de GLOBOCAN, que sí reporta ASR (Age-Standardized Rate).
- Años cubiertos: 2000–2023 (ampliable, ver sección "Incorporar datos recientes").

## 📐 Análisis de punto de quiebre — metodología

`stats_analysis.detect_breakpoint()` implementa una **regresión segmentada
de dos tramos**: prueba cada año posible como punto de corte (dejando un
mínimo de 3 años a cada lado), ajusta una recta a cada tramo y elige el año
que minimiza la suma de cuadrados residuales combinada. Luego aplica un
**test de Chow** para contrastar ese modelo segmentado contra un único
modelo lineal para toda la serie (H0: no hay cambio estructural); si
p < 0.05 el quiebre se reporta como estadísticamente significativo. El
análisis se calcula para una región a la vez (selector en la barra
lateral) para mantener el gráfico legible.

## 🧪 ¿Hay o no tendencia? — Mann-Kendall vs. regresión lineal

La regresión lineal simple (mínimos cuadrados) asume residuos
independientes. En conteos anuales de cáncer los residuos suelen estar
**autocorrelacionados** (un año alto tiende a seguir a otro año alto), lo
que subestima el error estándar de la pendiente y puede mostrar
significancia estadística artificial — una "tendencia" que en realidad es
solo ruido correlacionado.

`stats_analysis.mann_kendall_trend()` usa el test de **Mann-Kendall
modificado por Hamed & Rao (1998)**, que corrige explícitamente la varianza
por la autocorrelación de la serie y no asume una forma lineal. Cuando esa
corrección es numéricamente inestable (frecuente en series muy
autocorrelacionadas y cortas, como el total nacional), el módulo recurre
automáticamente a la modificación de **Yue-Wang (2004)**, otra corrección
robusta a autocorrelación — y siempre indica en el resultado qué variante
se usó. Se acompaña de la **pendiente de Sen** (mediana de las pendientes
entre todos los pares de puntos), robusta a valores atípicos, a diferencia
de la pendiente de mínimos cuadrados.

## 🔮 Proyección de casos — metodología y análisis de sensibilidad

Extrapolar una sola recta (u otra curva sin amortiguar) hacia el futuro
asume que el ritmo de cambio se mantendrá constante indefinidamente, lo que
casi nunca ocurre en series de salud (la cobertura de registro, la
capacidad diagnóstica, etc. tienden a saturarse) y típicamente
**sobreestima** a mediano plazo.

`stats_analysis.project_series()` entrega **3 escenarios** en vez de un
único número:

| Escenario | Método | Lectura |
|---|---|---|
| **Conservador** | Promedio de los últimos 3 años, sin tendencia | Cota inferior — asume que el crecimiento se detiene |
| **Recomendado** | Suavizado exponencial de Holt con **tendencia amortiguada** (damped trend) | Método estándar en pronósticos para no sobreestimar (Gardner & McKenzie); reduce gradualmente la pendiente proyectada |
| **Lineal (Sen)** | Extrapolación de la pendiente de Sen, sin amortiguar | Referencia de crecimiento sostenido; no siempre es el más alto |

El escenario recomendado incluye una **banda de incertidumbre al 90%**
calculada por bootstrap de los residuos del modelo (remuestreo con
reemplazo y re-proyección 300 veces), en vez de una fórmula analítica que
también podría subestimar el error por autocorrelación. El horizonte es
ajustable (1 a 10 años) desde la barra lateral.

## 🏛️ Quiebre por evento específico vs. detección automática

`detect_breakpoint()` busca automáticamente el año que mejor explica un
cambio de tendencia. `chow_test_arbitrary_break()` responde una pregunta
distinta: **"¿el cambio de tendencia coincide con un evento que yo ya
conozco?"** — útil para evaluar el impacto de una ley, una campaña de
tamizaje o la apertura de un centro oncológico. Se le indica el año del
evento y, opcionalmente, cuántos años de "implementación" excluir
inmediatamente después (periodo en que el efecto aún no se consolida), y
aplica el mismo test de Chow que la detección automática, pero anclado al
año que el usuario define.

## 🛠️ Stack técnico

Python · [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com/python/) · Pandas · SciPy (test de Chow) · [pymannkendall](https://pypi.org/project/pymannkendall/) · [statsmodels](https://www.statsmodels.org/) (Holt damped trend)
