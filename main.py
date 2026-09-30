"""
###############################################################################
#  GRUPO 7 - CLASIFICACIÓN DE PAÍSES SEGÚN TRANSICIÓN ENERGÉTICA  (Energía)  #
#  Machine Learning - Práctica 1 - Entregable 3 (guía de código en Python)   #
#  UNI - Facultad de Ingeniería Industrial y de Sistemas - Ciclo 2026-II     #
###############################################################################

Integrantes:
    Crisostomo Cahuas, Gabriel Andre        gabriel.crisostomo.c@uni.pe
    Curto Meneses, Alessandro del Piero     alessandro.curto.m@uni.pe
    Rubio Zeta, Luis Miguel                 luis.rubio.z@uni.pe
Profesor: Dr. Ing. Hilario Aradiel Castañeda

¿Qué hace este programa?
------------------------
Prepara los datos para que, en una etapa posterior, un clasificador supervisado
pueda ubicar a cada país y año en una de tres etapas de la transición
energética: INICIAL, INTERMEDIA o AVANZADA. Sigue la metodología CRISP-ML(Q)
(Studer et al., 2021, https://doi.org/10.48550/arXiv.2003.05155) y se detiene
en la tercera fase, como exige el curso:

    FASE 1  Comprensión del negocio  -> objetivos, regla de etiqueta, requisitos R1-R10
    FASE 2  Comprensión de los datos -> recolección, exploración y calidad por fuente
    FASE 3  Preparación de los datos -> selección, limpieza, atípicos, imputación,
                                        etiqueta, predictores, integración y partición

NO hay fase de modelado: aquí no se entrena ni se ejecuta ningún algoritmo de
aprendizaje automático. El producto es un archivo CSV país-año, sin faltantes,
etiquetado y verificado, listo para la fase de modelado.

Fuentes de datos (ambas abiertas, licencia CC BY 4.0):
    * Banco Mundial - World Development Indicators (API v2): economía, población,
      acceso a la electricidad, energía y CO2.
    * Our World in Data (OWID) - energy-data: composición de la generación
      eléctrica, que el Banco Mundial no publica y que es necesaria para la etiqueta.

Cómo se recorre el código (la función principal está al final):

    principal()
      |
      |-- fase1_comprension_negocio()        objetivos, regla de etiqueta y requisitos
      |
      |-- fase2_comprension_datos()
      |     |-- obtener_datos()              descarga (o lee la copia local) de BM y OWID
      |     |-- armar_panel()                une ambas fuentes en una tabla país-año
      |     |-- perfil_calidad()             faltantes, rangos, contraste entre fuentes
      |     `-- graficos_exploratorios()     figuras 1 a 5
      |
      `-- fase3_preparacion_datos()
            |-- seleccionar()                países, años y variables
            |-- limpiar_rangos()             valores imposibles -> vacío
            |-- filtro_hampel()              atípicos en ventanas móviles de 5 años
            |-- imputar()                    interpolación y mediana por grupo de ingreso
            |-- construir_etiqueta()         etapa 0/1/2 con la regla de la fase 1
            |-- construir_predictores()      logaritmos, variaciones y rezago de un año
            |-- integrar_y_particionar()     región, ingreso, banderas y conjuntos
            |-- controles_calidad()          R1-R10: si uno falla, no se guarda nada
            `-- graficos_preparacion()       figuras 6 a 8

Ejecución:
    python main.py                  # usa la copia local de datos/crudos (sin Internet)
    python main.py --descargar      # vuelve a descargar los datos de BM y OWID
    python main.py --pais CHL       # país que se muestra en la figura 4 (por defecto PER)
"""

# =============================================================================
# PASO 0. IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
# --- Biblioteca estándar de Python (no requiere instalación) -----------------
import argparse                     # opciones de la línea de comandos (--descargar, --pais)
import io                           # leer en memoria el CSV descargado de OWID
import json                         # guardar el resumen de la fase 1 en formato JSON
import sys                          # código de salida del programa
import time                         # esperas entre reintentos de descarga
from dataclasses import dataclass, asdict   # agrupar los requisitos de calidad
from datetime import datetime, timezone     # fecha de descarga de los datos
from pathlib import Path            # rutas que funcionan en Windows, Linux y macOS

# --- Librerías externas (se instalan con: pip install -r requirements.txt) ---
import numpy as np                  # cálculo numérico: logaritmos, medianas, NaN
import pandas as pd                 # tablas de datos (DataFrame) y series de tiempo
import requests                     # descarga de datos por HTTP (API del BM y OWID)
import matplotlib                   # gráficos
matplotlib.use("Agg")               # dibuja sin abrir ventanas; las figuras se guardan en PNG
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


# =============================================================================
# PASO 1. PARÁMETROS DEL PROYECTO
# =============================================================================
# Carpeta del proyecto. En Jupyter no existe __file__, por eso se usa la
# carpeta actual (la carpeta donde está main.ipynb).
RAIZ = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CARPETA_CRUDOS = RAIZ / "datos" / "crudos"          # datos tal como llegan de las fuentes
CARPETA_PREPARADOS = RAIZ / "datos" / "preparados"  # salida final de la fase 3
CARPETA_REPORTES = RAIZ / "resultados" / "reportes" # tablas CSV y reportes Markdown/JSON
CARPETA_FIGURAS = RAIZ / "resultados" / "figuras"   # gráficos PNG

# Direcciones de descarga
URL_API_BM = "https://api.worldbank.org/v2"
URL_OWID = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
ESPERA_MAX_SEG = 90     # tiempo máximo de espera por cada solicitud HTTP
INTENTOS = 3            # reintentos si la conexión falla

# Periodo de estudio
ANIO_INICIO, ANIO_FIN = 2000, 2024

# Indicadores del Banco Mundial: código -> (nombre en el proyecto, descripción)
INDICADORES_BM = {
    "NY.GDP.PCAP.KD":       ("pib_pc", "PIB per cápita (US$ constantes de 2015)"),
    "SP.POP.TOTL":          ("poblacion", "Población total (habitantes)"),
    "SP.URB.TOTL.IN.ZS":    ("urbano_pct", "Población urbana (% del total)"),
    "NV.IND.TOTL.ZS":       ("industria_pct", "Industria, valor agregado (% del PIB)"),
    "EG.FEC.RNEW.ZS":       ("renovable_pct", "Energía renovable (% del consumo final de energía)"),
    "EG.ELC.ACCS.ZS":       ("electricidad_pct", "Acceso a la electricidad (% de la población)"),
    "EN.GHG.CO2.PC.CE.AR5": ("co2_pc", "Emisiones de CO2 per cápita (t CO2e por habitante)"),
    "EG.USE.PCAP.KG.OE":    ("energia_pc_bm", "Uso de energía (kg de petróleo equivalente per cápita)"),
    "EG.USE.COMM.FO.ZS":    ("fosil_pct", "Energía de combustibles fósiles (% del total)"),
    "EG.ELC.COAL.ZS":       ("carbon_elec_pct", "Electricidad generada con carbón (% del total)"),
}
# Columnas de OWID: nombre original -> (nombre en el proyecto, descripción)
COLUMNAS_OWID = {
    "low_carbon_share_elec":  ("baja_carbono_elec_pct", "Electricidad baja en carbono: renovable + nuclear (% de la generación)"),
    "renewables_share_elec":  ("renovable_elec_pct", "Electricidad renovable (% de la generación)"),
    "fossil_share_elec":      ("fosil_elec_pct", "Electricidad fósil (% de la generación)"),
    "hydro_share_elec":       ("hidro_elec_pct", "Electricidad hidroeléctrica (% de la generación)"),
    "coal_share_elec":        ("carbon_elec_owid_pct", "Electricidad con carbón según OWID (% de la generación)"),
    "carbon_intensity_elec":  ("intensidad_carbono_elec", "Intensidad de carbono de la electricidad (gCO2/kWh)"),
    "energy_per_capita":      ("energia_pc", "Consumo de energía primaria per cápita (kWh)"),
    "per_capita_electricity": ("consumo_elec_pc", "Generación eléctrica per cápita (kWh)"),
    "energy_per_gdp":         ("intensidad_energetica", "Energía primaria por unidad de PIB (kWh/US$)"),
    "population":             ("poblacion_owid", "Población según OWID (habitantes)"),
}
# Columnas adicionales de OWID que solo se usan en las figuras
COLUMNAS_OWID_FIGURAS = [
    "gas_share_elec", "solar_share_elec", "wind_share_elec", "electricity_generation",
    "fossil_electricity", "nuclear_electricity", "hydro_electricity", "solar_electricity",
    "wind_electricity", "other_renewable_electricity",
]
VARS_BM = [v[0] for v in INDICADORES_BM.values()]
VARS_OWID = [v[0] for v in COLUMNAS_OWID.values()]
DESCRIPCION = {**{v[0]: v[1] for v in INDICADORES_BM.values()},
               **{v[0]: v[1] for v in COLUMNAS_OWID.values()}}

# Papel de cada variable en el proyecto
ETIQUETA_BASE = ["baja_carbono_elec_pct", "electricidad_pct"]   # insumos de la regla de etiqueta
COVARIABLES = ["pib_pc", "poblacion", "urbano_pct", "industria_pct",
               "renovable_pct", "co2_pc", "energia_pc", "consumo_elec_pc"]
# Variables de la matriz eléctrica: describen casi lo mismo que la etiqueta. Si
# entraran como predictores, el futuro modelo "vería la respuesta" (fuga de
# información), por eso se excluyen del conjunto final.
FUGA = ["renovable_elec_pct", "fosil_elec_pct", "hidro_elec_pct",
        "carbon_elec_owid_pct", "intensidad_carbono_elec", "carbon_elec_pct"]


@dataclass(frozen=True)
class Requisitos:
    """Requisitos de calidad fijados en la fase 1 (se verifican en la fase 3)."""
    min_paises: int = 120            # R1: países mínimos en el conjunto final
    cobertura_etiqueta: float = 0.85 # R2: años con dato de electricidad baja en carbono
    max_faltantes_var: float = 25.0  # R3: % máximo de vacíos para usar una variable
    max_faltantes_pais: float = 0.40 # un país sale si le falta > 40 % de años en una variable
    umbral_intermedia: float = 30.0  # b >= 30 %  -> al menos etapa intermedia
    umbral_avanzada: float = 60.0    # b >= 60 %  -> candidata a etapa avanzada ...
    acceso_minimo: float = 90.0      # ... solo si el acceso a la electricidad es >= 90 %
    min_clase: float = 10.0          # R7: % mínimo de filas por etapa
    tolerancia_fuentes: float = 5.0  # diferencia aceptada entre BM y OWID (% o p.p.)


REQ = Requisitos()

# Filtro de Hampel (detección de atípicos) e imputación
VENTANA_HAMPEL = 5          # años de la ventana móvil centrada
K_HAMPEL = 3.0              # número de desviaciones robustas para marcar un valor
DESVIO_MIN_PCT = 10.0       # además, al menos 10 puntos porcentuales ...
DESVIO_MIN_NIVEL = np.log(1.2)  # ... o 20 % en variables de nivel
ANIOS_CHOQUE = (2009, 2020) # crisis financiera y pandemia: cambios reales, no errores
MAX_HUECO = 3               # años seguidos que se pueden rellenar por interpolación

# Partición temporal (solo se ETIQUETAN las filas; no se entrena nada)
CORTES = {"entrenamiento": (2004, 2015), "validacion": (2016, 2019), "prueba": (2020, 2024)}

# Traducción de regiones y grupos de ingreso del Banco Mundial
REGIONES = {
    "Sub-Saharan Africa": ("África subsahariana", "africa_subsahariana"),
    "Latin America & Caribbean": ("América Latina y el Caribe", "america_latina_caribe"),
    "North America": ("América del Norte", "america_norte"),
    "East Asia & Pacific": ("Asia oriental y el Pacífico", "asia_oriental_pacifico"),
    "South Asia": ("Asia meridional", "asia_sur"),
    "Europe & Central Asia": ("Europa y Asia central", "europa_asia_central"),
    "Middle East, North Africa, Afghanistan & Pakistan":
        ("Medio Oriente, norte de África, Afganistán y Pakistán", "medio_oriente_norte_africa"),
}
INGRESOS = {
    "High income": ("Ingreso alto", "alto"),
    "Upper middle income": ("Ingreso medio alto", "medio_alto"),
    "Lower middle income": ("Ingreso medio bajo", "medio_bajo"),
    "Low income": ("Ingreso bajo", "bajo"),
}

# Colores de las figuras (uno por etapa)
COLOR_ETAPA = {"inicial": "#d1495b", "intermedia": "#edae49", "avanzada": "#00798c"}
AZUL, GRIS = "#30638e", "#8d99ae"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6, "savefig.dpi": 150,
})


# =============================================================================
# PASO 2. FUNCIONES DE APOYO
# =============================================================================
def seccion(texto: str) -> None:
    """Imprime un título de sección para seguir el avance en la consola."""
    print("\n" + "=" * 78 + f"\n {texto}\n" + "=" * 78)


def nota(texto: str) -> None:
    """Imprime un mensaje de avance con sangría."""
    print(f"   - {texto}")


def preparar_carpetas() -> None:
    """Crea las carpetas de salida si todavía no existen."""
    for carpeta in (CARPETA_CRUDOS, CARPETA_PREPARADOS, CARPETA_REPORTES, CARPETA_FIGURAS):
        carpeta.mkdir(parents=True, exist_ok=True)


def guardar_figura(fig, nombre: str) -> None:
    """Guarda una figura en resultados/figuras y libera la memoria."""
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / nombre, bbox_inches="tight")
    plt.close(fig)
    nota(f"figura guardada: resultados/figuras/{nombre}")


def tabla_md(df: pd.DataFrame) -> str:
    """Convierte un DataFrame en una tabla Markdown (sin depender de 'tabulate')."""
    columnas = [str(c) for c in df.columns]
    filas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for _, fila in df.iterrows():
        filas.append("| " + " | ".join(str(v) for v in fila.values) + " |")
    return "\n".join(filas)


def es_porcentaje(variable: str) -> bool:
    """Las variables que terminan en _pct están acotadas entre 0 y 100."""
    return variable.endswith("_pct")


def asignar_etapa(b: float, acceso: float) -> str | float:
    """
    REGLA DE ETIQUETA (definida en la fase 1).
        b      = electricidad baja en carbono, promedio de 3 años (%)
        acceso = población con acceso a la electricidad (%)
    Avanzada   : b >= 60 y acceso >= 90
    Intermedia : b >= 30 (y no cumple la condición de avanzada)
    Inicial    : b < 30
    Devuelve NaN si todavía no hay tres años para promediar.
    """
    if pd.isna(b):
        return np.nan
    if b >= REQ.umbral_avanzada and acceso >= REQ.acceso_minimo:
        return "avanzada"
    return "intermedia" if b >= REQ.umbral_intermedia else "inicial"


# =============================================================================
# PASO 3. FASE 1 - COMPRENSIÓN DEL NEGOCIO (Business Understanding)
# =============================================================================
def fase1_comprension_negocio() -> dict:
    """
    Deja por escrito QUÉ se quiere resolver, PARA QUIÉN y CÓMO se sabrá que la
    preparación de datos fue correcta. No usa datos todavía.

    Resultado esperado:
      * resultados/reportes/fase1_negocio.json y fase1_negocio.md con los
        objetivos, la regla de etiqueta y los requisitos R1-R10.
    """
    seccion("FASE 1 - COMPRENSIÓN DEL NEGOCIO")
    negocio = {
        "problema": ("Los organismos que financian o planifican la transición energética "
                     "necesitan saber, cada año y con un criterio común, en qué etapa está "
                     "cada país para priorizar recursos y cooperación."),
        "interesados": ["Ministerios de energía (en el Perú, el MINEM)", "Bancos de desarrollo",
                        "Investigadores y organismos internacionales"],
        "objetivo_negocio": ("Disponer cada año de una clasificación comparable de los países "
                             "según su avance en la transición energética."),
        "objetivo_mineria_datos": ("Preparar un conjunto país-año con la etapa de transición "
                                   "(inicial, intermedia, avanzada) como variable objetivo y "
                                   "predictores medidos el año anterior, apto para entrenar "
                                   "después un clasificador de tres clases."),
        "criterio_exito_modelo_futuro": "Exactitud balanceada >= 0,75 en los años de prueba (2020-2024).",
        "criterio_exito_negocio": "Cada clasificación debe poder explicarse con una regla pública.",
        "regla_etiqueta": {
            "b": "promedio de 3 años de la electricidad baja en carbono (% de la generación)",
            "avanzada": f"b >= {REQ.umbral_avanzada:.0f} y acceso >= {REQ.acceso_minimo:.0f}",
            "intermedia": f"b >= {REQ.umbral_intermedia:.0f} (sin cumplir avanzada)",
            "inicial": f"b < {REQ.umbral_intermedia:.0f}",
            "base": "ODS 7: electricidad limpia (7.2) y acceso universal (7.1)",
        },
        "requisitos": {
            "R1": f"Al menos {REQ.min_paises} países en el conjunto final",
            "R2": f"b disponible en >= {REQ.cobertura_etiqueta:.0%} de los años de cada país",
            "R3": f"Variables usadas con <= {REQ.max_faltantes_var:.0f} % de faltantes",
            "R4": "Sin valores nulos ni infinitos",
            "R5": "Una fila por país y año, sin saltos entre años",
            "R6": "Porcentajes dentro de [0, 100]",
            "R7": f"Cada etapa con >= {REQ.min_clase:.0f} % de las filas",
            "R8": "Ninguna columna derivada de la etiqueta; predictores del año anterior",
            "R9": "La etiqueta se reproduce exactamente al recalcularla",
            "R10": "Las tres etapas presentes en entrenamiento, validación y prueba",
        },
        "parametros": asdict(REQ),
    }
    (CARPETA_REPORTES / "fase1_negocio.json").write_text(
        json.dumps(negocio, indent=2, ensure_ascii=False), encoding="utf-8")

    reglas = pd.DataFrame([["Inicial (0)", negocio["regla_etiqueta"]["inicial"], "La generación depende sobre todo de combustibles fósiles"],
                           ["Intermedia (1)", negocio["regla_etiqueta"]["intermedia"], "Matriz mixta, o limpia pero con acceso incompleto"],
                           ["Avanzada (2)", negocio["regla_etiqueta"]["avanzada"], "Electricidad mayoritariamente limpia y casi universal"]],
                          columns=["Etapa", "Condición", "Lectura"])
    req = pd.DataFrame(list(negocio["requisitos"].items()), columns=["Código", "Requisito"])
    md = ["# Fase 1 - Comprensión del negocio", "",
          f"**Problema.** {negocio['problema']}", "",
          f"**Objetivo de negocio.** {negocio['objetivo_negocio']}", "",
          f"**Objetivo de minería de datos.** {negocio['objetivo_mineria_datos']}", "",
          f"**Criterios de éxito.** {negocio['criterio_exito_modelo_futuro']} "
          f"{negocio['criterio_exito_negocio']}", "",
          "## Regla de etiqueta", "", tabla_md(reglas), "",
          "## Requisitos de calidad", "", tabla_md(req), ""]
    (CARPETA_REPORTES / "fase1_negocio.md").write_text("\n".join(md), encoding="utf-8")

    nota("objetivo de negocio y de minería de datos definidos")
    nota("regla de etiqueta: inicial (<30 %), intermedia (>=30 %), avanzada (>=60 % y acceso >=90 %)")
    nota(f"{len(negocio['requisitos'])} requisitos de calidad (R1-R10) registrados")
    return negocio


# =============================================================================
# PASO 4. FASE 2 - COMPRENSIÓN DE LOS DATOS (Data Understanding)
# =============================================================================
# ---- 4.1 Recolección inicial -------------------------------------------------
def _pedir(sesion: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """Solicitud HTTP con reintentos: espera 3, 6 y 9 segundos entre intentos."""
    for intento in range(1, INTENTOS + 1):
        try:
            resp = sesion.get(url, params=params, timeout=ESPERA_MAX_SEG)
            resp.raise_for_status()
            return resp
        except requests.RequestException as error:
            nota(f"intento {intento}/{INTENTOS} fallido ({error})")
            time.sleep(3 * intento)
    raise ConnectionError(f"No se pudo descargar {url}")


def descargar_banco_mundial(sesion: requests.Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descarga los 10 indicadores (formato largo) y los metadatos de cada país."""
    partes = []
    for codigo, (nombre, _) in INDICADORES_BM.items():
        pagina, filas = 1, []
        while True:
            params = {"format": "json", "per_page": 20000, "page": pagina,
                      "date": f"{ANIO_INICIO}:{ANIO_FIN}"}
            info, registros = _pedir(sesion, f"{URL_API_BM}/country/all/indicator/{codigo}", params).json()
            for r in registros or []:
                filas.append({"iso3": r.get("countryiso3code") or "", "entidad": r["country"]["value"],
                              "anio": int(r["date"]), "indicador": codigo, "valor": r["value"]})
            if pagina >= int(info.get("pages", 1)):
                break
            pagina += 1
        nota(f"{codigo:<22} -> {nombre:<18} {len(filas):>6} registros")
        partes.append(pd.DataFrame(filas))
    _, registros = _pedir(sesion, f"{URL_API_BM}/country", {"format": "json", "per_page": 500}).json()
    paises = pd.DataFrame([{"iso3": r["id"], "pais": r["name"],
                            "region": r["region"]["value"].strip(),
                            "nivel_ingreso": r["incomeLevel"]["value"].strip(),
                            "es_agregado": r["region"]["value"].strip() == "Aggregates"}
                           for r in registros])
    return pd.concat(partes, ignore_index=True), paises


def descargar_owid(sesion: requests.Session) -> pd.DataFrame:
    """Descarga el CSV de energía de OWID y conserva solo columnas y años del estudio."""
    texto = _pedir(sesion, URL_OWID).text
    owid = pd.read_csv(io.StringIO(texto))
    columnas = ["country", "year", "iso_code"] + list(COLUMNAS_OWID) + COLUMNAS_OWID_FIGURAS
    owid = owid[owid["year"].between(ANIO_INICIO, ANIO_FIN)][columnas]
    nota(f"OWID energy-data -> {len(owid)} filas país-año")
    return owid


def obtener_datos(descargar: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recolección inicial. Si ya existe la copia local en datos/crudos y no se pidió
    --descargar, se usa esa copia: así el resultado es reproducible y el programa
    funciona sin Internet. Se lee con keep_default_na=False para que el código
    'NA' (Namibia) no se confunda con un valor vacío.
    """
    rutas = {"bm": CARPETA_CRUDOS / "bm_indicadores.csv", "paises": CARPETA_CRUDOS / "bm_paises.csv",
             "owid": CARPETA_CRUDOS / "owid_energia.csv"}
    if not descargar and all(r.exists() for r in rutas.values()):
        nota("usando la copia local de datos/crudos (use --descargar para actualizarla)")
    else:
        nota("descargando datos del Banco Mundial y de OWID...")
        with requests.Session() as sesion:
            bm, paises = descargar_banco_mundial(sesion)
            owid = descargar_owid(sesion)
        bm.to_csv(rutas["bm"], index=False)
        paises.to_csv(rutas["paises"], index=False)
        owid.to_csv(rutas["owid"], index=False)
        (CARPETA_CRUDOS / "fuentes.json").write_text(json.dumps({
            "banco_mundial": URL_API_BM, "owid": URL_OWID, "periodo": f"{ANIO_INICIO}-{ANIO_FIN}",
            "descargado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "licencia": "CC BY 4.0"}, indent=2), encoding="utf-8")
    leer = dict(keep_default_na=False, na_values=[""])
    return (pd.read_csv(rutas["bm"], **leer), pd.read_csv(rutas["paises"], **leer),
            pd.read_csv(rutas["owid"], **leer))


# ---- 4.2 Integración inicial: panel país-año ----------------------------------
def armar_panel(bm: pd.DataFrame, paises: pd.DataFrame, owid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Une las dos fuentes en una tabla con una fila por (país, año) y una columna
    por variable. Se quitan los agregados (Mundo, regiones, grupos de ingreso)
    y se conservan solo los países presentes en AMBAS fuentes.
    """
    paises["es_agregado"] = paises["es_agregado"].astype(str).str.lower().eq("true")
    solo_paises = paises[~paises["es_agregado"]].copy()
    owid_paises = owid[owid["iso_code"].notna() & ~owid["iso_code"].str.startswith("OWID", na=False)]
    comunes = sorted(set(solo_paises["iso3"]) & set(owid_paises["iso_code"]))

    bm = bm[bm["indicador"].isin(INDICADORES_BM) & bm["anio"].between(ANIO_INICIO, ANIO_FIN)
            & bm["iso3"].isin(comunes)]
    ancho_bm = (bm.pivot_table(index=["iso3", "anio"], columns="indicador", values="valor")
                .rename(columns={k: v[0] for k, v in INDICADORES_BM.items()}))
    ancho_owid = (owid_paises[owid_paises["iso_code"].isin(comunes)]
                  .rename(columns={"iso_code": "iso3", "year": "anio",
                                   **{k: v[0] for k, v in COLUMNAS_OWID.items()}})
                  .set_index(["iso3", "anio"])[VARS_OWID])
    indice = pd.MultiIndex.from_product([comunes, range(ANIO_INICIO, ANIO_FIN + 1)], names=["iso3", "anio"])
    panel = ancho_bm.reindex(indice).join(ancho_owid.reindex(indice))[VARS_BM + VARS_OWID]

    info = {"paises_bm": len(solo_paises), "agregados_bm": int(paises["es_agregado"].sum()),
            "paises_owid": owid_paises["iso_code"].nunique(), "paises_comunes": len(comunes),
            "filas_panel": panel.shape[0], "columnas_panel": panel.shape[1],
            "duplicados_bm": int(bm.duplicated(["iso3", "anio", "indicador"]).sum()),
            "duplicados_owid": int(owid_paises.duplicated(["iso_code", "year"]).sum())}
    nota(f"{info['paises_bm']} países BM ({info['agregados_bm']} agregados excluidos), "
         f"{info['paises_owid']} en OWID -> {len(comunes)} en común")
    nota(f"panel: {panel.shape[0]} filas x {panel.shape[1]} variables; duplicados BM/OWID: "
         f"{info['duplicados_bm']}/{info['duplicados_owid']}")
    return panel, solo_paises, info


# ---- 4.3 Exploración y calidad -----------------------------------------------
def perfil_calidad(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Evalúa la calidad de cada variable y de cada fuente:
      * % de faltantes por variable, por fuente y por año,
      * valores imposibles (porcentajes fuera de [0, 100], niveles <= 0),
      * consistencia entre fuentes (población y electricidad con carbón),
      * cobertura de la variable base de la etiqueta por país,
      * correlaciones de Spearman entre las variables principales.
    """
    filas = []
    for var in panel.columns:
        x = panel[var]
        fuera = int(((x < 0) | (x > 100)).sum()) if es_porcentaje(var) else 0
        no_pos = int((x <= 0).sum()) if not es_porcentaje(var) else 0
        filas.append({"variable": var, "fuente": "BM" if var in VARS_BM else "OWID",
                      "descripcion": DESCRIPCION[var], "faltantes_pct": round(x.isna().mean() * 100, 1),
                      "fuera_de_rango": fuera, "no_positivos": no_pos,
                      "minimo": round(x.min(), 3), "mediana": round(x.median(), 3), "maximo": round(x.max(), 3)})
    calidad = pd.DataFrame(filas)
    calidad.to_csv(CARPETA_REPORTES / "fase2_calidad_variables.csv", index=False)
    panel.isna().groupby(level="anio").mean().mul(100).round(1).to_csv(
        CARPETA_REPORTES / "fase2_faltantes_por_anio.csv")

    # Contraste entre fuentes: si dos fuentes miden lo mismo, deben coincidir.
    dif_pob = (panel["poblacion"] - panel["poblacion_owid"]).abs() / panel["poblacion"] * 100
    dif_carbon = (panel["carbon_elec_pct"] - panel["carbon_elec_owid_pct"]).abs()
    suma = (panel["baja_carbono_elec_pct"] + panel["fosil_elec_pct"] - 100).abs()
    cobertura = panel["baja_carbono_elec_pct"].unstack().notna().mean(axis=1)
    principales = ["baja_carbono_elec_pct", "renovable_pct", "pib_pc", "electricidad_pct", "urbano_pct",
                   "industria_pct", "energia_pc", "consumo_elec_pc", "co2_pc"]
    panel[principales].corr(method="spearman").round(2).to_csv(CARPETA_REPORTES / "fase2_correlaciones_spearman.csv")

    resumen = {
        "faltantes_por_fuente": {"BM": round(panel[VARS_BM].isna().mean().mean() * 100, 1),
                                 "OWID": round(panel[VARS_OWID].isna().mean().mean() * 100, 1)},
        "poblacion_dif_mayor_5pct": {"filas": int((dif_pob > REQ.tolerancia_fuentes).sum()),
                                     "paises": int(dif_pob[dif_pob > REQ.tolerancia_fuentes].index.get_level_values(0).nunique())},
        "carbon_dif_mayor_5pp": {"filas": int((dif_carbon > REQ.tolerancia_fuentes).sum()),
                                 "paises": int(dif_carbon[dif_carbon > REQ.tolerancia_fuentes].index.get_level_values(0).nunique()),
                                 "comparables": int(dif_carbon.notna().sum())},
        "baja_mas_fosil_distinto_de_100": int((suma > 1).sum()),
        "cobertura_etiqueta": cobertura,
        "paises_cobertura_ok": int((cobertura >= REQ.cobertura_etiqueta).sum()),
        "industria_cero": sorted(set(panel[panel["industria_pct"] <= 0].index.get_level_values(0))),
    }
    nota(f"faltantes promedio: BM {resumen['faltantes_por_fuente']['BM']} %, "
         f"OWID {resumen['faltantes_por_fuente']['OWID']} %")
    nota("variables con más de 25 % de faltantes: "
         + ", ".join(calidad.loc[calidad["faltantes_pct"] > REQ.max_faltantes_var, "variable"]))
    nota(f"población BM vs OWID: {resumen['poblacion_dif_mayor_5pct']['filas']} filas difieren > 5 %")
    nota(f"carbón BM vs OWID: {resumen['carbon_dif_mayor_5pp']['filas']} de "
         f"{resumen['carbon_dif_mayor_5pp']['comparables']} filas difieren > 5 p.p.")
    nota(f"países con b en >= 85 % de los años: {resumen['paises_cobertura_ok']} de {len(cobertura)}")
    return calidad, resumen


def graficos_exploratorios(panel: pd.DataFrame, owid: pd.DataFrame, pais_ref: str) -> None:
    """Figuras 1 a 5 de la comprensión de los datos."""
    comunes = set(panel.index.get_level_values(0))
    datos = panel.reset_index()

    # Figura 1: composición de la generación eléctrica mundial (suma de los países)
    x = owid[owid["iso_code"].isin(comunes)]
    fuentes = ["fossil_electricity", "nuclear_electricity", "hydro_electricity",
               "solar_electricity", "wind_electricity", "other_renewable_electricity"]
    g = x.groupby("year")[fuentes].sum(min_count=1)
    parte = g.div(g.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.stackplot(parte.index, parte["fossil_electricity"], parte["nuclear_electricity"], parte["hydro_electricity"],
                 parte["solar_electricity"] + parte["wind_electricity"], parte["other_renewable_electricity"],
                 colors=["#adb5bd", "#6d597a", AZUL, "#edae49", "#83c5be"],
                 labels=["Fósil", "Nuclear", "Hidroeléctrica", "Solar y eólica", "Otras renovables"])
    ax.set(xlim=(ANIO_INICIO, ANIO_FIN), ylim=(0, 100), xlabel="Año", ylabel="% de la generación",
           title="Fig. 1. Generación eléctrica mundial por fuente")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    guardar_figura(fig, "fig01_matriz_electrica_mundial.png")

    # Figura 2: distribución de la electricidad baja en carbono en 2024
    b = datos.loc[datos["anio"] == ANIO_FIN, "baja_carbono_elec_pct"].dropna()
    bordes = np.arange(0, 105, 5)
    conteo, _ = np.histogram(b, bins=bordes)
    colores = [COLOR_ETAPA["inicial"] if e < 30 else COLOR_ETAPA["intermedia"] if e < 60 else COLOR_ETAPA["avanzada"]
               for e in bordes[:-1]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(bordes[:-1] + 2.5, conteo, width=4.3, color=colores)
    ax.set(xlabel=f"Electricidad baja en carbono en {ANIO_FIN} (%)", ylabel="Países",
           title=f"Fig. 2. Distribución entre {len(b)} países")
    ax.legend(handles=[Patch(color=c, label=t) for c, t in zip(COLOR_ETAPA.values(), ["< 30 %", "30-60 %", ">= 60 %"])],
              frameon=False, ncol=3, loc="upper center")
    guardar_figura(fig, "fig02_distribucion_baja_carbono.png")

    # Figura 3: PIB per cápita vs. electricidad baja en carbono, color = acceso
    q = datos[datos["anio"] == ANIO_FIN - 1].dropna(subset=["pib_pc", "baja_carbono_elec_pct", "electricidad_pct"])
    mapa = LinearSegmentedColormap.from_list("acceso", list(COLOR_ETAPA.values()))
    fig, ax = plt.subplots(figsize=(7, 4))
    puntos = ax.scatter(q["pib_pc"], q["baja_carbono_elec_pct"], c=q["electricidad_pct"], cmap=mapa,
                        vmin=0, vmax=100, s=22, edgecolor="white", linewidth=0.4)
    destacados = {"COD": "#343a40", "ETH": "#343a40", "NOR": "#f08c00", pais_ref: "#c92a2a"}
    for iso, color in destacados.items():
        fila = q[q["iso3"] == iso]
        if fila.empty:
            continue
        r = fila.iloc[0]
        ax.scatter(r["pib_pc"], r["baja_carbono_elec_pct"], s=70, facecolor="none", edgecolor=color, linewidth=1.5)
        desplazamiento = (-8, -14) if iso == "COD" else (6, -14)
        ax.annotate(iso, (r["pib_pc"], r["baja_carbono_elec_pct"]), xytext=desplazamiento, textcoords="offset points",
                    color=color, fontweight="bold", fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel=f"PIB per cápita {ANIO_FIN - 1} (US$ de 2015, escala log.)", ylabel="Electricidad baja en carbono (%)",
           title="Fig. 3. Riqueza, electricidad limpia y acceso")
    fig.colorbar(puntos, ax=ax, label="Acceso a la electricidad (%)")
    guardar_figura(fig, "fig03_pib_vs_baja_carbono.png")

    # Figura 4: matriz eléctrica del país de referencia
    p = owid[owid["iso_code"] == pais_ref].set_index("year")
    if not p.empty:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(p.index, p["hydro_share_elec"], color=AZUL, lw=2, label="Hidroeléctrica")
        ax.plot(p.index, p["gas_share_elec"], color="#adb5bd", lw=2, label="Gas natural")
        ax.plot(p.index, p["solar_share_elec"].fillna(0) + p["wind_share_elec"].fillna(0),
                color="#edae49", lw=2, label="Solar y eólica")
        ax.plot(p.index, p["low_carbon_share_elec"], color=COLOR_ETAPA["avanzada"], ls="--", label="Total baja en carbono")
        ax.axhline(REQ.umbral_avanzada, color=COLOR_ETAPA["inicial"], lw=0.8, ls=":")
        ax.set(ylim=(0, 100), xlabel="Año", ylabel="% de la generación",
               title=f"Fig. 4. Matriz eléctrica de {pais_ref}")
        ax.legend(frameon=False, ncol=2, fontsize=8)
        guardar_figura(fig, "fig04_matriz_pais_referencia.png")

    # Figura 5: mapa de calor de faltantes (variable x año)
    faltantes = panel.isna().groupby(level="anio").mean().mul(100).T
    faltantes = faltantes.loc[faltantes.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    im = ax.imshow(faltantes.values, aspect="auto", vmin=0, vmax=100,
                   cmap=LinearSegmentedColormap.from_list("f", ["#f8f9fa", "#edae49", "#d1495b"]))
    ax.set_yticks(range(len(faltantes)), faltantes.index, fontsize=7.5)
    ax.set_xticks(range(0, len(faltantes.columns), 4), faltantes.columns[::4])
    ax.grid(False)
    ax.set(xlabel="Año", title="Fig. 5. Países sin dato por variable y año (%)")
    fig.colorbar(im, ax=ax, label="% de países sin dato")
    guardar_figura(fig, "fig05_mapa_faltantes.png")


def fase2_comprension_datos(descargar: bool, pais_ref: str) -> dict:
    """
    Recolecta, explora y evalúa la calidad de los datos, y comprueba si es
    factible cumplir los requisitos de la fase 1.

    Resultado esperado:
      * datos/crudos/: copia local de BM y OWID.
      * resultados/reportes/fase2_*.csv y fase2_comprension_datos.md.
      * resultados/figuras/fig01 a fig05.
      * En consola: 206 países en común, panel de 5 150 filas x 20 variables.
    """
    seccion("FASE 2 - COMPRENSIÓN DE LOS DATOS")
    bm, paises, owid = obtener_datos(descargar)
    panel, solo_paises, info = armar_panel(bm, paises, owid)
    calidad, resumen = perfil_calidad(panel)
    graficos_exploratorios(panel, owid, pais_ref)

    # Factibilidad: ¿alcanzan los datos para cumplir los requisitos?
    factible = resumen["paises_cobertura_ok"] >= REQ.min_paises
    nota(f"factibilidad (R1 con la cobertura de R2): {'SÍ' if factible else 'NO'}")
    if not factible:
        sys.exit("No hay suficientes países con dato de la etiqueta: revise la fase 1.")

    md = ["# Fase 2 - Comprensión de los datos", "",
          f"* Países del Banco Mundial: {info['paises_bm']} (se excluyeron {info['agregados_bm']} agregados).",
          f"* Países en OWID: {info['paises_owid']}. En común: {info['paises_comunes']}.",
          f"* Panel: {info['filas_panel']} filas x {info['columnas_panel']} variables ({ANIO_INICIO}-{ANIO_FIN}).",
          f"* Duplicados: BM {info['duplicados_bm']}, OWID {info['duplicados_owid']}.",
          f"* Faltantes promedio por fuente: BM {resumen['faltantes_por_fuente']['BM']} %, "
          f"OWID {resumen['faltantes_por_fuente']['OWID']} %.",
          f"* Población BM vs OWID con diferencia > 5 %: {resumen['poblacion_dif_mayor_5pct']['filas']} filas "
          f"({resumen['poblacion_dif_mayor_5pct']['paises']} países).",
          f"* Electricidad con carbón BM vs OWID con diferencia > 5 p.p.: {resumen['carbon_dif_mayor_5pp']['filas']} "
          f"de {resumen['carbon_dif_mayor_5pp']['comparables']} filas.",
          f"* Filas donde baja en carbono + fósil no suma 100 (±1): {resumen['baja_mas_fosil_distinto_de_100']}.",
          f"* Países con la variable de la etiqueta en >= 85 % de los años: {resumen['paises_cobertura_ok']}.",
          "", "## Calidad por variable", "",
          tabla_md(calidad[["variable", "fuente", "faltantes_pct", "fuera_de_rango", "no_positivos"]]), ""]
    (CARPETA_REPORTES / "fase2_comprension_datos.md").write_text("\n".join(md), encoding="utf-8")
    return {"panel": panel, "paises": solo_paises, "calidad": calidad, "resumen": resumen, "info": info}


# =============================================================================
# PASO 5. FASE 3 - PREPARACIÓN DE LOS DATOS (Data Preparation)
# =============================================================================
def seleccionar(panel: pd.DataFrame, calidad: pd.DataFrame, resumen: dict, bitacora: list) -> pd.DataFrame:
    """
    SELECCIÓN. Se conservan:
      * los países con la variable base de la etiqueta en >= 85 % de los años (R2),
      * las 2 variables de la etiqueta y 8 covariables. Se descartan las
        variables con > 25 % de faltantes (R3) y las de la matriz eléctrica (fuga).
    """
    cobertura = resumen["cobertura_etiqueta"]
    fuera = sorted(cobertura[cobertura < REQ.cobertura_etiqueta].index)
    retenidos = sorted(cobertura[cobertura >= REQ.cobertura_etiqueta].index)
    muchas_faltas = calidad.loc[calidad["faltantes_pct"] > REQ.max_faltantes_var, "variable"].tolist()
    df = panel.loc[retenidos, ETIQUETA_BASE + COVARIABLES].copy()
    bitacora.append({"paso": "selección", "detalle": f"{len(fuera)} países sin cobertura suficiente de b: {', '.join(fuera)}"})
    bitacora.append({"paso": "selección", "detalle": f"variables con > 25 % de faltantes descartadas: {', '.join(muchas_faltas)}"})
    bitacora.append({"paso": "selección", "detalle": f"variables de la matriz eléctrica excluidas por fuga: {', '.join(FUGA)}"})
    nota(f"{len(retenidos)} países y {df.shape[1]} variables seleccionadas ({len(fuera)} países fuera por cobertura)")
    return df


def limpiar_rangos(df: pd.DataFrame, bitacora: list) -> tuple[pd.DataFrame, list]:
    """
    LIMPIEZA. Un porcentaje fuera de [0, 100] o un nivel <= 0 no puede ser real:
    se vacía para tratarlo luego como faltante. La industria igual a 0 % (caso de
    Venezuela) también se considera un dato no válido.
    Además se identifican los países con más de 40 % de años vacíos en alguna
    variable: se retirarán porque rellenarlos sería inventar la serie.
    """
    antes = int(df.isna().sum().sum())
    for var in df.columns:
        if es_porcentaje(var):
            df.loc[(df[var] < 0) | (df[var] > 100), var] = np.nan
        else:
            df.loc[df[var] <= 0, var] = np.nan
    df.loc[df["industria_pct"] <= 0, "industria_pct"] = np.nan
    vaciados = int(df.isna().sum().sum()) - antes
    faltas_pais = df.isna().groupby(level="iso3").mean()
    excluir_40 = sorted(faltas_pais[(faltas_pais > REQ.max_faltantes_pais).any(axis=1)].index)
    bitacora.append({"paso": "limpieza", "detalle": f"{vaciados} valores imposibles convertidos en vacío"})
    bitacora.append({"paso": "limpieza", "detalle": f"{len(excluir_40)} países con > 40 % de años vacíos: {', '.join(excluir_40)}"})
    nota(f"{vaciados} valores imposibles vaciados; {len(excluir_40)} países con > 40 % de vacíos marcados")
    return df, excluir_40


def filtro_hampel(df: pd.DataFrame, bitacora: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ATÍPICOS con el filtro de Hampel, país por país:
        mediana móvil m (5 años centrados) y desviación robusta s = 1,4826 * MAD.
        Un valor x es atípico si |x - m| > 3 s  y además |x - m| >= 10 p.p.
        (porcentajes) o >= 20 % (niveles, analizados en logaritmo).
    Decisión según el caso:
        * 'etiqueta' : valor de la electricidad baja en carbono -> se conserva
                       (un año seco o una central nueva son cambios reales).
        * 'choque'   : ocurre en 2009 o 2020 -> se conserva (crisis y pandemia).
        * 'corregido': cualquier otro -> se reemplaza por la mediana móvil.
    """
    marcados_corregidos = pd.DataFrame(False, index=df.index, columns=df.columns)
    registro = []
    for var in df.columns:
        serie = df[var].unstack()                       # filas = países, columnas = años
        y = serie if es_porcentaje(var) else np.log(serie)
        mediana = y.T.rolling(VENTANA_HAMPEL, center=True, min_periods=VENTANA_HAMPEL).median().T
        mad = (y - mediana).abs().T.rolling(VENTANA_HAMPEL, center=True, min_periods=VENTANA_HAMPEL).median().T
        desvio = (y - mediana).abs()
        minimo = DESVIO_MIN_PCT if es_porcentaje(var) else DESVIO_MIN_NIVEL
        atipico = (desvio > K_HAMPEL * 1.4826 * mad) & (desvio >= minimo)
        for iso, anio in atipico.stack()[lambda s: s].index:
            anio = int(anio)
            tipo = "etiqueta" if var == "baja_carbono_elec_pct" else "choque" if anio in ANIOS_CHOQUE else "corregido"
            reemplazo = mediana.loc[iso, anio] if es_porcentaje(var) else np.exp(mediana.loc[iso, anio])
            registro.append({"iso3": iso, "anio": anio, "variable": var, "valor_original": serie.loc[iso, anio],
                             "mediana_movil": float(reemplazo), "decision": tipo})
    registro = pd.DataFrame(registro, columns=["iso3", "anio", "variable", "valor_original", "mediana_movil", "decision"])
    for r in registro[registro["decision"] == "corregido"].itertuples():
        df.loc[(r.iso3, r.anio), r.variable] = r.mediana_movil
        marcados_corregidos.loc[(r.iso3, r.anio), r.variable] = True
    nota(f"Hampel: {len(registro)} valores marcados; {int((registro['decision'] == 'corregido').sum())} corregidos")
    return df, registro, marcados_corregidos


def _rellenar_serie(serie: pd.Series, en_logaritmo: bool) -> pd.Series:
    """Interpola huecos internos de hasta 3 años y extiende los bordes hasta 3 años."""
    x = np.log(serie) if en_logaritmo else serie.copy()
    x = x.interpolate(limit=MAX_HUECO, limit_area="inside").ffill(limit=MAX_HUECO).bfill(limit=MAX_HUECO)
    return np.exp(x) if en_logaritmo else x


def imputar(df: pd.DataFrame, vacios_iniciales: pd.DataFrame, corregidos: pd.DataFrame,
            excluir_40: list, paises: pd.DataFrame, bitacora: list) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    IMPUTACIÓN en dos pasos:
      1) Dentro de cada país: interpolación (en logaritmo para los niveles).
      2) Porcentajes que siguen vacíos: mediana del mismo grupo de ingreso y año.
    Los países que aún tienen huecos, junto con los marcados por la regla del
    40 %, se retiran. Se devuelve una máscara con cada valor imputado o
    corregido para crear las banderas bandera_imp_*.
    """
    for var in df.columns:
        df[var] = df.groupby(level="iso3")[var].transform(
            lambda s: _rellenar_serie(s.droplevel(0), not es_porcentaje(var)).values)
    n_serie = int((vacios_iniciales & df.notna()).sum().sum())

    ingreso = paises.set_index("iso3")["nivel_ingreso"]
    grupo = df.index.get_level_values("iso3").map(ingreso)
    anios = df.index.get_level_values("anio")
    n_grupo = 0
    for var in [v for v in COVARIABLES if es_porcentaje(v)]:
        mediana = df.groupby([grupo, anios])[var].transform("median")
        llenar = df[var].isna() & mediana.notna()
        n_grupo += int(llenar.sum())
        df.loc[llenar, var] = mediana[llenar]

    imputados = (vacios_iniciales & df.notna()) | corregidos
    con_huecos = df.isna().groupby(level="iso3").sum().sum(axis=1)
    sin_llenar = sorted(set(con_huecos[con_huecos > 0].index) - set(excluir_40))
    retirar = sorted(set(excluir_40) | set(con_huecos[con_huecos > 0].index))
    df = df.drop(index=retirar, level="iso3")
    imputados = imputados.drop(index=retirar, level="iso3")
    bitacora.append({"paso": "imputación", "detalle": f"{n_serie} valores por interpolación dentro del país"})
    bitacora.append({"paso": "imputación", "detalle": f"{n_grupo} valores por mediana del grupo de ingreso y año"})
    bitacora.append({"paso": "imputación", "detalle": f"{len(sin_llenar)} países con huecos que no se pudieron llenar: {', '.join(sin_llenar)}"})
    nota(f"imputados: {n_serie} (interpolación) + {n_grupo} (mediana por ingreso); "
         f"quedan {df.index.get_level_values('iso3').nunique()} países")
    return df, imputados, retirar


def construir_etiqueta(df: pd.DataFrame) -> pd.DataFrame:
    """
    ETIQUETA. b = promedio de 3 años (t-2, t-1, t) de la electricidad baja en
    carbono; luego se aplica asignar_etapa(b, acceso). También se calcula la
    etapa con el valor anual (sin promediar) solo para mostrar cuánto ruido
    evita el promedio.
    """
    df = df.reset_index().sort_values(["iso3", "anio"])
    df["b_promedio_3"] = df.groupby("iso3")["baja_carbono_elec_pct"].transform(lambda s: s.rolling(3).mean())
    df["etapa"] = [asignar_etapa(b, a) for b, a in zip(df["b_promedio_3"], df["electricidad_pct"])]
    df["etapa_anual"] = [asignar_etapa(b, a) for b, a in zip(df["baja_carbono_elec_pct"], df["electricidad_pct"])]
    df["etapa_num"] = df["etapa"].map({"inicial": 0, "intermedia": 1, "avanzada": 2})
    for col in ("etapa", "etapa_anual"):
        previo = df.groupby("iso3")[col].shift()
        df["cambio_" + col] = (df[col] != previo) & previo.notna()
    return df


# Predictores base (antes del rezago)
PREDICTORES = ["pib_pc_log", "poblacion_log", "urbano_pct", "industria_pct", "renovable_pct",
               "co2_pc_log", "energia_pc_log", "consumo_elec_pc_log", "var_pib_pc_pct",
               "var_urbano_pp", "var3_renovable_pp"]


def construir_predictores(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    TRANSFORMACIÓN.
      * Logaritmo natural de las variables de nivel (PIB, población, CO2, energía):
        reduce la asimetría de variables que van de cientos a cientos de miles.
      * Variación anual del PIB per cápita (%), cambio anual de la urbanización
        (p.p.) y cambio en 3 años de la energía renovable (p.p.).
      * REZAGO de un año (sufijo _ant): la etapa del año t se explica con datos
        de t-1, como ocurriría en la práctica.
    Las filas sin historia suficiente (2000-2003) se retiran.
    """
    g = df.groupby("iso3")
    for var in ["pib_pc", "poblacion", "co2_pc", "energia_pc", "consumo_elec_pc"]:
        df[var + "_log"] = np.log(df[var])
    df["var_pib_pc_pct"] = g["pib_pc"].pct_change() * 100
    df["var_urbano_pp"] = g["urbano_pct"].diff()
    df["var3_renovable_pp"] = g["renovable_pct"].diff(3)
    for var in PREDICTORES:
        df[var + "_ant"] = df.groupby("iso3")[var].shift(1)
    df["anio_relativo"] = df["anio"] - CORTES["entrenamiento"][0]
    antes = len(df)
    df = df.dropna(subset=["etapa"] + [v + "_ant" for v in PREDICTORES]).copy()
    nota(f"predictores: {len(PREDICTORES)} con rezago de un año; {antes - len(df)} filas sin historia retiradas "
         f"(el conjunto empieza en {df['anio'].min()})")
    return df, antes - len(df)


def integrar_y_particionar(df: pd.DataFrame, paises: pd.DataFrame, imputados: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    INTEGRACIÓN: nombre del país, región y grupo de ingreso (en español) y sus
    variables indicadoras 0/1; banderas bandera_imp_* (1 = valor imputado o
    corregido). PARTICIÓN temporal: 2004-2015 entrenamiento, 2016-2019
    validación y 2020-2024 prueba. Solo se rotulan las filas.
    """
    meta = paises.set_index("iso3")
    df = df.rename(columns={"iso3": "codigo_iso3"})
    df["nombre_pais"] = df["codigo_iso3"].map(meta["pais"])
    region_en = df["codigo_iso3"].map(meta["region"])
    ingreso_en = df["codigo_iso3"].map(meta["nivel_ingreso"])
    df["region"] = region_en.map(lambda r: REGIONES[r][0])
    df["grupo_ingreso"] = ingreso_en.map(lambda i: INGRESOS[i][0])
    for original, (_, corto) in REGIONES.items():
        df["region_" + corto] = (region_en == original).astype(int)
    for original, (_, corto) in INGRESOS.items():
        df["ingreso_" + corto] = (ingreso_en == original).astype(int)
    banderas = imputados.rename_axis(["codigo_iso3", "anio"]).add_prefix("bandera_imp_").astype(int)
    df = df.merge(banderas.reset_index(), on=["codigo_iso3", "anio"], how="left")
    df["conjunto"] = "prueba"
    for nombre, (a0, a1) in CORTES.items():
        df.loc[df["anio"].between(a0, a1), "conjunto"] = nombre

    identificacion = ["codigo_iso3", "nombre_pais", "region", "grupo_ingreso", "anio"]
    objetivo = ["etapa", "etapa_num"]
    predictores = ([v + "_ant" for v in PREDICTORES] + ["anio_relativo"]
                   + ["region_" + c for _, c in REGIONES.values()] + ["ingreso_" + c for _, c in INGRESOS.values()])
    banderas_cols = ["bandera_imp_" + v for v in COVARIABLES]
    final = df[identificacion + objetivo + predictores + banderas_cols + ["conjunto"]].reset_index(drop=True)
    final["etapa_num"] = final["etapa_num"].astype(int)
    nota(f"conjunto final: {final.shape[0]} filas x {final.shape[1]} columnas")
    return final, predictores, df


def controles_calidad(final: pd.DataFrame, auditoria: pd.DataFrame, predictores: list,
                      cobertura: pd.Series, calidad: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica los requisitos R1-R10 de la fase 1 sobre el conjunto final.
    Resultado esperado: los 10 controles en 'CUMPLE'. Si alguno falla, el
    programa se detiene ANTES de guardar el archivo (compuerta de calidad).
    """
    numericas = final.select_dtypes("number")
    clases = final["etapa"].value_counts(normalize=True) * 100
    pct_cols = [c for c in predictores if c.endswith("_pct_ant") and not c.startswith("var")]
    prohibidas = FUGA + ETIQUETA_BASE + ["b_promedio_3"]
    fuga = [c for c in final.columns if any(c.startswith(p) for p in prohibidas)]
    recalculada = [asignar_etapa(b, a) for b, a in zip(auditoria["b_promedio_3"], auditoria["electricidad_pct"])]
    cob_final = cobertura.reindex(final["codigo_iso3"].unique()) * 100
    faltas_cov = calidad.set_index("variable").loc[COVARIABLES + ETIQUETA_BASE, "faltantes_pct"]
    n_conj = final["conjunto"].value_counts()
    controles = [
        ("R1", "Países >= 120", final["codigo_iso3"].nunique() >= REQ.min_paises,
         f"{final['codigo_iso3'].nunique()} países"),
        ("R2", "b disponible en >= 85 % de los años", bool((cob_final >= REQ.cobertura_etiqueta * 100).all()),
         f"mínimo {cob_final.min():.1f} % de los años"),
        ("R3", "Variables con <= 25 % de faltantes", bool((faltas_cov <= REQ.max_faltantes_var).all()),
         f"máximo {faltas_cov.max():.1f} % ({faltas_cov.idxmax()})"),
        ("R4", "Sin nulos ni infinitos", final.isna().sum().sum() == 0 and not np.isinf(numericas.values).any(), "0 y 0"),
        ("R5", "Una fila por país-año, sin saltos",
         (not final.duplicated(["codigo_iso3", "anio"]).any())
         and final.groupby("codigo_iso3")["anio"].apply(lambda s: (s.diff().dropna() == 1).all()).all(), "cumple"),
        ("R6", "Porcentajes en [0, 100]", all(final[c].between(0, 100).all() for c in pct_cols), f"{len(pct_cols)} variables"),
        ("R7", "Cada etapa >= 10 % de las filas", clases.min() >= REQ.min_clase, f"mínimo {clases.min():.1f} %"),
        ("R8", "Sin componentes de la etiqueta; predictores del año anterior",
         not fuga and all(c.endswith("_ant") for c in predictores[:len(PREDICTORES)]), "cumple"),
        ("R9", "La etiqueta se reproduce al recalcularla",
         all(r == e for r, e in zip(recalculada, auditoria["etapa"])), "100 % de coincidencia"),
        ("R10", "Tres etapas en cada conjunto",
         all(final.loc[final["conjunto"] == c, "etapa"].nunique() == 3 for c in CORTES),
         " / ".join(str(n_conj.get(c, 0)) for c in CORTES) + " filas"),
    ]
    tabla = pd.DataFrame(controles, columns=["codigo", "requisito", "cumple", "detalle"])
    tabla["cumple"] = tabla["cumple"].map(lambda v: "CUMPLE" if v else "NO CUMPLE")
    tabla.to_csv(CARPETA_REPORTES / "fase3_controles_calidad.csv", index=False)
    for c in tabla.itertuples():
        nota(f"{c.codigo:<4} {c.cumple:<10} {c.requisito} ({c.detalle})")
    if (tabla["cumple"] != "CUMPLE").any():
        sys.exit("Algún requisito de calidad falló: el conjunto NO se guardó.")
    return tabla


def graficos_preparacion(auditoria: pd.DataFrame, final: pd.DataFrame) -> None:
    """Figuras 6 a 8 de la preparación de los datos."""
    # Figura 6: por qué se promedia b en 3 años (ejemplo: Letonia)
    e = auditoria[auditoria["codigo_iso3"] == "LVA"].set_index("anio")
    if not e.empty:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1.2]})
        ax.plot(e.index, e["baja_carbono_elec_pct"], color=GRIS, marker="o", ms=3, label="Valor anual")
        ax.plot(e.index, e["b_promedio_3"], color=COLOR_ETAPA["avanzada"], lw=2, label="Promedio de 3 años")
        ax.axhline(REQ.umbral_avanzada, color=COLOR_ETAPA["inicial"], ls=":", lw=1)
        ax.set(ylabel="Elec. baja en carbono (%)", title="Fig. 6. Letonia: la etapa anual cambia mucho más que la suavizada")
        ax.legend(frameon=False, ncol=2)
        for k, col in enumerate(["etapa_anual", "etapa"]):
            for anio, etapa in e[col].items():
                ax2.barh(1 - k, 1, left=anio - 0.5, color=COLOR_ETAPA[etapa], height=0.8)
        ax2.set_yticks([1, 0], ["Anual", "Suavizada"])
        ax2.grid(False)
        ax2.set_xticks(range(CORTES["entrenamiento"][0], ANIO_FIN + 1, 4))
        ax2.set_xlabel("Año")
        guardar_figura(fig, "fig06_suavizado_etiqueta.png")

    # Figura 7: matriz de transición entre el primer y el último año
    orden = ["inicial", "intermedia", "avanzada"]
    a0, a1 = final["anio"].min(), final["anio"].max()
    tm = pd.crosstab(final[final["anio"] == a0].set_index("codigo_iso3")["etapa"],
                     final[final["anio"] == a1].set_index("codigo_iso3")["etapa"]).reindex(index=orden, columns=orden).fillna(0).astype(int)
    tm.to_csv(CARPETA_REPORTES / f"fase3_transiciones_{a0}_{a1}.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(tm.values, cmap=LinearSegmentedColormap.from_list("t", ["#f8f9fa", COLOR_ETAPA["avanzada"]]))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, tm.values[i, j], ha="center", va="center", fontsize=12,
                    color="white" if tm.values[i, j] > 40 else "black")
    ax.set_xticks(range(3), [o.capitalize() for o in orden])
    ax.set_yticks(range(3), [o.capitalize() for o in orden])
    ax.grid(False)
    ax.set(xlabel=f"Etapa en {a1}", ylabel=f"Etapa en {a0}", title="Fig. 7. Transiciones entre etapas")
    guardar_figura(fig, "fig07_matriz_transicion.png")

    # Figura 8: proporción de etapas por año y conjuntos
    pc = final.groupby("anio")["etapa"].value_counts(normalize=True).unstack().reindex(columns=orden).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    base = np.zeros(len(pc))
    for etapa in orden:
        ax.bar(pc.index, pc[etapa], bottom=base, color=COLOR_ETAPA[etapa], label=etapa.capitalize(), width=0.85)
        base += pc[etapa].values
    for nombre, (x0, x1) in CORTES.items():
        ax.annotate("", xy=(x0 - 0.45, 104), xytext=(x1 + 0.45, 104), arrowprops=dict(arrowstyle="|-|", lw=0.8))
        ax.text((x0 + x1) / 2, 107, {"validacion": "Validación"}.get(nombre, nombre.capitalize()), ha="center", fontsize=8)
    ax.set(ylim=(0, 118), xlabel="Año", ylabel="% de países", title="Fig. 8. Etapas por año y partición temporal")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    guardar_figura(fig, "fig08_etapas_por_anio.png")


def diccionario_datos(final: pd.DataFrame) -> pd.DataFrame:
    """Describe cada columna del conjunto final (tipo, papel y significado)."""
    filas = []
    for col in final.columns:
        base = col.removesuffix("_ant").removesuffix("_log").replace("bandera_imp_", "")
        if col in ("etapa", "etapa_num"):
            papel, texto = "objetivo", "Etapa de transición (0 inicial, 1 intermedia, 2 avanzada)"
        elif col.startswith("bandera_imp_"):
            papel, texto = "bandera", f"1 si {base} fue imputada o corregida en el año t"
        elif col.endswith("_ant"):
            papel = "predictor"
            texto = {"var_pib_pc_pct": "Variación anual del PIB per cápita (%)",
                     "var_urbano_pp": "Cambio anual de la población urbana (p.p.)",
                     "var3_renovable_pp": "Cambio en 3 años de la energía renovable (p.p.)"}.get(base, DESCRIPCION.get(base, base))
            texto += (" (logaritmo)" if "_log" in col else "") + ", año anterior"
        elif col.startswith(("region_", "ingreso_")):
            papel, texto = "predictor", "Indicadora 0/1 de " + col.replace("_", " ")
        elif col == "anio_relativo":
            papel, texto = "predictor", "Años transcurridos desde 2004"
        elif col == "conjunto":
            papel, texto = "partición", "entrenamiento (2004-2015), validacion (2016-2019), prueba (2020-2024)"
        else:
            papel, texto = "identificación", {"codigo_iso3": "Código ISO de 3 letras", "nombre_pais": "Nombre del país",
                                              "region": "Región del Banco Mundial", "grupo_ingreso": "Grupo de ingreso del Banco Mundial",
                                              "anio": "Año"}[col]
        filas.append({"columna": col, "tipo": str(final[col].dtype), "papel": papel, "descripcion": texto})
    dic = pd.DataFrame(filas)
    dic.to_csv(CARPETA_PREPARADOS / "diccionario_datos.csv", index=False)
    return dic


def fase3_preparacion_datos(fase2: dict) -> pd.DataFrame:
    """
    Convierte el panel crudo en el conjunto etiquetado listo para modelar.

    Resultado esperado:
      * datos/preparados/etapas_transicion_preparado.csv: 3 780 filas x 39
        columnas, 180 países, 2004-2024, sin faltantes (49,7 % inicial,
        33,8 % intermedia, 16,5 % avanzada).
      * datos/preparados/etapas_auditoria.csv y diccionario_datos.csv.
      * resultados/reportes/fase3_*.csv, fase3_preparacion.md.
      * resultados/figuras/fig06 a fig08.
    """
    seccion("FASE 3 - PREPARACIÓN DE LOS DATOS")
    bitacora = []
    df = seleccionar(fase2["panel"], fase2["calidad"], fase2["resumen"], bitacora)
    df, excluir_40 = limpiar_rangos(df, bitacora)
    vacios = df.isna()
    df, atipicos, corregidos = filtro_hampel(df, bitacora)
    df, imputados, retirados = imputar(df, vacios, corregidos, excluir_40, fase2["paises"], bitacora)

    atipicos = atipicos[~atipicos["iso3"].isin(retirados)]
    atipicos.to_csv(CARPETA_REPORTES / "fase3_atipicos_hampel.csv", index=False)
    resumen_at = atipicos.groupby(["variable", "decision"]).size().unstack(fill_value=0)
    bitacora.append({"paso": "atípicos", "detalle": "Hampel en países retenidos: "
                     + ", ".join(f"{k} {v}" for k, v in atipicos["decision"].value_counts().items())})

    df = construir_etiqueta(df)
    df, sin_historia = construir_predictores(df)
    bitacora.append({"paso": "transformación", "detalle": f"{sin_historia} filas de 2000-2003 sin historia retiradas"})
    final, predictores, auditoria = integrar_y_particionar(df, fase2["paises"], imputados)
    controles = controles_calidad(final, auditoria, predictores, fase2["resumen"]["cobertura_etiqueta"], fase2["calidad"])

    # Solo si los controles pasan, se guardan los archivos
    final.to_csv(CARPETA_PREPARADOS / "etapas_transicion_preparado.csv", index=False)
    auditoria.to_csv(CARPETA_PREPARADOS / "etapas_auditoria.csv", index=False)
    diccionario_datos(final)
    pd.DataFrame(bitacora).to_csv(CARPETA_REPORTES / "fase3_bitacora.csv", index=False)
    graficos_preparacion(auditoria, final)

    clases = final["etapa"].value_counts()
    cambios = auditoria[auditoria["anio"] > auditoria["anio"].min()]
    md = ["# Fase 3 - Preparación de los datos", "",
          f"* Filas: {final.shape[0]}; columnas: {final.shape[1]}; países: {final['codigo_iso3'].nunique()}; "
          f"años: {final['anio'].min()}-{final['anio'].max()}.",
          "* Etapas: " + ", ".join(f"{k} {v} ({v / len(final) * 100:.1f} %)" for k, v in clases.items()) + ".",
          "* Filas por conjunto: " + ", ".join(f"{k} {v}" for k, v in final["conjunto"].value_counts().items()) + ".",
          f"* Cambios de etapa: {int(cambios['cambio_etapa_anual'].sum())} con el valor anual y "
          f"{int(cambios['cambio_etapa'].sum())} con el promedio de 3 años.",
          "", "## Bitácora", "", tabla_md(pd.DataFrame(bitacora)), "",
          "## Atípicos (filtro de Hampel)", "", tabla_md(resumen_at.reset_index()), "",
          "## Controles de calidad", "", tabla_md(controles), ""]
    (CARPETA_REPORTES / "fase3_preparacion.md").write_text("\n".join(md), encoding="utf-8")
    nota("conjunto guardado en datos/preparados/etapas_transicion_preparado.csv")
    return final


# =============================================================================
# PASO 6. FLUJO PRINCIPAL
# =============================================================================
def principal(argumentos: list | None = None) -> int:
    """Ejecuta las tres fases en orden. Cada fase usa lo que produjo la anterior."""
    lector = argparse.ArgumentParser(description="Grupo 7 - CRISP-ML(Q) fases 1 a 3")
    lector.add_argument("--descargar", action="store_true", help="volver a descargar los datos")
    lector.add_argument("--pais", default="PER", help="código ISO3 del país de la figura 4")
    args = lector.parse_args(argumentos)

    preparar_carpetas()
    fase1_comprension_negocio()
    fase2 = fase2_comprension_datos(args.descargar, args.pais.upper())
    final = fase3_preparacion_datos(fase2)

    seccion("RESUMEN")
    nota(f"conjunto listo para modelar: {final.shape[0]} filas, {final.shape[1]} columnas, "
         f"{final['codigo_iso3'].nunique()} países")
    nota("fase de modelado: NO incluida (según las indicaciones del curso)")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
