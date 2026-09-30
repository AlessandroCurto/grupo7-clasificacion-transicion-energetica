# Grupo 7 · Clasificación de países según transición energética

**Curso:** Machine Learning · Práctica 1 · Entregable 3 (guía de código en Python)
**Universidad Nacional de Ingeniería**, Facultad de Ingeniería Industrial y de Sistemas, ciclo 2026-II
**Profesor:** Dr. Ing. Hilario Aradiel Castañeda · **Área:** Energía

| Integrante | Correo |
|---|---|
| Crisostomo Cahuas, Gabriel Andre | gabriel.crisostomo.c@uni.pe |
| Curto Meneses, Alessandro del Piero | alessandro.curto.m@uni.pe |
| Rubio Zeta, Luis Miguel | luis.rubio.z@uni.pe |

---

## 1. Descripción del proyecto

¿Qué países están dejando atrás la electricidad fósil y cuáles no? Este proyecto prepara los datos necesarios para que, en una etapa posterior, un **clasificador supervisado** ubique a cada país y año en una de tres **etapas de la transición energética**:

| Etapa | Condición | Lectura |
|---|---|---|
| 0 · Inicial | b̄ < 30 % | La generación eléctrica depende sobre todo de combustibles fósiles |
| 1 · Intermedia | b̄ ≥ 30 % y no cumple la condición de avanzada | Matriz mixta, o limpia pero con acceso incompleto |
| 2 · Avanzada | b̄ ≥ 60 % **y** acceso a la electricidad ≥ 90 % | Electricidad mayoritariamente limpia y casi universal |

donde **b̄** es el promedio de tres años de la participación de la electricidad baja en carbono (renovable + nuclear) en la generación del país. Las dos condiciones siguen las metas 7.1 (acceso) y 7.2 (energía limpia) del ODS 7.

Se trabaja con la metodología **CRISP-ML(Q)** ([Studer et al., 2021](https://doi.org/10.48550/arXiv.2003.05155)) y, según las indicaciones del curso, **solo hasta la fase 3**. **No se incluye la fase de modelado:** no se entrena ni se ejecuta ningún algoritmo. El producto es un conjunto de datos país-año limpio, etiquetado y verificado, listo para modelar.

**Fuentes de datos** (abiertas, licencia CC BY 4.0):

| Fuente | Qué aporta | Acceso |
|---|---|---|
| Banco Mundial · World Development Indicators | PIB, población, urbanización, industria, energía renovable, acceso a la electricidad, CO₂, uso de energía | API v2 · https://data.worldbank.org |
| Our World in Data · energy-data | Composición de la generación eléctrica (baja en carbono, fósil, hidro, carbón), consumo de energía y de electricidad per cápita | https://github.com/owid/energy-data |

El Banco Mundial no publica la composición de la generación eléctrica, por eso se integra OWID. Una copia de ambas fuentes (2000–2024) está en `datos/crudos/`, así el programa funciona sin Internet.

---

## 2. Fases trabajadas

### Fase 1 · Comprensión del negocio (`fase1_comprension_negocio`)
- **Objetivo de negocio:** que ministerios de energía (en el Perú, el MINEM), bancos de desarrollo e investigadores dispongan cada año de una clasificación comparable de los países según su avance en la transición, para orientar financiamiento y cooperación.
- **Objetivo de minería de datos:** preparar un conjunto país-año con la etapa de transición como variable objetivo y predictores medidos el año anterior, apto para entrenar después un clasificador de tres clases.
- **Criterios de éxito:** exactitud balanceada ≥ 0,75 en los años de prueba para el futuro modelo, y que cada clasificación se pueda explicar con una regla pública.
- **Regla de etiqueta** y **diez requisitos de calidad (R1–R10)**, que se verifican automáticamente al final de la fase 3.

### Fase 2 · Comprensión de los datos (`fase2_comprension_datos`)
- **Recolección:** 10 indicadores del Banco Mundial y 10 columnas de OWID; 206 países presentes en ambas fuentes (se excluyen 78 agregados como "Mundo" o "América Latina").
- **Exploración:** matriz eléctrica mundial, distribución de la electricidad baja en carbono, relación entre ingreso, electricidad limpia y acceso, serie del país de referencia y correlaciones de Spearman.
- **Calidad por fuente:** faltantes por variable, fuente y año; valores imposibles; contraste BM vs. OWID en población y electricidad con carbón; comprobación de que baja en carbono + fósil suma 100 %; cobertura de la variable de la etiqueta por país.

### Fase 3 · Preparación de los datos (`fase3_preparacion_datos`)
| Paso | Función | Qué hace |
|---|---|---|
| Selección | `seleccionar` | Conserva los países con b en ≥ 85 % de los años; descarta variables con > 25 % de faltantes y las de la matriz eléctrica (revelarían la etiqueta) |
| Limpieza | `limpiar_rangos` | Vacía porcentajes fuera de [0, 100] y niveles ≤ 0; marca países con > 40 % de años vacíos |
| Atípicos | `filtro_hampel` | Filtro de Hampel con ventana móvil de 5 años (3 desviaciones robustas y diferencia mínima de 10 p.p. o 20 %); corrige errores y conserva los valores de la etiqueta y los choques de 2009 y 2020 |
| Imputación | `imputar` | Interpolación de huecos de hasta 3 años dentro del país y mediana del grupo de ingreso y año; retira países imposibles de completar |
| Etiqueta | `construir_etiqueta` | Promedio de 3 años de b y regla de etapas |
| Transformación | `construir_predictores` | Logaritmos, variaciones y rezago de un año (sufijo `_ant`) |
| Integración y partición | `integrar_y_particionar` | Región e ingreso en español con indicadoras 0/1, banderas de imputación y conjuntos: entrenamiento 2004–2015, validación 2016–2019, prueba 2020–2024 |
| Control de calidad | `controles_calidad` | Verifica R1–R10; si alguno falla, **no se guarda el archivo** |

---

## 3. Estructura del repositorio

```
grupo7-clasificacion-transicion-energetica/
├── main.py                 # código completo de las fases 1 a 3 (un único archivo)
├── main.ipynb              # el mismo código en Jupyter, con tablas y figuras en pantalla
├── requirements.txt        # librerías necesarias
├── README.md
├── datos/
│   ├── crudos/             # copia local de las fuentes (Banco Mundial y OWID)
│   └── preparados/         # salida de la fase 3
│       ├── etapas_transicion_preparado.csv   # conjunto final listo para modelar
│       ├── etapas_auditoria.csv              # insumos de la etiqueta (para verificar R9)
│       └── diccionario_datos.csv             # descripción de cada columna
└── resultados/
    ├── reportes/           # fase1_*, fase2_*, fase3_* (Markdown, JSON y CSV)
    └── figuras/            # fig01 a fig08
```

---

## 4. Instrucciones de ejecución

### Opción A · Desde la terminal (`main.py`)

1. Instalar **Python 3.10 o superior** (https://www.python.org).
2. Descargar el proyecto: botón verde **Code → Download ZIP** en GitHub (y descomprimir), o bien
   ```bash
   git clone https://github.com/AlessandroCurto/grupo7-clasificacion-transicion-energetica.git
   cd grupo7-clasificacion-transicion-energetica
   ```
3. (Opcional) Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Linux / macOS
   ```
4. Instalar las librerías:
   ```bash
   pip install -r requirements.txt
   ```
5. Ejecutar:
   ```bash
   python main.py                  # usa la copia local de datos/crudos
   python main.py --descargar      # vuelve a descargar los datos (requiere Internet)
   python main.py --pais CHL       # cambia el país de la figura 4 (por defecto PER)
   ```
   Tarda unos segundos. En la consola se ve el avance de cada fase y el resultado de los diez controles.

### Opción B · Con Jupyter (`main.ipynb`) usando Anaconda

1. Abrir **Anaconda Navigator** y lanzar **Jupyter Notebook** (o escribir `jupyter notebook` en *Anaconda Prompt*).
2. En el navegador, entrar a la carpeta del proyecto y abrir `main.ipynb`. El notebook debe estar en la misma carpeta que la carpeta `datos/`.
3. Menú **Kernel → Restart Kernel and Run All Cells** (o ejecutar celda por celda con **Shift + Enter**).
4. Para cambiar las opciones, editar la celda de parámetros del Paso 1: `DESCARGAR = True` para volver a descargar, `PAIS_REF = "CHL"` para cambiar el país de la figura 4.

Anaconda ya trae pandas, numpy, matplotlib y requests; si faltara alguna, ejecutar `pip install -r requirements.txt` en *Anaconda Prompt*.

---

## 5. Resultados esperados

| Fase | Resultado |
|---|---|
| 1 | `resultados/reportes/fase1_negocio.md` y `.json`: objetivos, regla de etiqueta y requisitos R1–R10 |
| 2 | Panel de 5 150 filas (206 países × 25 años) y 20 variables; 3 variables con > 25 % de faltantes (`energia_pc_bm`, `fosil_pct`, `intensidad_energetica`); 200 países con la variable de la etiqueta en ≥ 85 % de los años; figuras 1 a 5 |
| 3 | `datos/preparados/etapas_transicion_preparado.csv`: **3 780 filas × 39 columnas, 180 países, 2004–2024, sin faltantes** |

Distribución de la etiqueta en el conjunto final: **49,7 % inicial, 33,8 % intermedia y 16,5 % avanzada**. Filas por conjunto: 2 160 entrenamiento, 720 validación y 900 prueba. Se retiraron 26 países: 6 por baja cobertura de la etiqueta, 15 por tener más de 40 % de años vacíos en alguna variable y 5 por huecos que no se pudieron llenar. Los diez requisitos R1–R10 terminan en **CUMPLE**.

Columnas del conjunto final (39):

| Grupo | Columnas |
|---|---|
| Identificación (5) | `codigo_iso3`, `nombre_pais`, `region`, `grupo_ingreso`, `anio` |
| Objetivo (2) | `etapa`, `etapa_num` |
| Predictores del año anterior (11) | `pib_pc_log_ant`, `poblacion_log_ant`, `urbano_pct_ant`, `industria_pct_ant`, `renovable_pct_ant`, `co2_pc_log_ant`, `energia_pc_log_ant`, `consumo_elec_pc_log_ant`, `var_pib_pc_pct_ant`, `var_urbano_pp_ant`, `var3_renovable_pp_ant` |
| Tiempo (1) | `anio_relativo` |
| Región (7) e ingreso (4) | `region_*`, `ingreso_*` (indicadoras 0/1) |
| Banderas (8) | `bandera_imp_*` (1 = valor imputado o corregido) |
| Partición (1) | `conjunto` |

> **Nota:** el trabajo llega solo hasta la fase 3 de CRISP-ML(Q). Está prohibido presentar el modelado o las corridas de los modelos.
