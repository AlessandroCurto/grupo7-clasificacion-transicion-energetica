# Fase 2 - Comprensión de los datos

* Países del Banco Mundial: 217 (se excluyeron 78 agregados).
* Países en OWID: 220. En común: 206.
* Panel: 5150 filas x 20 variables (2000-2024).
* Duplicados: BM 0, OWID 0.
* Faltantes promedio por fuente: BM 9.5 %, OWID 5.2 %.
* Población BM vs OWID con diferencia > 5 %: 166 filas (17 países).
* Electricidad con carbón BM vs OWID con diferencia > 5 p.p.: 126 de 4544 filas.
* Filas donde baja en carbono + fósil no suma 100 (±1): 0.
* Países con la variable de la etiqueta en >= 85 % de los años: 200.

## Calidad por variable

| variable | fuente | faltantes_pct | fuera_de_rango | no_positivos |
|---|---|---|---|---|
| pib_pc | BM | 3.7 | 0 | 0 |
| poblacion | BM | 0.0 | 0 | 0 |
| urbano_pct | BM | 0.0 | 0 | 0 |
| industria_pct | BM | 8.1 | 0 | 0 |
| renovable_pct | BM | 11.1 | 0 | 0 |
| electricidad_pct | BM | 1.0 | 0 | 0 |
| co2_pc | BM | 2.4 | 0 | 50 |
| energia_pc_bm | BM | 28.0 | 0 | 0 |
| fosil_pct | BM | 30.6 | 6 | 0 |
| carbon_elec_pct | BM | 9.8 | 0 | 0 |
| baja_carbono_elec_pct | OWID | 2.6 | 0 | 0 |
| renovable_elec_pct | OWID | 2.6 | 0 | 0 |
| fosil_elec_pct | OWID | 2.6 | 0 | 0 |
| hidro_elec_pct | OWID | 4.0 | 0 | 0 |
| carbon_elec_owid_pct | OWID | 3.3 | 0 | 0 |
| intensidad_carbono_elec | OWID | 2.6 | 0 | 22 |
| energia_pc | OWID | 3.9 | 0 | 48 |
| consumo_elec_pc | OWID | 2.5 | 0 | 2 |
| intensidad_energetica | OWID | 27.0 | 0 | 0 |
| poblacion_owid | OWID | 0.6 | 0 | 0 |
