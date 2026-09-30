# Fase 3 - Preparación de los datos

* Filas: 3780; columnas: 39; países: 180; años: 2004-2024.
* Etapas: inicial 1880 (49.7 %), intermedia 1278 (33.8 %), avanzada 622 (16.5 %).
* Filas por conjunto: entrenamiento 2160, prueba 900, validacion 720.
* Cambios de etapa: 188 con el valor anual y 112 con el promedio de 3 años.

## Bitácora

| paso | detalle |
|---|---|
| selección | 6 países sin cobertura suficiente de b: CUW, FSM, MNE, MNP, SSD, TUV |
| selección | variables con > 25 % de faltantes descartadas: energia_pc_bm, fosil_pct, intensidad_energetica |
| selección | variables de la matriz eléctrica excluidas por fuga: renovable_elec_pct, fosil_elec_pct, hidro_elec_pct, carbon_elec_owid_pct, intensidad_carbono_elec, carbon_elec_pct |
| limpieza | 39 valores imposibles convertidos en vacío |
| limpieza | 15 países con > 40 % de años vacíos: ARM, ASM, DJI, ERI, GIB, GUM, NCL, NRU, PRK, PSE, SOM, SRB, VEN, VGB, VIR |
| imputación | 823 valores por interpolación dentro del país |
| imputación | 299 valores por mediana del grupo de ingreso y año |
| imputación | 5 países con huecos que no se pudieron llenar: CYM, FRO, LBR, TCA, YEM |
| atípicos | Hampel en países retenidos: corregido 49, choque 24, etiqueta 18 |
| transformación | 720 filas de 2000-2003 sin historia retiradas |

## Atípicos (filtro de Hampel)

| variable | choque | corregido | etiqueta |
|---|---|---|---|
| baja_carbono_elec_pct | 0 | 0 | 18 |
| co2_pc | 9 | 10 | 0 |
| consumo_elec_pc | 3 | 24 | 0 |
| electricidad_pct | 0 | 3 | 0 |
| energia_pc | 2 | 10 | 0 |
| industria_pct | 3 | 1 | 0 |
| pib_pc | 7 | 1 | 0 |

## Controles de calidad

| codigo | requisito | cumple | detalle |
|---|---|---|---|
| R1 | Países >= 120 | CUMPLE | 180 países |
| R2 | b disponible en >= 85 % de los años | CUMPLE | mínimo 88.0 % de los años |
| R3 | Variables con <= 25 % de faltantes | CUMPLE | máximo 11.1 % (renovable_pct) |
| R4 | Sin nulos ni infinitos | CUMPLE | 0 y 0 |
| R5 | Una fila por país-año, sin saltos | CUMPLE | cumple |
| R6 | Porcentajes en [0, 100] | CUMPLE | 3 variables |
| R7 | Cada etapa >= 10 % de las filas | CUMPLE | mínimo 16.5 % |
| R8 | Sin componentes de la etiqueta; predictores del año anterior | CUMPLE | cumple |
| R9 | La etiqueta se reproduce al recalcularla | CUMPLE | 100 % de coincidencia |
| R10 | Tres etapas en cada conjunto | CUMPLE | 2160 / 720 / 900 filas |
