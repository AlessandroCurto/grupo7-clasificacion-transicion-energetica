# Fase 1 - Comprensión del negocio

**Problema.** Los organismos que financian o planifican la transición energética necesitan saber, cada año y con un criterio común, en qué etapa está cada país para priorizar recursos y cooperación.

**Objetivo de negocio.** Disponer cada año de una clasificación comparable de los países según su avance en la transición energética.

**Objetivo de minería de datos.** Preparar un conjunto país-año con la etapa de transición (inicial, intermedia, avanzada) como variable objetivo y predictores medidos el año anterior, apto para entrenar después un clasificador de tres clases.

**Criterios de éxito.** Exactitud balanceada >= 0,75 en los años de prueba (2020-2024). Cada clasificación debe poder explicarse con una regla pública.

## Regla de etiqueta

| Etapa | Condición | Lectura |
|---|---|---|
| Inicial (0) | b < 30 | La generación depende sobre todo de combustibles fósiles |
| Intermedia (1) | b >= 30 (sin cumplir avanzada) | Matriz mixta, o limpia pero con acceso incompleto |
| Avanzada (2) | b >= 60 y acceso >= 90 | Electricidad mayoritariamente limpia y casi universal |

## Requisitos de calidad

| Código | Requisito |
|---|---|
| R1 | Al menos 120 países en el conjunto final |
| R2 | b disponible en >= 85% de los años de cada país |
| R3 | Variables usadas con <= 25 % de faltantes |
| R4 | Sin valores nulos ni infinitos |
| R5 | Una fila por país y año, sin saltos entre años |
| R6 | Porcentajes dentro de [0, 100] |
| R7 | Cada etapa con >= 10 % de las filas |
| R8 | Ninguna columna derivada de la etiqueta; predictores del año anterior |
| R9 | La etiqueta se reproduce exactamente al recalcularla |
| R10 | Las tres etapas presentes en entrenamiento, validación y prueba |
