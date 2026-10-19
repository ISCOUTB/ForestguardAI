# Reporte ForestGuard-Agent: guaviare, predicción 2025

Generado: 2026-10-18 23:59

## Contexto de la zona

- Pérdida de bosque en 2024: 8,486 ha.
- Promedio anual 2001-2023: 9,157 ha.
- Razón último año / histórico: 0.93.

## Celdas prioritarias (top 20)

Cada celda mide ~1 km × 1 km. El riesgo es la probabilidad estimada de que la celda pierda bosque en 2025.

| # | Celda | Lat | Lon | Riesgo | Nivel | Alertas hist. | Última alerta | Pérdida último año (ha) |
|---|---|---|---|---|---|---|---|---|
| 1 | 9978 | 1.6945 | -72.6045 | 0.96 | alta | 2 | 2024 | 33.1 |
| 2 | 6217 | 1.9915 | -73.4865 | 0.96 | alta | 4 | 2024 | 20.8 |
| 3 | 2916 | 2.2615 | -73.2255 | 0.96 | alta | 4 | 2024 | 23.7 |
| 4 | 3139 | 2.2435 | -73.2165 | 0.96 | alta | 3 | 2024 | 13.5 |
| 5 | 9097 | 1.7665 | -72.5415 | 0.96 | alta | 2 | 2024 | 21.2 |
| 6 | 2917 | 2.2615 | -73.2165 | 0.96 | alta | 4 | 2024 | 23.7 |
| 7 | 8981 | 1.7755 | -72.5865 | 0.96 | alta | 3 | 2024 | 8.4 |
| 8 | 7663 | 1.8745 | -73.4595 | 0.96 | alta | 1 | 2024 | 27.6 |
| 9 | 9867 | 1.7035 | -72.6045 | 0.96 | alta | 3 | 2024 | 6.5 |
| 10 | 7774 | 1.8655 | -73.4595 | 0.96 | alta | 2 | 2024 | 15.8 |
| 11 | 5270 | 2.0725 | -73.0185 | 0.96 | alta | 4 | 2024 | 20.1 |
| 12 | 7664 | 1.8745 | -73.4505 | 0.96 | alta | 2 | 2024 | 19.7 |
| 13 | 2804 | 2.2705 | -73.2345 | 0.96 | alta | 3 | 2024 | 23.9 |
| 14 | 4936 | 2.0995 | -73.0275 | 0.96 | alta | 7 | 2024 | 7.5 |
| 15 | 5336 | 2.0635 | -73.4235 | 0.96 | alta | 3 | 2024 | 31.5 |
| 16 | 2032 | 2.3335 | -73.1895 | 0.96 | alta | 4 | 2024 | 18.0 |
| 17 | 5220 | 2.0725 | -73.4685 | 0.96 | alta | 4 | 2024 | 24.8 |
| 18 | 5106 | 2.0815 | -73.4955 | 0.96 | alta | 4 | 2024 | 13.5 |
| 19 | 9098 | 1.7665 | -72.5325 | 0.96 | alta | 2 | 2024 | 8.8 |
| 20 | 5334 | 2.0635 | -73.4415 | 0.96 | alta | 6 | 2024 | 21.9 |

Niveles: **alta** = riesgo sobre el umbral y la celda ya tuvo alertas antes (reincidente); **media** = riesgo sobre el umbral sin alertas previas; **baja** = riesgo bajo el umbral.

## Desempeño del modelo (años 2023-2024, nunca vistos al entrenar)

| Modelo | PR-AUC | Precisión | Recall |
|---|---|---|---|
| Aleatorio | 0.064 | - | - |
| Persistencia | 0.256 | 0.33 | 0.38 |
| Random Forest | 0.331 | 0.35 | 0.44 |

Inspecciones usadas como retroalimentación en esta corrida: 0.

## Limitaciones

- Los datos de Hansen miden pérdida de cobertura arbórea (incluye incendios y plantaciones), no deforestación ilegal.
- Son datos anuales de 30 m: sirven para priorizar inspecciones, no para alertas en tiempo real.
- El modelo se validó solo en esta zona; no se puede asumir lo mismo para otras regiones.
