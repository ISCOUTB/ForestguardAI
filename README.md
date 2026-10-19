# ForestGuard-Agent

Agente inteligente que monitorea la pérdida de cobertura forestal en una región piloto de
Guaviare (Colombia) a partir de los datos **Hansen Global Forest Change (GFC-2024-v1.12)**.
Divide la región en celdas de ~1 km, estima para cada una el **riesgo de perder bosque el año
siguiente**, prioriza las celdas que la autoridad ambiental debería inspeccionar y genera
reportes geolocalizados (CSV, GeoJSON, mapa interactivo y resumen). Aprende de los resultados de
las inspecciones para mejorar sus predicciones.

Proyecto del curso de Inteligencia Artificial — Universidad Tecnológica de Bolívar.

---

## 1. Requisitos

- Python 3.11 o superior (probado con Python 3.13 y 3.14).
- Unos 2 GB libres en disco para las teselas de Hansen.
- Conexión a internet para descargar los datos la primera vez.

## 2. Instalación

```bash
git clone <URL-del-repositorio>
cd forestguard-agent

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-notebooks.txt   # solo si vas a abrir los notebooks
```

Todos los comandos de este README se ejecutan **desde la raíz del repositorio**.

## 3. Datos

Fuente: https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/

| Capa | Contenido |
|---|---|
| `treecover2000` | % de cobertura arbórea en el año 2000 (0–100) |
| `lossyear` | 0 = sin pérdida; 1–24 = año de pérdida (2001–2024) |
| `datamask` | 0 = sin datos, 1 = tierra, 2 = agua |

Descargar la tesela que contiene la región y recortar la zona piloto (una sola vez):

```bash
python -c "from src.forestguard.data import descargar_tesela, recortar; descargar_tesela('10N_080W'); recortar('10N_080W', bbox=(-73.5, 1.5, -72.5, 2.5), zona='guaviare')"
```

- Las teselas completas quedan en `data/raw/` y los recortes en `data/processed/guaviare_<capa>.tif`.
- `bbox = (lon_min, lat_min, lon_max, lat_max)` en grados (EPSG:4326).
- Las teselas completas **no se suben a git** (ver `.gitignore`). Los tres recortes de la zona piloto
  (`data/processed/guaviare_*.tif`) sí están en el repositorio, porque los usa la aplicación desplegada;
  con ellos se puede ejecutar el agente sin descargar nada.

## 4. Ejecutar el agente

```bash
python scripts/run_agent.py                   # una corrida completa
python scripts/run_agent.py --zona guaviare --k 20
```

La primera corrida construye la tabla de celdas (`data/processed/panel_guaviare.csv`); las
siguientes la reutilizan. Al terminar, la consola muestra los archivos generados y el tiempo de
ejecución.

### Ciclo del agente

Implementado en `src/forestguard/agent.py`, método `ForestGuardAgent.ejecutar()`:

| Paso | Método | Qué hace |
|---|---|---|
| Recordar | `recordar()` | Lee el historial de corridas y el feedback de inspecciones (`memory/`) |
| Percibir | `percibir()` | Lee los ráster y calcula las variables de cada celda hasta el último año disponible |
| Aprender | `aprender()` | Evalúa el modelo con división temporal, elige el umbral de riesgo en validación y entrena el modelo final sumando las inspecciones confirmadas |
| Razonar | `razonar()` | Estima el riesgo de cada celda y lo contrasta con sus alertas históricas |
| Decidir | `decidir()` | Prioriza las K celdas de mayor riesgo y les asigna nivel alta / media / baja |
| Actuar | `actuar()` | Genera los reportes geolocalizados en `outputs/` |
| Memorizar | `memorizar()` | Guarda la corrida en `memory/historial.csv` y el modelo entrenado |

### Ciclo de retroalimentación (aprendizaje)

1. Ejecutar el agente: genera `outputs/plantilla_feedback_guaviare_<año>.csv`.
2. Anotar los resultados de inspección en `memory/feedback.csv` (`confirmado`: 1 = sí hubo pérdida, 0 = no). Dos formas de obtenerlos sin inspecciones reales:
   - **Datos reales de 2025** (Hansen v1.13):
     ```bash
     python scripts/verificar_con_v113.py --zona guaviare --tesela 10N_080W --guardar-feedback
     ```
   - **Simulados**, solo para demostrar el mecanismo:
     ```bash
     python scripts/run_agent.py --simular-feedback
     ```
3. Volver a ejecutar `python scripts/run_agent.py`: el modelo se reentrena incluyendo esas inspecciones.

## 5. Aplicación web (evidencia de funcionamiento)

**Enlace público:** <(https://github.com/ISCOUTB/ForestguardAI)> 

La aplicación (`streamlit_app.py`) ejecuta el agente real del repositorio. Cada corrida empieza desde cero en
una carpeta temporal que solo contiene los tres rásteres recortados. Tiene tres pestañas:

| Pestaña | Qué permite |
|---|---|
| 1 · Ejecutar el agente | Escenario típico: ciclo completo en vivo, celdas priorizadas, mapa, evaluación del modelo, registro con marca de tiempo y descarga de reportes. Botón para simular inspecciones y reentrenar (ciclo de aprendizaje) |
| 2 · Escenario de falla | Inyecta una entrada inválida (capa faltante, tamaños distintos, valores fuera de rango, archivo corrupto, zona sin datos o 40 % sin dato) y ejecuta el agente: debe detenerse con un error controlado y sin reportes |
| 3 · Pruebas de desempeño | Tabla oficial de resultados, ejecución en vivo de un escenario de backtest (con la misma semilla reproduce la bitácora) y la suite abreviada de pruebas |

La pestaña **Inicio** incluye las instrucciones y los casos de prueba sugeridos.

Para ejecutarla en tu computador:

```bash
streamlit run streamlit_app.py
```

## 6. Pruebas de desempeño (Fase 3)

```bash
python scripts/pruebas_desempeno.py --zona guaviare                   # completas: 5 repeticiones
python scripts/pruebas_desempeno.py --zona guaviare --repeticiones 1 --sin-agente   # prueba rápida
```

- Los umbrales de aceptación están fijados al inicio del script (`REGLAS FIJADAS`) y la bitácora
  guarda un hash del script en cada ejecución.
- Escenarios: 6 normales (predecir 2019–2024), 3 límite, 4 de punta a punta del agente, 5 fallas
  esperadas y 1 entorno adverso.
- Resultados en `outputs/pruebas/`:
  - `registro_pruebas_guaviare.txt` — bitácora acumulada con la tabla de resultados.
  - `corridas_guaviare_<fecha>.csv` — una fila por corrida.
  - `evidencia_<fecha>/` — reportes reales generados por el agente durante las pruebas.

## 7. Salidas

| Archivo | Contenido |
|---|---|
| `outputs/prioridades_<zona>_<año>.csv` | Celdas priorizadas: ranking, coordenadas, riesgo, nivel, historial |
| `outputs/prioridades_<zona>_<año>.geojson` | Las mismas celdas como polígonos (QGIS, geojson.io) |
| `outputs/mapa_prioridades_<zona>_<año>.html` | Mapa interactivo con capa satelital |
| `outputs/resumen_<zona>_<año>.md` | Resumen para quien decide dónde inspeccionar |
| `outputs/plantilla_feedback_<zona>_<año>.csv` | Plantilla para anotar resultados de inspección |
| `memory/historial.csv`, `memory/feedback.csv` | Memoria del agente |
| `memory/modelo_<zona>.joblib` | Modelo entrenado y su umbral |
| `logs/ejecucion_<zona>.jsonl` | **Registro de ejecución con marca de tiempo** (ver abajo) |

### Registro de ejecución

Cada corrida escribe en `logs/ejecucion_<zona>.jsonl` un evento por paso, con fecha y hora:

```json
{"ts": "2026-10-19T10:15:02.481", "ejecucion": "20261019_101458_312", "fase": "decision",
 "evento": "celdas_priorizadas", "datos": {"k": 20, "niveles": {"alta": 3, "media": 17}, "celdas": [...]}}
```

Fases: `sistema` (inicio, fin o interrupción del ciclo), `memoria`, `percepcion`, `aprendizaje`,
`decision` y `accion`. Si el ciclo falla, queda un evento `ciclo_interrumpido` con el error.

## 8. Estructura

```
forestguard-agent/
├── README.md
├── requirements.txt
├── requirements-notebooks.txt
├── streamlit_app.py    # aplicación web desplegada
├── data/
│   ├── raw/            # teselas descargadas (no se versionan)
│   └── processed/      # recortes de la zona (versionados) y tabla de celdas
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_construccion_features.ipynb
│   └── 03_modelo_y_evaluacion.ipynb
├── src/forestguard/
│   ├── data.py         # descarga y recorte
│   ├── features.py     # celdas, variables y etiquetas
│   ├── model.py        # entrenar, elegir umbral y evaluar
│   ├── agent.py        # ciclo del agente
│   ├── registro.py     # log de ejecución con marca de tiempo
│   └── report.py       # CSV, GeoJSON, mapa HTML y resumen
├── scripts/
│   ├── run_agent.py              # ejecuta el agente
│   ├── pruebas_desempeno.py      # pruebas de la Fase 3
│   └── verificar_con_v113.py     # validación con los datos reales de 2025
├── memory/             # historial, feedback y modelo (lo crea el agente)
├── logs/               # registro de ejecución (lo crea el agente)
└── outputs/            # reportes, mapas y resultados de pruebas
```

## 9. Limitaciones

- Hansen mide pérdida de cobertura arbórea (incluye incendios y plantaciones), no deforestación ilegal.
- Datos anuales de 30 m: sirven para priorizar inspecciones, no para alertas en tiempo real.
- El modelo se validó en una sola zona de Guaviare; no se puede asumir el mismo desempeño en otras regiones.
- Las inspecciones de campo se reemplazan por datos de Hansen v1.13 o por feedback simulado.
