"""Generación de reportes geolocalizados: CSV, GeoJSON, mapa HTML y resumen en texto.

Todas las funciones reciben el DataFrame de prioridades que produce el agente
(una fila por celda, con lat, lon, riesgo, nivel, etc.) y la meta de las celdas
(`meta_celdas`, con el tamaño de cada celda en grados).
"""
import json
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd

COLORES_NIVEL = {"alta": "red", "media": "orange", "baja": "blue"}

COLUMNAS_CSV = [
    "ranking", "celda_id", "lat", "lon", "riesgo", "nivel",
    "n_alertas_hist", "ultima_alerta", "perdida_ha_ultimo", "perdida_ha_ult3",
    "pct_bosque_restante", "anio_obs", "anio_pred",
]


def _limites(fila, meta_celdas):
    """Esquinas (sur, oeste, norte, este) de la celda."""
    dlat, dlon = meta_celdas["lado_lat"], meta_celdas["lado_lon"]
    return (
        fila["lat"] - dlat / 2,
        fila["lon"] - dlon / 2,
        fila["lat"] + dlat / 2,
        fila["lon"] + dlon / 2,
    )


def guardar_csv(prioridades, ruta):
    """Tabla de celdas prioritarias, lista para abrir en Excel."""
    columnas = [c for c in COLUMNAS_CSV if c in prioridades.columns]
    prioridades[columnas].round(4).to_csv(ruta, index=False)


def guardar_geojson(prioridades, meta_celdas, ruta):
    """Cada celda prioritaria como polígono, para abrir en QGIS, geojson.io, etc."""
    features = []
    for _, f in prioridades.iterrows():
        s, w, n, e = _limites(f, meta_celdas)
        ultima = None if pd.isna(f["ultima_alerta"]) else int(f["ultima_alerta"])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
            },
            "properties": {
                "celda_id": int(f["celda_id"]),
                "ranking": int(f["ranking"]),
                "riesgo": round(float(f["riesgo"]), 4),
                "nivel": f["nivel"],
                "n_alertas_hist": int(f["n_alertas_hist"]),
                "ultima_alerta": ultima,
                "perdida_ha_ultimo": round(float(f["perdida_ha_ultimo"]), 2),
            },
        })
    coleccion = {"type": "FeatureCollection", "features": features}
    Path(ruta).write_text(json.dumps(coleccion, indent=2), encoding="utf-8")


def crear_mapa(prioridades, meta_celdas, ruta):
    """Mapa HTML interactivo: rojo = nivel alto, naranja = medio, azul = bajo."""
    mapa = folium.Map(
        location=[prioridades["lat"].mean(), prioridades["lon"].mean()],
        zoom_start=10,
        tiles="OpenStreetMap",
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite",
    ).add_to(mapa)

    for _, f in prioridades.iterrows():
        s, w, n, e = _limites(f, meta_celdas)
        folium.Rectangle(
            bounds=[[s, w], [n, e]],
            color=COLORES_NIVEL.get(f["nivel"], "blue"),
            fill=True,
            fill_opacity=0.35,
            popup=(
                f"#{int(f['ranking'])} · celda {int(f['celda_id'])}<br>"
                f"Riesgo: {f['riesgo']:.2f} ({f['nivel']})<br>"
                f"Alertas históricas: {int(f['n_alertas_hist'])}<br>"
                f"Pérdida último año: {f['perdida_ha_ultimo']:.1f} ha"
            ),
        ).add_to(mapa)

    folium.LayerControl().add_to(mapa)
    mapa.save(str(ruta))


def guardar_plantilla_feedback(prioridades, zona, ruta):
    """Plantilla para anotar los resultados de las inspecciones (confirmado: 1 = sí hubo pérdida, 0 = no)."""
    plantilla = pd.DataFrame({
        "zona": zona,
        "celda_id": prioridades["celda_id"],
        "anio_pred": prioridades["anio_pred"],
        "lat": prioridades["lat"].round(4),
        "lon": prioridades["lon"].round(4),
        "confirmado": "",
    })
    plantilla.to_csv(ruta, index=False)


def escribir_resumen(ruta, zona, prioridades, resumen_zona, resultados, anio_pred, n_feedback):
    """Resumen en texto (Markdown) para quien tenga que decidir dónde inspeccionar."""
    lineas = [
        f"# Reporte ForestGuard-Agent: {zona}, predicción {anio_pred}",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Contexto de la zona",
        "",
        f"- Pérdida de bosque en {resumen_zona['anio']}: {resumen_zona['perdida_ha']:,.0f} ha.",
        f"- Promedio anual 2001-{resumen_zona['anio'] - 1}: {resumen_zona['promedio_historico_ha']:,.0f} ha.",
        f"- Razón último año / histórico: {resumen_zona['razon']:.2f}.",
        "",
        f"## Celdas prioritarias (top {len(prioridades)})",
        "",
        "Cada celda mide ~1 km × 1 km. El riesgo es la probabilidad estimada de que la celda "
        f"pierda bosque en {anio_pred}.",
        "",
        "| # | Celda | Lat | Lon | Riesgo | Nivel | Alertas hist. | Última alerta | Pérdida último año (ha) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in prioridades.iterrows():
        ultima = "-" if pd.isna(f["ultima_alerta"]) else str(int(f["ultima_alerta"]))
        lineas.append(
            f"| {int(f['ranking'])} | {int(f['celda_id'])} | {f['lat']:.4f} | {f['lon']:.4f} | "
            f"{f['riesgo']:.2f} | {f['nivel']} | {int(f['n_alertas_hist'])} | {ultima} | "
            f"{f['perdida_ha_ultimo']:.1f} |"
        )

    lineas += [
        "",
        "Niveles: **alta** = riesgo sobre el umbral y la celda ya tuvo alertas antes (reincidente); "
        "**media** = riesgo sobre el umbral sin alertas previas; **baja** = riesgo bajo el umbral.",
        "",
        "## Desempeño del modelo (años 2023-2024, nunca vistos al entrenar)",
        "",
        "| Modelo | PR-AUC | Precisión | Recall |",
        "|---|---|---|---|",
    ]
    prueba = resultados[resultados["conjunto"] == "prueba"]
    for _, f in prueba.iterrows():
        precision = "-" if pd.isna(f["precisión"]) else f"{f['precisión']:.2f}"
        recall = "-" if pd.isna(f["recall"]) else f"{f['recall']:.2f}"
        lineas.append(f"| {f['modelo']} | {f['PR-AUC']:.3f} | {precision} | {recall} |")

    lineas += [
        "",
        f"Inspecciones usadas como retroalimentación en esta corrida: {n_feedback}.",
        "",
        "## Limitaciones",
        "",
        "- Los datos de Hansen miden pérdida de cobertura arbórea (incluye incendios y plantaciones), "
        "no deforestación ilegal.",
        "- Son datos anuales de 30 m: sirven para priorizar inspecciones, no para alertas en tiempo real.",
        "- El modelo se validó solo en esta zona; no se puede asumir lo mismo para otras regiones.",
        "",
    ]
    Path(ruta).write_text("\n".join(lineas), encoding="utf-8")
