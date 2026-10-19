"""Construcción de celdas, features y etiquetas a partir de los ráster.

Uso rápido (desde la raíz del repo):

    from src.forestguard.features import construir_panel

    panel, meta_celdas = construir_panel("guaviare")

El panel tiene una fila por celda de ~1 km y año de observación T. Las features usan
solo información hasta T y la etiqueta `alerta` corresponde al año T + 1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROCESSED_DIR = Path("data/processed")

BOSQUE_MIN = 30    # % mínimo de cobertura para llamar "bosque" a un píxel
CELDA_PX = 36      # 36 píxeles ≈ 1 km por lado (36 × ~28 m)
UMBRAL_HA = 2.0    # alerta si la celda pierde >= 2 ha de bosque en el año siguiente

CAPAS = ("treecover2000", "lossyear", "datamask")
ANIO_ULTIMO = 24   # lossyear llega hasta 24 (año 2024)

# Columnas que usa el modelo. lat, lon, fila y col solo sirven para dibujar mapas.
FEATURES = [
    "pct_bosque_2000",
    "pct_bosque_restante",
    "perdida_acum_pct",
    "perdida_ha_ultimo",
    "perdida_ha_ult3",
    "perdida_vecinas_ha_ultimo",
    "perdida_vecinas_ha_ult3",
]

# División temporal, según el año que se quiere predecir
ANIO_TRAIN_MAX = 2020
ANIO_VAL_MAX = 2022


def leer_capas(zona, procesado=PROCESSED_DIR):
    """Lee las 3 capas recortadas de data/processed/ y calcula las hectáreas por píxel."""
    procesado = Path(procesado)
    capas = {}
    meta = None
    for capa in CAPAS:
        ruta = procesado / f"{zona}_{capa}.tif"
        if not ruta.exists():
            raise FileNotFoundError(f"No encuentro {ruta}. ¿Ejecutaste descargar_tesela() y recortar()?")
        with rasterio.open(ruta) as src:
            capas[capa] = src.read(1)
            if meta is None:
                meta = {"bounds": src.bounds, "res": src.res}

    tc, ly, dm = capas["treecover2000"], capas["lossyear"], capas["datamask"]
    if not (tc.shape == ly.shape == dm.shape):
        raise ValueError("Las tres capas deben tener la misma forma")

    lat_centro = (meta["bounds"].top + meta["bounds"].bottom) / 2
    ancho_m = meta["res"][0] * 111_320 * np.cos(np.radians(lat_centro))
    alto_m = meta["res"][1] * 111_320
    meta["ha_por_pixel"] = float(ancho_m * alto_m / 10_000)
    return tc, ly, dm, meta


def sumar_por_celda(mascara, n_f, n_c, celda_px):
    """Cuenta los píxeles True dentro de cada celda -> matriz (n_f, n_c)."""
    return mascara.reshape(n_f, celda_px, n_c, celda_px).sum(axis=(1, 3))


def media_vecinas(mapa):
    """Promedio de las 8 celdas vecinas (fuera del borde cuenta como 0)."""
    p = np.pad(mapa.astype(float), 1)
    suma = sum(
        p[1 + di:1 + di + mapa.shape[0], 1 + dj:1 + dj + mapa.shape[1]]
        for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)
    )
    return suma / 8


def asignar_conjunto(anio_pred):
    """train / val / test según el año que se quiere predecir."""
    if anio_pred <= ANIO_TRAIN_MAX:
        return "train"
    if anio_pred <= ANIO_VAL_MAX:
        return "val"
    return "test"


def calcular_cubos(zona, bosque_min=BOSQUE_MIN, celda_px=CELDA_PX, procesado=PROCESSED_DIR):
    """Agrupa los ráster en celdas.

    Devuelve un diccionario con:
      perdida[a, i, j]  píxeles de bosque perdidos en la celda (i, j) en el año 2000 + a
      bosque_2000[i, j] píxeles de bosque en 2000
      acum[a, i, j]     píxeles perdidos desde 2001 hasta 2000 + a
      y la geometría de la cuadrícula (centros de celda, tamaño en grados, ha por píxel).
    """
    tc, ly, dm, meta = leer_capas(zona, procesado)

    n_f = tc.shape[0] // celda_px
    n_c = tc.shape[1] // celda_px
    alto, ancho = n_f * celda_px, n_c * celda_px
    tc, ly, dm = tc[:alto, :ancho], ly[:alto, :ancho], dm[:alto, :ancho]

    bosque = (tc >= bosque_min) & (dm == 1)
    bosque_2000 = sumar_por_celda(bosque, n_f, n_c, celda_px)

    perdida = np.zeros((ANIO_ULTIMO + 1, n_f, n_c), dtype=np.int32)   # el índice 0 no se usa
    for a in range(1, ANIO_ULTIMO + 1):
        perdida[a] = sumar_por_celda(bosque & (ly == a), n_f, n_c, celda_px)

    if perdida.sum() != (bosque & (ly > 0)).sum():
        raise ValueError("El agrupado por celdas no coincide con el total del ráster")

    b = meta["bounds"]
    lado_lon = celda_px * meta["res"][0]
    lado_lat = celda_px * meta["res"][1]
    return {
        "perdida": perdida,
        "bosque_2000": bosque_2000,
        "acum": np.cumsum(perdida, axis=0),
        "ha_por_pixel": meta["ha_por_pixel"],
        "celda_px": celda_px,
        "lado_lon": lado_lon,
        "lado_lat": lado_lat,
        "lon_centro": b.left + (np.arange(n_c) + 0.5) * lado_lon,
        "lat_centro": b.top - (np.arange(n_f) + 0.5) * lado_lat,
    }


def tabla_features(cubos, T, umbral_ha=UMBRAL_HA):
    """Tabla de features observando hasta el año 2000 + T.

    Si T < 24 incluye también `perdida_ha_siguiente` y la etiqueta `alerta` (año T + 1).
    Con T = 24 no hay año siguiente en los datos: es la tabla para predecir 2025.
    """
    if not 3 <= T <= ANIO_ULTIMO:
        raise ValueError(f"T debe estar entre 3 y {ANIO_ULTIMO}")

    perdida, bosque_2000, acum = cubos["perdida"], cubos["bosque_2000"], cubos["acum"]
    ha = cubos["ha_por_pixel"]
    n_f, n_c = bosque_2000.shape
    pix_celda = cubos["celda_px"] ** 2

    filas_idx, cols_idx = np.meshgrid(np.arange(n_f), np.arange(n_c), indexing="ij")
    filas_idx, cols_idx = filas_idx.ravel(), cols_idx.ravel()

    p_ultimo = perdida[T]
    p_ult3 = perdida[T - 2:T + 1].sum(axis=0)
    restante = bosque_2000 - acum[T]
    acum_pct = np.divide(acum[T], bosque_2000, out=np.zeros(bosque_2000.shape), where=bosque_2000 > 0) * 100

    tabla = pd.DataFrame({
        "celda_id": filas_idx * n_c + cols_idx,
        "fila": filas_idx,
        "col": cols_idx,
        "lat": cubos["lat_centro"][filas_idx],
        "lon": cubos["lon_centro"][cols_idx],
        "anio_obs": 2000 + T,
        "anio_pred": 2000 + T + 1,
        "pct_bosque_2000": bosque_2000.ravel() / pix_celda * 100,
        "pct_bosque_restante": restante.ravel() / pix_celda * 100,
        "perdida_acum_pct": acum_pct.ravel(),
        "perdida_ha_ultimo": p_ultimo.ravel() * ha,
        "perdida_ha_ult3": p_ult3.ravel() * ha,
        "perdida_vecinas_ha_ultimo": media_vecinas(p_ultimo).ravel() * ha,
        "perdida_vecinas_ha_ult3": media_vecinas(p_ult3).ravel() * ha,
    })

    if T < ANIO_ULTIMO:
        tabla["perdida_ha_siguiente"] = perdida[T + 1].ravel() * ha
        tabla["alerta"] = (tabla["perdida_ha_siguiente"] >= umbral_ha).astype(int)

    # Solo celdas que todavía tienen bosque que perder (>= 1 % de la celda)
    return tabla[tabla["pct_bosque_restante"] >= 1]


def construir_panel(zona, bosque_min=BOSQUE_MIN, celda_px=CELDA_PX, umbral_ha=UMBRAL_HA,
                    procesado=PROCESSED_DIR, guardar=True):
    """Construye el panel completo (celdas × años 2003-2023) con etiqueta y conjunto.

    Si guardar=True escribe panel_<zona>.csv y meta_celdas_<zona>.json en data/processed/.
    Devuelve (panel, meta_celdas).
    """
    procesado = Path(procesado)
    cubos = calcular_cubos(zona, bosque_min, celda_px, procesado)

    tablas = [tabla_features(cubos, T, umbral_ha) for T in range(3, ANIO_ULTIMO)]
    panel = pd.concat(tablas, ignore_index=True)
    panel["conjunto"] = panel["anio_pred"].map(asignar_conjunto)

    if not (panel["anio_obs"] < panel["anio_pred"]).all():
        raise ValueError("Fuga de datos: una feature usa un año posterior al observado")

    meta_celdas = {
        "zona": zona,
        "celda_px": celda_px,
        "lado_lon": cubos["lado_lon"],
        "lado_lat": cubos["lado_lat"],
        "bosque_min": bosque_min,
        "umbral_ha": umbral_ha,
        "ha_por_pixel": cubos["ha_por_pixel"],
    }

    if guardar:
        procesado.mkdir(parents=True, exist_ok=True)
        panel.to_csv(procesado / f"panel_{zona}.csv", index=False)
        (procesado / f"meta_celdas_{zona}.json").write_text(json.dumps(meta_celdas, indent=2))

    return panel, meta_celdas


def cargar_panel(zona, procesado=PROCESSED_DIR):
    """Lee el panel y la meta guardados por construir_panel()."""
    procesado = Path(procesado)
    panel = pd.read_csv(procesado / f"panel_{zona}.csv")
    meta_celdas = json.loads((procesado / f"meta_celdas_{zona}.json").read_text())
    return panel, meta_celdas


def features_ultimo_anio(zona, bosque_min=BOSQUE_MIN, celda_px=CELDA_PX, umbral_ha=UMBRAL_HA,
                         procesado=PROCESSED_DIR):
    """Features observando hasta 2024, sin etiqueta: es lo que el agente usa para predecir 2025."""
    cubos = calcular_cubos(zona, bosque_min, celda_px, procesado)
    return tabla_features(cubos, ANIO_ULTIMO, umbral_ha).reset_index(drop=True)
