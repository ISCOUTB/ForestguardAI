"""Descarga de teselas Hansen y recorte de la zona de interés.

Uso rápido (desde la raíz del repo):

    from src.forestguard.data import descargar_tesela, recortar

    descargar_tesela("10N_080W")
    recortar("10N_080W", bbox=(-73.5, 1.5, -72.5, 2.5), zona="guaviare")

bbox = (lon_min, lat_min, lon_max, lat_max), en grados.

Para otra versión de Hansen (por ejemplo la v1.13, que incluye 2025) se pasa
`version="GFC-2025-v1.13"`. Por defecto se usa la v1.12, con la que se entrena.
"""
from pathlib import Path

import rasterio
import requests
from rasterio.windows import Window, from_bounds

BASE = "https://storage.googleapis.com/earthenginepartners-hansen"
VERSION = "GFC-2024-v1.12"
BASE_URL = f"{BASE}/{VERSION}"
CAPAS = ("treecover2000", "lossyear", "datamask")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Teselas de 10x10 grados que cubren Colombia continental.
# El nombre es la esquina superior izquierda: latitud (2 dígitos + N/S) y longitud (3 dígitos + E/W).
TESELAS_COLOMBIA = ("20N_080W", "10N_080W", "10N_070W", "00N_080W", "00N_070W")


def nombre_archivo(capa, tesela, version=VERSION):
    return f"Hansen_{version}_{capa}_{tesela}.tif"


def url_capa(capa, tesela, version=VERSION):
    return f"{BASE}/{version}/{nombre_archivo(capa, tesela, version)}"


def tamano_remoto_mb(capa, tesela, version=VERSION):
    """Pregunta cuánto pesa el archivo SIN descargarlo."""
    r = requests.head(url_capa(capa, tesela, version), allow_redirects=True, timeout=30)
    r.raise_for_status()
    return int(r.headers["Content-Length"]) / 1e6


def descargar_tesela(tesela, capas=CAPAS, destino=RAW_DIR, version=VERSION):
    """Descarga las capas de una tesela a data/raw/. Si ya existen, no las repite."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for capa in capas:
        ruta = destino / nombre_archivo(capa, tesela, version)
        rutas[capa] = ruta
        if ruta.exists():
            print(f"[ya existe] {ruta.name}")
            continue
        parcial = ruta.with_suffix(".part")  # evita dejar un .tif a medias si se corta
        print(f"[descargando] {ruta.name}")
        with requests.get(url_capa(capa, tesela, version), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(parcial, "wb") as f:
                for trozo in r.iter_content(chunk_size=1024 * 1024):
                    f.write(trozo)
        parcial.rename(ruta)
    return rutas


def recortar(tesela, bbox, zona, origen=RAW_DIR, destino=PROCESSED_DIR):
    """Recorta las 3 capas de una tesela a un bbox y las guarda en data/processed/."""
    origen, destino = Path(origen), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    salidas = {}
    for capa in CAPAS:
        with rasterio.open(origen / nombre_archivo(capa, tesela)) as src:
            ventana = from_bounds(*bbox, transform=src.transform)
            ventana = ventana.round_offsets().round_lengths()
            datos = src.read(1, window=ventana)
            perfil = src.profile.copy()
            perfil.update(
                height=datos.shape[0],
                width=datos.shape[1],
                transform=src.window_transform(ventana),
                tiled=False,
                compress="lzw",
            )
            perfil.pop("blockxsize", None)
            perfil.pop("blockysize", None)
        ruta_salida = destino / f"{zona}_{capa}.tif"
        with rasterio.open(ruta_salida, "w", **perfil) as dst:
            dst.write(datos, 1)
        salidas[capa] = ruta_salida
        print(f"[recortado] {ruta_salida.name} {datos.shape}")
    return salidas


def leer_alineado(ruta_origen, ruta_referencia):
    """Lee de `ruta_origen` exactamente los píxeles que cubre `ruta_referencia`.

    Sirve para sacar de otra tesela (por ejemplo la v1.13) la misma zona que ya recortaste,
    píxel por píxel, sin volver a escribir el bbox. Falla si las rejillas no coinciden.
    """
    with rasterio.open(ruta_referencia) as ref:
        alto, ancho, t_ref = ref.height, ref.width, ref.transform

    with rasterio.open(ruta_origen) as src:
        t = src.transform
        if abs(t.a - t_ref.a) > 1e-9 or abs(t.e - t_ref.e) > 1e-9:
            raise ValueError("Las dos capas no tienen el mismo tamaño de píxel.")

        col = (t_ref.c - t.c) / t.a
        fila = (t_ref.f - t.f) / t.e
        col_i, fila_i = round(col), round(fila)
        if abs(col - col_i) > 1e-3 or abs(fila - fila_i) > 1e-3:
            raise ValueError("Las capas no están alineadas a la misma rejilla de píxeles.")
        if col_i < 0 or fila_i < 0 or col_i + ancho > src.width or fila_i + alto > src.height:
            raise ValueError("La zona queda fuera de esta tesela. ¿Es la tesela correcta?")

        return src.read(1, window=Window(col_i, fila_i, ancho, alto))
