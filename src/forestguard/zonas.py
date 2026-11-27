"""Zonas de trabajo del agente: las del Hito 2 y cualquier recuadro dentro de Colombia.

Zonas fijas: se leen de config/hito2.yaml (entrenamiento y retenidas) y usan los recortes que ya están
en data/processed/<zona>_<capa>.tif (ver scripts/preparar_zonas_app.py).

Recuadro libre: el usuario da (lon_min, lat_min, lon_max, lat_max). Se valida, se averigua la tesela de
Hansen que lo contiene y se recorta por ventana, sin descargar la tesela completa:
  - si la tesela está en data/raw/ (descargada antes), se lee de ahí;
  - si no, se lee a distancia desde el servidor de Hansen con GDAL (/vsicurl/), pidiendo solo los
    bytes de la ventana. Esto es lo que usa la aplicación desplegada.

    from src.forestguard.zonas import validar_recuadro, recortar_recuadro

    bbox = validar_recuadro((-73.2, 1.8, -72.7, 2.3))
    info = recortar_recuadro(bbox, destino=Path("tmp/data/processed"))
"""
import math
import time
from pathlib import Path

import rasterio
import yaml
from rasterio.windows import from_bounds

from .data import BASE, TESELAS_COLOMBIA, VERSION, nombre_archivo
from .features import CAPAS

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "config" / "hito2.yaml"
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"

LADO_MAX = 1.0     # grados: la meta de tiempo del Hito 1 se fijó para recuadros de 1° × 1°
LADO_MIN = 0.2     # grados: por debajo quedan muy pocas celdas (~22 × 22) y suelen faltar alertas
ZONA_RECUADRO = "recuadro"

# Opciones de GDAL para leer un GeoTIFF remoto pidiendo solo los bytes necesarios
GDAL_REMOTO = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}


class RecuadroInvalido(ValueError):
    """El recuadro no se puede procesar (fuera de Colombia, mal ordenado, demasiado grande...)."""


# ------------------------------------------------------------------------- zonas fijas
def zonas_hito2(config=CONFIG):
    """{zona: {"bbox": [...], "rol": "entrenamiento" | "retenida"}} según config/hito2.yaml."""
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    zonas = {}
    for nombre, bbox in cfg["zonas"]["entrenamiento"].items():
        zonas[nombre] = {"bbox": list(bbox), "rol": "entrenamiento"}
    for nombre, bbox in cfg["zonas"]["retenidas"].items():
        zonas[nombre] = {"bbox": list(bbox), "rol": "retenida"}
    return zonas


def recorte_disponible(zona, procesado=PROCESSED_DIR):
    return all((Path(procesado) / f"{zona}_{capa}.tif").exists() for capa in CAPAS)


# ----------------------------------------------------------------------- recuadro libre
def tesela_de(lon, lat):
    """Nombre de la tesela Hansen 10°×10° que contiene el punto (esquina superior izquierda)."""
    norte = math.ceil(lat / 10) * 10
    oeste = math.floor(lon / 10) * 10
    if lat == norte:            # un punto justo en el borde norte pertenece a la tesela de arriba
        norte += 10
    lat_txt = f"{abs(norte):02d}{'N' if norte >= 0 else 'S'}"
    lon_txt = f"{abs(oeste):03d}{'E' if oeste >= 0 else 'W'}"
    return f"{lat_txt}_{lon_txt}"


def validar_recuadro(bbox):
    """Revisa el recuadro y lo devuelve como tupla de float. Lanza RecuadroInvalido si no sirve."""
    try:
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise RecuadroInvalido("El recuadro debe tener 4 números: lon_min, lat_min, lon_max, lat_max.")

    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise RecuadroInvalido("Coordenadas mal ordenadas: se espera lon_min < lon_max y lat_min < lat_max.")
    ancho, alto = lon_max - lon_min, lat_max - lat_min
    if ancho > LADO_MAX + 1e-9 or alto > LADO_MAX + 1e-9:
        raise RecuadroInvalido(f"El recuadro mide {ancho:.2f}° × {alto:.2f}°; el máximo es "
                               f"{LADO_MAX:.0f}° × {LADO_MAX:.0f}°.")
    if ancho < LADO_MIN - 1e-9 or alto < LADO_MIN - 1e-9:
        raise RecuadroInvalido(f"El recuadro mide {ancho:.2f}° × {alto:.2f}°; el mínimo es "
                               f"{LADO_MIN}° × {LADO_MIN}° para tener suficientes celdas.")

    esquina = tesela_de(lon_min, lat_max - 1e-9)
    if esquina != tesela_de(lon_max - 1e-9, lat_min):
        raise RecuadroInvalido("El recuadro cruza el borde entre dos teselas de Hansen (múltiplos de 10°). "
                               "Muévelo para que quede dentro de una sola.")
    if esquina not in TESELAS_COLOMBIA:
        raise RecuadroInvalido(f"El recuadro cae en la tesela {esquina}, fuera de Colombia continental "
                               f"({', '.join(TESELAS_COLOMBIA)}).")
    return lon_min, lat_min, lon_max, lat_max


def _fuente(capa, tesela, raw, base, version):
    local = Path(raw) / nombre_archivo(capa, tesela, version)
    if local.exists():
        return str(local), "local"
    return f"/vsicurl/{base}/{version}/{nombre_archivo(capa, tesela, version)}", "remoto"


def recortar_recuadro(bbox, destino, zona=ZONA_RECUADRO, raw=RAW_DIR, base=BASE, version=VERSION):
    """Recorta las 3 capas del recuadro y las escribe en destino/<zona>_<capa>.tif.

    Devuelve {"tesela", "origen" (local/remoto), "segundos", "forma"}.
    """
    bbox = validar_recuadro(bbox)
    tesela = tesela_de(bbox[0], bbox[3] - 1e-9)
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)

    inicio = time.perf_counter()
    origenes = set()
    forma = None
    with rasterio.Env(**GDAL_REMOTO):
        for capa in CAPAS:
            ruta, origen = _fuente(capa, tesela, raw, base, version)
            origenes.add(origen)
            try:
                src = rasterio.open(ruta)
            except rasterio.errors.RasterioIOError as error:
                raise ConnectionError(f"No pude abrir la capa {capa} de la tesela {tesela} ({origen}): "
                                      f"{str(error)[:150]}") from error
            with src:
                ventana = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
                datos = src.read(1, window=ventana)
                perfil = src.profile.copy()
                perfil.update(height=datos.shape[0], width=datos.shape[1],
                              transform=src.window_transform(ventana), tiled=False, compress="lzw",
                              driver="GTiff")
                for clave in ("blockxsize", "blockysize", "interleave"):
                    perfil.pop(clave, None)
            with rasterio.open(destino / f"{zona}_{capa}.tif", "w", **perfil) as dst:
                dst.write(datos, 1)
            forma = datos.shape
    return {"tesela": tesela, "origen": "/".join(sorted(origenes)), "forma": forma,
            "segundos": time.perf_counter() - inicio}
