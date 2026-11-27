"""Deja listos en data/processed/ los recortes de las 4 zonas del Hito 2 que usa la aplicación.

    python scripts/preparar_zonas_app.py

Si una zona ya está recortada no hace nada. Si falta, descarga la tesela 10N_080W (v1.12) a data/raw/
(una sola vez, varios cientos de MB) y recorta. Los recortes (~6 MB por zona) se suben al repositorio
para que la aplicación desplegada no tenga que descargar nada (ver excepciones en .gitignore).
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.forestguard.data import descargar_tesela, recortar  # noqa: E402
from src.forestguard.features import CAPAS  # noqa: E402
from src.forestguard.zonas import PROCESSED_DIR, RAW_DIR, recorte_disponible, tesela_de, zonas_hito2  # noqa: E402


def main():
    for zona, info in zonas_hito2().items():
        if recorte_disponible(zona):
            print(f"[listo] {zona}")
            continue
        bbox = tuple(info["bbox"])
        tesela = tesela_de(bbox[0], bbox[3] - 1e-9)
        print(f"[recortando] {zona} {bbox} de la tesela {tesela}")
        descargar_tesela(tesela, destino=RAW_DIR)
        recortar(tesela, bbox, zona, origen=RAW_DIR, destino=PROCESSED_DIR)

    print("\nArchivos para subir al repositorio:")
    for zona in zonas_hito2():
        for capa in CAPAS:
            ruta = PROCESSED_DIR / f"{zona}_{capa}.tif"
            print(f"  {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
