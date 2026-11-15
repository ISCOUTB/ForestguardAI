"""Hito 2 · Paso 1: revisar zonas candidatas para elegir las de entrenamiento y las retenidas.

Usa SOLO la versión v1.12 (pérdida 2001-2024). No abre la v1.13 ni mira 2025:
las zonas se eligen sin conocer el año de prueba.

    python scripts/tamizaje_zonas.py

Salida: outputs/hito2/tamizaje_zonas.csv y la decisión impresa en pantalla.

IMPORTANTE: los CRITERIOS de abajo se escriben y se suben a git ANTES de ejecutar este script.
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.forestguard.data import descargar_tesela, recortar  # noqa: E402
from src.forestguard.features import calcular_cubos, tabla_features  # noqa: E402

RAW = RAIZ / "data" / "raw"
PROC = RAIZ / "data" / "processed"
SALIDA = RAIZ / "outputs" / "hito2"
TESELA = "10N_080W"   # cubre lat 0..10 N, lon -80..-70

# ======================================================================================
# CANDIDATAS (bbox = lon_min, lat_min, lon_max, lat_max), en ORDEN DE PREFERENCIA.
# El orden se fija aquí y sirve de desempate: no se escoge "la que salga más fácil".
# ======================================================================================
CANDIDATAS_ENTRENAMIENTO = {
    "guaviare": (-73.5, 1.5, -72.5, 2.5),   # zona piloto (obligatoria)
    "macarena": (-74.5, 2.0, -73.5, 3.0),   # Meta, frente de pérdida vecino a Guaviare
}
CANDIDATAS_RETENIDAS = {
    "chaira":     (-75.0, 0.5, -74.0, 1.5),   # Caquetá, Cartagena del Chairá
    "putumayo":   (-77.0, 0.0, -76.0, 1.0),   # Putumayo, Puerto Asís
    "bajo_cauca": (-75.5, 7.0, -74.5, 8.0),   # Antioquia, región Caribe/andina (reserva)
    "catatumbo":  (-73.5, 8.0, -72.5, 9.0),   # Norte de Santander (reserva; roza frontera con Venezuela)
}

# ======================================================================================
# CRITERIOS (fijados antes de ver el tamizaje)
# ======================================================================================
MIN_CELDAS_BOSQUE = 3000        # celdas de ~1 km con bosque restante en 2024
MIN_ALERTAS_POR_ANIO = 50       # alertas (>= 2 ha) en CADA año 2022, 2023 y 2024
ANIOS_REVISADOS = (2022, 2023, 2024)
N_RETENIDAS = 2                 # se toman las 2 primeras retenidas (en orden) que cumplan
# Una retenida no puede tocar (ni por borde ni por esquina) una zona de entrenamiento,
# ni otra retenida, para que no compartan el mismo frente de deforestación.
# ======================================================================================


def separadas(a, b):
    """True si los dos recuadros no se tocan ni por borde ni por esquina."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def revisar_zona(zona, bbox):
    recortar(TESELA, bbox, zona, origen=RAW, destino=PROC)
    cubos = calcular_cubos(zona, procesado=PROC)
    fila = {"zona": zona, "lon_min": bbox[0], "lat_min": bbox[1], "lon_max": bbox[2], "lat_max": bbox[3]}
    # T = año observado - 2000; la etiqueta es del año T + 1
    for anio in ANIOS_REVISADOS:
        t = tabla_features(cubos, anio - 2001)
        fila[f"alertas_{anio}"] = int(t["alerta"].sum())
        fila[f"prevalencia_{anio}_%"] = round(100 * t["alerta"].mean(), 2)
    fila["celdas_bosque_2024"] = len(tabla_features(cubos, 24))
    fila["cumple"] = bool(
        fila["celdas_bosque_2024"] >= MIN_CELDAS_BOSQUE
        and all(fila[f"alertas_{a}"] >= MIN_ALERTAS_POR_ANIO for a in ANIOS_REVISADOS)
    )
    return fila


def main():
    descargar_tesela(TESELA, destino=RAW)   # si ya está en data/raw/ no se repite

    todas = {**CANDIDATAS_ENTRENAMIENTO, **CANDIDATAS_RETENIDAS}
    filas = []
    for zona, bbox in todas.items():
        fila = revisar_zona(zona, bbox)
        fila["rol_candidato"] = "entrenamiento" if zona in CANDIDATAS_ENTRENAMIENTO else "retenida"
        filas.append(fila)
    df = pd.DataFrame(filas)

    # Comprobación de que ninguna pareja se solapa (sería un error de bbox)
    for (za, a), (zb, b) in combinations(todas.items(), 2):
        solapan = a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
        if solapan:
            sys.exit(f"ERROR: {za} y {zb} se solapan. Corrige los bbox.")

    # Decisión con la regla fijada
    entrenamiento = list(CANDIDATAS_ENTRENAMIENTO)
    fallan = df[(df["rol_candidato"] == "entrenamiento") & ~df["cumple"]]["zona"].tolist()
    if fallan:
        print(f"AVISO: zonas de entrenamiento que no cumplen los mínimos: {fallan}")

    retenidas = []
    for zona, bbox in CANDIDATAS_RETENIDAS.items():
        if len(retenidas) == N_RETENIDAS:
            break
        ok = df.loc[df["zona"] == zona, "cumple"].item()
        lejos = all(separadas(bbox, todas[z]) for z in entrenamiento + retenidas)
        if ok and lejos:
            retenidas.append(zona)

    df["rol_final"] = df["zona"].map(
        lambda z: "entrenamiento" if z in entrenamiento else ("retenida" if z in retenidas else "descartada")
    )
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / "tamizaje_zonas.csv"
    df.to_csv(ruta, index=False)

    pd.set_option("display.width", 200)
    print(df.drop(columns=["lon_min", "lat_min", "lon_max", "lat_max"]).to_string(index=False))
    print(f"\nEntrenamiento: {entrenamiento}")
    print(f"Retenidas:     {retenidas}")
    if len(retenidas) < N_RETENIDAS:
        print("NO hay suficientes retenidas que cumplan: agrega candidatas al FINAL de la lista y vuelve a correr.")
    print(f"Tabla guardada en {ruta}")


if __name__ == "__main__":
    main()
