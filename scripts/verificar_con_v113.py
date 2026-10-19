"""Valida las predicciones del agente con datos reales de Hansen v1.13 (que incluye 2025).

El agente usa datos hasta 2024 (v1.12) para estimar qué celdas de ~1 km perderán bosque en 2025.
Este script descarga el `lossyear` de v1.13, mira qué pasó realmente en 2025 y calcula:

  1. `confirmado` (1/0) para las celdas que el agente priorizó -> feedback para el agente.
  2. Las métricas del objetivo SMART (accuracy, recall, falsos positivos) sobre TODAS las celdas.

Uso (desde la raíz del repo, después de haber ejecutado el agente al menos una vez):

    python scripts/verificar_con_v113.py --zona guaviare --tesela 10N_080W
    python scripts/verificar_con_v113.py --zona guaviare --tesela 10N_080W --guardar-feedback

Salidas en outputs/:
    validacion_<zona>_2025.md                 resumen con la tabla del objetivo SMART
    validacion_<zona>_2025.csv                una fila por celda: riesgo, pérdida real y alerta real
    feedback_v113_<zona>_2025.csv             `confirmado` de las celdas priorizadas por el agente
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.forestguard.data import descargar_tesela, leer_alineado  # noqa: E402
from src.forestguard.features import (  # noqa: E402
    cargar_panel,
    construir_panel,
    features_ultimo_anio,
    leer_capas,
    sumar_por_celda,
)
from src.forestguard.model import (  # noqa: E402
    K,
    entrenar,
    entrenar_y_evaluar,
    precision_top_k,
    predecir_riesgo,
    separar_conjuntos,
)

VERSION_NUEVA = "GFC-2025-v1.13"
ANIO_REAL = 2025
CODIGO_ANIO = ANIO_REAL - 2000   # valor de lossyear que corresponde a 2025

# Objetivo SMART
OBJ_ACCURACY = 0.85
OBJ_RECALL = 0.85
OBJ_FPR = 0.15


# ------------------------------------------------------------------ métricas
def umbral_para_recall(y_val, score_val, recall_obj):
    """Mayor umbral de riesgo que todavía logra el recall objetivo, medido en validación."""
    _, r, u = precision_recall_curve(y_val, score_val)
    ok = np.where(r[:-1] >= recall_obj)[0]
    if len(ok) == 0:
        return float(u[0])   # ni con el umbral más bajo se llega al recall pedido
    return float(u[ok[-1]])


def metricas_binarias(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    def div(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "accuracy": div(tp + tn, len(y)),
        "recall": div(tp, tp + fn),
        "fpr": div(fp, fp + tn),                    # falsos positivos / todas las celdas sin pérdida real
        "falsas_entre_alertas": div(fp, fp + tp),   # de las celdas marcadas, cuántas eran falsas
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def cumple_smart(fila):
    return bool(
        fila["accuracy"] >= OBJ_ACCURACY and fila["recall"] >= OBJ_RECALL and fila["fpr"] < OBJ_FPR
    )


def pct(x):
    return "n/d" if pd.isna(x) else f"{x * 100:.1f}%"


def si_no(ok):
    return "cumple" if ok else "NO cumple"


# ------------------------------------------------------------------- reporte
def escribir_resumen(ruta, zona, umbral_ha, n_celdas, n_alertas, base_acc, coincide, tabla, top_rf, top_pers):
    lineas = [
        f"# Validación de ForestGuard-Agent con Hansen v1.13 ({zona}, {ANIO_REAL})",
        "",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"El agente usó datos hasta 2024 (v1.12) para estimar qué celdas de ~1 km perderían bosque en {ANIO_REAL}. "
        f"Lo que realmente pasó sale del `lossyear` de v1.13: una celda es **alerta real** si perdió "
        f">= {umbral_ha} ha de bosque en {ANIO_REAL}.",
        "",
        f"- Celdas evaluadas: {n_celdas:,}",
        f"- Alertas reales en {ANIO_REAL}: {n_alertas:,} ({pct(n_alertas / n_celdas)})",
        f"- Accuracy de la referencia \"nunca alertar\": {pct(base_acc)}. "
        "Con tan pocas alertas, la accuracy sola casi no distingue un buen modelo de uno inútil; por eso "
        "importan también el recall y los falsos positivos.",
        f"- Coincidencia de v1.12 y v1.13 en la pérdida 2001-2024: {pct(coincide)}.",
        "",
        "## Objetivo SMART",
        "",
        f"Objetivos: accuracy >= {OBJ_ACCURACY:.0%}, recall >= {OBJ_RECALL:.0%}, "
        f"falsos positivos < {OBJ_FPR:.0%}.",
        "",
        "| Umbral de riesgo | Valor | Accuracy | Recall | FPR | Falsas entre alertas | TP | FP | FN | TN | Objetivo |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, f in tabla.iterrows():
        lineas.append(
            f"| {f['nombre']} | {f['umbral']:.3f} | {pct(f['accuracy'])} | {pct(f['recall'])} | "
            f"{pct(f['fpr'])} | {pct(f['falsas_entre_alertas'])} | {f['TP']} | {f['FP']} | {f['FN']} | "
            f"{f['TN']} | {si_no(cumple_smart(f))} |"
        )
    lineas += [
        "",
        "- **FPR** = falsos positivos / celdas sin pérdida real. **Falsas entre alertas** = falsos positivos / "
        "celdas marcadas. \"Falsos positivos < 15 %\" admite las dos lecturas; la tabla muestra ambas.",
        "- El segundo umbral se eligió **en validación** (años 2021-2022) para alcanzar el recall pedido, "
        f"sin mirar {ANIO_REAL}. Ajustar el umbral con los datos de {ANIO_REAL} sería hacer trampa.",
        "",
        f"## Priorización (top {K})",
        "",
        f"De las {K} celdas con mayor riesgo, tuvieron pérdida real: Random Forest {pct(top_rf)} | "
        f"Persistencia (repetir el último año) {pct(top_pers)} | Azar {pct(n_alertas / n_celdas)}.",
        "",
        "## Notas",
        "",
        "- La unidad de evaluación es la celda de ~1 km, no el píxel.",
        "- Una alerta real significa pérdida de cobertura arbórea (incluye incendios y plantaciones), "
        "no necesariamente deforestación ilegal.",
        "- Resultado de una sola zona y un solo año: no se puede generalizar a otras regiones.",
        "",
    ]
    Path(ruta).write_text("\n".join(lineas), encoding="utf-8")


def leer_csv_si_existe(ruta):
    ruta = Path(ruta)
    if ruta.exists() and ruta.stat().st_size > 0:
        return pd.read_csv(ruta)
    return pd.DataFrame()


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Valida ForestGuard-Agent con Hansen v1.13")
    parser.add_argument("--zona", default="guaviare", help="nombre usado en recortar(..., zona=...)")
    parser.add_argument("--tesela", default="10N_080W", help="tesela de Hansen que contiene la zona")
    parser.add_argument(
        "--guardar-feedback",
        action="store_true",
        help="agrega el `confirmado` calculado a memory/feedback.csv (no pisa filas que ya existan)",
    )
    args = parser.parse_args()

    procesado = RAIZ / "data" / "processed"
    raw = RAIZ / "data" / "raw"
    salidas = RAIZ / "outputs"
    memoria = RAIZ / "memory"
    salidas.mkdir(exist_ok=True)

    # 1. lossyear de v1.13 (una sola capa)
    ruta_v113 = descargar_tesela(args.tesela, capas=("lossyear",), destino=raw, version=VERSION_NUEVA)["lossyear"]

    # 2. panel con las mismas reglas con las que entrena el agente
    if (procesado / f"panel_{args.zona}.csv").exists():
        panel, meta_celdas = cargar_panel(args.zona, procesado)
    else:
        print("No hay panel guardado, construyéndolo (tarda un poco)...")
        panel, meta_celdas = construir_panel(args.zona, procesado=procesado)
    bosque_min = meta_celdas["bosque_min"]
    celda_px = meta_celdas["celda_px"]
    umbral_ha = meta_celdas["umbral_ha"]

    # 3. lo que realmente pasó en 2025, por celda
    tc, ly12, dm, meta = leer_capas(args.zona, procesado)
    ly13 = leer_alineado(ruta_v113, procesado / f"{args.zona}_lossyear.tif")

    n_f, n_c = tc.shape[0] // celda_px, tc.shape[1] // celda_px
    alto, ancho = n_f * celda_px, n_c * celda_px
    tc, ly12, ly13, dm = (a[:alto, :ancho] for a in (tc, ly12, ly13, dm))
    bosque = (tc >= bosque_min) & (dm == 1)   # mismo bosque que usa el agente (de v1.12)

    perdio12 = bosque & (ly12 > 0)
    coincide = float((ly13[perdio12] == ly12[perdio12]).mean()) if perdio12.any() else float("nan")
    print(f"Coincidencia v1.12 vs v1.13 en la pérdida 2001-2024: {pct(coincide)}")
    if coincide < 0.95:
        print("  AVISO: las versiones difieren bastante. No mezcles v1.12 y v1.13 para entrenar.")

    if not (bosque & (ly13 == CODIGO_ANIO)).any():
        raise ValueError(
            f"En v1.13 no hay píxeles de bosque con lossyear == {CODIGO_ANIO} en esta zona. "
            "Revisa la codificación con np.unique(ly13): las notas de v1.13 dicen 1-20 pero el catálogo de "
            "Earth Engine dice 1-25."
        )

    perdida_px = sumar_por_celda(bosque & (ly13 == CODIGO_ANIO), n_f, n_c, celda_px).ravel()
    real = pd.DataFrame({
        "celda_id": np.arange(n_f * n_c),   # mismo orden que usa features.py (fila * n_c + col)
        "perdida_ha_2025": perdida_px * meta["ha_por_pixel"],
    })
    real["alerta_real"] = (real["perdida_ha_2025"] >= umbral_ha).astype(int)

    # 4. predicciones: mismo procedimiento que el agente (sin feedback)
    train, val, test = separar_conjuntos(panel)
    evaluacion = entrenar_y_evaluar(panel, umbral_ha)
    umbral_f1 = evaluacion["umbral_riesgo"]
    score_val = predecir_riesgo(evaluacion["modelo"], val)
    umbral_rec = umbral_para_recall(val["alerta"].values, score_val, OBJ_RECALL)

    modelo = entrenar(panel)
    nuevo = features_ultimo_anio(
        args.zona, bosque_min=bosque_min, celda_px=celda_px, umbral_ha=umbral_ha, procesado=procesado
    )
    nuevo["riesgo"] = predecir_riesgo(modelo, nuevo)
    datos = nuevo.merge(real, on="celda_id", how="left")
    if datos["alerta_real"].isna().any():
        raise ValueError("Hay celdas sin resultado real: la cuadrícula no coincide con la del panel.")

    # 5. métricas del objetivo SMART sobre todas las celdas
    y = datos["alerta_real"].values
    umbrales = [
        ("Umbral del agente (mejor F1 en validación)", umbral_f1),
        (f"Umbral para recall >= {OBJ_RECALL:.0%} (elegido en validación)", umbral_rec),
    ]
    filas = []
    for nombre, umbral in umbrales:
        pred = (datos["riesgo"].values >= umbral).astype(int)
        datos[f"alerta_pred_{len(filas) + 1}"] = pred
        filas.append({"nombre": nombre, "umbral": umbral, **metricas_binarias(y, pred)})
    tabla = pd.DataFrame(filas)

    n_celdas, n_alertas = len(datos), int(y.sum())
    base_acc = 1 - n_alertas / n_celdas
    top_in = datos.assign(alerta=datos["alerta_real"])
    top_rf = precision_top_k(top_in, "riesgo", K)
    top_pers = precision_top_k(top_in, "perdida_ha_ultimo", K)

    datos.rename(columns={"alerta_pred_1": "alerta_pred_agente", "alerta_pred_2": "alerta_pred_recall"})[
        ["celda_id", "lat", "lon", "riesgo", "alerta_pred_agente", "alerta_pred_recall",
         "perdida_ha_2025", "alerta_real"]
    ].round(4).to_csv(salidas / f"validacion_{args.zona}_{ANIO_REAL}.csv", index=False)

    escribir_resumen(
        salidas / f"validacion_{args.zona}_{ANIO_REAL}.md",
        args.zona, umbral_ha, n_celdas, n_alertas, base_acc, coincide, tabla, top_rf, top_pers,
    )

    print(f"\nCeldas evaluadas: {n_celdas:,} | alertas reales en {ANIO_REAL}: {n_alertas:,} ({pct(n_alertas / n_celdas)})")
    print(f"Accuracy de 'nunca alertar': {pct(base_acc)}")
    for _, f in tabla.iterrows():
        print(
            f"\n{f['nombre']} (umbral {f['umbral']:.3f})\n"
            f"  accuracy {pct(f['accuracy'])} | recall {pct(f['recall'])} | FPR {pct(f['fpr'])} | "
            f"falsas entre alertas {pct(f['falsas_entre_alertas'])} -> objetivo SMART: {si_no(cumple_smart(f))}"
        )
    print(f"\nTop {K}: Random Forest {pct(top_rf)} | Persistencia {pct(top_pers)} | Azar {pct(n_alertas / n_celdas)}")

    # 6. `confirmado` para las celdas que el agente priorizó
    historial = leer_csv_si_existe(memoria / "historial.csv")
    if historial.empty:
        print("\nNo hay memory/historial.csv: ejecuta primero `python scripts/run_agent.py` para obtener `confirmado`.")
    else:
        h = historial[(historial["zona"] == args.zona) & (historial["anio_pred"] == ANIO_REAL)]
        if h.empty:
            print(f"\nEl historial no tiene corridas de {args.zona} que predigan {ANIO_REAL}.")
        else:
            ultima = h[h["ejecucion"] == h["ejecucion"].max()]
            fb = ultima[["celda_id", "lat", "lon"]].merge(real[["celda_id", "alerta_real"]], on="celda_id")
            fb.insert(0, "zona", args.zona)
            fb["anio_pred"] = ANIO_REAL
            fb["confirmado"] = fb["alerta_real"].astype(int)
            fb = fb[["zona", "celda_id", "anio_pred", "lat", "lon", "confirmado"]]

            ruta_fb = salidas / f"feedback_v113_{args.zona}_{ANIO_REAL}.csv"
            fb.to_csv(ruta_fb, index=False)
            print(f"\nDe las {len(fb)} celdas priorizadas por el agente, {int(fb['confirmado'].sum())} "
                  f"tuvieron pérdida real (>= {umbral_ha} ha) en {ANIO_REAL}.")
            print(f"Guardado: {ruta_fb}")

            if args.guardar_feedback:
                memoria.mkdir(exist_ok=True)
                existente = leer_csv_si_existe(memoria / "feedback.csv")
                combinado = pd.concat([existente, fb], ignore_index=True)
                combinado = combinado.drop_duplicates(["zona", "celda_id", "anio_pred"], keep="first")
                nuevas = len(combinado) - len(existente)
                combinado.to_csv(memoria / "feedback.csv", index=False)
                print(f"memory/feedback.csv: {nuevas} filas nuevas (las que ya existían no se tocaron).")
                print("OJO: si reentrenas con este feedback, no vuelvas a evaluar con los mismos datos de "
                      f"{ANIO_REAL}: el modelo ya habría visto la respuesta.")

    print(f"\nResumen: {salidas / f'validacion_{args.zona}_{ANIO_REAL}.md'}")


if __name__ == "__main__":
    main()
