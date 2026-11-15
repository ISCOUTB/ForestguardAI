"""Hito 2 · Línea base multizona de ForestGuard-Agent.

Entrena la línea base (Random Forest de src/forestguard/model.py) con las zonas de ENTRENAMIENTO,
fija el umbral de riesgo en VALIDACIÓN y mide en las zonas RETENIDAS, prediciendo 2025 con las
etiquetas de Hansen v1.13. Todas las reglas salen de config/hito2.yaml: este script no decide nada.

    python scripts/linea_base_multizona.py            # corrida oficial (5 semillas)
    python scripts/linea_base_multizona.py --rapido   # prueba local con 1 semilla (archivos con prefijo prueba_)

Salidas en outputs/hito2/:
    registro_linea_base.txt     bitácora acumulada (cada corrida agrega un bloque)
    linea_base_corridas.csv     una fila por modelo x zona x semilla
    linea_base_resumen.csv      media ± desviación por modelo y zona, y media de zonas retenidas
"""
import argparse
import hashlib
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import sklearn
import yaml
from sklearn.metrics import average_precision_score, roc_curve

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from src.forestguard.data import descargar_tesela, leer_alineado, recortar  # noqa: E402
from src.forestguard.features import (  # noqa: E402
    CAPAS,
    FEATURES,
    construir_panel,
    features_ultimo_anio,
    leer_capas,
    sumar_por_celda,
)
from src.forestguard.model import entrenar, predecir_riesgo  # noqa: E402
from verificar_con_v113 import metricas_binarias  # noqa: E402

RAW = RAIZ / "data" / "raw"
PROC = RAIZ / "data" / "processed"
SALIDA = RAIZ / "outputs" / "hito2"
ANIO_PRUEBA = 2025
METRICAS = ["pr_auc", "top_k", "recall", "fpr", "accuracy", "falsas_entre_alertas"]


# ------------------------------------------------------------------------- utilidades
class Bitacora:
    """Imprime en pantalla y agrega al archivo de texto."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, texto=""):
        print(texto)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(texto + "\n")


def pct(x):
    return "n/d" if pd.isna(x) else f"{x * 100:.1f}%"


def pct_ms(m, s):
    if pd.isna(m):
        return "n/d"
    return pct(m) if (pd.isna(s) or s == 0) else f"{pct(m)} ± {s * 100:.1f}"


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def cargar_config(ruta):
    contenido = Path(ruta).read_bytes()
    return yaml.safe_load(contenido), hashlib.sha256(contenido).hexdigest()[:12]


# ------------------------------------------------------------------------------ datos
def asegurar_recorte(zona, bbox, tesela):
    """Recorta la zona con v1.12 si todavía no está en data/processed/."""
    if all((PROC / f"{zona}_{c}.tif").exists() for c in CAPAS):
        return
    descargar_tesela(tesela, destino=RAW)
    recortar(tesela, tuple(bbox), zona, origen=RAW, destino=PROC)


def etiquetas_2025(zona, ruta_v113, celdas):
    """Pérdida real de 2025 por celda (v1.13), con el mismo bosque y la misma cuadrícula que v1.12."""
    tc, ly12, dm, meta = leer_capas(zona, PROC)
    ly13 = leer_alineado(ruta_v113, PROC / f"{zona}_lossyear.tif")

    cp = celdas["celda_px"]
    n_f, n_c = tc.shape[0] // cp, tc.shape[1] // cp
    alto, ancho = n_f * cp, n_c * cp
    tc, ly12, ly13, dm = (a[:alto, :ancho] for a in (tc, ly12, ly13, dm))
    bosque = (tc >= celdas["bosque_min_pct"]) & (dm == 1)

    perdio12 = bosque & (ly12 > 0)
    coincide = float((ly13[perdio12] == ly12[perdio12]).mean()) if perdio12.any() else float("nan")

    codigo = ANIO_PRUEBA - 2000
    if not (bosque & (ly13 == codigo)).any():
        raise ValueError(f"{zona}: v1.13 no tiene píxeles de bosque con lossyear == {codigo}. "
                         "Revisa la codificación con np.unique(ly13).")

    px = sumar_por_celda(bosque & (ly13 == codigo), n_f, n_c, cp).ravel()
    real = pd.DataFrame({"celda_id": np.arange(n_f * n_c), "perdida_ha_2025": px * meta["ha_por_pixel"]})
    real["alerta"] = (real["perdida_ha_2025"] >= celdas["alerta_ha"]).astype(int)
    return real, coincide


def panel_entrenamiento(zonas, celdas):
    paneles = []
    for zona in zonas:
        panel, _ = construir_panel(zona, bosque_min=celdas["bosque_min_pct"], celda_px=celdas["celda_px"],
                                   umbral_ha=celdas["alerta_ha"], procesado=PROC, guardar=False)
        paneles.append(panel.assign(zona=zona))
    return pd.concat(paneles, ignore_index=True)


def tabla_prueba(zona, real, celdas):
    """Features observando hasta 2024 + alerta real de 2025."""
    nuevo = features_ultimo_anio(zona, bosque_min=celdas["bosque_min_pct"], celda_px=celdas["celda_px"],
                                 umbral_ha=celdas["alerta_ha"], procesado=PROC)
    datos = nuevo.merge(real, on="celda_id", how="left", validate="one_to_one")
    if datos["alerta"].isna().any():
        raise ValueError(f"{zona}: hay celdas sin etiqueta 2025; la cuadrícula no coincide.")
    return datos.assign(zona=zona)


# --------------------------------------------------------------------------- métricas
def umbral_por_fpr(y, score, fpr_max):
    """Umbral con MAYOR recall que deja FPR <= fpr_max. Si hay empate, el umbral más alto."""
    fpr, tpr, umbrales = roc_curve(y, score, drop_intermediate=False)
    ok = np.flatnonzero(fpr <= fpr_max)
    mejor = tpr[ok].max()
    i = ok[np.flatnonzero(tpr[ok] == mejor)[0]]   # umbrales viene de mayor a menor
    return float(umbrales[i]), float(tpr[i]), float(fpr[i])


def top_k(y, score, k):
    orden = np.argsort(-score, kind="stable")[:k]
    return float(np.asarray(y)[orden].mean())


def evaluar(y, score, pred, k):
    return {"pr_auc": average_precision_score(y, score), "top_k": top_k(y, score, k),
            **metricas_binarias(y, pred)}


# ------------------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="Línea base multizona (Hito 2)")
    parser.add_argument("--config", default=str(RAIZ / "config" / "hito2.yaml"))
    parser.add_argument("--rapido", action="store_true", help="1 semilla y archivos con prefijo prueba_")
    args = parser.parse_args()

    cfg, hash_cfg = cargar_config(args.config)
    celdas, zonas, div = cfg["celdas"], cfg["zonas"], cfg["division_temporal"]
    base = cfg["linea_base"]
    semillas = base["semillas"][:1] if args.rapido else base["semillas"]
    k = cfg["metricas"]["precision_top_k"]["k"]
    fpr_max = cfg["umbral_riesgo"]["fpr_max_validacion"]
    entrenamiento, retenidas = list(zonas["entrenamiento"]), list(zonas["retenidas"])

    prefijo = "prueba_" if args.rapido else ""
    log = Bitacora(SALIDA / f"{prefijo}registro_linea_base.txt")
    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    inicio = time.perf_counter()

    # ---- 0) encabezado y trazabilidad
    commit = git("rev-parse", "--short", "HEAD") or "desconocido"
    config_sucio = bool(git("status", "--porcelain", "--", "config"))
    log("=" * 100)
    log(f"LÍNEA BASE MULTIZONA · ForestGuard-Agent · ejecución {marca}{'  [MODO RÁPIDO]' if args.rapido else ''}")
    log("=" * 100)
    log(f"Entorno: Python {platform.python_version()} · numpy {np.__version__} · pandas {pd.__version__} · "
        f"scikit-learn {sklearn.__version__} · rasterio {rasterio.__version__} · {platform.platform()}")
    log(f"Commit: {commit} | config/hito2.yaml sha256 {hash_cfg} | reglas fijadas: {cfg.get('fecha_fijacion')}")
    if config_sucio:
        log("AVISO: config/ tiene cambios sin commit. La corrida oficial debe usar las reglas tal como están en git.")
    log(f"Entrenamiento: {entrenamiento} | retenidas: {retenidas} | semillas: {semillas}")
    log(f"Años predichos -> entrenamiento {div['entrenamiento']}, validación {div['validacion']}, prueba {div['prueba']}")
    log(f"Alerta real: pérdida >= {celdas['alerta_ha']} ha en la celda | umbral: mayor recall con FPR <= {fpr_max:.0%} en validación")
    log("Modelo de prueba: el mismo entrenado hasta el año de corte de entrenamiento (no se reentrena con validación).")

    if base["features"] != FEATURES:
        sys.exit("ERROR: las features del YAML no coinciden con FEATURES de features.py.")

    # ---- 1) datos: recortes y etiquetas 2025
    log("")
    log("-" * 100)
    log("1) DATOS")
    log("-" * 100)
    tesela = cfg["datos"]["tesela"]
    for zona, bbox in {**zonas["entrenamiento"], **zonas["retenidas"]}.items():
        asegurar_recorte(zona, bbox, tesela)
    ruta_v113 = descargar_tesela(tesela, capas=("lossyear",), destino=RAW,
                                 version=cfg["datos"]["version_etiquetas_2025"])["lossyear"]

    prueba = {}
    for zona in entrenamiento + retenidas:
        real, coincide = etiquetas_2025(zona, ruta_v113, celdas)
        log(f"  {zona:<12} coincidencia v1.12 = v1.13 en 2001-2024: {pct(coincide)}")
        if coincide < 0.95:
            log("    AVISO: las versiones difieren bastante en esta zona.")
        if zona in retenidas:   # solo se arma la tabla de prueba de las retenidas
            prueba[zona] = tabla_prueba(zona, real, celdas)
            d = prueba[zona]
            log(f"    prueba 2025: {len(d):,} celdas, {int(d['alerta'].sum()):,} alertas reales "
                f"({pct(d['alerta'].mean())})")

    panel = panel_entrenamiento(entrenamiento, celdas)
    e0, e1 = div["entrenamiento"]
    v0, v1 = div["validacion"]
    train = panel[panel["anio_pred"].between(e0, e1)]
    val = panel[panel["anio_pred"].between(v0, v1)]
    for nombre, df in (("entrenamiento", train), ("validación", val)):
        if df["alerta"].sum() == 0:
            sys.exit(f"ERROR: el conjunto de {nombre} no tiene alertas.")
    log(f"  entrenamiento: {len(train):,} filas, {int(train['alerta'].sum()):,} alertas ({pct(train['alerta'].mean())})")
    log(f"  validación:    {len(val):,} filas, {int(val['alerta'].sum()):,} alertas ({pct(val['alerta'].mean())})")

    # ---- 2) referencias (no dependen de la semilla)
    filas = []
    for zona, d in prueba.items():
        y = d["alerta"].values
        prev = float(y.mean())
        filas.append({"modelo": "azar", "zona": zona, "semilla": -1, "pr_auc": prev, "top_k": prev})
        puntaje = d["perdida_ha_ultimo"].values + d["perdida_ha_ult3"].values * 1e-6   # desempate
        pred = (d["perdida_ha_ultimo"].values >= celdas["alerta_ha"]).astype(int)
        filas.append({"modelo": "persistencia", "zona": zona, "semilla": -1, **evaluar(y, puntaje, pred, k)})

    # ---- 3) línea base, una corrida por semilla
    log("")
    log("-" * 100)
    log("2) LÍNEA BASE (Random Forest)")
    log("-" * 100)
    for semilla in semillas:
        t0 = time.perf_counter()
        modelo = entrenar(train, semilla=semilla)
        params = modelo.get_params()
        for p in ("n_estimators", "max_depth", "min_samples_leaf", "class_weight"):
            if params[p] != base[p]:
                sys.exit(f"ERROR: model.py usa {p}={params[p]} y el YAML dice {base[p]}.")

        s_val = predecir_riesgo(modelo, val)
        umbral, rec_val, fpr_val = umbral_por_fpr(val["alerta"].values, s_val, fpr_max)
        if not np.isfinite(umbral):
            log(f"  AVISO semilla {semilla}: ningún umbral cumple FPR <= {fpr_max:.0%} en validación.")
        linea = f"  semilla {semilla}: umbral {umbral:.4f} (validación: recall {pct(rec_val)}, FPR {pct(fpr_val)})"

        for zona, d in prueba.items():
            y = d["alerta"].values
            s = predecir_riesgo(modelo, d)
            r = evaluar(y, s, (s >= umbral).astype(int), k)
            filas.append({"modelo": "linea_base_rf", "zona": zona, "semilla": semilla, "umbral": umbral,
                          "recall_val": rec_val, "fpr_val": fpr_val, **r})
            linea += f" | {zona}: PR-AUC {r['pr_auc']:.3f}, top-{k} {pct(r['top_k'])}, recall {pct(r['recall'])}, FPR {pct(r['fpr'])}"
        log(linea + f" | {time.perf_counter() - t0:.1f} s")

    # ---- 4) resumen
    df = pd.DataFrame(filas)
    por_zona = df.groupby(["modelo", "zona"])[METRICAS].agg(["mean", "std"])
    # media de retenidas: primero se promedian las zonas en cada semilla, luego media ± desv. entre semillas
    media = (df.groupby(["modelo", "semilla"])[METRICAS].mean()
               .groupby("modelo")[METRICAS].agg(["mean", "std"]))
    media.index = pd.MultiIndex.from_product([media.index, ["media_retenidas"]], names=["modelo", "zona"])
    resumen = pd.concat([por_zona, media]).sort_index()

    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA / f"{prefijo}linea_base_corridas.csv", index=False)
    plano = resumen.copy()
    plano.columns = [f"{m}_{e}" for m, e in plano.columns]
    plano.reset_index().to_csv(SALIDA / f"{prefijo}linea_base_resumen.csv", index=False)

    log("")
    log("=" * 100)
    log(f"TABLA DE RESULTADOS · predicción {ANIO_PRUEBA} en zonas retenidas · media ± desv. ({len(semillas)} semillas)")
    log("=" * 100)
    nombres = {"pr_auc": "PR-AUC", "top_k": f"Top-{k}", "recall": "Recall", "fpr": "FPR",
               "accuracy": "Accuracy*", "falsas_entre_alertas": "Falsas entre alertas*"}
    log("| Zona | Modelo | " + " | ".join(nombres.values()) + " |")
    log("|" + "---|" * (2 + len(nombres)))
    for zona in retenidas + ["media_retenidas"]:
        for modelo in ("azar", "persistencia", "linea_base_rf"):
            if (modelo, zona) not in resumen.index:
                continue
            fila = resumen.loc[(modelo, zona)]
            celdas_txt = []
            for m in nombres:
                valor, desv = fila[(m, "mean")], fila[(m, "std")]
                if m == "pr_auc":
                    celdas_txt.append(f"{valor:.3f}" + ("" if pd.isna(desv) or desv == 0 else f" ± {desv:.3f}"))
                else:
                    celdas_txt.append(pct_ms(valor, desv))
            log(f"| {zona} | {modelo} | " + " | ".join(celdas_txt) + " |")
    log("* informativas: no deciden nada.")
    log("Azar: PR-AUC y top-k = proporción de alertas de la zona (valor esperado); no tiene recall/FPR.")
    log("")
    log(f"Corridas: {SALIDA / f'{prefijo}linea_base_corridas.csv'}")
    log(f"Resumen:  {SALIDA / f'{prefijo}linea_base_resumen.csv'}")
    log(f"Duración total: {(time.perf_counter() - inicio) / 60:.1f} min")
    log("")


if __name__ == "__main__":
    main()
