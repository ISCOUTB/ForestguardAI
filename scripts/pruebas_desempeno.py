"""Pruebas de desempeño de ForestGuard-Agent (Fase 3).

Convierte las medidas SMART en pruebas con resultado numérico y guarda en un archivo de texto todo lo
que pide la tabla de resultados. Cada ejecución AGREGA un bloque nuevo al final de la bitácora; no se
borra lo anterior.

    python scripts/pruebas_desempeno.py --zona guaviare
    python scripts/pruebas_desempeno.py --zona guaviare --repeticiones 1 --sin-agente   # prueba rápida

Requisitos: haber ejecutado antes los notebooks o el agente, para que exista data/processed/<zona>_*.tif.

Salidas en outputs/pruebas/:
    registro_pruebas_<zona>.txt     bitácora acumulada: reglas, escenarios, resultados y tabla final
    corridas_<zona>_<fecha>.csv     una fila por corrida individual (escenario x repetición)
    evidencia_<fecha>/              reporte real que generó el agente en la primera corrida (CSV, GeoJSON, mapa)

IMPORTANTE: los umbrales de aceptación están escritos más abajo, en REGLAS FIJADAS. Defínelos ANTES de la
primera ejecución y no los cambies después de ver los resultados. El script guarda un hash de sí mismo en
cada bloque de la bitácora, así que un cambio posterior queda registrado.
"""
import argparse
import contextlib
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import sklearn
from rasterio.errors import RasterioError
from rasterio.windows import Window

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # para importar verificar_con_v113

from src.forestguard.agent import ForestGuardAgent  # noqa: E402
from src.forestguard.features import (  # noqa: E402
    BOSQUE_MIN,
    CAPAS,
    CELDA_PX,
    cargar_panel,
    construir_panel,
    features_ultimo_anio,
    leer_capas,
)
from src.forestguard.model import K, elegir_umbral, entrenar, precision_top_k, predecir_riesgo  # noqa: E402
from verificar_con_v113 import metricas_binarias, umbral_para_recall  # noqa: E402

# =====================================================================================================
# REGLAS FIJADAS ANTES DE EJECUTAR LAS PRUEBAS
# =====================================================================================================
OBJ_ACCURACY = 0.85          # accuracy >= 85 %
OBJ_RECALL = 0.85            # recall >= 85 %
OBJ_FPR = 0.15               # falsos positivos / celdas sin pérdida real < 15 %
OBJ_SEGUNDOS = 24 * 3600     # tiempo de respuesta <= 24 h
OBJ_GEO = 1.0                # 100 % de las alertas con coordenadas válidas
OBJ_FALLAS = 1.0             # 100 % de las fallas inyectadas se manejan sin generar reportes

# Punto de operación: el umbral de riesgo se elige SOLO con validación, como el mayor umbral que logra este recall
RECALL_OPERACION = 0.85

# Escenarios normales: predecir cada año Y entrenando con años <= Y-3 y validando con Y-2 y Y-1
ANIOS_NORMALES = [2019, 2020, 2021, 2022, 2023, 2024]

# Escenarios de punta a punta del agente: (id, fracción de la zona, celdas a priorizar)
ESCENARIOS_E2E = [
    ("E-completa-k20", 1.0, 20),
    ("E-completa-k100", 1.0, 100),
    ("E-mitad-k20", 0.5, 20),
    ("E-cuarto-k20", 0.25, 20),
]

FRACCION_FALLAS = 0.25       # tamaño de la zona recortada para las pruebas de falla (para que sean rápidas)
# =====================================================================================================


class Bitacora:
    """Imprime en pantalla y agrega al archivo de texto."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, texto=""):
        print(texto)
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(texto + "\n")


def estadisticas(valores):
    v = np.array([x for x in valores if not pd.isna(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0), len(v)


def pct(x):
    return "n/d" if pd.isna(x) else f"{x * 100:.1f}%"


def pct_ms(valores):
    m, s, n = estadisticas(valores)
    return f"{pct(m)} ± {s * 100:.1f} (n={n})" if n else "sin datos"


# ------------------------------------------------------------------ zonas de prueba
def copiar_zona(origen, destino, zona, fraccion=1.0, modificar=None, omitir=()):
    """Copia las 3 capas de la zona (o su esquina superior izquierda) a otra carpeta.

    `modificar(capas)` recibe y devuelve un dict {capa: matriz}, para inyectar problemas.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    capas, perfil, ventana = {}, None, None
    for capa in CAPAS:
        with rasterio.open(Path(origen) / f"{zona}_{capa}.tif") as src:
            if perfil is None:
                alto, ancho = int(src.height * fraccion), int(src.width * fraccion)
                ventana = Window(0, 0, ancho, alto)
                perfil = src.profile.copy()
                perfil.update(height=alto, width=ancho, transform=src.window_transform(ventana))
            capas[capa] = src.read(1, window=ventana)
    if modificar:
        capas = modificar(capas)
    for capa, datos in capas.items():
        if capa in omitir:
            continue
        p = perfil.copy()
        p.update(height=datos.shape[0], width=datos.shape[1])
        with rasterio.open(destino / f"{zona}_{capa}.tif", "w", **p) as dst:
            dst.write(datos, 1)


# ------------------------------------------------------------- escenarios de backtest
def definir_escenarios():
    esc = []
    for y in ANIOS_NORMALES:
        esc.append(dict(id=f"N-{y}", tipo="normal", anio=y, umbral_ha=None, ventana=None,
                        descr=f"predecir {y} (entrena <= {y - 3}, valida {y - 2}-{y - 1})"))
    esc.append(dict(id="L-1ha", tipo="limite", anio=2024, umbral_ha=1.0, ventana=None,
                    descr="alerta >= 1 ha: más celdas positivas (predecir 2024)"))
    esc.append(dict(id="L-5ha", tipo="limite", anio=2024, umbral_ha=5.0, ventana=None,
                    descr="alerta >= 5 ha: pocas alertas, mayor desbalance (predecir 2024)"))
    esc.append(dict(id="L-5anios", tipo="limite", anio=2024, umbral_ha=None, ventana=5,
                    descr="solo 5 años de entrenamiento (2017-2021) para predecir 2024"))
    return esc


def correr_backtest(panel, esc, umbral_base, semilla):
    u = esc["umbral_ha"] or umbral_base
    p = panel.assign(alerta=(panel["perdida_ha_siguiente"] >= u).astype(int))
    y = esc["anio"]
    desde = y - 3 - (esc["ventana"] - 1) if esc["ventana"] else -np.inf

    train = p[(p["anio_pred"] <= y - 3) & (p["anio_pred"] >= desde)]
    val = p[p["anio_pred"].isin([y - 2, y - 1])]
    test = p[p["anio_pred"] == y]
    if min(len(train), len(val), len(test)) == 0 or min(train["alerta"].sum(), val["alerta"].sum(), test["alerta"].sum()) == 0:
        return {"estado": "no_evaluable", "detalle": "algún conjunto no tiene alertas"}

    t0 = time.perf_counter()
    modelo = entrenar(train, semilla=semilla)
    s_val = predecir_riesgo(modelo, val)
    s_test = predecir_riesgo(modelo, test)
    u_rec = umbral_para_recall(val["alerta"].values, s_val, RECALL_OPERACION)
    u_f1 = elegir_umbral(val["alerta"].values, s_val)
    segundos = time.perf_counter() - t0

    y_t = test["alerta"].values
    m = metricas_binarias(y_t, (s_test >= u_rec).astype(int))
    m_f1 = metricas_binarias(y_t, (s_test >= u_f1).astype(int))
    return {
        "estado": "ok", "detalle": "",
        **m,
        "umbral_riesgo": u_rec,
        "acc_f1": m_f1["accuracy"], "recall_f1": m_f1["recall"], "fpr_f1": m_f1["fpr"],
        "accuracy_nunca_alertar": 1 - y_t.mean(),
        "top_k": precision_top_k(test.assign(riesgo=s_test), "riesgo", K),
        "n_train": len(train), "n_test": len(test), "alertas_test": int(y_t.sum()),
        "segundos": segundos,
    }


# ----------------------------------------------------------------------- punta a punta
def validar_geojson(ruta, limites, tol=1e-6):
    """Cuenta las alertas con polígono cerrado dentro de la zona y riesgo entre 0 y 1."""
    izq, abajo, der, arriba = limites
    features = json.loads(Path(ruta).read_text(encoding="utf-8"))["features"]
    validas = 0
    for f in features:
        try:
            ok = f["geometry"]["type"] == "Polygon" and 0 <= f["properties"]["riesgo"] <= 1
            anillo = f["geometry"]["coordinates"][0]
            ok = ok and anillo[0] == anillo[-1]
            for lon, lat in anillo:
                ok = ok and (izq - tol <= lon <= der + tol) and (abajo - tol <= lat <= arriba + tol)
        except Exception:
            ok = False
        validas += bool(ok)
    return len(features), validas


def correr_agente(zona, procesado, fraccion, k, evidencia=None):
    """Ejecuta el agente completo en una carpeta temporal (arranque en frío: construye el panel)."""
    raiz = Path(tempfile.mkdtemp(prefix="fg_e2e_"))
    try:
        proc = raiz / "data" / "processed"
        copiar_zona(procesado, proc, zona, fraccion)
        with rasterio.open(proc / f"{zona}_lossyear.tif") as r:
            limites = tuple(r.bounds)

        agente = ForestGuardAgent(raiz, zona=zona, k=k)
        salida = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(salida):
            rutas = agente.ejecutar()
        segundos = time.perf_counter() - t0

        n, validas = validar_geojson(rutas["geojson"], limites)
        if evidencia is not None:
            Path(evidencia).mkdir(parents=True, exist_ok=True)
            for ruta in rutas.values():
                shutil.copy2(ruta, Path(evidencia) / Path(ruta).name)
        return {"estado": "ok", "detalle": "", "segundos": segundos, "alertas": n,
                "alertas_validas": validas, "log": salida.getvalue()}
    except Exception as e:
        return {"estado": "error", "detalle": f"{type(e).__name__}: {e}", "log": ""}
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


# ------------------------------------------------------------------- fallas esperadas
def casos_de_falla(zona):
    def ejecutar_agente(raiz):
        with contextlib.redirect_stdout(io.StringIO()):
            ForestGuardAgent(raiz, zona=zona).ejecutar()

    def tamanos_distintos(capas):
        capas["treecover2000"] = capas["treecover2000"][:-50, :]
        return capas

    def fuera_de_rango(capas):
        bosque = (capas["treecover2000"] >= BOSQUE_MIN) & (capas["datamask"] == 1)
        idx = np.flatnonzero(bosque.ravel())[:50]
        ly = capas["lossyear"].copy()
        ly.reshape(-1)[idx] = 99
        capas["lossyear"] = ly
        return capas

    def sin_datos(capas):
        capas["datamask"] = np.zeros_like(capas["datamask"])
        return capas

    def leer(raiz):
        leer_capas(zona, raiz / "data" / "processed")

    def panel(raiz):
        construir_panel(zona, procesado=raiz / "data" / "processed", guardar=False)

    return [
        dict(id="F-1", descr="falta la capa lossyear", mod=None, omitir=("lossyear",), corromper=None,
             ejecutar=leer, esperadas=(FileNotFoundError,)),
        dict(id="F-2", descr="capas con tamaños distintos", mod=tamanos_distintos, omitir=(), corromper=None,
             ejecutar=leer, esperadas=(ValueError,)),
        dict(id="F-3", descr="lossyear con valores fuera de rango (99)", mod=fuera_de_rango, omitir=(), corromper=None,
             ejecutar=panel, esperadas=(ValueError,)),
        dict(id="F-4", descr="archivo lossyear corrupto", mod=None, omitir=(), corromper="lossyear",
             ejecutar=leer, esperadas=(OSError, RasterioError)),
        dict(id="F-5", descr="zona sin datos (datamask = 0 en todo)", mod=sin_datos, omitir=(), corromper=None,
             ejecutar=ejecutar_agente, esperadas=(ValueError,)),
    ]


def correr_falla(zona, procesado, caso):
    raiz = Path(tempfile.mkdtemp(prefix="fg_falla_"))
    try:
        proc = raiz / "data" / "processed"
        copiar_zona(procesado, proc, zona, FRACCION_FALLAS, modificar=caso["mod"], omitir=caso["omitir"])
        if caso["corromper"]:
            (proc / f"{zona}_{caso['corromper']}.tif").write_bytes(b"esto no es un tif")
        try:
            caso["ejecutar"](raiz)
        except caso["esperadas"] as e:
            reportes = list((raiz / "outputs").glob("*")) if (raiz / "outputs").exists() else []
            if reportes:
                return False, f"falló como se esperaba ({type(e).__name__}) pero generó reportes"
            return True, f"{type(e).__name__}: {str(e)[:110]}"
        except Exception as e:
            return False, f"error inesperado {type(e).__name__}: {str(e)[:110]}"
        return False, "no falló: aceptó una entrada inválida"
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


def correr_sin_datos_parcial(zona, procesado):
    """Entorno adverso: 40 % de la zona sin dato. El agente no debe producir celdas en esa franja."""
    raiz = Path(tempfile.mkdtemp(prefix="fg_sindato_"))
    try:
        proc = raiz / "data" / "processed"

        def modificar(capas):
            ancho = capas["datamask"].shape[1]
            c = (int(0.4 * ancho) // CELDA_PX) * CELDA_PX
            capas["datamask"] = capas["datamask"].copy()
            capas["datamask"][:, :c] = 0
            modificar.corte = c
            return capas

        copiar_zona(procesado, proc, zona, FRACCION_FALLAS, modificar=modificar)
        corte = modificar.corte
        construir_panel(zona, procesado=proc, guardar=False)
        nuevo = features_ultimo_anio(zona, procesado=proc)
        if nuevo.empty:
            return False, "no devolvió ninguna celda"
        ok = int(nuevo["col"].min()) >= corte // CELDA_PX
        return ok, f"{len(nuevo)} celdas, columna mínima {int(nuevo['col'].min())} (franja sin dato: columnas < {corte // CELDA_PX})"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:110]}"
    finally:
        shutil.rmtree(raiz, ignore_errors=True)


# ----------------------------------------------------------------------------- tabla
def fila_tabla(*celdas):
    return "| " + " | ".join(str(c) for c in celdas) + " |"


def main():
    parser = argparse.ArgumentParser(description="Pruebas de desempeño de ForestGuard-Agent (Fase 3)")
    parser.add_argument("--zona", default="guaviare")
    parser.add_argument("--repeticiones", type=int, default=5, help="repeticiones por escenario (mínimo 5 para entregar)")
    parser.add_argument("--semilla-base", type=int, default=42)
    parser.add_argument("--sin-agente", action="store_true", help="omite las corridas punta a punta (más rápido)")
    args = parser.parse_args()

    zona, R = args.zona, args.repeticiones
    semillas = [args.semilla_base + i for i in range(R)]
    procesado = RAIZ / "data" / "processed"
    if not (procesado / f"{zona}_lossyear.tif").exists():
        sys.exit(f"No encuentro data/processed/{zona}_*.tif. Ejecuta antes descargar_tesela() y recortar().")

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    carpeta = RAIZ / "outputs" / "pruebas"
    log = Bitacora(carpeta / f"registro_pruebas_{zona}.txt")
    ruta_csv = carpeta / f"corridas_{zona}_{marca}.csv"
    carpeta_evidencia = carpeta / f"evidencia_{marca}"
    hash_script = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    inicio_total = time.perf_counter()

    log("=" * 100)
    log(f"PRUEBAS DE DESEMPEÑO · ForestGuard-Agent · zona {zona}")
    log(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   |   ID de ejecución: {marca}")
    log("=" * 100)
    log(f"Entorno: Python {platform.python_version()} · numpy {np.__version__} · pandas {pd.__version__} · "
        f"scikit-learn {sklearn.__version__} · rasterio {rasterio.__version__} · {platform.platform()}")
    log(f"Script: pruebas_desempeno.py (sha256 {hash_script}). Si este hash cambia entre ejecuciones, cambió algo del script.")
    log(f"Repeticiones por escenario: {R} (semillas del Random Forest: {semillas[0]}..{semillas[-1]}). Los datos son fijos: "
        "la única aleatoriedad es la del modelo; la desviación estándar es muestral (n-1).")
    log("")
    log("REGLAS FIJADAS ANTES DE EJECUTAR (constantes al inicio del script)")
    log(f"  Accuracy                 = (TP+TN)/(TP+TN+FP+FN)                      umbral >= {OBJ_ACCURACY:.0%}")
    log(f"  Recall                   = TP/(TP+FN)                                 umbral >= {OBJ_RECALL:.0%}")
    log(f"  Falsos positivos (FPR)   = FP/(FP+TN)  (informativo: FP/(FP+TP))      umbral <  {OBJ_FPR:.0%}")
    log(f"  Tiempo de respuesta      = segundos de agente.ejecutar(), desde los ráster recortados hasta el reporte   umbral <= {OBJ_SEGUNDOS // 3600} h")
    log(f"  Alertas geolocalizadas   = alertas con polígono cerrado dentro de la zona y riesgo en [0,1] / alertas generadas   umbral = {OBJ_GEO:.0%}")
    log(f"  Manejo de fallas         = fallas inyectadas que terminan en error controlado y sin reportes / fallas inyectadas   umbral = {OBJ_FALLAS:.0%}")
    log(f"  Unidad de evaluación: celda de ~1 km. Alerta real = pérdida de bosque >= umbral de hectáreas del panel en el año predicho.")
    log(f"  Punto de operación: umbral de riesgo = el mayor que logra recall >= {RECALL_OPERACION:.0%} en VALIDACIÓN (nunca en el año de prueba).")
    log("  Criterio de aprobación: la media sobre los escenarios NORMALES (todas sus repeticiones) cumple el umbral. "
        "Los escenarios LÍMITE se reportan aparte; los de FALLA se evalúan en la fila de manejo de fallas.")
    log("")

    # ---- panel y parámetros
    if (procesado / f"panel_{zona}.csv").exists():
        panel, meta_celdas = cargar_panel(zona, procesado)
    else:
        log("No hay panel guardado: construyéndolo...")
        panel, meta_celdas = construir_panel(zona, procesado=procesado)
    umbral_base = meta_celdas["umbral_ha"]
    log(f"Panel: {len(panel):,} filas | alerta = pérdida >= {umbral_base} ha en el año siguiente | zona de {len(panel['celda_id'].unique()):,} celdas con bosque")

    filas = []

    # ---- 1) backtests temporales (accuracy, recall, FPR)
    log("")
    log("-" * 100)
    log("1) ESCENARIOS DE BACKTEST (accuracy, recall, FPR; tiempo de entrenamiento)")
    log("-" * 100)
    for esc in definir_escenarios():
        log(f"\n[{esc['id']}] ({esc['tipo']}) {esc['descr']}")
        rs = []
        for s in semillas:
            r = correr_backtest(panel, esc, umbral_base, s)
            r.update(prueba="backtest", escenario=esc["id"], tipo=esc["tipo"], semilla=s)
            rs.append(r)
            filas.append(r)
        ok = [r for r in rs if r["estado"] == "ok"]
        if not ok:
            log(f"  NO EVALUABLE: {rs[0]['detalle']}")
            continue
        log(f"  celdas de prueba: {ok[0]['n_test']:,} | alertas reales: {ok[0]['alertas_test']} | entrenamiento: {ok[0]['n_train']:,} filas")
        log(f"  accuracy {pct_ms([r['accuracy'] for r in ok])} | recall {pct_ms([r['recall'] for r in ok])} | "
            f"FPR {pct_ms([r['fpr'] for r in ok])} | falsas entre alertas {pct_ms([r['falsas_entre_alertas'] for r in ok])}")
        log(f"  referencia 'nunca alertar': accuracy {pct(ok[0]['accuracy_nunca_alertar'])} | "
            f"top-{K} precisión {pct_ms([r['top_k'] for r in ok])} | tiempo entrenar+predecir "
            f"{np.mean([r['segundos'] for r in ok]):.1f} s")
        log(f"  (con el umbral de mejor F1, que usa el agente hoy: accuracy {pct(np.mean([r['acc_f1'] for r in ok]))} | "
            f"recall {pct(np.mean([r['recall_f1'] for r in ok]))} | FPR {pct(np.mean([r['fpr_f1'] for r in ok]))})")

    # ---- 2) punta a punta
    log("")
    log("-" * 100)
    log("2) ESCENARIOS PUNTA A PUNTA DEL AGENTE (tiempo de respuesta, geolocalización)")
    log("-" * 100)
    primer_log = None
    if args.sin_agente:
        log("Omitido por --sin-agente.")
    else:
        for id_esc, fraccion, k in ESCENARIOS_E2E:
            log(f"\n[{id_esc}] zona al {fraccion:.0%}, top {k}, arranque en frío (construye el panel)")
            rs = []
            for i in range(R):
                destino_evidencia = carpeta_evidencia if (primer_log is None and i == 0) else None
                r = correr_agente(zona, procesado, fraccion, k, destino_evidencia)
                if r["estado"] == "ok" and primer_log is None:
                    primer_log = (id_esc, r["log"])
                r.pop("log", None)
                r.update(prueba="e2e", escenario=id_esc, tipo="normal", semilla=i)
                rs.append(r)
                filas.append(r)
                estado = f"{r['segundos']:.1f} s, {r['alertas_validas']}/{r['alertas']} alertas válidas" if r["estado"] == "ok" else f"ERROR {r['detalle']}"
                log(f"  rep {i + 1}/{R}: {estado}")
            ok = [r for r in rs if r["estado"] == "ok"]
            if ok:
                m, s, n = estadisticas([r["segundos"] for r in ok])
                log(f"  tiempo: media {m:.1f} s ± {s:.1f} (n={n}) | máx {max(r['segundos'] for r in ok):.1f} s")

        if primer_log:
            log(f"\nEvidencia: salida del agente en la primera corrida ({primer_log[0]})")
            for linea in primer_log[1].strip().splitlines():
                log(f"  | {linea}")
            log(f"Reportes de esa corrida copiados a: {carpeta_evidencia}")

    # ---- 3) fallas y entorno adverso
    log("")
    log("-" * 100)
    log("3) FALLAS ESPERADAS Y ENTORNO ADVERSO")
    log("-" * 100)
    fallas = []
    for caso in casos_de_falla(zona):
        manejado, detalle = correr_falla(zona, procesado, caso)
        fallas.append(manejado)
        filas.append(dict(prueba="falla", escenario=caso["id"], tipo="falla", estado="ok" if manejado else "no_manejada",
                          detalle=detalle, semilla=0))
        log(f"[{caso['id']}] {caso['descr']}: {'MANEJADA' if manejado else 'NO MANEJADA'} -> {detalle}")
    manejado, detalle = correr_sin_datos_parcial(zona, procesado)
    fallas.append(manejado)
    filas.append(dict(prueba="falla", escenario="A-1", tipo="falla", estado="ok" if manejado else "no_manejada",
                      detalle=detalle, semilla=0))
    log(f"[A-1] entorno adverso: 40% de la zona sin dato: {'MANEJADA' if manejado else 'NO MANEJADA'} -> {detalle}")

    # ---- tabla de resultados
    df = pd.DataFrame(filas)
    df.to_csv(ruta_csv, index=False)

    bt = df[(df["prueba"] == "backtest") & (df["estado"] == "ok")]
    normales, limites = bt[bt["tipo"] == "normal"], bt[bt["tipo"] == "limite"]
    n_esc_norm, n_esc_lim = normales["escenario"].nunique(), limites["escenario"].nunique()
    e2e = df[(df["prueba"] == "e2e")]
    e2e_ok = e2e[e2e["estado"] == "ok"]

    def evaluar(col, obj, mayor):
        m, s, n = estadisticas(normales[col]) if len(normales) else (float("nan"), float("nan"), 0)
        por_esc = normales.groupby("escenario")[col].mean() if len(normales) else pd.Series(dtype=float)
        cumplen = int((por_esc >= obj).sum() if mayor else (por_esc < obj).sum())
        if pd.isna(m):
            return "sin datos", "n/d"
        ok = (m >= obj) if mayor else (m < obj)
        texto = (f"Normales: {pct(m)} ± {s * 100:.1f} (n={n}); {cumplen}/{len(por_esc)} escenarios cumplen. "
                 f"Límite: {pct_ms(limites[col]) if len(limites) else 'sin datos'}")
        return texto, "Sí" if ok else "No"

    casos_bt = (f"Normales: {n_esc_norm} escenarios × {R} rep.; límite: {n_esc_lim} × {R}; "
                f"falla esperada: {len(fallas)} (ver fila de fallas)")
    evid = f"{log.ruta.name} (ejecución {marca}); {ruta_csv.name}; video: min ___ (completar)"

    r_acc, c_acc = evaluar("accuracy", OBJ_ACCURACY, True)
    r_rec, c_rec = evaluar("recall", OBJ_RECALL, True)
    r_fpr, c_fpr = evaluar("fpr", OBJ_FPR, False)

    if len(e2e_ok):
        m, s, n = estadisticas(e2e_ok["segundos"])
        mx = float(e2e_ok["segundos"].max())
        r_t = f"media {m:.1f} s ± {s:.1f} (n={n}); máx {mx:.1f} s; {len(e2e) - len(e2e_ok)} corridas con error"
        c_t = "Sí" if mx <= OBJ_SEGUNDOS else "No"
        casos_t = f"{e2e['escenario'].nunique()} escenarios × {R} rep. punta a punta (+ tiempos de entrenamiento de los backtests en el log)"
        tot, val = int(e2e_ok["alertas"].sum()), int(e2e_ok["alertas_validas"].sum())
        r_g = f"{val}/{tot} alertas válidas ({pct(val / tot) if tot else 'n/d'})"
        c_g = "Sí" if tot and val / tot >= OBJ_GEO else "No"
        casos_g = f"{len(e2e_ok)} reportes generados"
    else:
        r_t = r_g = "no ejecutado o todas las corridas fallaron"
        c_t = c_g = "n/d"
        casos_t = casos_g = "0"

    r_f = f"{sum(fallas)}/{len(fallas)} manejadas"
    c_f = "Sí" if sum(fallas) / len(fallas) >= OBJ_FALLAS else "No"

    log("")
    log("=" * 100)
    log(f"TABLA DE RESULTADOS · ejecución {marca}")
    log("=" * 100)
    log(fila_tabla("Medida de desempeño", "Fórmula", "Umbral", "Casos ejecutados", "Resultado obtenido", "¿Cumple?", "Evidencia"))
    log(fila_tabla(*["---"] * 7))
    log(fila_tabla("Accuracy", "(TP+TN)/(TP+TN+FP+FN)", f">= {OBJ_ACCURACY:.0%}", casos_bt, r_acc, c_acc, evid))
    log(fila_tabla("Recall", "TP/(TP+FN)", f">= {OBJ_RECALL:.0%}", casos_bt, r_rec, c_rec, evid))
    log(fila_tabla("Falsos positivos (FPR)", "FP/(FP+TN)", f"< {OBJ_FPR:.0%}", casos_bt, r_fpr, c_fpr, evid))
    log(fila_tabla("Tiempo de respuesta", "segundos de agente.ejecutar() (ráster recortado -> reporte)", f"<= {OBJ_SEGUNDOS // 3600} h", casos_t, r_t, c_t, evid))
    log(fila_tabla("Alertas geolocalizadas válidas (complementaria)", "alertas válidas / alertas generadas", f"{OBJ_GEO:.0%}", casos_g, r_g, c_g, f"{evid}; {carpeta_evidencia.name}/"))
    log(fila_tabla("Manejo de fallas esperadas (complementaria)", "fallas controladas / fallas inyectadas", f"{OBJ_FALLAS:.0%}", f"{len(fallas)} (5 entradas inválidas + 1 entorno adverso)", r_f, c_f, evid))

    log("")
    log("NOTAS PARA EL INFORME")
    log("- Las etiquetas salen de Hansen (lossyear), no de inspección en campo: se mide qué tan bien el modelo reproduce el "
        "producto satelital, no la deforestación ilegal real.")
    log("- 'Falsos positivos < 15 %' admite dos lecturas (FP/(FP+TN) o FP/(FP+TP)); la bitácora muestra ambas. Indica cuál usaste.")
    log("- Con pocas alertas, la accuracy es alta incluso si nunca se alerta (ver 'nunca alertar' arriba): no basta por sí sola.")
    log("- Los casos de falla y el 40 % sin dato son SIMULADOS (Hansen no tiene sensores propios): inyectan archivos inválidos o vacíos.")
    log("- El tiempo medido es el de procesamiento; no incluye la descarga de datos ni el tiempo que tarde alguien en ejecutar el agente.")
    log("- Una sola zona y datos fijos: la desviación estándar solo refleja la aleatoriedad del modelo, no variación entre regiones.")
    log("")
    log(f"Duración total de las pruebas: {(time.perf_counter() - inicio_total) / 60:.1f} min")
    log(f"Corridas individuales: {ruta_csv}")
    log("")


if __name__ == "__main__":
    main()
