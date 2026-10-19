"""Bucle del agente: recordar, percibir, aprender, razonar, decidir, actuar y memorizar.

Uso rápido (desde la raíz del repo, o mejor con scripts/run_agent.py):

    from src.forestguard.agent import ForestGuardAgent

    agente = ForestGuardAgent(".", zona="guaviare")
    agente.ejecutar()

Cómo se corresponde con el enunciado:
  percibir  -> lee los ráster y calcula el estado actual de cada celda
  recordar  -> lee el historial de corridas y el feedback de inspecciones (memory/)
  aprender  -> entrena el modelo, incorporando el feedback
  razonar   -> estima el riesgo y lo contrasta con las alertas históricas
  decidir   -> prioriza las K celdas y les asigna un nivel
  actuar    -> genera los reportes geolocalizados (outputs/)
  memorizar -> guarda esta corrida en el historial (memory/)

Registro de ejecución: cada paso escribe un evento con marca de tiempo en
logs/ejecucion_<zona>.jsonl (ver registro.py).
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import report
from .features import ANIO_ULTIMO, FEATURES, calcular_cubos, cargar_panel, construir_panel, tabla_features
from .model import K, entrenar, entrenar_y_evaluar, guardar_modelo, predecir_riesgo
from .registro import Registro

COLUMNAS_FEEDBACK = ["zona", "celda_id", "anio_pred", "confirmado"]
PESO_FEEDBACK = 20   # cada inspección confirmada cuenta como 20 filas al entrenar (si no, se pierde entre miles)


class ForestGuardAgent:
    def __init__(self, raiz, zona="guaviare", k=K):
        self.raiz = Path(raiz)
        self.zona = zona
        self.k = k

        self.processed = self.raiz / "data" / "processed"
        self.memoria = self.raiz / "memory"
        self.salidas = self.raiz / "outputs"
        self.ruta_historial = self.memoria / "historial.csv"
        self.ruta_feedback = self.memoria / "feedback.csv"
        self.ruta_log = self.raiz / "logs" / f"ejecucion_{zona}.jsonl"
        self.registro = Registro(self.ruta_log)

        self.historial = pd.DataFrame()
        self.feedback = pd.DataFrame(columns=COLUMNAS_FEEDBACK)

    # ------------------------------------------------------------------ memoria
    @staticmethod
    def _leer_csv(ruta, columnas=None):
        ruta = Path(ruta)
        if not ruta.exists() or ruta.stat().st_size == 0:
            return pd.DataFrame(columns=columnas or [])
        return pd.read_csv(ruta)

    def recordar(self):
        """Lee lo que el agente sabe de corridas anteriores."""
        self.historial = self._leer_csv(self.ruta_historial)
        self.feedback = self._leer_csv(self.ruta_feedback, COLUMNAS_FEEDBACK)
        con_resultado = int(self.feedback["confirmado"].notna().sum()) if len(self.feedback) else 0
        print(f"[recordar] historial: {len(self.historial)} filas | feedback con resultado: {con_resultado}")
        self.registro.evento("memoria", "memoria_leida", filas_historial=len(self.historial),
                             feedback_total=len(self.feedback), feedback_con_resultado=con_resultado)

    def _filas_feedback(self):
        """Convierte las inspecciones confirmadas en filas de entrenamiento (features + alerta)."""
        vacio = pd.DataFrame(columns=[*FEATURES, "alerta"])
        if self.feedback.empty or self.historial.empty:
            return vacio

        fb = self.feedback.dropna(subset=["confirmado"])
        if fb.empty:
            return vacio
        if not fb["confirmado"].isin([0, 1]).all():
            raise ValueError("En feedback.csv la columna 'confirmado' solo puede tener 1 (sí) o 0 (no).")

        # Las features de cada celda se guardaron en el historial cuando el agente la priorizó
        base = self.historial[["zona", "celda_id", "anio_pred", *FEATURES]]
        base = base.drop_duplicates(["zona", "celda_id", "anio_pred"], keep="last")
        filas = fb.merge(base, on=["zona", "celda_id", "anio_pred"], how="inner")
        filas["alerta"] = filas["confirmado"].astype(int)
        return filas[[*FEATURES, "alerta"]]

    # ----------------------------------------------------------------- percibir
    def percibir(self):
        """Lee los ráster y calcula las features de cada celda hasta el último año disponible."""
        if (self.processed / f"panel_{self.zona}.csv").exists():
            self.panel, self.meta = cargar_panel(self.zona, self.processed)
            origen_panel = "guardado"
        else:
            print("[percibir] no hay panel guardado, construyéndolo (tarda un poco)...")
            self.panel, self.meta = construir_panel(self.zona, procesado=self.processed)
            origen_panel = "construido"
        self.umbral_ha = self.meta["umbral_ha"]

        cubos = calcular_cubos(
            self.zona,
            bosque_min=self.meta["bosque_min"],
            celda_px=self.meta["celda_px"],
            procesado=self.processed,
        )
        self.nuevo = tabla_features(cubos, ANIO_ULTIMO, self.umbral_ha).reset_index(drop=True)
        if self.nuevo.empty:
            raise ValueError("No quedan celdas con bosque para evaluar en esta zona.")
        self.anio_obs = int(self.nuevo["anio_obs"].iloc[0])
        self.anio_pred = int(self.nuevo["anio_pred"].iloc[0])

        # Comparación de periodos a nivel de zona: último año frente al promedio histórico
        perdida_anual = pd.Series(
            cubos["perdida"].sum(axis=(1, 2))[1:] * cubos["ha_por_pixel"],
            index=range(2001, 2001 + ANIO_ULTIMO),
        )
        historico = perdida_anual.loc[2001:self.anio_obs - 1].mean()
        self.resumen_zona = {
            "anio": self.anio_obs,
            "perdida_ha": float(perdida_anual.loc[self.anio_obs]),
            "promedio_historico_ha": float(historico),
            "razon": float(perdida_anual.loc[self.anio_obs] / historico) if historico > 0 else float("nan"),
        }
        print(f"[percibir] {len(self.nuevo):,} celdas con bosque | pérdida {self.anio_obs}: "
              f"{self.resumen_zona['perdida_ha']:,.0f} ha ({self.resumen_zona['razon']:.2f}x el promedio histórico)")
        self.registro.evento("percepcion", "rasters_percibidos", zona=self.zona, panel=origen_panel,
                             filas_panel=len(self.panel), celdas_con_bosque=len(self.nuevo),
                             anio_observado=self.anio_obs, anio_a_predecir=self.anio_pred,
                             umbral_ha=self.umbral_ha, **{f"zona_{k}": v for k, v in self.resumen_zona.items()})

    # ----------------------------------------------------------------- aprender
    def aprender(self):
        """Entrena el modelo. Si hay feedback de inspecciones, lo suma a los datos de entrenamiento."""
        # 1) Evaluación honesta (división por tiempo) y elección del umbral de riesgo
        evaluacion = entrenar_y_evaluar(self.panel, self.umbral_ha)
        self.umbral_riesgo = evaluacion["umbral_riesgo"]
        self.resultados = evaluacion["resultados"]

        # 2) Modelo final: todos los años con etiqueta + las inspecciones confirmadas
        fb = self._filas_feedback()
        self.n_feedback = len(fb)
        datos = pd.concat([self.panel, *([fb] * PESO_FEEDBACK)], ignore_index=True) if len(fb) else self.panel
        self.modelo = entrenar(datos)
        print(f"[aprender] umbral de riesgo {self.umbral_riesgo:.3f} | inspecciones usadas: {self.n_feedback}")
        rf_prueba = self.resultados[(self.resultados["conjunto"] == "prueba")
                                    & (self.resultados["modelo"] == "Random Forest")].iloc[0]
        self.registro.evento("aprendizaje", "modelo_entrenado", umbral_riesgo=round(self.umbral_riesgo, 4),
                             inspecciones_usadas=self.n_feedback, filas_entrenamiento=len(datos),
                             prueba_pr_auc=round(float(rf_prueba["PR-AUC"]), 4),
                             prueba_precision=round(float(rf_prueba["precisión"]), 4),
                             prueba_recall=round(float(rf_prueba["recall"]), 4))

    # ------------------------------------------------------------------ razonar
    def razonar(self):
        """Estima el riesgo de cada celda y lo contrasta con sus alertas históricas."""
        self.nuevo["riesgo"] = predecir_riesgo(self.modelo, self.nuevo)

        alertas = (
            self.panel[self.panel["alerta"] == 1]
            .groupby("celda_id")["anio_pred"]
            .agg(n_alertas_hist="count", ultima_alerta="max")
        )
        self.nuevo = self.nuevo.merge(alertas, left_on="celda_id", right_index=True, how="left")
        self.nuevo["n_alertas_hist"] = self.nuevo["n_alertas_hist"].fillna(0).astype(int)
        self.nuevo["ultima_alerta"] = self.nuevo["ultima_alerta"].astype("Int64")
        print(f"[razonar] riesgo estimado para {len(self.nuevo):,} celdas")
        riesgo = self.nuevo["riesgo"]
        self.registro.evento("decision", "riesgo_estimado", celdas=len(self.nuevo),
                             riesgo_medio=round(float(riesgo.mean()), 4),
                             riesgo_maximo=round(float(riesgo.max()), 4),
                             celdas_sobre_umbral=int((riesgo >= self.umbral_riesgo).sum()),
                             celdas_con_alertas_previas=int((self.nuevo["n_alertas_hist"] > 0).sum()))

    # ------------------------------------------------------------------- decidir
    def decidir(self):
        """Se queda con las K celdas de mayor riesgo y les asigna un nivel de prioridad."""
        top = self.nuevo.sort_values("riesgo", ascending=False).head(self.k).copy()
        top["ranking"] = np.arange(1, len(top) + 1)

        sobre_umbral = top["riesgo"] >= self.umbral_riesgo
        reincidente = top["n_alertas_hist"] >= 1
        top["nivel"] = np.select([sobre_umbral & reincidente, sobre_umbral], ["alta", "media"], default="baja")

        self.prioridades = top.reset_index(drop=True)
        niveles = self.prioridades["nivel"].value_counts().to_dict()
        print(f"[decidir] top {len(self.prioridades)} celdas | niveles: {niveles}")
        self.registro.evento(
            "decision", "celdas_priorizadas", k=self.k, umbral_riesgo=round(self.umbral_riesgo, 4),
            niveles=niveles,
            celdas=[{"ranking": int(f.ranking), "celda_id": int(f.celda_id), "lat": round(float(f.lat), 5),
                     "lon": round(float(f.lon), 5), "riesgo": round(float(f.riesgo), 4), "nivel": f.nivel,
                     "n_alertas_hist": int(f.n_alertas_hist)}
                    for f in self.prioridades.itertuples()],
        )

    # --------------------------------------------------------------------- actuar
    def actuar(self):
        """Genera los reportes geolocalizados en outputs/."""
        self.salidas.mkdir(parents=True, exist_ok=True)
        sufijo = f"{self.zona}_{self.anio_pred}"
        rutas = {
            "csv": self.salidas / f"prioridades_{sufijo}.csv",
            "geojson": self.salidas / f"prioridades_{sufijo}.geojson",
            "mapa": self.salidas / f"mapa_prioridades_{sufijo}.html",
            "resumen": self.salidas / f"resumen_{sufijo}.md",
            "plantilla_feedback": self.salidas / f"plantilla_feedback_{sufijo}.csv",
        }
        report.guardar_csv(self.prioridades, rutas["csv"])
        report.guardar_geojson(self.prioridades, self.meta, rutas["geojson"])
        report.crear_mapa(self.prioridades, self.meta, rutas["mapa"])
        report.escribir_resumen(
            rutas["resumen"], self.zona, self.prioridades, self.resumen_zona,
            self.resultados, self.anio_pred, self.n_feedback,
        )
        report.guardar_plantilla_feedback(self.prioridades, self.zona, rutas["plantilla_feedback"])
        print(f"[actuar] reportes guardados en {self.salidas}")
        self.registro.evento("accion", "reportes_generados", celdas_reportadas=len(self.prioridades),
                             archivos={nombre: str(ruta) for nombre, ruta in rutas.items()})
        return rutas

    # ------------------------------------------------------------------ memorizar
    def memorizar(self):
        """Guarda esta corrida en el historial y el modelo entrenado en memory/."""
        self.memoria.mkdir(parents=True, exist_ok=True)

        columnas = ["ejecucion", "zona", "anio_obs", "anio_pred", "celda_id", "lat", "lon",
                    "ranking", "riesgo", "nivel", "n_alertas_hist", *FEATURES]
        corrida = self.prioridades.assign(
            ejecucion=datetime.now().isoformat(timespec="seconds"), zona=self.zona
        )[columnas]
        historial = corrida if self.historial.empty else pd.concat([self.historial, corrida], ignore_index=True)
        historial.to_csv(self.ruta_historial, index=False)

        if not self.ruta_feedback.exists():
            pd.DataFrame(columns=COLUMNAS_FEEDBACK).to_csv(self.ruta_feedback, index=False)

        ruta_modelo = self.memoria / f"modelo_{self.zona}.joblib"
        guardar_modelo(self.modelo, self.umbral_riesgo, ruta_modelo)
        print(f"[memorizar] historial: {len(historial)} filas")
        self.registro.evento("memoria", "corrida_guardada", filas_agregadas=len(corrida),
                             filas_historial=len(historial), modelo=str(ruta_modelo))

    # ------------------------------------------------------------------- ejecutar
    def ejecutar(self):
        """Corre el ciclo completo una vez y devuelve las rutas de los reportes."""
        self.registro.nueva_ejecucion()
        inicio = time.perf_counter()
        self.registro.evento("sistema", "inicio_ciclo", zona=self.zona, k=self.k)
        try:
            self.recordar()
            self.percibir()
            self.aprender()
            self.razonar()
            self.decidir()
            rutas = self.actuar()
            self.memorizar()
        except Exception as error:
            # Las fallas también quedan registradas: el agente se detiene sin generar reportes.
            self.registro.evento("sistema", "ciclo_interrumpido", error=type(error).__name__,
                                 detalle=str(error)[:300], segundos=round(time.perf_counter() - inicio, 3))
            raise
        self.registro.evento("sistema", "fin_ciclo", segundos=round(time.perf_counter() - inicio, 3),
                             anio_predicho=self.anio_pred)
        return rutas

    # ------------------------------------------------------- demostración (simulada)
    def simular_feedback(self, semilla=42):
        """SOLO PARA DEMOSTRAR el mecanismo: inventa resultados de inspección.

        Para cada celda priorizada en la última corrida, sortea confirmado = 1 con probabilidad
        igual a su riesgo. Son datos simulados, no inspecciones reales. Si ya hay feedback real
        en memory/feedback.csv, no lo toca.
        """
        self.registro.nueva_ejecucion()
        self.recordar()
        if self.historial.empty:
            raise RuntimeError("Primero ejecuta el agente al menos una vez (python scripts/run_agent.py).")
        if not self.feedback.empty and self.feedback["confirmado"].notna().any():
            raise RuntimeError("Ya hay resultados en memory/feedback.csv; no los sobrescribo.")

        h = self.historial[self.historial["zona"] == self.zona]
        ultima = h[h["ejecucion"] == h["ejecucion"].max()]
        rng = np.random.default_rng(semilla)
        fb = pd.DataFrame({
            "zona": ultima["zona"].values,
            "celda_id": ultima["celda_id"].values,
            "anio_pred": ultima["anio_pred"].values,
            "confirmado": (rng.random(len(ultima)) < ultima["riesgo"].values).astype(int),
        })
        self.memoria.mkdir(parents=True, exist_ok=True)
        fb.to_csv(self.ruta_feedback, index=False)
        print(f"[simular_feedback] {len(fb)} inspecciones SIMULADAS escritas en {self.ruta_feedback}")
        self.registro.evento("aprendizaje", "feedback_simulado", inspecciones=len(fb),
                             confirmadas=int(fb["confirmado"].sum()), semilla=semilla,
                             archivo=str(self.ruta_feedback))
