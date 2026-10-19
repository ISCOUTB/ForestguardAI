"""Entrenamiento y evaluación del modelo de riesgo de pérdida forestal.

Uso rápido (desde la raíz del repo):

    from src.forestguard.features import cargar_panel
    from src.forestguard.model import entrenar_y_evaluar

    panel, meta = cargar_panel("guaviare")
    res = entrenar_y_evaluar(panel, meta["umbral_ha"])
    res["resultados"]
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

from .features import FEATURES

SEMILLA = 42   # fija el azar para que los resultados se puedan repetir
K = 20         # cuántas celdas puede inspeccionar la autoridad


def separar_conjuntos(panel):
    """Devuelve (train, val, test) usando la columna `conjunto` del panel."""
    train = panel[panel["conjunto"] == "train"]
    val = panel[panel["conjunto"] == "val"]
    test = panel[panel["conjunto"] == "test"]
    for nombre, df in (("train", train), ("val", val), ("test", test)):
        if df["alerta"].sum() == 0:
            raise ValueError(
                f"El conjunto '{nombre}' no tiene alertas. Baja el umbral de hectáreas "
                "(umbral_ha) al construir el panel y vuelve a construirlo."
            )
    return train, val, test


def entrenar(train, features=FEATURES, semilla=SEMILLA):
    """Entrena un Random Forest. `train` puede ser el panel de entrenamiento más el feedback."""
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=semilla,
        n_jobs=-1,
    )
    modelo.fit(train[features], train["alerta"])
    return modelo


def predecir_riesgo(modelo, df, features=FEATURES):
    """Probabilidad de alerta (0 a 1) para cada fila de df."""
    return modelo.predict_proba(df[features])[:, 1]


def elegir_umbral(y_val, score_val):
    """Umbral de riesgo con mejor F1. Se elige solo con validación, nunca con test."""
    p, r, u = precision_recall_curve(y_val, score_val)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(u[np.argmax(f1)])


def metricas(y, score, umbral):
    pred = (score >= umbral).astype(int)
    return {
        "PR-AUC": average_precision_score(y, score),
        "precisión": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def tabla_resultados(val, test, score_val, score_test, umbral_riesgo, umbral_ha):
    """Compara Aleatorio, Persistencia y Random Forest en validación y prueba."""
    resultados = []
    for conjunto, df, s_rf in (("validación", val, score_val), ("prueba", test, score_test)):
        y = df["alerta"].values
        resultados.append({"conjunto": conjunto, "modelo": "Aleatorio", "PR-AUC": y.mean(),
                           "precisión": np.nan, "recall": np.nan})
        resultados.append({"conjunto": conjunto, "modelo": "Persistencia",
                           **metricas(y, df["perdida_ha_ultimo"].values, umbral_ha)})
        resultados.append({"conjunto": conjunto, "modelo": "Random Forest",
                           **metricas(y, s_rf, umbral_riesgo)})
    return pd.DataFrame(resultados)


def precision_top_k(df, columna, k=K):
    """De las k filas con mayor `columna`, qué fracción era alerta real."""
    return df.nlargest(k, columna)["alerta"].mean()


def top_k_por_anio(test, k=K):
    """Precisión en las k celdas prioritarias de cada año de prueba. Requiere la columna `riesgo`."""
    filas = []
    for anio, g in test.groupby("anio_pred"):
        filas.append({
            "anio_pred": anio,
            "random_forest": precision_top_k(g, "riesgo", k),
            "persistencia": precision_top_k(g, "perdida_ha_ultimo", k),
            "azar": g["alerta"].mean(),
        })
    return pd.DataFrame(filas)


def importancias(modelo, features=FEATURES):
    """Importancia de cada variable, ordenada de menor a mayor."""
    return pd.Series(modelo.feature_importances_, index=features).sort_values()


def entrenar_y_evaluar(panel, umbral_ha):
    """Pipeline completo: separar, entrenar, elegir umbral y evaluar.

    Devuelve un diccionario con: modelo, umbral_riesgo, resultados (tabla) y
    test (el conjunto de prueba con la columna `riesgo`).
    """
    train, val, test = separar_conjuntos(panel)
    modelo = entrenar(train)

    score_val = predecir_riesgo(modelo, val)
    score_test = predecir_riesgo(modelo, test)
    umbral_riesgo = elegir_umbral(val["alerta"], score_val)

    return {
        "modelo": modelo,
        "umbral_riesgo": umbral_riesgo,
        "resultados": tabla_resultados(val, test, score_val, score_test, umbral_riesgo, umbral_ha),
        "test": test.assign(riesgo=score_test),
    }


def guardar_modelo(modelo, umbral_riesgo, ruta):
    """Guarda el modelo y su umbral para que el agente lo reutilice."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "umbral_riesgo": umbral_riesgo, "features": FEATURES}, ruta)


def cargar_modelo(ruta):
    """Devuelve el diccionario guardado por guardar_modelo()."""
    return joblib.load(ruta)
