"""Qué variables pesaron en el riesgo de cada celda (explicación local del Random Forest).

Usa la descomposición por caminos de decisión (Saabas): en cada árbol se sigue el camino de la celda
desde la raíz hasta la hoja y, en cada división, la variable usada se lleva el cambio de probabilidad
entre el nodo padre y el hijo. Promediando los árboles se cumple, para cada celda:

    riesgo = base + suma(contribuciones)

donde `base` es la probabilidad media del bosque en la raíz. Una contribución positiva empuja el riesgo
hacia arriba y una negativa hacia abajo. No necesita librerías adicionales (solo scikit-learn).

    from src.forestguard.explicar import contribuciones, factores_principales

    contrib, base = contribuciones(modelo, prioridades)
    prioridades["factores_principales"] = factores_principales(contrib, prioridades)
"""
import numpy as np
import pandas as pd

from .features import FEATURES

NOMBRES = {
    "pct_bosque_2000": "bosque en 2000",
    "pct_bosque_restante": "bosque que queda",
    "perdida_acum_pct": "pérdida acumulada",
    "perdida_ha_ultimo": "pérdida del último año",
    "perdida_ha_ult3": "pérdida de los últimos 3 años",
    "perdida_vecinas_ha_ultimo": "pérdida en celdas vecinas (último año)",
    "perdida_vecinas_ha_ult3": "pérdida en celdas vecinas (3 años)",
}

CORTOS = {
    "pct_bosque_2000": "bosque 2000",
    "pct_bosque_restante": "bosque restante",
    "perdida_acum_pct": "pérdida acumulada",
    "perdida_ha_ultimo": "pérdida último año",
    "perdida_ha_ult3": "pérdida 3 años",
    "perdida_vecinas_ha_ultimo": "vecinas último año",
    "perdida_vecinas_ha_ult3": "vecinas 3 años",
}

UNIDADES = {
    "pct_bosque_2000": "%",
    "pct_bosque_restante": "%",
    "perdida_acum_pct": "%",
    "perdida_ha_ultimo": "ha",
    "perdida_ha_ult3": "ha",
    "perdida_vecinas_ha_ultimo": "ha",
    "perdida_vecinas_ha_ult3": "ha",
}


def _prob_positiva(arbol):
    """Probabilidad de la clase 1 en cada nodo del árbol (normalizada, igual que predict_proba)."""
    valores = arbol.tree_.value[:, 0, :]
    return valores[:, 1] / valores.sum(axis=1)


def contribuciones(modelo, df, features=FEATURES):
    """Devuelve (contribuciones, base).

    contribuciones: DataFrame (filas de df × features) con cuánto sumó cada variable al riesgo.
    base: probabilidad media en la raíz de los árboles (punto de partida antes de mirar la celda).
    """
    X = df[features].to_numpy(dtype=np.float32)
    total = np.zeros((len(X), len(features)))
    base = 0.0
    for arbol in modelo.estimators_:
        prob = _prob_positiva(arbol)
        izq, der = arbol.tree_.children_left, arbol.tree_.children_right
        var, umbral = arbol.tree_.feature, arbol.tree_.threshold
        base += prob[0]
        for i, x in enumerate(X):
            nodo = 0
            while izq[nodo] != -1:
                hijo = izq[nodo] if x[var[nodo]] <= umbral[nodo] else der[nodo]
                total[i, var[nodo]] += prob[hijo] - prob[nodo]
                nodo = hijo
    n = len(modelo.estimators_)
    return pd.DataFrame(total / n, columns=features, index=df.index), base / n


def factores_principales(contrib, df, n=3):
    """Texto corto por celda con las n variables que más subieron el riesgo, con su valor."""
    textos = []
    for idx, fila in contrib.iterrows():
        top = fila[fila > 0].sort_values(ascending=False).head(n)
        partes = [
            f"{NOMBRES.get(v, v)} = {df.at[idx, v]:,.1f} {UNIDADES.get(v, '')} (+{c * 100:.0f} pts)"
            for v, c in top.items()
        ]
        textos.append("; ".join(partes) if partes else "ninguna variable subió el riesgo")
    return textos


def tabla_larga(contrib, df, celda_id):
    """Contribuciones de una celda, ordenadas, listas para un gráfico de barras."""
    idx = df.index[df["celda_id"] == celda_id][0]
    fila = contrib.loc[idx]
    return pd.DataFrame({
        "variable": [NOMBRES.get(v, v) for v in fila.index],
        "corto": [CORTOS.get(v, v) for v in fila.index],
        "valor": [df.at[idx, v] for v in fila.index],
        "unidad": [UNIDADES.get(v, "") for v in fila.index],
        "contribucion_pts": (fila.values * 100).round(1),
    }).sort_values("contribucion_pts", ascending=False, key=abs).reset_index(drop=True)


# ------------------------------------------------------------- en lenguaje simple
def _perdida(v_ultimo, v_ult3, en_top):
    """'X ha el último año (Y ha en 3 años)' según cuáles de las dos variables están entre las que pesan."""
    if "ultimo" in en_top and "ult3" in en_top:
        return f"{v_ultimo:,.0f} ha el último año ({v_ult3:,.0f} ha en 3 años)"
    if "ultimo" in en_top:
        return f"{v_ultimo:,.0f} ha el último año"
    return f"{v_ult3:,.0f} ha en los últimos 3 años"


def frase(contrib, df, idx, n=3):
    """Explicación de una celda en una frase, a partir de las n variables que más subieron su riesgo."""
    fila = contrib.loc[idx]
    top = set(fila[fila > 0].sort_values(ascending=False).head(n).index)
    v = df.loc[idx]
    partes = []
    propia = {k for k in ("ultimo", "ult3") if f"perdida_ha_{k}" in top}
    if propia:
        partes.append("perdió " + _perdida(v["perdida_ha_ultimo"], v["perdida_ha_ult3"], propia))
    vecinas = {k for k in ("ultimo", "ult3") if f"perdida_vecinas_ha_{k}" in top}
    if vecinas:
        partes.append("sus vecinas perdieron en promedio "
                      + _perdida(v["perdida_vecinas_ha_ultimo"], v["perdida_vecinas_ha_ult3"], vecinas))
    if "perdida_acum_pct" in top:
        partes.append(f"ya perdió el {v['perdida_acum_pct']:,.0f} % del bosque que tenía en 2000")
    if "pct_bosque_restante" in top:
        partes.append(f"todavía conserva {v['pct_bosque_restante']:,.0f} % de bosque")
    if "pct_bosque_2000" in top:
        partes.append(f"era {v['pct_bosque_2000']:,.0f} % bosque en 2000")

    texto = ("La celda " + _unir(partes)) if partes else "Ninguna variable por sí sola explica el riesgo"
    n_alertas = int(v.get("n_alertas_hist", 0) or 0)
    if n_alertas:
        ultima = v.get("ultima_alerta")
        texto += (f". Ya tuvo alertas en {n_alertas} año{'s' if n_alertas > 1 else ''}"
                  + (f", la última en {int(ultima)}" if pd.notna(ultima) else ""))
    return texto + "."


def _unir(partes):
    if len(partes) <= 1:
        return "".join(partes)
    return ", ".join(partes[:-1]) + " y " + partes[-1]
