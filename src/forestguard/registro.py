"""Registro (log) de ejecución del agente, con marca de tiempo.

Cada vez que el agente percibe, decide o actúa se escribe una línea en un archivo JSONL
(un JSON por línea) en logs/ejecucion_<zona>.jsonl. Ejemplo de línea:

    {"ts": "2026-10-19T10:15:02.481", "ejecucion": "20261019_101458", "fase": "decision",
     "evento": "celdas_priorizadas", "datos": {"k": 20, "niveles": {"alta": 7, ...}}}

Campos:
    ts          fecha y hora del evento (milisegundos)
    ejecucion   identificador de la corrida (todas las líneas de una misma corrida lo comparten)
    fase        memoria | percepcion | aprendizaje | decision | accion | sistema
    evento      qué pasó
    datos       valores relevantes del evento

El archivo se va acumulando entre corridas: nunca se borra lo anterior.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _a_json(valor):
    """Convierte tipos de NumPy / pandas para que json.dumps los acepte."""
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return None if np.isnan(valor) else float(valor)
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if valor is pd.NA or valor is pd.NaT:
        return None
    if isinstance(valor, Path):
        return str(valor)
    return str(valor)


def _limpiar(valor):
    """Cambia NaN por None (null en JSON) en cualquier nivel de diccionarios y listas."""
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpiar(v) for v in valor]
    if isinstance(valor, (float, np.floating)) and np.isnan(valor):
        return None
    return valor


class Registro:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.ejecucion = datetime.now().strftime("%Y%m%d_%H%M%S")

    def nueva_ejecucion(self):
        """Marca el inicio de una corrida nueva (nuevo identificador)."""
        self.ejecucion = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        return self.ejecucion

    def evento(self, fase, evento, **datos):
        linea = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "ejecucion": self.ejecucion,
            "fase": fase,
            "evento": evento,
            "datos": _limpiar(datos),
        }
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(linea, ensure_ascii=False, default=_a_json) + "\n")
        return linea
