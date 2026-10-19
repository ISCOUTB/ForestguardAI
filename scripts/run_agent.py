"""Ejecuta el agente de principio a fin.

    python scripts/run_agent.py                      # corrida normal
    python scripts/run_agent.py --simular-feedback   # inventa inspecciones (solo para demostrar)
    python scripts/run_agent.py --zona guaviare --k 20
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.forestguard.agent import ForestGuardAgent  # noqa: E402

OBJ_HORAS = 24   # objetivo SMART: reporte dentro de las 24 h posteriores a tener los datos


def main():
    parser = argparse.ArgumentParser(description="ForestGuard-Agent")
    parser.add_argument("--zona", default="guaviare", help="nombre usado en recortar(..., zona=...)")
    parser.add_argument("--k", type=int, default=20, help="cuántas celdas priorizar")
    parser.add_argument(
        "--simular-feedback",
        action="store_true",
        help="escribe inspecciones SIMULADAS en memory/feedback.csv y termina (solo para demostrar)",
    )
    args = parser.parse_args()

    agente = ForestGuardAgent(RAIZ, zona=args.zona, k=args.k)
    if args.simular_feedback:
        agente.simular_feedback()
        print(f"Registro de ejecución: {agente.ruta_log}")
        return

    inicio = time.perf_counter()
    rutas = agente.ejecutar()
    segundos = time.perf_counter() - inicio

    print("\nListo. Archivos generados:")
    for nombre, ruta in rutas.items():
        print(f"  {nombre}: {ruta}")
    print(f"  registro de ejecución: {agente.ruta_log}")

    cumple = segundos <= OBJ_HORAS * 3600
    print(
        f"\nTiempo de ejecución del agente: {segundos:.1f} s ({segundos / 3600:.3f} h) "
        f"-> objetivo <= {OBJ_HORAS} h: {'cumple' if cumple else 'NO cumple'}"
    )


if __name__ == "__main__":
    main()
