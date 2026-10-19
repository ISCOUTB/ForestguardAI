"""Aplicación web de ForestGuard-Agent (evidencia de funcionamiento, Opción A).

Ejecuta el agente real del repositorio sobre los recortes de Hansen de la zona piloto:
cada corrida empieza desde cero en una carpeta temporal (sin tabla de celdas, sin memoria ni modelo).

    streamlit run streamlit_app.py
"""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from src.forestguard.agent import ForestGuardAgent  # noqa: E402
from src.forestguard.features import CAPAS, CELDA_PX, construir_panel  # noqa: E402

import pruebas_desempeno as pruebas  # noqa: E402

ZONA = "guaviare"
DATOS = RAIZ / "data" / "processed"
CARPETA_PRUEBAS = RAIZ / "outputs" / "pruebas"
REPO_URL = "https://github.com/ISCOUTB/ForestguardAI"
OBJ_SEGUNDOS = 24 * 3600

st.set_page_config(page_title="ForestGuard-Agent", page_icon="🌳", layout="wide")


# ---------------------------------------------------------------------- utilidades
class Consola(io.TextIOBase):
    """Recibe los print() del agente y los muestra en vivo en la página."""

    def __init__(self, caja):
        self.caja = caja
        self.texto = ""

    def write(self, s):
        self.texto += s
        self.caja.code(self.texto or " ", language=None)
        return len(s)


def carpeta_de_trabajo():
    """Carpeta temporal nueva con solo los 3 rásteres recortados: arranque desde cero."""
    raiz = Path(tempfile.mkdtemp(prefix="fg_app_"))
    destino = raiz / "data" / "processed"
    destino.mkdir(parents=True)
    for capa in CAPAS:
        shutil.copy2(DATOS / f"{ZONA}_{capa}.tif", destino)
    return raiz


def eventos_de(agente):
    """Eventos del registro de ejecución (logs/*.jsonl) que corresponden a la última corrida."""
    if not agente.ruta_log.exists():
        return pd.DataFrame()
    filas = [json.loads(l) for l in agente.ruta_log.read_text(encoding="utf-8").splitlines() if l.strip()]
    filas = [f for f in filas if f["ejecucion"] == agente.registro.ejecucion]
    return pd.DataFrame(filas)


def tabla_eventos(df):
    if df.empty:
        st.info("Sin eventos registrados.")
        return
    vista = df[["ts", "fase", "evento"]].copy()
    vista["datos"] = df["datos"].map(lambda d: json.dumps(d, ensure_ascii=False)[:160])
    st.dataframe(vista, hide_index=True, width="stretch")


def ejecutar_con_consola(funcion, titulo):
    with st.status(titulo, expanded=True) as estado:
        consola = Consola(st.empty())
        with contextlib.redirect_stdout(consola):
            inicio = time.perf_counter()
            try:
                resultado = funcion()
            except Exception as error:
                estado.update(label=f"El ciclo se detuvo: {type(error).__name__}", state="error")
                return None, error, time.perf_counter() - inicio, consola.texto
        estado.update(label=f"{titulo}: listo en {time.perf_counter() - inicio:.1f} s", state="complete")
    return resultado, None, time.perf_counter() - inicio, consola.texto


@st.cache_resource(show_spinner="Construyendo la tabla de celdas para las pruebas...")
def panel_para_pruebas():
    return construir_panel(ZONA, procesado=DATOS, guardar=False)


def si_no(ok):
    return "✅ Cumple" if ok else "❌ No cumple"


# ------------------------------------------------------------------- comprobación
if not all((DATOS / f"{ZONA}_{capa}.tif").exists() for capa in CAPAS):
    st.error(
        f"No encuentro los recortes `data/processed/{ZONA}_*.tif`. Genera los recortes con "
        "`descargar_tesela()` y `recortar()` (ver README §3) antes de abrir la aplicación."
    )
    st.stop()

# ---------------------------------------------------------------------- encabezado
st.title("🌳 ForestGuard-Agent")
st.caption(
    "Agente que prioriza las celdas de ~1 km con mayor riesgo de perder bosque el próximo año en una "
    "región piloto de Guaviare (Colombia), a partir de Hansen Global Forest Change v1.12."
)

with st.sidebar:
    st.header("Proyecto")
    st.markdown(
        f"- Repositorio: [{REPO_URL.split('/')[-1]}]({REPO_URL})\n"
        "- Curso de Inteligencia Artificial, Universidad Tecnológica de Bolívar\n"
        "- Zona piloto: bbox (-73.5, 1.5, -72.5, 2.5), 12,321 celdas"
    )
    st.header("Cómo usarla")
    st.markdown(
        "1. **Ejecutar el agente**: escenario típico desde cero.\n"
        "2. **Escenario de falla**: entrada inválida inyectada.\n"
        "3. **Pruebas de desempeño**: tabla oficial y pruebas en vivo."
    )

inicio, tab_agente, tab_falla, tab_pruebas = st.tabs(
    ["Inicio", "1 · Ejecutar el agente", "2 · Escenario de falla", "3 · Pruebas de desempeño"]
)

# ----------------------------------------------------------------------------- inicio
with inicio:
    st.subheader("Qué hace el agente")
    st.markdown(
        "En cada corrida el agente lee los tres rásteres de Hansen (cobertura arbórea 2000, año de pérdida y "
        "máscara de datos), divide la zona en celdas de ~1 km, entrena un Random Forest con división temporal, "
        "estima el riesgo de pérdida de cada celda para el año siguiente, prioriza las K celdas que la autoridad "
        "ambiental debería inspeccionar y genera reportes geolocalizados. Aprende de los resultados de las inspecciones."
    )
    st.table(pd.DataFrame([
        ("Recordar", "recordar()", "Lee el historial de corridas y el feedback de inspecciones"),
        ("Percibir", "percibir()", "Lee los rásteres y calcula las variables de cada celda"),
        ("Aprender", "aprender()", "Evalúa el modelo, elige el umbral de riesgo y entrena incluyendo el feedback"),
        ("Razonar", "razonar()", "Estima el riesgo y lo contrasta con las alertas históricas"),
        ("Decidir", "decidir()", "Prioriza las K celdas y asigna nivel alta / media / baja"),
        ("Actuar", "actuar()", "Genera CSV, GeoJSON, mapa y resumen"),
        ("Memorizar", "memorizar()", "Guarda la corrida y el modelo"),
    ], columns=["Paso", "Método", "Qué hace"]))

    st.subheader("Casos de prueba sugeridos")
    st.table(pd.DataFrame([
        ("Típico", "1 · Ejecutar el agente", "K = 20 → **Ejecutar desde cero**",
         "Ciclo completo en segundos (< 24 h), 20 celdas priorizadas con nivel, mapa y registro"),
        ("Típico (más celdas)", "1 · Ejecutar el agente", "K = 100 → **Ejecutar desde cero**",
         "100 celdas, todas con polígono dentro de la zona"),
        ("Aprendizaje", "1 · Ejecutar el agente", "Después de una corrida → **Simular inspecciones y reentrenar**",
         "La segunda corrida usa las inspecciones (“inspecciones usadas: K”) y el ranking puede cambiar"),
        ("Falla esperada", "2 · Escenario de falla", "F-1 falta la capa lossyear",
         "Error controlado (FileNotFoundError), evento ciclo_interrumpido y ningún reporte"),
        ("Falla esperada", "2 · Escenario de falla", "F-4 archivo corrupto o F-5 zona sin datos",
         "Error controlado y ningún reporte"),
        ("Entorno adverso", "2 · Escenario de falla", "A-1 40 % de la zona sin dato",
         "El agente continúa y no prioriza celdas dentro de la franja sin dato"),
        ("Pruebas", "3 · Pruebas de desempeño", "Escenario N-2024, semilla 42",
         "Reproduce los mismos valores de la bitácora oficial"),
    ], columns=["Tipo", "Pestaña", "Qué hacer", "Resultado esperado"]))
    st.caption(
        "Nota: los datos son anuales; el agente predice la pérdida del año siguiente al último disponible (2025). "
        "Las inspecciones simuladas solo demuestran el mecanismo de aprendizaje."
    )

# --------------------------------------------------------------------- 1 · agente
with tab_agente:
    st.subheader("Escenario típico: ciclo completo desde cero")
    st.markdown(
        "Cada corrida usa una carpeta nueva que solo contiene los tres rásteres recortados: el agente construye "
        "la tabla de celdas, entrena el modelo y genera los reportes en vivo. Tarda alrededor de un minuto."
    )
    k = st.slider("Celdas a priorizar (K)", 5, 100, 20, step=5)
    col_a, col_b = st.columns(2)
    correr = col_a.button("▶ Ejecutar desde cero", type="primary", width="stretch")
    reentrenar = col_b.button(
        "🔁 Simular inspecciones y reentrenar", width="stretch",
        disabled="agente" not in st.session_state,
        help="Inventa resultados de inspección para las celdas priorizadas (solo para demostrar el mecanismo) "
             "y vuelve a ejecutar el ciclo usando esa memoria.",
    )

    if correr:
        anterior = st.session_state.pop("raiz", None)
        if anterior:
            shutil.rmtree(anterior, ignore_errors=True)
        raiz = carpeta_de_trabajo()
        agente = ForestGuardAgent(raiz, zona=ZONA, k=k)
        rutas, error, segundos, texto = ejecutar_con_consola(agente.ejecutar, "Ejecutando el ciclo del agente")
        if error:
            st.error(f"{type(error).__name__}: {error}")
        else:
            st.session_state.update(raiz=raiz, agente=agente, rutas=rutas, segundos=segundos, consola=texto,
                                    corrida="Corrida 1 (sin inspecciones)", anterior=None)
            st.rerun()

    if reentrenar and "agente" in st.session_state:
        previo = st.session_state["agente"]
        agente = ForestGuardAgent(st.session_state["raiz"], zona=ZONA, k=previo.k)

        def ciclo_con_feedback():
            agente.simular_feedback()
            return agente.ejecutar()

        rutas, error, segundos, texto = ejecutar_con_consola(ciclo_con_feedback, "Reentrenando con inspecciones")
        if error:
            st.error(f"{type(error).__name__}: {error}")
        else:
            st.session_state.update(agente=agente, rutas=rutas, segundos=segundos, consola=texto,
                                    corrida="Corrida 2 (con inspecciones simuladas)",
                                    anterior=previo.prioridades[["ranking", "celda_id", "riesgo"]])
            st.rerun()

    if "agente" in st.session_state:
        agente = st.session_state["agente"]
        rutas = st.session_state["rutas"]
        segundos = st.session_state["segundos"]
        st.success(f"{st.session_state['corrida']}: ciclo completo en {segundos:.1f} s")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tiempo de respuesta", f"{segundos:.1f} s")
        m2.metric("Celdas analizadas", f"{len(agente.nuevo):,}")
        m3.metric("Año predicho", agente.anio_pred)
        m4.metric("Umbral de riesgo", f"{agente.umbral_riesgo:.3f}")
        m5.metric("Inspecciones usadas", agente.n_feedback)
        st.caption(f"Medida de tiempo de respuesta (≤ 24 h): {si_no(segundos <= OBJ_SEGUNDOS)}")

        with st.expander("Salida del agente (consola)", expanded=False):
            st.code(st.session_state["consola"], language=None)

        st.markdown(f"#### Celdas priorizadas (top {agente.k})")
        niveles = agente.prioridades["nivel"].value_counts().to_dict()
        st.caption("Niveles: " + ", ".join(f"{n}: {c}" for n, c in niveles.items())
                   + " · alta = riesgo sobre el umbral y con alertas previas; media = sobre el umbral; baja = resto")
        prioridades = pd.read_csv(rutas["csv"])
        if st.session_state.get("anterior") is not None:
            antes = st.session_state["anterior"].rename(columns={"ranking": "ranking_antes", "riesgo": "riesgo_antes"})
            prioridades = prioridades.merge(antes, on="celda_id", how="left")
            nuevas = int(prioridades["ranking_antes"].isna().sum())
            st.info(f"Con las inspecciones, {nuevas} de {len(prioridades)} celdas del top son distintas a las de la corrida anterior.")
        st.dataframe(prioridades, hide_index=True, width="stretch", height=300)

        st.markdown("#### Mapa de prioridades")
        st.iframe(Path(rutas["mapa"]), height=520)

        izq, der = st.columns(2)
        with izq:
            st.markdown("#### Evaluación del modelo (división temporal)")
            st.dataframe(agente.resultados.round(3), hide_index=True, width="stretch")
        with der:
            st.markdown("#### Resumen de la zona")
            r = agente.resumen_zona
            st.markdown(
                f"- Pérdida en {r['anio']}: **{r['perdida_ha']:,.0f} ha**\n"
                f"- Promedio histórico: {r['promedio_historico_ha']:,.0f} ha/año\n"
                f"- Razón frente al histórico: {r['razon']:.2f}×"
            )

        st.markdown("#### Registro de ejecución con marca de tiempo")
        tabla_eventos(eventos_de(agente))

        st.markdown("#### Descargar reportes")
        cols = st.columns(len(rutas) + 1)
        for col, (nombre, ruta) in zip(cols, rutas.items()):
            col.download_button(nombre, Path(ruta).read_bytes(), file_name=Path(ruta).name, width="stretch")
        cols[-1].download_button("registro .jsonl", agente.ruta_log.read_bytes(), file_name=agente.ruta_log.name,
                                 width="stretch")

# ---------------------------------------------------------------------- 2 · falla
with tab_falla:
    st.subheader("Escenario de falla esperada")
    st.markdown(
        "Se copia una esquina de la zona (25 %) a una carpeta temporal, se inyecta el problema elegido y se ejecuta "
        "el **ciclo completo del agente**. Lo correcto es que se detenga con un error controlado, que quede el evento "
        "`ciclo_interrumpido` en el registro y que **no** genere reportes."
    )
    casos = {c["id"]: c for c in pruebas.casos_de_falla(ZONA)}

    def quitar_franja(capas):
        ancho = capas["datamask"].shape[1]
        corte = (int(0.4 * ancho) // CELDA_PX) * CELDA_PX
        capas["datamask"] = capas["datamask"].copy()
        capas["datamask"][:, :corte] = 0
        return capas

    casos["A-1"] = dict(id="A-1", descr="entorno adverso: 40 % de la zona sin dato", mod=quitar_franja,
                        omitir=(), corromper=None)
    eleccion = st.selectbox("Falla a inyectar", list(casos), format_func=lambda i: f"{i} · {casos[i]['descr']}")

    if st.button("⚠ Inyectar y ejecutar el agente", type="primary"):
        caso = casos[eleccion]
        raiz = Path(tempfile.mkdtemp(prefix="fg_falla_app_"))
        try:
            proc = raiz / "data" / "processed"
            pruebas.copiar_zona(DATOS, proc, ZONA, pruebas.FRACCION_FALLAS, modificar=caso["mod"],
                                omitir=caso["omitir"])
            if caso["corromper"]:
                (proc / f"{ZONA}_{caso['corromper']}.tif").write_bytes(b"esto no es un tif")
            agente = ForestGuardAgent(raiz, zona=ZONA)
            rutas, error, segundos, _ = ejecutar_con_consola(agente.ejecutar, f"Ejecutando el agente con {eleccion}")
            reportes = sorted(p.name for p in (raiz / "outputs").glob("*")) if (raiz / "outputs").exists() else []

            if eleccion == "A-1":
                if error:
                    st.error(f"El agente no debía detenerse: {type(error).__name__}: {error}")
                else:
                    limite = int(0.4 * 1000) // CELDA_PX
                    col_min = int(agente.nuevo["col"].min())
                    st.success(f"Manejado: el agente continuó con {len(agente.nuevo):,} celdas válidas. Columna mínima "
                               f"evaluada: {col_min} (la franja sin dato son las columnas < {limite}).")
            elif error is None:
                st.error("El agente aceptó una entrada inválida (esto sería una falla de la prueba).")
            else:
                st.success(f"Falla controlada en {segundos:.1f} s → **{type(error).__name__}**: {str(error)[:200]}")
                st.markdown(f"Reportes generados: **{len(reportes)}** " + ("✅" if not reportes else "❌"))
            st.markdown("#### Registro de ejecución")
            tabla_eventos(eventos_de(agente))
        finally:
            shutil.rmtree(raiz, ignore_errors=True)

# ------------------------------------------------------------------- 3 · pruebas
with tab_pruebas:
    st.subheader("Resultados oficiales de las pruebas")
    ruta_registro = CARPETA_PRUEBAS / f"registro_pruebas_{ZONA}.txt"
    csvs = sorted(CARPETA_PRUEBAS.glob(f"corridas_{ZONA}_*.csv"))
    if ruta_registro.exists():
        texto = ruta_registro.read_text(encoding="utf-8")
        bloque = texto.split("TABLA DE RESULTADOS")[-1]
        cabecera = bloque.splitlines()[0].strip(" ·")
        tabla = "\n".join(l for l in bloque.splitlines() if l.startswith("|"))
        st.caption(f"Bitácora `outputs/pruebas/registro_pruebas_{ZONA}.txt` · {cabecera} · 5 repeticiones por escenario")
        st.markdown(tabla)
        d1, d2 = st.columns(2)
        d1.download_button("Descargar bitácora (.txt)", ruta_registro.read_bytes(), file_name=ruta_registro.name)
        if csvs:
            d2.download_button("Descargar corridas (.csv)", csvs[-1].read_bytes(), file_name=csvs[-1].name)
        with st.expander("Ver la bitácora completa"):
            st.code(texto, language=None)
    else:
        st.info("No se encontró la bitácora oficial en outputs/pruebas/.")

    st.divider()
    st.subheader("Ejecutar una prueba aquí")
    st.markdown(
        "Corre un escenario de backtest con las mismas reglas del script `pruebas_desempeno.py`. Con la misma semilla "
        "debe reproducir los valores de la bitácora oficial."
    )
    escenarios = {e["id"]: e for e in pruebas.definir_escenarios()}
    c1, c2 = st.columns(2)
    esc_id = c1.selectbox("Escenario", list(escenarios), index=5,
                          format_func=lambda i: f"{i} · {escenarios[i]['descr']}")
    semilla = c2.number_input("Semilla del Random Forest", 42, 46, 42)

    if st.button("▶ Ejecutar el escenario", type="primary"):
        panel, meta = panel_para_pruebas()
        with st.spinner("Entrenando y evaluando..."):
            r = pruebas.correr_backtest(panel, escenarios[esc_id], meta["umbral_ha"], int(semilla))
        if r["estado"] != "ok":
            st.warning(f"No evaluable: {r['detalle']}")
        else:
            filas = [
                ("Accuracy", r["accuracy"], f"≥ {pruebas.OBJ_ACCURACY:.0%}", r["accuracy"] >= pruebas.OBJ_ACCURACY),
                ("Recall", r["recall"], f"≥ {pruebas.OBJ_RECALL:.0%}", r["recall"] >= pruebas.OBJ_RECALL),
                ("Falsos positivos (FPR)", r["fpr"], f"< {pruebas.OBJ_FPR:.0%}", r["fpr"] < pruebas.OBJ_FPR),
            ]
            oficial = None
            if csvs:
                todas = pd.read_csv(csvs[-1])
                sel = todas[(todas["escenario"] == esc_id) & (todas["semilla"] == int(semilla))]
                oficial = sel.iloc[0] if len(sel) else None
            col_of = {"Accuracy": "accuracy", "Recall": "recall", "Falsos positivos (FPR)": "fpr"}
            st.table(pd.DataFrame([
                {"Medida": m, "Resultado ahora": f"{v:.1%}", "Umbral": u, "¿Cumple?": si_no(ok),
                 "Bitácora oficial (misma semilla)": f"{oficial[col_of[m]]:.1%}" if oficial is not None else "n/d"}
                for m, v, u, ok in filas
            ]))
            st.markdown(
                f"TP {r['TP']:,} · FP {r['FP']:,} · FN {r['FN']:,} · TN {r['TN']:,} · "
                f"top-20: {r['top_k']:.0%} · accuracy de “nunca alertar”: {r['accuracy_nunca_alertar']:.1%} · "
                f"entrenar y predecir: {r['segundos']:.1f} s"
            )

    st.divider()
    st.subheader("Suite completa abreviada (1 repetición)")
    st.markdown(
        "Ejecuta `scripts/pruebas_desempeno.py --repeticiones 1` en una carpeta temporal y muestra su salida en vivo. "
        "La versión oficial usa 5 repeticiones; esta sirve para ver el proceso completo. Tarda varios minutos."
    )
    con_agente = st.checkbox("Incluir las corridas punta a punta del agente (más lento)", value=False)
    if st.button("▶ Ejecutar la suite"):
        tmp = Path(tempfile.mkdtemp(prefix="fg_suite_"))
        try:
            shutil.copytree(RAIZ / "src", tmp / "src")
            shutil.copytree(RAIZ / "scripts", tmp / "scripts")
            (tmp / "data" / "processed").mkdir(parents=True)
            for capa in CAPAS:
                shutil.copy2(DATOS / f"{ZONA}_{capa}.tif", tmp / "data" / "processed")
            cmd = [sys.executable, "-u", str(tmp / "scripts" / "pruebas_desempeno.py"), "--zona", ZONA,
                   "--repeticiones", "1"] + ([] if con_agente else ["--sin-agente"])
            caja = st.empty()
            salida = ""
            with st.status("Ejecutando las pruebas...", expanded=True) as estado:
                proceso = subprocess.Popen(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for linea in proceso.stdout:
                    salida += linea
                    caja.code(salida[-6000:], language=None)
                proceso.wait()
                estado.update(label="Pruebas terminadas" if proceso.returncode == 0 else "Las pruebas fallaron",
                              state="complete" if proceso.returncode == 0 else "error")
            tabla = "\n".join(l for l in salida.split("TABLA DE RESULTADOS")[-1].splitlines() if l.startswith("|"))
            if tabla:
                st.markdown(tabla)
            st.download_button("Descargar la salida", salida.encode("utf-8"), file_name="pruebas_app.txt")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
