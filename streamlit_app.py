"""Aplicación web de ForestGuard-Agent (evidencia de funcionamiento, Opción A).

Ejecuta el agente real del repositorio sobre una zona del Hito 2 o sobre un recuadro que el usuario dibuja
en el mapa: cada corrida empieza desde cero en una carpeta temporal (sin tabla de celdas, sin memoria ni modelo).

    streamlit run streamlit_app.py
"""
import base64
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from src.forestguard.agent import ForestGuardAgent  # noqa: E402
from src.forestguard.explicar import frase, tabla_larga  # noqa: E402
from src.forestguard.features import CAPAS, CELDA_PX, construir_panel  # noqa: E402
from src.forestguard.zonas import (  # noqa: E402
    LADO_MAX,
    LADO_MIN,
    ZONA_RECUADRO,
    RecuadroInvalido,
    recortar_recuadro,
    recorte_disponible,
    tesela_de,
    validar_recuadro,
    zonas_hito2,
)

import pruebas_desempeno as pruebas  # noqa: E402

ZONA_PILOTO = "guaviare"            # zona de las pruebas oficiales de la Fase 3 y de los casos de falla
DATOS = RAIZ / "data" / "processed"
RAW = RAIZ / "data" / "raw"
ASSETS = RAIZ / "assets"
CARPETA_PRUEBAS = RAIZ / "outputs" / "pruebas"
CARPETA_HITO2 = RAIZ / "outputs" / "hito2"
CARPETA_RUTA = RAIZ / "outputs" / "ruta_espacial"
REPO_URL = "https://github.com/ISCOUTB/ForestguardAI"
META_TIEMPO_S = 5 * 60 + 12         # Hito 1: ≤ 5 min 12 s desde un recuadro nuevo hasta el reporte
RECUADRO_EJEMPLO = (-73.2, 1.8, -72.7, 2.3)
NOMBRES_ZONA = {"guaviare": "Guaviare", "macarena": "Macarena", "chaira": "Chairá", "putumayo": "Putumayo"}
RESULTADOS_INSPECCION = {"Sin inspeccionar": None, "✔ Confirmada (hubo pérdida)": 1, "✘ No confirmada": 0}
NIVELES = {"alta": "🔴 Alta", "media": "🟠 Media", "baja": "🔵 Baja"}
TOP_TARJETAS = 5

# Pasos del ciclo: el agente imprime "[paso] ..." al terminar cada uno
PASOS = [
    ("recordar", "Recordar", "Revisando la memoria: corridas anteriores e inspecciones reportadas"),
    ("percibir", "Percibir", "Leyendo las imágenes de Hansen y dividiendo la zona en celdas de ~1 km"),
    ("aprender", "Aprender", "Entrenando el modelo con la historia de pérdida de la zona"),
    ("razonar", "Razonar", "Calculando el riesgo de cada celda"),
    ("decidir", "Decidir", "Eligiendo las celdas que conviene inspeccionar y explicando por qué"),
    ("actuar", "Actuar", "Generando el mapa y los reportes"),
    ("memorizar", "Memorizar", "Guardando la corrida en la memoria del agente"),
]
ORDEN_PASOS = [p[0] for p in PASOS]

st.set_page_config(page_title="ForestGuard-Agent", page_icon=str(ASSETS / "icono.png"), layout="wide")
st.logo(str(ASSETS / "logo.png"), icon_image=str(ASSETS / "icono.png"), size="large", link=REPO_URL)

ZONAS = zonas_hito2()


# ---------------------------------------------------------------------- utilidades
class Progreso(io.TextIOBase):
    """Recibe los print() del agente y los convierte en una barra con los 7 pasos del ciclo."""

    def __init__(self, barra, lista):
        self.barra, self.lista = barra, lista
        self.texto = ""
        self.hechos = 0
        self.inicio = time.perf_counter()
        self._pintar()

    def _pintar(self):
        if self.hechos < len(PASOS):
            _, nombre, que = PASOS[self.hechos]
            etiqueta = f"Paso {self.hechos + 1} de {len(PASOS)} · {nombre}: {que}…"
        else:
            etiqueta = f"Ciclo completo en {time.perf_counter() - self.inicio:.1f} s"
        self.barra.progress(self.hechos / len(PASOS), text=etiqueta)
        self.lista.markdown("  ".join(
            ("✅" if i < self.hechos else "⏳" if i == self.hechos else "⬜") + f" {nombre}"
            for i, (_, nombre, _) in enumerate(PASOS)
        ))

    def write(self, s):
        self.texto += s
        for linea in s.splitlines():
            if linea.startswith("[") and "]" in linea:
                paso = linea[1:linea.index("]")]
                if paso in ORDEN_PASOS:
                    self.hechos = max(self.hechos, ORDEN_PASOS.index(paso) + 1)
                    self._pintar()
        return len(s)


def ejecutar_con_progreso(funcion, titulo):
    """Corre el ciclo del agente mostrando el avance paso a paso. Devuelve (resultado, error, segundos, texto)."""
    with st.status(titulo, expanded=True) as estado:
        progreso = Progreso(st.empty(), st.empty())
        with contextlib.redirect_stdout(progreso):
            inicio = time.perf_counter()
            try:
                resultado = funcion()
            except Exception as error:
                estado.update(label=f"El agente se detuvo: {type(error).__name__}", state="error")
                return None, error, time.perf_counter() - inicio, progreso.texto
        segundos = time.perf_counter() - inicio
        estado.update(label=f"{titulo}: listo en {segundos:.1f} s", state="complete", expanded=False)
    return resultado, None, segundos, progreso.texto


def nombre_zona(zona):
    return NOMBRES_ZONA.get(zona, zona)


def carpeta_con_zona(zona):
    """Carpeta temporal nueva con solo los 3 rásteres recortados de una zona: arranque desde cero."""
    raiz = Path(tempfile.mkdtemp(prefix="fg_app_"))
    destino = raiz / "data" / "processed"
    destino.mkdir(parents=True)
    for capa in CAPAS:
        shutil.copy2(DATOS / f"{zona}_{capa}.tif", destino)
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


# Compatibilidad con la pestaña de fallas (usa el mismo nombre que antes)
ejecutar_con_consola = ejecutar_con_progreso


@st.cache_resource(show_spinner="Construyendo la tabla de celdas para las pruebas...")
def panel_para_pruebas():
    return construir_panel(ZONA_PILOTO, procesado=DATOS, guardar=False)


def si_no(ok):
    return "✅ Cumple" if ok else "❌ No cumple"


def url_mapa(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat:.4f},{lon:.4f}"


def limpiar_corrida():
    anterior = st.session_state.pop("corrida", None)
    if anterior:
        shutil.rmtree(anterior["raiz"], ignore_errors=True)


def prioridades_amigables(agente):
    """Tabla de prioridades con frase, nivel con color y enlace al mapa."""
    p = agente.prioridades.copy()
    p["por_que"] = [frase(agente.contribuciones, p, i) for i in p.index]
    p["prioridad"] = p["nivel"].map(NIVELES)
    p["ubicacion"] = [url_mapa(la, lo) for la, lo in zip(p["lat"], p["lon"])]
    p["riesgo_pct"] = (p["riesgo"] * 100).round(0)
    return p


def ajustar_a_maximo(bbox):
    """Recorta un recuadro demasiado grande a LADO_MAX por lado, manteniendo el centro."""
    lon_min, lat_min, lon_max, lat_max = bbox
    clon, clat = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    a = min(lon_max - lon_min, LADO_MAX) / 2
    b = min(lat_max - lat_min, LADO_MAX) / 2
    return round(clon - a, 3), round(clat - b, 3), round(clon + a, 3), round(clat + b, 3)


def poner_recuadro(bbox):
    for clave, valor in zip(("lon_min", "lat_min", "lon_max", "lat_max"), bbox):
        st.session_state[clave] = float(valor)


def mapa_para_dibujar(bbox_actual):
    """Mapa de Colombia con las zonas del Hito 2 de referencia, el recuadro actual y la herramienta de dibujo."""
    lon_min, lat_min, lon_max, lat_max = bbox_actual
    mapa = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=7,
                      tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satélite",
    ).add_to(mapa)
    for zona, info in ZONAS.items():
        w, s, e, n = info["bbox"]
        folium.Rectangle([[s, w], [n, e]], color="#6B4F3A", weight=1, dash_array="4", fill=False,
                         tooltip=f"{nombre_zona(zona)} ({info['rol']})").add_to(mapa)
    folium.Rectangle([[lat_min, lon_min], [lat_max, lon_max]], color="#2F6B45", weight=3, fill=True,
                     fill_opacity=0.15, tooltip="Recuadro actual").add_to(mapa)
    Draw(
        draw_options={"rectangle": {"shapeOptions": {"color": "#C68A1E", "weight": 3}},
                      "polyline": False, "polygon": False, "circle": False, "marker": False,
                      "circlemarker": False},
        edit_options={"edit": False, "remove": False},
    ).add_to(mapa)
    folium.LayerControl().add_to(mapa)
    return mapa


def limites_de(geojson):
    coords = geojson["geometry"]["coordinates"][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    return round(min(lons), 3), round(min(lats), 3), round(max(lons), 3), round(max(lats), 3)


def banner():
    imagen = ASSETS / "banner_guaviare.jpg"
    fondo = ""
    if imagen.exists():
        datos = base64.b64encode(imagen.read_bytes()).decode()
        fondo = f"background-image:linear-gradient(90deg, rgba(20,45,30,.92) 0%, rgba(20,45,30,.70) 45%, " \
                f"rgba(20,45,30,.05) 100%), url(data:image/jpeg;base64,{datos});"
    st.html(f"""
    <div style="{fondo} background-color:#1E3A2A; background-size:cover; background-position:center;
                border-radius:18px; padding:34px 36px 30px; color:#F3F6EE; margin-bottom:6px;">
      <div style="font-size:13px; letter-spacing:3px; text-transform:uppercase; color:#A9D6A0;">
        Monitoreo de pérdida de bosque · Colombia</div>
      <div style="font-family:Fraunces, Georgia, serif; font-size:44px; font-weight:700; line-height:1.15;
                  margin:6px 0 10px;">ForestGuard-Agent</div>
      <div style="font-size:17px; max-width:620px; line-height:1.5;">
        Te dice <b>a qué celdas de ~1 km ir primero</b> para encontrar pérdida de bosque el próximo año,
        explica por qué y aprende de lo que reportan tus inspecciones.</div>
      <div style="margin-top:16px; font-size:13px; opacity:.85;">
        🛰 Hansen GFC v1.12 &nbsp;·&nbsp; 🌲 Random Forest &nbsp;·&nbsp; 🧭 20 inspecciones por ciclo</div>
    </div>""")
    st.caption("Fondo: cobertura arbórea del año 2000 (verde) y pérdida 2001–2024 (ocre a rojo) en Guaviare. "
               "Datos: Hansen et al. (2013), UMD/Google/USGS/NASA, CC BY 4.0.")


# ------------------------------------------------------------------- comprobación
if not recorte_disponible(ZONA_PILOTO, DATOS):
    st.error(
        f"No encuentro los recortes `data/processed/{ZONA_PILOTO}_*.tif`. Genera los recortes con "
        "`python scripts/preparar_zonas_app.py` (ver README) antes de abrir la aplicación."
    )
    st.stop()

ZONAS_LISTAS = [z for z in ZONAS if recorte_disponible(z, DATOS)]

# ---------------------------------------------------------------------- encabezado
banner()

with st.sidebar:
    st.markdown("### Cómo usarla")
    st.markdown(
        "1. **🌲 Analizar una zona**: elige una zona o dibuja tu recuadro y ejecuta el agente.\n"
        "2. Revisa **dónde inspeccionar primero** y por qué.\n"
        "3. Marca qué encontraste en cada celda y **reentrena**.\n\n"
        "**⚠ Fallas** y **📊 Pruebas** muestran cómo responde el agente ante errores y cuánto acierta."
    )
    st.divider()
    st.markdown(
        f"**Proyecto** · [Repositorio]({REPO_URL})  \n"
        "Inteligencia Artificial · Universidad Tecnológica de Bolívar  \n"
        "Juan Diego Serpa Medina · Jesús David Julio Hawasly"
    )
    st.caption(f"Zonas listas: {', '.join(nombre_zona(z) for z in ZONAS_LISTAS)}")

# Navegación por secciones (en vez de st.tabs: solo se dibuja la sección activa, lo que evita que el mapa
# de dibujo quede con altura 0 dentro de una pestaña oculta y hace la página más rápida)
SECCIONES = ["🏠 Inicio", "🌲 Analizar una zona", "⚠ Escenario de falla", "📊 Pruebas de desempeño"]
if "seccion" not in st.session_state:
    st.session_state["seccion"] = SECCIONES[0]


def ir_a(seccion):
    st.session_state["seccion"] = seccion


seccion = st.segmented_control("Sección", SECCIONES, key="seccion", required=True, label_visibility="collapsed",
                               width="stretch")

# ----------------------------------------------------------------------------- inicio
if seccion == "🏠 Inicio":
    st.subheader("Cómo funciona")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("#### 1 · Elige el lugar")
        st.markdown("Una de las 4 zonas del proyecto o **un recuadro que dibujas en el mapa**, de "
                    f"{LADO_MIN}° a {LADO_MAX:.0f}° por lado (hasta ~110 km).")
    with c2.container(border=True):
        st.markdown("#### 2 · El agente analiza")
        st.markdown("Lee 24 años de imágenes satelitales, aprende cómo avanza la pérdida de bosque y estima el "
                    "riesgo de cada celda de ~1 km para el próximo año.")
    with c3.container(border=True):
        st.markdown("#### 3 · Inspecciona y reporta")
        st.markdown("Recibes las celdas a visitar con su ubicación y el porqué. Lo que reportes de cada visita "
                    "entra al siguiente entrenamiento.")
    st.button("🌲 Empezar: analizar una zona", type="primary", on_click=ir_a, args=("🌲 Analizar una zona",))
    st.caption("Con Guaviare y 20 celdas tarda menos de un minuto.")

    with st.expander("Ciclo del agente (los 7 pasos)"):
        st.table(pd.DataFrame([
            ("Recordar", "recordar()", "Lee el historial de corridas y las inspecciones reportadas"),
            ("Percibir", "percibir()", "Lee los rásteres del recuadro y calcula las variables de cada celda"),
            ("Aprender", "aprender()", "Evalúa el modelo, elige el umbral de riesgo y entrena incluyendo las inspecciones"),
            ("Razonar", "razonar()", "Estima el riesgo y lo contrasta con las alertas históricas"),
            ("Decidir", "decidir()", "Prioriza las K celdas, asigna nivel y explica qué variables pesaron"),
            ("Actuar", "actuar()", "Genera CSV, GeoJSON, mapa y resumen"),
            ("Memorizar", "memorizar()", "Guarda la corrida y el modelo"),
        ], columns=["Paso", "Método", "Qué hace"]))

    with st.expander("Casos de prueba sugeridos", expanded=True):
        st.table(pd.DataFrame([
            ("Típico", "🌲 Analizar", "Zona Guaviare, 20 celdas → **Analizar desde cero**",
             "20 celdas priorizadas con su porqué, mapa y registro; tiempo ≤ 5 min 12 s"),
            ("Zona no vista", "🌲 Analizar", "Zona Chairá o Putumayo (retenidas en el Hito 2)",
             "El agente corre igual en una zona que no se usó para fijar las reglas"),
            ("Recuadro nuevo", "🌲 Analizar", "Recuadro personalizado: dibújalo con ▭ en el mapa",
             "El agente recorta el recuadro de la tesela y corre de punta a punta; se compara el tiempo con la meta"),
            ("Aprendizaje", "🌲 Analizar", "Marcar celdas como ✔ / ✘ → **Guardar inspecciones y reentrenar**",
             "La nueva corrida dice cuántas inspecciones usó y qué celdas cambiaron en el top"),
            ("Falla esperada", "⚠ Fallas", "F-1 falta la capa lossyear",
             "Error controlado (FileNotFoundError), evento ciclo_interrumpido y ningún reporte"),
            ("Recuadro imposible", "⚠ Fallas", "R-1 recuadro fuera de Colombia o R-2 demasiado grande",
             "El agente rechaza el recuadro antes de leer datos y explica por qué"),
            ("Entorno adverso", "⚠ Fallas", "A-1 40 % de la zona sin dato",
             "El agente continúa y no prioriza celdas dentro de la franja sin dato"),
            ("Pruebas", "📊 Pruebas", "Escenario N-2024, semilla 42",
             "Reproduce los mismos valores de la bitácora oficial"),
        ], columns=["Tipo", "Pestaña", "Qué hacer", "Resultado esperado"]))
    st.caption(
        "Los datos son anuales: el agente predice la pérdida del año siguiente al último disponible (2025). "
        "La pérdida de Hansen incluye incendios y plantaciones, no solo deforestación ilegal. Las inspecciones que "
        "se marcan en la app son de demostración."
    )

# --------------------------------------------------------------------- 1 · agente
if seccion == "🌲 Analizar una zona":
    st.subheader("¿Dónde quieres buscar?")
    modo = st.radio("Lugar", ["📍 Una zona del proyecto", "▭ Dibujar mi recuadro"], horizontal=True,
                    label_visibility="collapsed")
    bbox, zona_elegida = None, None

    if modo.startswith("📍"):
        zona_elegida = st.selectbox(
            "Zona", ZONAS_LISTAS,
            format_func=lambda z: f"{nombre_zona(z)} · {'usada para entrenar' if ZONAS[z]['rol'] == 'entrenamiento' else 'no vista al entrenar'}",
        )
        faltan = [nombre_zona(z) for z in ZONAS if z not in ZONAS_LISTAS]
        if faltan:
            st.caption(f"Sin recorte en el repositorio: {', '.join(faltan)} (ejecuta scripts/preparar_zonas_app.py).")
    else:
        if "lon_min" not in st.session_state:
            poner_recuadro(RECUADRO_EJEMPLO)
        actual = tuple(st.session_state[k] for k in ("lon_min", "lat_min", "lon_max", "lat_max"))
        st.markdown(
            "Usa el botón **▭** de la izquierda del mapa y arrastra para dibujar. El recuadro verde es el que se "
            f"analizará; los punteados son las zonas del proyecto. Tamaño permitido: {LADO_MIN}° a {LADO_MAX:.0f}° "
            "por lado."
        )
        dibujo = st_folium(mapa_para_dibujar(actual), height=430, use_container_width=True,
                           returned_objects=["last_active_drawing"], key="mapa_dibujo")
        if dibujo and dibujo.get("last_active_drawing"):
            nuevo = limites_de(dibujo["last_active_drawing"])
            if nuevo != st.session_state.get("ultimo_dibujo"):
                st.session_state["ultimo_dibujo"] = nuevo
                poner_recuadro(nuevo)
                st.rerun()

        with st.expander("Coordenadas exactas (opcional)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("Oeste (lon_min)", -180.0, 180.0, step=0.1, format="%.3f", key="lon_min")
            c2.number_input("Sur (lat_min)", -90.0, 90.0, step=0.1, format="%.3f", key="lat_min")
            c3.number_input("Este (lon_max)", -180.0, 180.0, step=0.1, format="%.3f", key="lon_max")
            c4.number_input("Norte (lat_max)", -90.0, 90.0, step=0.1, format="%.3f", key="lat_max")
            st.caption("Grados decimales. Al oeste de Greenwich la longitud es negativa; al sur del ecuador, la latitud.")

        actual = tuple(st.session_state[k] for k in ("lon_min", "lat_min", "lon_max", "lat_max"))
        try:
            bbox = validar_recuadro(actual)
            tesela = tesela_de(bbox[0], bbox[3] - 1e-9)
            local = all((RAW / f"Hansen_GFC-2024-v1.12_{c}_{tesela}.tif").exists() for c in CAPAS)
            km = lambda g: g * 111  # noqa: E731
            st.success(
                f"✔ Recuadro listo: {bbox[2] - bbox[0]:.2f}° × {bbox[3] - bbox[1]:.2f}° "
                f"(~{km(bbox[2] - bbox[0]):.0f} × {km(bbox[3] - bbox[1]):.0f} km). "
                + ("Se leerá de la tesela descargada." if local else
                   "Se leerá del servidor de Hansen solo la parte del recuadro.")
            )
        except RecuadroInvalido as error:
            st.warning(f"Este recuadro no se puede analizar: {error}")
            a1, a2 = st.columns(2)
            ancho, alto = actual[2] - actual[0], actual[3] - actual[1]
            if ancho > 0 and alto > 0 and (ancho > LADO_MAX or alto > LADO_MAX):
                a1.button(f"Ajustar a {LADO_MAX:.0f}° × {LADO_MAX:.0f}° manteniendo el centro",
                          on_click=poner_recuadro, args=(ajustar_a_maximo(actual),))
            a2.button(f"Usar el recuadro de ejemplo {RECUADRO_EJEMPLO}", on_click=poner_recuadro,
                      args=(RECUADRO_EJEMPLO,))

    k = st.slider("¿Cuántas celdas pueden inspeccionar en este ciclo?", 5, 100, 20, step=5,
                  help="El proyecto supone 20 inspecciones por ciclo para una corporación autónoma regional.")
    correr = st.button("🌲 Analizar desde cero", type="primary", width="stretch",
                       disabled=modo.startswith("▭") and bbox is None)

    if correr:
        limpiar_corrida()
        inicio_total = time.perf_counter()
        if zona_elegida:
            raiz = carpeta_con_zona(zona_elegida)
            zona_agente, etiqueta, origen = zona_elegida, nombre_zona(zona_elegida), "recorte del repositorio"
        else:
            raiz = Path(tempfile.mkdtemp(prefix="fg_app_"))
            zona_agente, etiqueta = ZONA_RECUADRO, f"Recuadro {tuple(round(v, 3) for v in bbox)}"
            with st.status("Preparando el recuadro: recortando las imágenes de Hansen...", expanded=False) as estado:
                try:
                    info = recortar_recuadro(bbox, raiz / "data" / "processed", raw=RAW)
                except Exception as error:   # sin conexión, servidor caído, etc.
                    estado.update(label="No se pudieron leer las imágenes del recuadro", state="error")
                    st.error(f"No pude descargar el recuadro ({type(error).__name__}). Revisa la conexión e "
                             "inténtalo de nuevo, o usa una de las zonas del proyecto, que ya están en el repositorio.")
                    shutil.rmtree(raiz, ignore_errors=True)
                    st.stop()
                origen = f"tesela {info['tesela']} ({info['origen']})"
                estado.update(label=f"Recuadro listo: {info['forma'][0]:,} × {info['forma'][1]:,} píxeles en "
                                    f"{info['segundos']:.1f} s", state="complete")
        t_preparar = time.perf_counter() - inicio_total

        agente = ForestGuardAgent(raiz, zona=zona_agente, k=k)
        rutas, error, t_ciclo, texto = ejecutar_con_progreso(agente.ejecutar, "Analizando")
        if error:
            st.error(f"El agente se detuvo sin generar reportes: {error}")
            st.caption(f"Tipo de error: {type(error).__name__}. Quedó registrado como `ciclo_interrumpido`. "
                       "Suele pasar con recuadros sin bosque o sin pérdida reciente: prueba otro lugar.")
            with st.expander("Registro técnico"):
                tabla_eventos(eventos_de(agente))
            shutil.rmtree(raiz, ignore_errors=True)
        else:
            st.session_state["corrida"] = dict(
                raiz=raiz, zona=zona_agente, etiqueta=etiqueta, origen=origen, agente=agente, rutas=rutas,
                consola=texto, t_preparar=t_preparar, t_ciclo=t_ciclo, t_total=t_preparar + t_ciclo,
                titulo="Primera corrida", anterior=None,
            )
            st.session_state["ultimo_tiempo"] = (etiqueta, t_preparar + t_ciclo)
            st.rerun()

    c = st.session_state.get("corrida")
    if c:
        agente, rutas = c["agente"], c["rutas"]
        p = prioridades_amigables(agente)
        st.divider()

        # ---- resumen en una línea
        st.subheader(f"Resultado · {c['etiqueta']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tiempo de análisis", f"{c['t_total']:.0f} s",
                  delta="dentro de la meta" if c["t_total"] <= META_TIEMPO_S else "fuera de la meta",
                  delta_color="normal" if c["t_total"] <= META_TIEMPO_S else "inverse",
                  help=f"Desde el recuadro hasta el reporte. Meta del proyecto: ≤ 5 min 12 s. "
                       f"Preparar datos {c['t_preparar']:.1f} s + ciclo {c['t_ciclo']:.1f} s.")
        m2.metric("Celdas revisadas", f"{len(agente.nuevo):,}", help="Celdas de ~1 km con bosque que se puede perder")
        m3.metric("Para inspeccionar", f"{agente.k}",
                  help=", ".join(f"{NIVELES[n]}: {v}" for n, v in agente.prioridades["nivel"].value_counts().items()))
        m4.metric("Año que se predice", agente.anio_pred)
        if agente.n_feedback:
            st.info(f"🔁 {c['titulo']}: el modelo aprendió de **{agente.n_feedback} inspecciones** reportadas.")

        # ---- dónde ir primero
        st.markdown(f"### 🧭 Dónde inspeccionar primero")
        cambios = set()
        if c.get("anterior") is not None:
            cambios = set(p["celda_id"]) - set(c["anterior"]["celda_id"])
            st.caption(f"Con las inspecciones, {len(cambios)} de {len(p)} celdas del top son nuevas (marcadas con ✨).")
        cols = st.columns(TOP_TARJETAS)
        for col, (_, f) in zip(cols, p.head(TOP_TARJETAS).iterrows()):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**#{int(f['ranking'])}** · {f['prioridad']}" + (" ✨" if f["celda_id"] in cambios else ""))
                st.progress(float(f["riesgo"]), text=f"Riesgo {f['riesgo']:.0%}")
                st.markdown(f["por_que"])
                st.link_button("📍 Ver en el mapa", f["ubicacion"], width="stretch")

        st.markdown("#### Mapa de las celdas priorizadas")
        st.iframe(Path(rutas["mapa"]), height=480)
        st.caption("🔴 alta: riesgo sobre el umbral y con alertas previas · 🟠 media: sobre el umbral · "
                   "🔵 baja: el resto. Haz clic en una celda para ver qué pesó.")

        st.markdown(f"#### Todas las celdas priorizadas ({len(p)})")
        st.dataframe(
            p.assign(nueva=p["celda_id"].isin(cambios).map({True: "✨", False: ""}))[
                ["ranking", "prioridad", "riesgo_pct", "por_que", "ubicacion"] + (["nueva"] if cambios else [])],
            hide_index=True, width="stretch", height=min(38 * len(p) + 40, 420),
            column_config={
                "ranking": st.column_config.NumberColumn("#", width="small"),
                "prioridad": st.column_config.TextColumn("Prioridad", width="small"),
                "riesgo_pct": st.column_config.ProgressColumn("Riesgo", min_value=0, max_value=100, format="%d %%",
                                                              width="small"),
                "por_que": st.column_config.TextColumn("Por qué", width="large"),
                "ubicacion": st.column_config.LinkColumn("Ubicación", display_text="📍 Abrir mapa", width="small"),
                "nueva": st.column_config.TextColumn("", width="small"),
            },
        )

        with st.expander("🔎 Ver en detalle qué pesó en una celda"):
            etiquetas = {f"#{int(f.ranking)} · {NIVELES[f.nivel]} · riesgo {f.riesgo:.0%}": int(f.celda_id)
                         for f in agente.prioridades.itertuples()}
            celda = etiquetas[st.selectbox("Celda", list(etiquetas))]
            st.caption(f"Riesgo = punto de partida del modelo ({agente.base_riesgo:.0%}) + aporte de cada variable. "
                       "Rojo sube el riesgo, azul lo baja (puntos de probabilidad).")
            detalle = tabla_larga(agente.contribuciones, agente.prioridades, celda)
            detalle["sube el riesgo"] = detalle["contribucion_pts"].clip(lower=0)
            detalle["baja el riesgo"] = detalle["contribucion_pts"].clip(upper=0)
            g, t = st.columns([3, 2])
            with g:
                st.bar_chart(detalle, x="corto", y=["sube el riesgo", "baja el riesgo"], horizontal=True,
                             color=["#B4492D", "#3D6F8C"], x_label="puntos de probabilidad", y_label="", sort=False)
            with t:
                st.dataframe(detalle[["variable", "valor", "unidad", "contribucion_pts"]].round(2), hide_index=True,
                             width="stretch", column_config={"contribucion_pts": "aporte (pts)"})

        # ---- inspecciones y reentrenamiento
        st.markdown("### 📝 ¿Qué encontraron en las visitas?")
        st.caption("Marca el resultado de las celdas que se visitaron y guarda: el agente vuelve a entrenar usando "
                   "esas respuestas (memory/feedback.csv) y actualiza la lista.")
        editable = p[["ranking", "prioridad", "riesgo_pct", "ubicacion", "celda_id", "anio_pred"]].copy()
        editable["resultado"] = "Sin inspeccionar"
        editado = st.data_editor(
            editable, hide_index=True, width="stretch", key=f"insp_{agente.registro.ejecucion}",
            column_order=["ranking", "prioridad", "riesgo_pct", "ubicacion", "resultado"],
            disabled=["ranking", "prioridad", "riesgo_pct", "ubicacion"],
            column_config={
                "ranking": st.column_config.NumberColumn("#", width="small"),
                "prioridad": st.column_config.TextColumn("Prioridad", width="small"),
                "riesgo_pct": st.column_config.ProgressColumn("Riesgo", min_value=0, max_value=100, format="%d %%"),
                "ubicacion": st.column_config.LinkColumn("Ubicación", display_text="📍 Abrir mapa"),
                "resultado": st.column_config.SelectboxColumn("Resultado de la visita",
                                                              options=list(RESULTADOS_INSPECCION), required=True),
            },
        )
        marcadas = editado[editado["resultado"] != "Sin inspeccionar"]
        b1, b2 = st.columns([2, 1])
        guardar = b1.button(
            f"💾 Guardar {len(marcadas)} visita{'s' if len(marcadas) != 1 else ''} y reentrenar" if len(marcadas)
            else "💾 Marca al menos una visita para reentrenar",
            type="primary", width="stretch", disabled=marcadas.empty)
        simular = b2.button("🎲 Simular visitas (demo)", width="stretch", disabled=agente.n_feedback > 0,
                            help="Inventa resultados para todo el top (confirmada con probabilidad = riesgo). "
                                 "Solo se puede usar antes de guardar visitas.")

        if guardar or simular:
            nuevo_agente = ForestGuardAgent(c["raiz"], zona=c["zona"], k=agente.k)

            def ciclo_con_feedback():
                if guardar:
                    fb = marcadas.assign(confirmado=marcadas["resultado"].map(RESULTADOS_INSPECCION))
                    nuevo_agente.registrar_inspecciones(fb[["celda_id", "anio_pred", "confirmado"]])
                else:
                    nuevo_agente.simular_feedback()
                return nuevo_agente.ejecutar()

            rutas2, error, t_ciclo, texto = ejecutar_con_progreso(ciclo_con_feedback, "Reentrenando con las visitas")
            if error:
                st.error(f"No se pudo reentrenar: {error}")
            else:
                c.update(agente=nuevo_agente, rutas=rutas2, consola=texto, t_ciclo=t_ciclo,
                         titulo="Corrida con visitas simuladas" if simular else "Corrida con visitas reportadas",
                         anterior=agente.prioridades[["ranking", "celda_id", "riesgo"]])
                st.rerun()

        # ---- descargas y detalle técnico
        st.markdown("### ⬇ Reportes")
        nombres = {"csv": "📄 Tabla", "geojson": "🗺 GeoJSON", "mapa": "🌐 Mapa", "resumen": "📝 Resumen",
                   "plantilla_feedback": "✍ Plantilla"}
        cols = st.columns(len(rutas) + 1)
        for col, (nombre, ruta) in zip(cols, rutas.items()):
            col.download_button(nombres.get(nombre, nombre), Path(ruta).read_bytes(), file_name=Path(ruta).name,
                                width="stretch")
        cols[-1].download_button("🧾 Registro", agente.ruta_log.read_bytes(), file_name=agente.ruta_log.name,
                                 width="stretch")

        with st.expander("⚙ Detalle técnico (para la evaluación)"):
            t1, t2, t3 = st.columns(3)
            t1.metric("Umbral de riesgo", f"{agente.umbral_riesgo:.3f}")
            t2.metric("Inspecciones usadas", agente.n_feedback)
            t3.metric("Datos", c["origen"])
            st.caption(f"Meta de tiempo del Hito 1 (≤ 5 min 12 s desde un recuadro nuevo hasta el reporte): "
                       f"{si_no(c['t_total'] <= META_TIEMPO_S)} · preparar {c['t_preparar']:.1f} s + ciclo "
                       f"{c['t_ciclo']:.1f} s")
            izq, der = st.columns(2)
            with izq:
                st.markdown("**Evaluación del modelo (división temporal)**")
                st.dataframe(agente.resultados.round(3), hide_index=True, width="stretch")
            with der:
                st.markdown("**Resumen de la zona**")
                r = agente.resumen_zona
                st.markdown(
                    f"- Pérdida en {r['anio']}: **{r['perdida_ha']:,.0f} ha**\n"
                    f"- Promedio histórico: {r['promedio_historico_ha']:,.0f} ha/año\n"
                    f"- Razón frente al histórico: {r['razon']:.2f}×"
                )
            st.markdown("**Tabla completa del reporte**")
            st.dataframe(pd.read_csv(rutas["csv"]), hide_index=True, width="stretch", height=260)
            st.markdown("**Registro de ejecución con marca de tiempo**")
            tabla_eventos(eventos_de(agente))
            st.markdown("**Salida del agente (consola)**")
            st.code(c["consola"], language=None)

# ---------------------------------------------------------------------- 2 · falla
if seccion == "⚠ Escenario de falla":
    st.subheader("Escenario de falla esperada")
    st.markdown(
        "**Entradas inválidas (F-x, A-1):** se copia una esquina de Guaviare (25 %) a una carpeta temporal, se "
        "inyecta el problema y se ejecuta el **ciclo completo del agente**. Lo correcto es que se detenga con un "
        "error controlado, que quede el evento `ciclo_interrumpido` en el registro y que **no** genere reportes.\n\n"
        "**Recuadros imposibles (R-x):** el agente debe rechazar el recuadro antes de leer datos y decir por qué."
    )
    casos = {c["id"]: c for c in pruebas.casos_de_falla(ZONA_PILOTO)}

    def quitar_franja(capas):
        ancho = capas["datamask"].shape[1]
        corte = (int(0.4 * ancho) // CELDA_PX) * CELDA_PX
        capas["datamask"] = capas["datamask"].copy()
        capas["datamask"][:, :corte] = 0
        return capas

    casos["A-1"] = dict(id="A-1", descr="entorno adverso: 40 % de la zona sin dato", mod=quitar_franja,
                        omitir=(), corromper=None)
    recuadros_malos = {
        "R-1": ("recuadro fuera de Colombia (Venezuela/Guyana)", (-60.0, 5.0, -59.5, 5.5)),
        "R-2": ("recuadro de 2° × 1° (más grande que la meta)", (-75.0, 1.0, -73.0, 2.0)),
        "R-3": ("recuadro que cruza el borde de dos teselas", (-70.5, 1.0, -69.5, 2.0)),
        "R-4": ("coordenadas al revés (lon_min > lon_max)", (-72.0, 1.0, -73.0, 2.0)),
    }
    for rid, (descr, caja) in recuadros_malos.items():
        casos[rid] = dict(id=rid, descr=f"{descr}: {caja}", bbox=caja)

    eleccion = st.selectbox("Falla a inyectar", list(casos), format_func=lambda i: f"{i} · {casos[i]['descr']}")

    if st.button("⚠ Inyectar y ejecutar el agente", type="primary"):
        caso = casos[eleccion]
        if "bbox" in caso:
            raiz = Path(tempfile.mkdtemp(prefix="fg_falla_app_"))
            try:
                inicio_r = time.perf_counter()
                recortar_recuadro(caso["bbox"], raiz / "data" / "processed", raw=RAW)
                st.error("El agente aceptó un recuadro imposible (esto sería una falla de la prueba).")
            except RecuadroInvalido as error:
                st.success(f"Rechazado en {time.perf_counter() - inicio_r:.3f} s → **RecuadroInvalido**: {error}")
                reportes = list((raiz / "outputs").glob("*")) if (raiz / "outputs").exists() else []
                datos = list((raiz / "data" / "processed").glob("*.tif")) if (raiz / "data" / "processed").exists() \
                    else []
                st.markdown(f"Datos leídos: **{len(datos)}** · reportes generados: **{len(reportes)}** "
                            + ("✅" if not (datos or reportes) else "❌"))
            finally:
                shutil.rmtree(raiz, ignore_errors=True)
        else:
            raiz = Path(tempfile.mkdtemp(prefix="fg_falla_app_"))
            try:
                proc = raiz / "data" / "processed"
                pruebas.copiar_zona(DATOS, proc, ZONA_PILOTO, pruebas.FRACCION_FALLAS, modificar=caso["mod"],
                                    omitir=caso["omitir"])
                if caso["corromper"]:
                    (proc / f"{ZONA_PILOTO}_{caso['corromper']}.tif").write_bytes(b"esto no es un tif")
                agente = ForestGuardAgent(raiz, zona=ZONA_PILOTO)
                rutas, error, segundos, _ = ejecutar_con_consola(agente.ejecutar, f"Ejecutando el agente con {eleccion}")
                reportes = sorted(p.name for p in (raiz / "outputs").glob("*")) if (raiz / "outputs").exists() else []

                if eleccion == "A-1":
                    if error:
                        st.error(f"El agente no debía detenerse: {type(error).__name__}: {error}")
                    else:
                        limite = int(0.4 * 1000) // CELDA_PX
                        col_min = int(agente.nuevo["col"].min())
                        st.success(f"Manejado: el agente continuó con {len(agente.nuevo):,} celdas válidas. Columna "
                                   f"mínima evaluada: {col_min} (la franja sin dato son las columnas < {limite}).")
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
if seccion == "📊 Pruebas de desempeño":
    # ---- metas del Hito 1 en zonas retenidas
    st.subheader("Metas del Hito 1 en zonas retenidas (Chairá y Putumayo, año 2025)")
    resumen_rf = CARPETA_HITO2 / "linea_base_resumen.csv"
    corridas_bo = CARPETA_RUTA / "corridas.csv"
    if resumen_rf.exists() and corridas_bo.exists():
        rf = pd.read_csv(resumen_rf)
        rf_z = rf[rf["modelo"] == "linea_base_rf"].set_index("zona")
        pers = rf[rf["modelo"] == "persistencia"].set_index("zona")
        bo = pd.read_csv(corridas_bo)
        bo = bo[bo["modelo"] == "boosting_espacial"]
        bo_z = bo.groupby("zona")[["pr_auc", "top_k", "recall", "fpr"]].mean()
        bo_m = bo.groupby("semilla")[["pr_auc", "top_k", "recall", "fpr"]].mean().mean()
        rf_m, pers_m = rf_z.loc["media_retenidas"], pers.loc["media_retenidas"]

        razon = bo_m["pr_auc"] / rf_m["pr_auc_mean"]
        ninguna_peor = all(bo_z.loc[z, "pr_auc"] >= rf_z.loc[z, "pr_auc_mean"] for z in bo_z.index)
        ultimo = st.session_state.get("ultimo_tiempo")
        filas = [
            ("PR-AUC", "Ruta ≥ 1,10 × línea base en la media y no menor en ninguna zona",
             f"{rf_m['pr_auc_mean']:.3f}", f"{bo_m['pr_auc']:.3f} ({razon:.3f}×)", si_no(razon >= 1.10 and ninguna_peor)),
            ("Precisión top-20", f"≥ 75 % y mayor que persistencia ({pers_m['top_k_mean']:.0%})",
             f"{rf_m['top_k_mean']:.0%}", f"{bo_m['top_k']:.0%}",
             si_no(bo_m["top_k"] >= 0.75 and bo_m["top_k"] > pers_m["top_k_mean"])),
            ("Recall con FPR ≤ 15 %", "Recall ≥ 75 % y falsos positivos ≤ 15 %",
             f"{rf_m['recall_mean']:.1%} con FPR {rf_m['fpr_mean']:.1%}",
             f"{bo_m['recall']:.1%} con FPR {bo_m['fpr']:.1%}",
             si_no(bo_m["recall"] >= 0.75 and bo_m["fpr"] <= 0.15)),
            ("Tiempo de respuesta", "≤ 5 min 12 s desde un recuadro nuevo hasta el reporte",
             "se mide en 🌲 Analizar", f"{ultimo[1]:.1f} s ({ultimo[0]})" if ultimo else "analiza una zona primero",
             si_no(ultimo[1] <= META_TIEMPO_S) if ultimo else "—"),
        ]
        st.table(pd.DataFrame(filas, columns=["Métrica", "Meta (fijada en el Hito 1)", "Línea base RF",
                                              "Ruta: boosting espacial", "¿Cumple la ruta?"]))
        st.caption(
            "Media de 5 semillas en las dos zonas retenidas, umbral fijado en validación (Guaviare y Macarena). "
            "La aplicación usa el Random Forest: el boosting casi empata y no mejora lo suficiente para reemplazarlo. "
            "Fuentes: `outputs/hito2/linea_base_resumen.csv` y `outputs/ruta_espacial/corridas.csv`."
        )
        with st.expander("Por zona"):
            por_zona = pd.DataFrame({
                "PR-AUC RF": rf_z.loc[bo_z.index, "pr_auc_mean"], "PR-AUC boosting": bo_z["pr_auc"],
                "top-20 RF": rf_z.loc[bo_z.index, "top_k_mean"], "top-20 boosting": bo_z["top_k"],
                "FPR RF": rf_z.loc[bo_z.index, "fpr_mean"], "FPR boosting": bo_z["fpr"],
            }).rename(index=nombre_zona)
            st.dataframe(por_zona.round(3), width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("Descargar resumen línea base (.csv)", resumen_rf.read_bytes(), file_name=resumen_rf.name)
        d2.download_button("Descargar corridas boosting (.csv)", corridas_bo.read_bytes(), file_name=corridas_bo.name)
    else:
        st.info("No se encontraron los resultados del Hito 2 en outputs/hito2/ y outputs/ruta_espacial/.")

    st.divider()
    st.subheader("Resultados oficiales de las pruebas de la Fase 3 (Guaviare)")
    ruta_registro = CARPETA_PRUEBAS / f"registro_pruebas_{ZONA_PILOTO}.txt"
    csvs = sorted(CARPETA_PRUEBAS.glob(f"corridas_{ZONA_PILOTO}_*.csv"))
    if ruta_registro.exists():
        texto = ruta_registro.read_text(encoding="utf-8")
        bloque = texto.split("TABLA DE RESULTADOS")[-1]
        cabecera = bloque.splitlines()[0].strip(" ·")
        tabla = "\n".join(l for l in bloque.splitlines() if l.startswith("|"))
        st.caption(f"Bitácora `outputs/pruebas/registro_pruebas_{ZONA_PILOTO}.txt` · {cabecera} · "
                   "5 repeticiones por escenario")
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
                shutil.copy2(DATOS / f"{ZONA_PILOTO}_{capa}.tif", tmp / "data" / "processed")
            cmd = [sys.executable, "-u", str(tmp / "scripts" / "pruebas_desempeno.py"), "--zona", ZONA_PILOTO,
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
