import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import Voronoi
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ================================================================
# MOTOR DE DIBUJO DE CANCHA
# ================================================================
PALETA = {
    "fondo": "#ffffff", "cancha": "#f7f9fb", "grid": "#e3e8ee",
    "lineas": "#2b3a4a", "texto": "#1a2530", "texto_tenue": "#6b7785",
    "acento": "#0066cc", "aliado": "#0066cc", "rival": "#e8553a",
    "balon": "#2b3a4a", "exito": "#1aa179", "fallo": "#d64545", "aviso": "#e0a020",
}
CANCHA = {"largo": 219, "ancho": 158, "area_largo": 25, "area_ancho": 80,
          "porteria_ancho": 60, "circulo_radio": 30, "margen": 12, "grid_paso": 20}


def _fuente():
    preferidas = ["DejaVu Sans", "Helvetica Neue", "Arial", "Liberation Sans"]
    disponibles = {f.name for f in fm.fontManager.ttflist}
    for f in preferidas:
        if f in disponibles:
            return f
    return "sans-serif"


FUENTE = _fuente()


def _titulo(ax, titulo, subtitulo=None):
    ax.text(0, CANCHA["ancho"] + CANCHA["margen"] + 5, titulo, ha="left",
            va="bottom", fontsize=14.5, fontweight="bold",
            color=PALETA["texto"], family=FUENTE)
    if subtitulo:
        ax.text(0, CANCHA["ancho"] + CANCHA["margen"] + 1, subtitulo,
                ha="left", va="bottom", fontsize=9.5,
                color=PALETA["texto_tenue"], family=FUENTE, style="italic")


def _firma(ax):
    ax.text(CANCHA["largo"], -CANCHA["margen"] - 4,
            "Copa FutBotMX · Equipo AZTEM", ha="right", va="top",
            fontsize=7.5, color=PALETA["texto_tenue"], family=FUENTE)


def _ejes_tecnicos(ax):
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    paso = CANCHA["grid_paso"] * 2
    for x in range(0, largo + 1, paso):
        ax.text(x, -4, str(x), ha="center", va="top", fontsize=6.5,
                color=PALETA["texto_tenue"], family=FUENTE)
        ax.plot([x, x], [0, -1.5], color=PALETA["texto_tenue"], lw=0.6)
    for y in range(0, ancho + 1, paso):
        ax.text(-4, y, str(y), ha="right", va="center", fontsize=6.5,
                color=PALETA["texto_tenue"], family=FUENTE)
        ax.plot([0, -1.5], [y, y], color=PALETA["texto_tenue"], lw=0.6)
    ax.text(largo / 2, -11, "x (cm)", ha="center", va="top", fontsize=7.5,
            color=PALETA["texto_tenue"], family=FUENTE)
    ax.text(-11, ancho / 2, "y (cm)", ha="center", va="center", rotation=90,
            fontsize=7.5, color=PALETA["texto_tenue"], family=FUENTE)


def dibujar_cancha(ax=None, figsize=(13, 9), con_grid=True, con_ejes=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(PALETA["fondo"])
    ax.set_facecolor(PALETA["fondo"])
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    lin = PALETA["lineas"]; lw = 1.6
    ax.add_patch(patches.Rectangle((0, 0), largo, ancho,
                 facecolor=PALETA["cancha"], edgecolor="none", zorder=0))
    if con_grid:
        paso = CANCHA["grid_paso"]
        for x in range(paso, largo, paso):
            ax.plot([x, x], [0, ancho], color=PALETA["grid"], lw=0.7, zorder=1)
        for y in range(paso, ancho, paso):
            ax.plot([0, largo], [y, y], color=PALETA["grid"], lw=0.7, zorder=1)
    ax.add_patch(patches.Rectangle((0, 0), largo, ancho, fill=False,
                 edgecolor=lin, linewidth=lw, zorder=2))
    ax.plot([largo / 2, largo / 2], [0, ancho], color=lin, linewidth=lw, zorder=2)

    # Círculo central: solo borde, garantizado sin relleno
    ax.add_patch(patches.Circle((largo / 2, ancho / 2), CANCHA["circulo_radio"],
                 fill=False, edgecolor=lin, linewidth=lw, zorder=2))

    area_y = (ancho - CANCHA["area_ancho"]) / 2
    for x0 in (0, largo - CANCHA["area_largo"]):
        ax.add_patch(patches.Rectangle((x0, area_y), CANCHA["area_largo"],
                     CANCHA["area_ancho"], fill=False, edgecolor=lin,
                     linewidth=lw, zorder=2))
    port_y = (ancho - CANCHA["porteria_ancho"]) / 2
    for x0 in (0, largo):
        ax.plot([x0, x0], [port_y, port_y + CANCHA["porteria_ancho"]],
                color=PALETA["acento"], linewidth=4.5, zorder=3,
                solid_capstyle="round")
    if con_ejes:
        _ejes_tecnicos(ax)
    m = CANCHA["margen"]
    ax.set_xlim(-m, largo + m); ax.set_ylim(-m - 8, ancho + m + 16)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _leyenda(ax, items, loc="upper right"):
    # Si detectamos una línea (para la red de pases), ajustamos el handle
    handles = []
    for c, lbl, mk in items:
        if mk in ["-", "--"]:
            handles.append(plt.Line2D([0], [0], color=c, linestyle=mk, lw=2.5, label=lbl))
        else:
            handles.append(plt.Line2D([0], [0], marker=mk, color="none", markerfacecolor=c,
                                      markeredgecolor="none", markersize=11, label=lbl))

    leg = ax.legend(handles=handles, loc=loc, frameon=True, fontsize=9,
                    labelcolor=PALETA["texto"], handletextpad=0.4)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor(PALETA["grid"])
    leg.get_frame().set_alpha(0.95)


def heatmap(xs, ys, titulo="Mapa de calor", subtitulo=None, bins=40, mostrar_cbar=True):
    fig, ax = dibujar_cancha(con_grid=False)
    heat, _, _ = np.histogram2d(xs, ys, bins=bins,
        range=[[0, CANCHA["largo"]], [0, CANCHA["ancho"]]])
    if SCIPY_OK:
        heat = gaussian_filter(heat, sigma=1.4)
    cmap = LinearSegmentedColormap.from_list("calor", [
        (0.00, (1, 1, 1, 0)), (0.20, (0.30, 0.55, 0.85, 0.45)),
        (0.50, (0.45, 0.35, 0.75, 0.70)), (0.78, (0.90, 0.45, 0.30, 0.85)),
        (1.00, (0.80, 0.10, 0.15, 0.95))])

    # IMPORTANTE: aspect="equal" para que no deforme el mapa respecto a las otras vistas
    im = ax.imshow(heat.T, extent=[0, CANCHA["largo"], 0, CANCHA["ancho"]],
                   origin="lower", cmap=cmap, interpolation="bilinear",
                   zorder=1, aspect="equal")

    if mostrar_cbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Densidad de presencia", fontsize=8, color=PALETA["texto_tenue"])
        cbar.ax.tick_params(labelsize=6, colors=PALETA["texto_tenue"])

    _titulo(ax, titulo, subtitulo); _firma(ax)
    return fig, ax


def trayectoria(xs, ys, titulo="Trayectoria", subtitulo=None, color=None):
    fig, ax = dibujar_cancha()
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 2:
        _titulo(ax, titulo, "Sin datos suficientes"); _firma(ax)
        return fig, ax
    puntos = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    col_fin = color or PALETA["acento"]
    cmap = LinearSegmentedColormap.from_list("ruta", ["#cfe0f2", col_fin])
    lc = LineCollection(segmentos, cmap=cmap, linewidth=2.4, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segmentos)))
    ax.add_collection(lc)
    ax.scatter(xs[0], ys[0], s=120, c="#ffffff", edgecolor=col_fin, linewidth=2, zorder=4)
    ax.scatter(xs[-1], ys[-1], s=160, c=col_fin, edgecolor="#ffffff", linewidth=2, zorder=4)
    ax.text(xs[0], ys[0] - 7, "inicio", ha="center", fontsize=7.5,
            color=PALETA["texto_tenue"], family=FUENTE)
    ax.text(xs[-1], ys[-1] + 6, "fin", ha="center", fontsize=7.5,
            color=col_fin, family=FUENTE, fontweight="bold")
    _titulo(ax, titulo, subtitulo); _firma(ax)
    return fig, ax


def mapa_pases(pases, titulo="Mapa de pases", subtitulo=None):
    """
    Dibuja cada pase como una flecha sobre la trayectoria del balón:
    verde = completado, rojo punteado = interceptado/errado.
    """
    fig, ax = dibujar_cancha()
    completados = sum(1 for p in pases if p["exito"]); total = len(pases)

    for p in pases:
        x1, y1 = p["origen"]
        x2, y2 = p["destino"]
        ok = p["exito"]
        color = PALETA["exito"] if ok else PALETA["fallo"]
        estilo = "-" if ok else "--"
        # arc3 separa ligeramente las flechas de ida y vuelta entre los mismos puntos
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                    alpha=0.9, shrinkA=4, shrinkB=4,
                                    linestyle=estilo,
                                    connectionstyle="arc3,rad=0.10"), zorder=3)
        ax.scatter(x1, y1, s=45, c=color, zorder=4, edgecolor="#ffffff", linewidth=1.0)

    _leyenda(ax, [(PALETA["exito"], "Pase completado", "o"),
                  (PALETA["fallo"], "Pase interceptado o errado", "o")])

    pct = (completados / total * 100) if total else 0
    sub = subtitulo or f"Efectividad: {completados}/{total} pases ({pct:.0f}%)"
    _titulo(ax, titulo, sub); _firma(ax)
    return fig, ax


def mapa_tiros(tiros, titulo="Mapa de tiros", subtitulo=None):
    fig, ax = dibujar_cancha()
    estilos = {"gol": {"c": PALETA["exito"], "mk": "*", "s": 420},
               "atajado": {"c": PALETA["aviso"], "mk": "o", "s": 190},
               "fuera": {"c": PALETA["fallo"], "mk": "X", "s": 190}}
    for x, y, res in tiros:
        e = estilos.get(res, estilos["fuera"])
        ax.scatter(x, y, c=e["c"], marker=e["mk"], s=e["s"],
                   edgecolor="#ffffff", linewidth=1.5, zorder=4)
    _leyenda(ax, [(estilos["gol"]["c"], "Gol", "*"),
                  (estilos["atajado"]["c"], "Atajado", "o"),
                  (estilos["fuera"]["c"], "Fuera", "X")])
    goles = sum(1 for *_, r in tiros if r == "gol")
    sub = subtitulo or f"Goles: {goles} · Tiros: {len(tiros)}"
    _titulo(ax, titulo, sub); _firma(ax)
    return fig, ax


def _recortar_poligono(poligono, largo, ancho):
    def dentro(p, borde):
        x, y = p
        if borde == "izq": return x >= 0
        if borde == "der": return x <= largo
        if borde == "abajo": return y >= 0
        if borde == "arriba": return y <= ancho

    def interseccion(p1, p2, borde):
        x1, y1 = p1; x2, y2 = p2
        if borde in ("izq", "der"):
            xb = 0 if borde == "izq" else largo
            t = (xb - x1) / (x2 - x1) if x2 != x1 else 0
            return (xb, y1 + t * (y2 - y1))
        else:
            yb = 0 if borde == "abajo" else ancho
            t = (yb - y1) / (y2 - y1) if y2 != y1 else 0
            return (x1 + t * (x2 - x1), yb)

    salida = list(poligono)
    for borde in ("izq", "der", "abajo", "arriba"):
        if not salida: break
        entrada = salida; salida = []
        for i in range(len(entrada)):
            actual = entrada[i]; previo = entrada[i - 1]
            if dentro(actual, borde):
                if not dentro(previo, borde):
                    salida.append(interseccion(previo, actual, borde))
                salida.append(actual)
            elif dentro(previo, borde):
                salida.append(interseccion(previo, actual, borde))
    return salida


def voronoi_control(aliados, rivales, titulo="Control de espacio", subtitulo=None):
    fig, ax = dibujar_cancha(con_grid=False)
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    puntos = [(x, y) for x, y in aliados] + [(x, y) for x, y in rivales]
    equipo_de = [0] * len(aliados) + [1] * len(rivales)
    n_real = len(puntos)
    if n_real < 2 or not SCIPY_OK:
        for x, y in aliados:
            ax.scatter(x, y, s=170, c=PALETA["aliado"], edgecolor="#ffffff",
                       linewidth=2, zorder=4)
        for x, y in rivales:
            ax.scatter(x, y, s=170, c=PALETA["rival"], edgecolor="#ffffff",
                       linewidth=2, zorder=4)
        _titulo(ax, titulo, subtitulo); _firma(ax)
        return fig, ax
    extra = [(-largo * 3, -ancho * 3), (largo * 4, -ancho * 3),
             (-largo * 3, ancho * 4), (largo * 4, ancho * 4)]
    vor = Voronoi(np.array(puntos + extra))
    color_eq = {0: PALETA["aliado"], 1: PALETA["rival"]}
    area_aliado = 0.0; area_total = largo * ancho
    for idx_punto in range(n_real):
        region_idx = vor.point_region[idx_punto]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0: continue
        poligono = [tuple(vor.vertices[v]) for v in region]
        recortado = _recortar_poligono(poligono, largo, ancho)
        if len(recortado) < 3: continue
        eq = equipo_de[idx_punto]
        ax.add_patch(patches.Polygon(recortado, closed=True,
                     facecolor=color_eq[eq], alpha=0.22,
                     edgecolor=color_eq[eq], linewidth=1.4, zorder=1))
        xs = [p[0] for p in recortado]; ys = [p[1] for p in recortado]
        area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                             xs[(i + 1) % len(xs)] * ys[i]
                             for i in range(len(xs))))
        if eq == 0: area_aliado += area
    for x, y in aliados:
        ax.scatter(x, y, s=170, c=PALETA["aliado"], edgecolor="#ffffff",
                   linewidth=2, zorder=4)
    for x, y in rivales:
        ax.scatter(x, y, s=170, c=PALETA["rival"], edgecolor="#ffffff",
                   linewidth=2, zorder=4)
    pct_aliado = area_aliado / area_total * 100
    _leyenda(ax, [(PALETA["aliado"], "Aliados", "o"),
                  (PALETA["rival"], "Rivales", "o")])
    sub = subtitulo or (f"Control: Aliados {pct_aliado:.0f}% · "
                        f"Rivales {100 - pct_aliado:.0f}%")
    _titulo(ax, titulo, sub); _firma(ax)
    return fig, ax


def posesion_zonas(aliado_xy, rival_xy, titulo="Posesion por zonas", subtitulo=None, cols=6, filas=4):
    fig, ax = dibujar_cancha(con_grid=False)
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    cw, ch = largo / cols, ancho / filas
    a = np.array(aliado_xy) if len(aliado_xy) else np.empty((0, 2))
    r = np.array(rival_xy) if len(rival_xy) else np.empty((0, 2))
    cmap = LinearSegmentedColormap.from_list("divergente",
        [PALETA["rival"], "#f2f2f2", PALETA["aliado"]])
    for i in range(cols):
        for j in range(filas):
            x0, y0 = i * cw, j * ch
            na = np.sum((a[:, 0] >= x0) & (a[:, 0] < x0 + cw) &
                        (a[:, 1] >= y0) & (a[:, 1] < y0 + ch)) if len(a) else 0
            nr = np.sum((r[:, 0] >= x0) & (r[:, 0] < x0 + cw) &
                        (r[:, 1] >= y0) & (r[:, 1] < y0 + ch)) if len(r) else 0
            total = na + nr
            if total == 0: continue
            frac = na / total
            ax.add_patch(patches.Rectangle((x0, y0), cw, ch,
                         facecolor=cmap(frac), alpha=0.78,
                         edgecolor="#ffffff", linewidth=1.0, zorder=1))
            ax.text(x0 + cw / 2, y0 + ch / 2, f"{frac*100:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color=PALETA["texto"], family=FUENTE, zorder=2,
                    fontweight="bold")
    _leyenda(ax, [(PALETA["aliado"], "Dominio aliado", "s"),
                  (PALETA["rival"], "Dominio rival", "s")])
    _titulo(ax, titulo, subtitulo or "% de presencia aliada por zona")
    _firma(ax)
    return fig, ax


def red_pases(posiciones, conexiones, mapeo_robots, titulo="Análisis de Red de Pases", subtitulo=None):
    fig, ax = dibujar_cancha()
    if not conexiones:
        _titulo(ax, titulo, "Sin datos de conexiones"); _firma(ax)
        return fig, ax

    max_pases = max(conexiones.values(), default=1)

    for (o, d, exito), cant in conexiones.items():
        if o not in posiciones or d not in posiciones: continue
        x1, y1 = posiciones[o]; x2, y2 = posiciones[d]

        grosor = 1.5 + 4.5 * (cant / max_pases)
        color = PALETA["exito"] if exito else PALETA["fallo"]
        estilo = "solid" if exito else "dashed"

        # Dibujar línea de pase
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=grosor,
                alpha=0.6, zorder=3, linestyle=estilo, solid_capstyle="round")

        # Mostrar cantidad de pases en esa conexión
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(cant), fontsize=8.5,
                color=color, family=FUENTE, ha="center", va="center", zorder=5,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec=color, lw=0.8))

    for rid, (x, y) in posiciones.items():
        nombre = mapeo_robots.get(rid, str(rid))
        # El prefijo del mapeo ("A·R1" / "B·R1") define el equipo de forma robusta
        es_equipo_a = nombre.startswith("A")
        color_nodo = PALETA["aliado"] if es_equipo_a else PALETA["rival"]

        ax.scatter(x, y, s=800, c=color_nodo, edgecolor="#ffffff",
                   linewidth=2.5, zorder=4)
        ax.text(x, y, nombre, ha="center", va="center", fontsize=9.5,
                color="#ffffff", family=FUENTE, fontweight="bold", zorder=5)

    _leyenda(ax, [(PALETA["exito"], "Conexión Exitosa", "-"),
                  (PALETA["fallo"], "Pase Interceptado", "--")])
    _titulo(ax, titulo, subtitulo)
    _firma(ax)
    return fig, ax


def guardar(nombre, dpi=170):
    plt.savefig(nombre, dpi=dpi, bbox_inches="tight", facecolor=PALETA["fondo"])
    plt.close()


DPI_EXPORTACION = 170

# ================================================================
# RUTAS DEL PROYECTO
# ================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAIZ_PROYECTO = os.path.dirname(BASE_DIR)

CARPETA_METRICAS         = os.path.join(RAIZ_PROYECTO, "results", "metrics")
CARPETA_VISUALIZACIONES  = os.path.join(RAIZ_PROYECTO, "results", "visualizations")

ARCHIVO_TRACKING   = os.path.join(CARPETA_METRICAS, "tracking_data.csv")
ARCHIVO_EVENTOS    = os.path.join(CARPETA_METRICAS, "game_events.csv")
CARPETA_RESULTADOS = CARPETA_VISUALIZACIONES

VENTANA_ESTELA = 10
FPS            = 15

EQUIPO_A = "Equipo A"
EQUIPO_B = "Equipo B"

COLOR_EQUIPO = {
    EQUIPO_A: PALETA["aliado"],
    EQUIPO_B: PALETA["rival"],
}

COLOR_BALON = "black"

EVENTO_TIRO    = "Tiro a Gol / Despeje"
EVENTO_CONTROL = "Control de Balon"


# -------------------------------------------------------------
# CARGA Y NORMALIZACIÓN
# -------------------------------------------------------------
def cargar_tracking(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()
    if "id_objeto" in df.columns and "objeto_id" not in df.columns:
        df.rename(columns={"id_objeto": "objeto_id"}, inplace=True)
    df["equipo"] = df["equipo"].astype(str).str.strip()
    df["tipo"]   = df["tipo"].astype(str).str.strip().str.lower()

    def _clasificar(fila):
        if fila["tipo"] == "balon": return "balon"
        if fila["equipo"] == EQUIPO_A: return "robot_aliado"
        if fila["equipo"] == EQUIPO_B: return "robot_rival"
        return "desconocido"

    df["tipo_ext"] = df.apply(_clasificar, axis=1)
    for col, default in [("frame", 0), ("tiempo", 0.0),
                         ("objeto_id", 0), ("intervencion_humana", "no")]:
        if col not in df.columns: df[col] = default
    df = df.dropna(subset=["x", "y"])
    return df


def cargar_eventos(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()
    if "robot_implicado" in df.columns and "robot_id" not in df.columns:
        df.rename(columns={"robot_implicado": "robot_id"}, inplace=True)
    for col, default in [("frame", 0), ("tiempo", 0.0),
                         ("robot_id", -1), ("equipo", ""), ("detalles", "")]:
        if col not in df.columns: df[col] = default
    df["robot_id"] = pd.to_numeric(df["robot_id"], errors="coerce").fillna(-1).astype(int)
    df["evento"]   = df["evento"].str.strip()
    df["equipo"]   = df["equipo"].str.strip()
    return df


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def _split(df: pd.DataFrame):
    return df[df["tipo_ext"] == "robot_aliado"], df[df["tipo_ext"] == "robot_rival"]

def _balon(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tipo_ext"] == "balon"]

def _robots(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["tipo_ext"].isin(["robot_aliado", "robot_rival"])]

def _xy(df: pd.DataFrame):
    return df["x"].to_numpy(), df["y"].to_numpy()

def _posicion_robot_en_frame(tracking: pd.DataFrame, frame: int, robot_id: int):
    sub = tracking[tracking["objeto_id"] == robot_id]
    if sub.empty: return None
    idx = (sub["frame"] - frame).abs().argsort()
    row = sub.iloc[idx.iloc[0]]
    return float(row["x"]), float(row["y"])

def _posicion_balon_en_frame(tracking: pd.DataFrame, frame: int):
    """
    Posición del BALÓN en (o lo más cerca de) un frame.

    Es la MISMA fuente de datos que dibuja el GIF, así que cualquier evento que
    ubiquemos con esta función cae exactamente donde se ve en el GIF/video, sin
    depender de que el id de game_events.csv coincida con el id de tracking_data.csv.
    """
    sub = tracking[tracking["tipo_ext"] == "balon"]
    if sub.empty: return None
    idx = (sub["frame"] - frame).abs().argsort()
    row = sub.iloc[idx.iloc[0]]
    return float(row["x"]), float(row["y"])

def _guardar(fig, nombre: str):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    guardar(ruta, dpi=DPI_EXPORTACION)
    print(f"  [OK] {ruta}")

def generar_mapeo_robots(tracking: pd.DataFrame):
    """Genera un diccionario para identificar a los robots como 'A·R1', 'B·R2', etc."""
    robots_df = _robots(tracking)
    mapeo = {}
    for eq, prefijo in [(EQUIPO_A, "A"), (EQUIPO_B, "B")]:
        ids = sorted(robots_df[robots_df["equipo"] == eq]["objeto_id"].unique())
        for i, rid in enumerate(ids, 1):
            mapeo[int(rid)] = f"{prefijo}·R{i}"
    return mapeo

def mapeo_desde_eventos(eventos: pd.DataFrame):
    """
    Etiqueta de robot ('A·R1', 'B·R2', ...) a partir del equipo declarado en los
    PROPIOS eventos. No depende del tracking, así que el id del evento nunca se
    confunde con el id del tracking.
    """
    df = eventos[eventos["robot_id"] >= 0][["robot_id", "equipo"]].drop_duplicates()
    mapeo = {}
    for eq, prefijo in [(EQUIPO_A, "A"), (EQUIPO_B, "B")]:
        ids = sorted(df[df["equipo"] == eq]["robot_id"].unique())
        for i, rid in enumerate(ids, 1):
            mapeo[int(rid)] = f"{prefijo}·R{i}"
    return mapeo

def posiciones_control_balon(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Por cada robot, la posición promedio del BALÓN en los frames donde ese robot
    controló la pelota. Ancla los nodos de la Red de pases donde de verdad ocurrió
    la acción (igual que el GIF) y, de nuevo, sin cruzar ids con el tracking.
    """
    controles = eventos[eventos["evento"] == EVENTO_CONTROL]
    acumulado = {}
    for _, ev in controles.iterrows():
        rid = int(ev["robot_id"])
        pos = _posicion_balon_en_frame(tracking, int(ev["frame"]))
        if pos is None: continue
        acumulado.setdefault(rid, []).append(pos)
    return {rid: (float(np.mean([p[0] for p in pts])),
                  float(np.mean([p[1] for p in pts])))
            for rid, pts in acumulado.items() if pts}

def extraer_pases(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Detecta pases a partir de los cambios de 'Control de Balon' (game_events.csv),
    pero ubica cada pase con la posición del BALÓN en el tracking — la misma fuente
    que el GIF. De esta forma el mapa coincide con el GIF/video y no importa si el
    id del evento no empata con el id del tracking.
    """
    controles = eventos[eventos["evento"] == EVENTO_CONTROL].sort_values("frame").reset_index(drop=True)
    tiros_frames = sorted(eventos[eventos["evento"] == EVENTO_TIRO]["frame"].tolist())
    pases = []

    for i in range(len(controles) - 1):
        ev_o = controles.iloc[i]
        ev_d = controles.iloc[i + 1]

        rob_o = int(ev_o["robot_id"])
        rob_d = int(ev_d["robot_id"])
        if rob_o == rob_d:
            continue  # el mismo robot sigue conduciendo: no es un pase

        # Si hubo un tiro entre ambos controles, el cambio de posesión vino del
        # disparo (rebote/despeje), no de un pase. No lo contamos como pase.
        if any(ev_o["frame"] < tf < ev_d["frame"] for tf in tiros_frames):
            continue

        pos_o = _posicion_balon_en_frame(tracking, int(ev_o["frame"]))
        pos_d = _posicion_balon_en_frame(tracking, int(ev_d["frame"]))
        if pos_o is None or pos_d is None:
            continue

        eq_o = ev_o["equipo"].strip()
        eq_d = ev_d["equipo"].strip()
        exito = (eq_o == eq_d)  # mismo equipo = completado · distinto = interceptado

        pases.append({
            "origen": pos_o, "destino": pos_d,
            "r_origen": rob_o, "r_destino": rob_d,
            "eq_origen": eq_o, "eq_destino": eq_d,
            "exito": exito
        })
    return pases


# -------------------------------------------------------------
# VISTAS
# -------------------------------------------------------------
def vista_heatmap(tracking: pd.DataFrame, _eventos):
    robots = _robots(tracking)
    if robots.empty:
        print("  [INFO] Sin datos de robots para el heatmap.")
        return
    n_a = robots[robots["tipo_ext"] == "robot_aliado"]["objeto_id"].nunique()
    n_b = robots[robots["tipo_ext"] == "robot_rival"]["objeto_id"].nunique()

    # Le indicamos mostrar_cbar=False para que no altere las proporciones
    fig, _ = heatmap(
        *_xy(robots),
        titulo="Mapa de calor · Todos los robots",
        subtitulo=(f"Actividad global en cancha · Equipo A ({n_a}) vs Equipo B ({n_b})"),
        mostrar_cbar=False
    )
    _guardar(fig, "01_heatmap.png")


def vista_trayectoria(tracking: pd.DataFrame, _eventos):
    if tracking.empty: return
    fig, ax = dibujar_cancha()
    robots_df = _robots(tracking)
    mapeo = generar_mapeo_robots(tracking)

    robots_dibujados = 0
    for obj_id, grupo in robots_df.groupby("objeto_id"):
        grupo = grupo.sort_values("frame")
        if len(grupo) < 2: continue

        equipo = grupo["equipo"].iloc[0]
        color  = COLOR_EQUIPO.get(equipo, PALETA["acento"])
        x, y   = grupo["x"].to_numpy(), grupo["y"].to_numpy()
        label  = mapeo.get(int(obj_id), str(int(obj_id)))

        ax.plot(x, y, color=color, linewidth=2, alpha=0.8, zorder=4)
        ax.scatter(x[0], y[0], color=color, edgecolors="white", linewidths=1.5, s=90, marker="o", zorder=5)
        ax.scatter(x[-1], y[-1], color=color, edgecolors="white", linewidths=1.5, s=140, marker="X", zorder=5)
        ax.annotate(label, (x[-1], y[-1]), textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=9, fontweight="bold", family=FUENTE, zorder=6)
        robots_dibujados += 1

    balon = _balon(tracking).sort_values("frame")
    if not balon.empty:
        bx, by = balon["x"].to_numpy(), balon["y"].to_numpy()
        ax.plot(bx, by, color=COLOR_BALON, linewidth=1.5, alpha=0.55, linestyle=":", zorder=3)
        ax.scatter(bx[-1], by[-1], color=COLOR_BALON, edgecolors="white", linewidths=1.2, s=100, marker="o", zorder=6)

    _titulo(ax, "Trayectorias de los robots", f"{robots_dibujados} trayectorias graficadas")
    _firma(ax)

    # Leyenda muy simplificada para el usuario final
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETA["aliado"], markeredgecolor="white", label="Equipo A"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETA["rival"], markeredgecolor="white", label="Equipo B"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", markeredgecolor="white", label="Inicio"),
        plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="#888888", markeredgecolor="none", label="Fin"),
    ]
    if not balon.empty:
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_BALON, markeredgecolor="white", label="Balón"))

    leg = ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9, labelcolor=PALETA["texto"])
    leg.get_frame().set_edgecolor(PALETA["grid"])
    leg.get_frame().set_alpha(0.95)
    _guardar(fig, "02_trayectorias.png")


def vista_pases(tracking: pd.DataFrame, eventos: pd.DataFrame):
    pases = extraer_pases(tracking, eventos)
    if not pases:
        print("  [INFO] No se detectaron pases entre robots.")
        return
    comp = sum(1 for p in pases if p["exito"])
    print(f"  [INFO] {len(pases)} pases ({comp} completados, {len(pases)-comp} perdidos) "
          f"ubicados sobre la trayectoria del balón (misma fuente que el GIF).")
    fig, _ = mapa_pases(pases, titulo="Dirección y Éxito de Pases",
                        subtitulo=f"{len(pases)} pases · {comp} completados")
    _guardar(fig, "03_pases.png")


def vista_tiros(tracking: pd.DataFrame, eventos: pd.DataFrame):
    tiros_ev = eventos[eventos["evento"] == EVENTO_TIRO].copy()
    if tiros_ev.empty: return
    tiros = []
    for _, fila in tiros_ev.iterrows():
        pos = _posicion_balon_en_frame(tracking, int(fila["frame"]))
        if pos is None: continue
        detalles = str(fila.get("detalles", "")).lower()
        if "gol" in detalles: resultado = "gol"
        elif "ataj" in detalles or "salv" in detalles or "block" in detalles: resultado = "atajado"
        else: resultado = "fuera"
        tiros.append((*pos, resultado))
    if not tiros: return
    goles = sum(1 for *_, r in tiros if r == "gol")
    fig, _ = mapa_tiros(tiros, titulo="Mapa de tiros", subtitulo=f"Disparos: {len(tiros)}  ·  Goles detectados: {goles}")
    _guardar(fig, "04_tiros.png")


def vista_voronoi(tracking: pd.DataFrame, _eventos):
    aliados, rivales = _split(tracking)
    pos_aliados = [(g["x"].mean(), g["y"].mean()) for _, g in aliados.groupby("objeto_id")]
    pos_rivales = [(g["x"].mean(), g["y"].mean()) for _, g in rivales.groupby("objeto_id")]
    if not pos_aliados and not pos_rivales: return
    fig, _ = voronoi_control(pos_aliados, pos_rivales, titulo="Control de espacio · Voronoi")
    _guardar(fig, "05_voronoi.png")


def vista_posesion(tracking: pd.DataFrame, eventos: pd.DataFrame):
    controles = eventos[eventos["evento"] == EVENTO_CONTROL].sort_values("frame")
    aliado_xy, rival_xy = [], []
    for _, fila in controles.iterrows():
        pos = _posicion_balon_en_frame(tracking, int(fila["frame"]))
        if pos is None: continue
        if fila["equipo"].strip() == EQUIPO_A: aliado_xy.append(pos)
        else: rival_xy.append(pos)

    if not aliado_xy and not rival_xy:
        aliados, rivales = _split(tracking)
        aliado_xy = list(zip(*_xy(aliados))) if not aliados.empty else []
        rival_xy  = list(zip(*_xy(rivales))) if not rivales.empty else []
        subtitulo = "Posesión por presencia de robots en zona"
    else:
        total = len(aliado_xy) + len(rival_xy)
        pct = len(aliado_xy) / total * 100 if total else 0
        subtitulo = f"Equipo A {pct:.0f}%  ·  Equipo B {100-pct:.0f}%"

    fig, _ = posesion_zonas(aliado_xy, rival_xy, titulo="Posesión por zonas", subtitulo=subtitulo)
    _guardar(fig, "06_posesion.png")


def vista_red_pases(tracking: pd.DataFrame, eventos: pd.DataFrame):
    pases = extraer_pases(tracking, eventos)
    if not pases: return

    # Etiquetas y posiciones derivadas SOLO de eventos + balón (igual que el mapa y el GIF)
    mapeo = mapeo_desde_eventos(eventos)
    posiciones = posiciones_control_balon(tracking, eventos)

    # Agrupar las conexiones y contar cantidad
    conexiones = {}
    for p in pases:
        clave = (p["r_origen"], p["r_destino"], p["exito"])
        conexiones[clave] = conexiones.get(clave, 0) + 1

    fig, _ = red_pases(
        posiciones, conexiones, mapeo,
        titulo="Red de pases de todo el partido",
        subtitulo="Posición = dónde cada robot tocó el balón · Grosor = nº de pases"
    )
    _guardar(fig, "07_red_pases.png")


def vista_gif(tracking: pd.DataFrame, eventos: pd.DataFrame):
    eventos_por_frame = eventos.groupby("frame").apply(
        lambda g: g[["evento", "robot_id", "equipo"]].to_dict("records")
    ).to_dict()
    frames_unicos = sorted(tracking["frame"].unique())
    fig, ax = dibujar_cancha()

    def dibujar_frame(num_frame):
        ax.clear()
        dibujar_cancha(ax=ax)
        datos_frame = tracking[tracking["frame"] == num_frame]
        t_val = datos_frame["tiempo"]
        t_txt = f"t = {t_val.iloc[0]:.2f} s" if len(t_val) else ""
        evs_frame = eventos_por_frame.get(num_frame, [])
        ev_txt = f"  ⚡ {evs_frame[0]['evento']} — Robot {evs_frame[0]['robot_id']}" if evs_frame else ""

        _titulo(ax, "Rastreo de robots", f"Frame {num_frame}  ·  {t_txt}{ev_txt}")
        _firma(ax)

        for _, entidad in datos_frame.iterrows():
            r_id = int(entidad["objeto_id"])
            tipo_ext = entidad["tipo_ext"]
            bx, by = entidad["x"], entidad["y"]

            if tipo_ext == "balon":
                historial_b = tracking[(tracking["tipo_ext"] == "balon") &
                                       (tracking["frame"] <= num_frame) &
                                       (tracking["frame"] > num_frame - VENTANA_ESTELA)].sort_values("frame")
                if len(historial_b) > 1:
                    ax.plot(historial_b["x"], historial_b["y"], color=COLOR_BALON, alpha=0.35, linewidth=1.5, linestyle=":", zorder=4)
                ax.scatter(bx, by, color=COLOR_BALON, s=140, edgecolors="white", linewidths=1.5, zorder=6)
                continue

            equipo = entidad["equipo"]
            color = COLOR_EQUIPO.get(equipo, PALETA["acento"])

            historial = tracking[(tracking["objeto_id"] == r_id) & (tracking["tipo_ext"] != "balon") &
                                 (tracking["frame"] <= num_frame) & (tracking["frame"] > num_frame - VENTANA_ESTELA)].sort_values("frame")
            if len(historial) > 1:
                ax.plot(historial["x"], historial["y"], color=color, alpha=0.30, linewidth=2, linestyle="--", zorder=4)

            ax.scatter(bx, by, color=color, s=360, edgecolors="white", linewidths=2, zorder=5)
            ax.text(bx, by, str(r_id), color="white", ha="center", va="center", fontweight="bold", fontsize=10, zorder=6, family=FUENTE)

            for ev in evs_frame:
                if ev["robot_id"] == r_id:
                    ax.scatter(bx, by, s=800, edgecolors=PALETA["aviso"], facecolors="none", linewidths=3, zorder=7)
                    ax.text(bx, by + 7, ev["evento"][:14], color=PALETA["aviso"], ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=8, family=FUENTE)

            if str(entidad.get("intervencion_humana", "")).lower() == "si":
                ax.scatter(bx, by, s=720, edgecolors=PALETA["aviso"], facecolors="none", linewidths=3, zorder=4)
                ax.text(bx, by + 6, "MANO", color=PALETA["aviso"], ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=6, family=FUENTE)

        _leyenda(ax, [(PALETA["aliado"], "Equipo A", "o"), (PALETA["rival"],  "Equipo B", "o"), (COLOR_BALON, "Balón", "o")], loc="upper right")

    print(f"  [INFO] Renderizando {len(frames_unicos)} frames para el GIF...")
    cuadros_png = []
    for num_frame in frames_unicos:
        dibujar_frame(num_frame)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI_EXPORTACION, bbox_inches="tight", facecolor=PALETA["fondo"])
        buf.seek(0)
        cuadros_png.append(Image.open(buf).convert("RGB"))
    plt.close(fig)

    ancho_max = max(im.width for im in cuadros_png)
    alto_max  = max(im.height for im in cuadros_png)
    cuadros = []
    for im in cuadros_png:
        lienzo = Image.new("RGB", (ancho_max, alto_max), PALETA["fondo"])
        x = (ancho_max - im.width) // 2
        y = (alto_max - im.height) // 2
        lienzo.paste(im, (x, y))
        cuadros.append(lienzo)

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta_gif = os.path.join(CARPETA_RESULTADOS, "tracking_visualization.gif")
    cuadros[0].save(ruta_gif, save_all=True, append_images=cuadros[1:], duration=int(1000 / FPS), loop=0)
    print(f"  [ÉXITO] GIF guardado en: {ruta_gif}")


def main():
    print("=" * 64)
    print("=== Copa FutBotMX · AZTEM — Generador unificado de vistas ===")
    print("=" * 64)

    if not os.path.exists(ARCHIVO_TRACKING):
        print(f"[ERROR] No se encontró: {ARCHIVO_TRACKING}")
        return
    tracking = cargar_tracking(ARCHIVO_TRACKING)
    if tracking.empty:
        return

    eventos = cargar_eventos(ARCHIVO_EVENTOS) if os.path.exists(ARCHIVO_EVENTOS) else pd.DataFrame(columns=["frame", "tiempo", "evento", "robot_id", "equipo", "detalles"])

    pasos = [
        ("Heatmap",            vista_heatmap),
        ("Trayectorias",       vista_trayectoria),
        ("Mapa de pases",      vista_pases),
        ("Mapa de tiros",      vista_tiros),
        ("Voronoi",            vista_voronoi),
        ("Posesión por zonas", vista_posesion),
        ("Total de pases",     vista_red_pases),
        ("GIF de animación",   vista_gif),
    ]

    for nombre, func in pasos:
        print(f"\n── {nombre} ──")
        try:
            func(tracking, eventos)
        except Exception as e:
            print(f"  [WARN] {nombre} falló: {e}")

    print("\n" + "=" * 64)
    print(f"[LISTO] Todas las vistas guardadas en '{CARPETA_VISUALIZACIONES}'")
    print("=" * 64)

if __name__ == "__main__":
    main()