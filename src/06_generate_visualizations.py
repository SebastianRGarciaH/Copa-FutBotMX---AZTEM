"""
================================================================
 06_generate_visualizations.py — Copa FutBotMX · Equipo AZTEM
 Motor unificado de visualizaciones + animación GIF
================================================================
 UBICACIÓN ESPERADA: <raíz_del_proyecto>/src/06_generate_visualizations.py

 ESTRUCTURA DE CARPETAS:
   <raíz_del_proyecto>/
     results/
       metrics/
         tracking_data.csv
         game_events.csv
       visualizations/        <- aquí se guardan TODAS las salidas
     src/
       06_generate_visualizations.py   <- este archivo

 Las rutas se calculan a partir de la ubicación de este archivo
 (no del directorio desde el que se ejecute), así que el script
 funciona sin importar desde dónde se llame.

 FUENTES DE DATOS:
   - results/metrics/tracking_data.csv → posiciones frame a frame
       Columnas reales: frame, tiempo, id_objeto, tipo, equipo, x, y, area, intervencion_humana
       tipo   = "robot" | "balon"
       equipo = "Equipo A" | "Equipo B" | "balon"
       ⚠ El balón SÍ viene como una fila más (tipo == "balon",
         id_objeto == 0) y se clasifica aparte para que nunca se
         dibuje como si fuera un robot.

   - results/metrics/game_events.csv → eventos reales del partido
       Columnas reales: frame, tiempo, evento, robot_implicado, equipo, detalles
       evento = "Control de Balon" | "Tiro a Gol / Despeje"
================================================================
 NOTA SOBRE EL MOTOR DE DIBUJO:
 Antes vivía en cancha_template.py (módulo aparte). Ahora está
 incluido en este mismo archivo para no depender de un import
 externo.

 NOTA SOBRE EL GIF:
 Las vistas estáticas se exportan con guardar(), que usa
 bbox_inches="tight" para recortar el margen blanco sobrante.
 matplotlib.animation no soporta ese recorte por cuadro, así que
 el GIF ya NO se genera con FuncAnimation: cada cuadro se renderiza
 y se recorta exactamente igual que un PNG (mismo bbox_inches
 "tight", mismo dpi) y luego se compone con Pillow. Así el GIF
 queda con las mismas proporciones que el resto de las vistas.
================================================================
"""

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
# MOTOR DE DIBUJO DE CANCHA (antes cancha_template.py)
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
    ax.add_patch(patches.Circle((largo / 2, ancho / 2), CANCHA["circulo_radio"],
                 fill=False, edgecolor=lin, linewidth=lw, zorder=2))
    ax.add_patch(patches.Circle((largo / 2, ancho / 2), 1.6, color=lin, zorder=2))
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
    leg = ax.legend(handles=[
        plt.Line2D([0], [0], marker=mk, color="none", markerfacecolor=c,
                   markeredgecolor="none", markersize=11, label=lbl)
        for c, lbl, mk in items], loc=loc, frameon=True, fontsize=9,
        labelcolor=PALETA["texto"], handletextpad=0.4)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor(PALETA["grid"])
    leg.get_frame().set_alpha(0.95)


def heatmap(xs, ys, titulo="Mapa de calor", subtitulo=None, bins=40):
    fig, ax = dibujar_cancha(con_grid=False)
    heat, _, _ = np.histogram2d(xs, ys, bins=bins,
        range=[[0, CANCHA["largo"]], [0, CANCHA["ancho"]]])
    if SCIPY_OK:
        heat = gaussian_filter(heat, sigma=1.4)
    cmap = LinearSegmentedColormap.from_list("calor", [
        (0.00, (1, 1, 1, 0)), (0.20, (0.30, 0.55, 0.85, 0.45)),
        (0.50, (0.45, 0.35, 0.75, 0.70)), (0.78, (0.90, 0.45, 0.30, 0.85)),
        (1.00, (0.80, 0.10, 0.15, 0.95))])
    im = ax.imshow(heat.T, extent=[0, CANCHA["largo"], 0, CANCHA["ancho"]],
                   origin="lower", cmap=cmap, interpolation="bilinear",
                   zorder=1, aspect="auto")
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
    fig, ax = dibujar_cancha()
    completados = sum(1 for *_, ok in pases if ok); total = len(pases)
    for x1, y1, x2, y2, ok in pases:
        color = PALETA["exito"] if ok else PALETA["fallo"]
        estilo = "-" if ok else "--"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                    alpha=0.9, shrinkA=2, shrinkB=2,
                                    linestyle=estilo), zorder=3)
        ax.scatter(x1, y1, s=20, c=color, zorder=4,
                   edgecolor="#ffffff", linewidth=0.8)
    _leyenda(ax, [(PALETA["exito"], "Completado", "o"),
                  (PALETA["fallo"], "Fallido", "o")])
    pct = (completados / total * 100) if total else 0
    sub = subtitulo or f"Precision: {completados}/{total} ({pct:.0f}%)"
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


def posesion_zonas(aliado_xy, rival_xy, titulo="Posesion por zonas",
                   subtitulo=None, cols=6, filas=4):
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


def red_pases(posiciones, conexiones, titulo="Red de pases", subtitulo=None):
    fig, ax = dibujar_cancha()
    max_pases = max((c for *_, c in conexiones), default=1)
    for o, d, cant in conexiones:
        if o not in posiciones or d not in posiciones: continue
        x1, y1 = posiciones[o]; x2, y2 = posiciones[d]
        grosor = 1 + 6 * (cant / max_pases)
        ax.plot([x1, x2], [y1, y2], color=PALETA["acento"],
                linewidth=grosor, alpha=0.45, zorder=3, solid_capstyle="round")
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(cant), fontsize=7.5,
                color=PALETA["acento"], family=FUENTE, ha="center",
                va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff",
                          ec=PALETA["grid"], lw=0.6))
    for rid, (x, y) in posiciones.items():
        ax.scatter(x, y, s=620, c=PALETA["acento"], edgecolor="#ffffff",
                   linewidth=2.5, zorder=4)
        ax.text(x, y, str(rid), ha="center", va="center", fontsize=11,
                color="#ffffff", family=FUENTE, fontweight="bold", zorder=5)
    _titulo(ax, titulo, subtitulo or "Grosor de linea = cantidad de pases")
    _firma(ax)
    return fig, ax


def dashboard(metricas, titulo="Resumen del partido"):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(PALETA["fondo"])
    ax.set_facecolor(PALETA["fondo"])
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.text(0.6, 7.4, titulo, ha="left", fontsize=20, fontweight="bold",
            color=PALETA["texto"], family=FUENTE)
    ax.text(0.6, 6.95, "Copa FutBotMX · Equipo AZTEM", ha="left", fontsize=10,
            color=PALETA["acento"], family=FUENTE)
    ax.plot([0.6, 11.4], [6.75, 6.75], color=PALETA["acento"], lw=2)
    items = list(metricas.items())
    cols = 3; cw, ch = 3.5, 2.1; x0, y0 = 0.6, 4.2
    gap_x, gap_y = 0.35, 0.4
    for idx, (clave, valor) in enumerate(items):
        col = idx % cols; fila = idx // cols
        x = x0 + col * (cw + gap_x); y = y0 - fila * (ch + gap_y)
        ax.add_patch(patches.FancyBboxPatch((x, y), cw, ch,
                     boxstyle="round,pad=0.06,rounding_size=0.12",
                     facecolor=PALETA["cancha"], edgecolor=PALETA["grid"],
                     linewidth=1.3))
        ax.add_patch(patches.Rectangle((x, y + ch - 0.12), cw, 0.12,
                     facecolor=PALETA["acento"], edgecolor="none"))
        ax.text(x + cw / 2, y + ch * 0.58, str(valor), ha="center",
                fontsize=27, fontweight="bold", color=PALETA["acento"],
                family=FUENTE)
        ax.text(x + cw / 2, y + ch * 0.20, clave, ha="center", fontsize=10.5,
                color=PALETA["texto_tenue"], family=FUENTE)
    return fig, ax


def guardar(nombre, dpi=170):
    plt.savefig(nombre, dpi=dpi, bbox_inches="tight", facecolor=PALETA["fondo"])
    plt.close()


# Mismo dpi que usa guardar() — se reutiliza para que los cuadros del
# GIF salgan con la misma resolución/proporción que los PNG estáticos.
DPI_EXPORTACION = 170


# ================================================================
# RUTAS DEL PROYECTO
# ================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # .../src
RAIZ_PROYECTO = os.path.dirname(BASE_DIR)                     # carpeta del proyecto

CARPETA_METRICAS         = os.path.join(RAIZ_PROYECTO, "results", "metrics")
CARPETA_VISUALIZACIONES  = os.path.join(RAIZ_PROYECTO, "results", "visualizations")

ARCHIVO_TRACKING   = os.path.join(CARPETA_METRICAS, "tracking_data.csv")
ARCHIVO_EVENTOS    = os.path.join(CARPETA_METRICAS, "game_events.csv")
CARPETA_RESULTADOS = CARPETA_VISUALIZACIONES  # nombre usado por _guardar()

VENTANA_ESTELA = 10
FPS            = 12

EQUIPO_A = "Equipo A"
EQUIPO_B = "Equipo B"

# Colores por equipo (mapeados a aliado/rival del template)
COLOR_EQUIPO = {
    EQUIPO_A: PALETA["aliado"],
    EQUIPO_B: PALETA["rival"],
}

# El balón siempre se dibuja en negro, nunca con color de equipo
COLOR_BALON = "black"

EVENTO_TIRO    = "Tiro a Gol / Despeje"
EVENTO_CONTROL = "Control de Balon"


# -------------------------------------------------------------
# CARGA Y NORMALIZACIÓN — TRACKING
# -------------------------------------------------------------
def cargar_tracking(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()

    # Renombrar id_objeto → objeto_id para consistencia interna
    if "id_objeto" in df.columns and "objeto_id" not in df.columns:
        df.rename(columns={"id_objeto": "objeto_id"}, inplace=True)

    # Normalizar equipo y tipo
    df["equipo"] = df["equipo"].astype(str).str.strip()
    df["tipo"]   = df["tipo"].astype(str).str.strip().str.lower()

    # Crear columna tipo_ext: "robot_aliado" / "robot_rival" / "balon"
    def _clasificar(fila):
        if fila["tipo"] == "balon":
            return "balon"
        if fila["equipo"] == EQUIPO_A:
            return "robot_aliado"
        if fila["equipo"] == EQUIPO_B:
            return "robot_rival"
        return "desconocido"

    df["tipo_ext"] = df.apply(_clasificar, axis=1)

    for col, default in [("frame", 0), ("tiempo", 0.0),
                         ("objeto_id", 0), ("intervencion_humana", "no")]:
        if col not in df.columns:
            df[col] = default

    df = df.dropna(subset=["x", "y"])
    return df


# -------------------------------------------------------------
# CARGA Y NORMALIZACIÓN — EVENTOS
# -------------------------------------------------------------
def cargar_eventos(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip().str.lower()

    # robot_implicado → robot_id
    if "robot_implicado" in df.columns and "robot_id" not in df.columns:
        df.rename(columns={"robot_implicado": "robot_id"}, inplace=True)

    for col, default in [("frame", 0), ("tiempo", 0.0),
                         ("robot_id", -1), ("equipo", ""),
                         ("detalles", "")]:
        if col not in df.columns:
            df[col] = default

    df["robot_id"] = pd.to_numeric(df["robot_id"], errors="coerce").fillna(-1).astype(int)
    df["evento"]   = df["evento"].str.strip()
    df["equipo"]   = df["equipo"].str.strip()
    return df


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def _split(df: pd.DataFrame):
    """Divide tracking en (aliados_df, rivales_df) por tipo_ext."""
    aliados = df[df["tipo_ext"] == "robot_aliado"]
    rivales = df[df["tipo_ext"] == "robot_rival"]
    return aliados, rivales


def _balon(df: pd.DataFrame) -> pd.DataFrame:
    """Filas correspondientes al balón (tipo_ext == 'balon')."""
    return df[df["tipo_ext"] == "balon"]


def _robots(df: pd.DataFrame) -> pd.DataFrame:
    """Filas correspondientes solo a robots (excluye el balón)."""
    return df[df["tipo_ext"].isin(["robot_aliado", "robot_rival"])]


def _xy(df: pd.DataFrame):
    return df["x"].to_numpy(), df["y"].to_numpy()


def _posicion_robot_en_frame(tracking: pd.DataFrame, frame: int, robot_id: int):
    """Devuelve (x, y) del robot robot_id en el frame más cercano disponible."""
    sub = tracking[tracking["objeto_id"] == robot_id]
    if sub.empty:
        return None
    idx = (sub["frame"] - frame).abs().argsort()
    row = sub.iloc[idx.iloc[0]]
    return float(row["x"]), float(row["y"])


def _guardar(fig, nombre: str):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre)
    guardar(ruta, dpi=DPI_EXPORTACION)
    print(f"  [OK] {ruta}")


# -------------------------------------------------------------
# VISTA 1 — HEATMAP (uno solo, combinando todos los robots)
# -------------------------------------------------------------
def vista_heatmap(tracking: pd.DataFrame, _eventos):
    """Genera UN SOLO heatmap con la actividad combinada de todos los
    robots (Equipo A + Equipo B). El balón se excluye porque este mapa
    es de presencia de robots, no de la pelota."""
    robots = _robots(tracking)
    if robots.empty:
        print("  [INFO] Sin datos de robots para el heatmap.")
        return

    n_a = robots[robots["tipo_ext"] == "robot_aliado"]["objeto_id"].nunique()
    n_b = robots[robots["tipo_ext"] == "robot_rival"]["objeto_id"].nunique()

    fig, _ = heatmap(
        *_xy(robots),
        titulo="Mapa de calor · Todos los robots",
        subtitulo=f"Actividad global en cancha · Equipo A ({n_a}) vs Equipo B ({n_b})",
    )
    _guardar(fig, "01_heatmap.png")


# -------------------------------------------------------------
# VISTA 2 — TRAYECTORIAS (una sola gráfica con todos los robots)
# -------------------------------------------------------------
def vista_trayectoria(tracking: pd.DataFrame, _eventos):
    """Genera UNA SOLA gráfica con las trayectorias de todos los robots
    superpuestas sobre la cancha, coloreadas por equipo. El balón se
    dibuja por separado como una pelota negra, nunca como un robot."""
    if tracking.empty:
        print("  [INFO] Sin datos de tracking para trayectorias.")
        return

    fig, ax = dibujar_cancha()

    robots_dibujados = 0
    for obj_id, grupo in _robots(tracking).groupby("objeto_id"):
        grupo = grupo.sort_values("frame")
        if len(grupo) < 2:
            continue

        equipo = grupo["equipo"].iloc[0]
        color  = COLOR_EQUIPO.get(equipo, PALETA["acento"])
        x, y = grupo["x"].to_numpy(), grupo["y"].to_numpy()

        ax.plot(x, y, color=color, linewidth=2, alpha=0.8, zorder=4)
        ax.scatter(x[0], y[0], color=color, edgecolors="white",
                   linewidths=1.5, s=90, marker="o", zorder=5)
        ax.scatter(x[-1], y[-1], color=color, edgecolors="white",
                   linewidths=1.5, s=140, marker="X", zorder=5)
        ax.annotate(str(int(obj_id)), (x[-1], y[-1]),
                    textcoords="offset points", xytext=(6, 6),
                    color=color, fontsize=9, fontweight="bold",
                    family=FUENTE, zorder=6)
        robots_dibujados += 1

    # Trayectoria del balón: línea punteada negra, sin ID ni marca de robot
    balon = _balon(tracking).sort_values("frame")
    if not balon.empty:
        bx, by = balon["x"].to_numpy(), balon["y"].to_numpy()
        ax.plot(bx, by, color=COLOR_BALON, linewidth=1.5, alpha=0.55,
                linestyle=":", zorder=3)
        ax.scatter(bx[-1], by[-1], color=COLOR_BALON, edgecolors="white",
                   linewidths=1.2, s=100, marker="o", zorder=6)

    n_a = tracking[tracking["tipo_ext"] == "robot_aliado"]["objeto_id"].nunique()
    n_b = tracking[tracking["tipo_ext"] == "robot_rival"]["objeto_id"].nunique()

    _titulo(ax, "Trayectorias de todos los robots",
           f"Equipo A: {n_a} robots  ·  Equipo B: {n_b} robots  ·  "
           f"{robots_dibujados} trayectorias graficadas "
           f"(○ inicio, ✕ final)")
    _firma(ax)
    leyenda = [
        (PALETA["aliado"], "Equipo A", "o"),
        (PALETA["rival"],  "Equipo B", "o"),
    ]
    if not balon.empty:
        leyenda.append((COLOR_BALON, "Balón", "o"))
    _leyenda(ax, leyenda, loc="upper right")

    _guardar(fig, "02_trayectorias.png")


# -------------------------------------------------------------
# VISTA 3 — MAPA DE PASES  (desde game_events.csv)
# -------------------------------------------------------------
def vista_pases(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Detecta transferencias de control entre robots distintos.
    Usa la posición del robot implicado en tracking como origen/destino.
    """
    controles = eventos[eventos["evento"] == EVENTO_CONTROL] \
                       .sort_values("frame").reset_index(drop=True)

    if len(controles) < 2:
        print("  [INFO] Sin suficientes eventos de control para mapa de pases.")
        return

    pases = []
    for i in range(len(controles) - 1):
        fo = controles.iloc[i]
        fd = controles.iloc[i + 1]
        if fo["robot_id"] == fd["robot_id"]:
            continue  # mismo robot, no es pase

        pos_o = _posicion_robot_en_frame(tracking, fo["frame"], fo["robot_id"])
        pos_d = _posicion_robot_en_frame(tracking, fd["frame"], fd["robot_id"])
        if pos_o is None or pos_d is None:
            continue

        mismo_equipo = (fo["equipo"].strip() == fd["equipo"].strip())
        pases.append((*pos_o, *pos_d, mismo_equipo))

    if not pases:
        print("  [INFO] No se construyeron pases entre robots distintos.")
        return

    fig, _ = mapa_pases(
        pases,
        titulo="Mapa de pases · Datos reales",
        subtitulo=f"{len(pases)} transferencias de control detectadas",
    )
    _guardar(fig, "03_pases.png")


# -------------------------------------------------------------
# VISTA 4 — MAPA DE TIROS  (desde game_events.csv)
# -------------------------------------------------------------
def vista_tiros(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Filtra eventos 'Tiro a Gol / Despeje' y cruza con tracking
    para obtener la posición real del robot disparador.
    """
    tiros_ev = eventos[eventos["evento"] == EVENTO_TIRO].copy()

    if tiros_ev.empty:
        print("  [INFO] Sin eventos de tiro en game_events.csv.")
        return

    tiros = []
    for _, fila in tiros_ev.iterrows():
        rid = int(fila["robot_id"])
        pos = _posicion_robot_en_frame(tracking, int(fila["frame"]), rid)
        if pos is None:
            continue

        detalles = str(fila.get("detalles", "")).lower()
        if "gol" in detalles:
            resultado = "gol"
        elif "ataj" in detalles or "salv" in detalles or "block" in detalles:
            resultado = "atajado"
        else:
            resultado = "fuera"

        tiros.append((*pos, resultado))

    if not tiros:
        print("  [INFO] No se cruzaron tiros con posiciones de tracking.")
        return

    goles = sum(1 for *_, r in tiros if r == "gol")
    fig, _ = mapa_tiros(
        tiros,
        titulo="Mapa de tiros · Datos reales",
        subtitulo=f"Disparos: {len(tiros)}  ·  Goles detectados: {goles}",
    )
    _guardar(fig, "04_tiros.png")


# -------------------------------------------------------------
# VISTA 5 — VORONOI
# -------------------------------------------------------------
def vista_voronoi(tracking: pd.DataFrame, _eventos):
    aliados, rivales = _split(tracking)

    pos_aliados = [(g["x"].mean(), g["y"].mean())
                   for _, g in aliados.groupby("objeto_id")]
    pos_rivales = [(g["x"].mean(), g["y"].mean())
                   for _, g in rivales.groupby("objeto_id")]

    if not pos_aliados and not pos_rivales:
        print("  [INFO] Sin posiciones para Voronoi.")
        return

    fig, _ = voronoi_control(
        pos_aliados, pos_rivales,
        titulo="Control de espacio · Voronoi",
        subtitulo="Posición promedio por robot · Equipo A vs Equipo B",
    )
    _guardar(fig, "05_voronoi.png")


# -------------------------------------------------------------
# VISTA 6 — POSESIÓN POR ZONAS  (desde game_events.csv)
# -------------------------------------------------------------
def vista_posesion(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Usa eventos 'Control de Balon' para determinar posesión y
    la posición del robot implicado como punto de posesión.
    """
    controles = eventos[eventos["evento"] == EVENTO_CONTROL].sort_values("frame")

    aliado_xy, rival_xy = [], []
    for _, fila in controles.iterrows():
        pos = _posicion_robot_en_frame(tracking, int(fila["frame"]),
                                       int(fila["robot_id"]))
        if pos is None:
            continue
        if fila["equipo"].strip() == EQUIPO_A:
            aliado_xy.append(pos)
        else:
            rival_xy.append(pos)

    if not aliado_xy and not rival_xy:
        # Fallback: presencia de robots en cada zona
        aliados, rivales = _split(tracking)
        aliado_xy = list(zip(*_xy(aliados))) if not aliados.empty else []
        rival_xy  = list(zip(*_xy(rivales))) if not rivales.empty else []
        subtitulo = "Posesión por presencia de robots en zona"
    else:
        total     = len(aliado_xy) + len(rival_xy)
        pct       = len(aliado_xy) / total * 100 if total else 0
        subtitulo = (f"Eventos reales · "
                     f"Equipo A {pct:.0f}%  ·  Equipo B {100-pct:.0f}%")

    fig, _ = posesion_zonas(aliado_xy, rival_xy,
                            titulo="Posesión por zonas",
                            subtitulo=subtitulo)
    _guardar(fig, "06_posesion.png")


# -------------------------------------------------------------
# VISTA 7 — RED DE PASES  (desde game_events.csv)
# -------------------------------------------------------------
def vista_red_pases(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Cuenta transferencias de control entre robots distintos.
    Los nodos usan la posición promedio de cada robot en tracking.
    """
    controles = eventos[eventos["evento"] == EVENTO_CONTROL] \
                       .sort_values("frame").reset_index(drop=True)

    if len(controles) < 2:
        print("  [INFO] Sin suficientes eventos para red de pases.")
        return

    # Posiciones promedio de todos los robots
    posiciones = {}
    for rid, grupo in tracking.groupby("objeto_id"):
        posiciones[int(rid)] = (grupo["x"].mean(), grupo["y"].mean())

    conexiones: dict[tuple, int] = {}
    for i in range(len(controles) - 1):
        o = int(controles.iloc[i]["robot_id"])
        d = int(controles.iloc[i + 1]["robot_id"])
        if o != d and o in posiciones and d in posiciones:
            clave = (o, d)
            conexiones[clave] = conexiones.get(clave, 0) + 1

    if not conexiones:
        print("  [INFO] No se detectaron pases entre robots distintos.")
        return

    lista_con = [(o, d, c) for (o, d), c in conexiones.items()]
    total_pases = sum(c for *_, c in lista_con)
    fig, _ = red_pases(
        posiciones, lista_con,
        titulo="Red de pases · Eventos reales",
        subtitulo=f"{total_pases} transferencias de control detectadas",
    )
    _guardar(fig, "07_red_pases.png")


# -------------------------------------------------------------
# VISTA 8 — DASHBOARD
# -------------------------------------------------------------
def vista_dashboard(tracking: pd.DataFrame, eventos: pd.DataFrame):
    aliados, rivales = _split(tracking)

    n_aliados  = aliados["objeto_id"].nunique()
    n_rivales  = rivales["objeto_id"].nunique()
    n_frames   = tracking["frame"].nunique()
    n_tiros    = int((eventos["evento"] == EVENTO_TIRO).sum())
    n_controles = int((eventos["evento"] == EVENTO_CONTROL).sum())
    intervenciones = int(
        (tracking["intervencion_humana"].astype(str).str.lower() == "si").sum()
    )

    # Distancia total recorrida por cada equipo
    def dist_total(df):
        total = 0.0
        for _, g in df.groupby("objeto_id"):
            g = g.sort_values("frame")
            total += np.hypot(g["x"].diff(), g["y"].diff()).sum()
        return total

    dist_a = dist_total(aliados)
    dist_b = dist_total(rivales)

    # Posesión desde eventos
    ctrl = eventos[eventos["evento"] == EVENTO_CONTROL]
    pct_a = 0
    if len(ctrl):
        pct_a = round((ctrl["equipo"].str.strip() == EQUIPO_A).sum() / len(ctrl) * 100)

    metricas = {
        "Robots Eq. A":       n_aliados,
        "Robots Eq. B":       n_rivales,
        "Frames analizados":  n_frames,
        "Dist. A (cm)":       f"{dist_a:.0f}",
        "Dist. B (cm)":       f"{dist_b:.0f}",
        "Tiros a gol":        n_tiros,
        "Controles balón":    n_controles,
        f"Posesión A":        f"{pct_a}%",
        "Intervenciones":     intervenciones,
    }
    fig, _ = dashboard(metricas, titulo="Resumen del partido · AZTEM")
    _guardar(fig, "08_dashboard.png")


# -------------------------------------------------------------
# VISTA 9 — GIF  (robots diferenciados por equipo + eventos)
# -------------------------------------------------------------
def vista_gif(tracking: pd.DataFrame, eventos: pd.DataFrame):
    """
    Genera el GIF de rastreo. Cada cuadro se recorta con
    bbox_inches="tight" (igual que guardar()) para que el GIF tenga
    EXACTAMENTE las mismas proporciones que el resto de las vistas;
    matplotlib.animation no soporta ese recorte por cuadro, así que
    en vez de FuncAnimation se renderiza manualmente cuadro por
    cuadro y se compone con Pillow.
    """
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
        ev_txt = ""
        if evs_frame:
            ev = evs_frame[0]
            ev_txt = f"  ⚡ {ev['evento']} — Robot {ev['robot_id']}"

        _titulo(ax, "Rastreo de robots",
               f"Frame {num_frame}  ·  {t_txt}{ev_txt}")
        _firma(ax)

        for _, entidad in datos_frame.iterrows():
            r_id     = int(entidad["objeto_id"])
            tipo_ext = entidad["tipo_ext"]
            bx, by   = entidad["x"], entidad["y"]

            if tipo_ext == "balon":
                # El balón se dibuja como una pelota negra, nunca como robot
                historial_b = tracking[
                    (tracking["tipo_ext"] == "balon") &
                    (tracking["frame"] <= num_frame) &
                    (tracking["frame"] >  num_frame - VENTANA_ESTELA)
                ].sort_values("frame")
                if len(historial_b) > 1:
                    ax.plot(historial_b["x"], historial_b["y"],
                            color=COLOR_BALON, alpha=0.35, linewidth=1.5,
                            linestyle=":", zorder=4)
                ax.scatter(bx, by, color=COLOR_BALON, s=140,
                           edgecolors="white", linewidths=1.5, zorder=6)
                continue

            equipo = entidad["equipo"]
            color  = COLOR_EQUIPO.get(equipo, PALETA["acento"])

            # Estela
            historial = tracking[
                (tracking["objeto_id"] == r_id) &
                (tracking["tipo_ext"] != "balon") &
                (tracking["frame"] <= num_frame) &
                (tracking["frame"] >  num_frame - VENTANA_ESTELA)
            ].sort_values("frame")
            if len(historial) > 1:
                ax.plot(historial["x"], historial["y"],
                        color=color, alpha=0.30, linewidth=2,
                        linestyle="--", zorder=4)

            # Robot
            ax.scatter(bx, by, color=color, s=360,
                       edgecolors="white", linewidths=2, zorder=5)
            ax.text(bx, by, str(r_id), color="white",
                    ha="center", va="center", fontweight="bold",
                    fontsize=10, zorder=6, family=FUENTE)

            # Resaltar robot implicado en evento
            for ev in evs_frame:
                if ev["robot_id"] == r_id:
                    ax.scatter(bx, by, s=800,
                               edgecolors=PALETA["aviso"],
                               facecolors="none", linewidths=3, zorder=7)
                    ax.text(bx, by + 7, ev["evento"][:14],
                            color=PALETA["aviso"], ha="center",
                            va="bottom", fontsize=7, fontweight="bold",
                            zorder=8, family=FUENTE)

            # Intervención humana
            if str(entidad.get("intervencion_humana", "")).lower() == "si":
                ax.scatter(bx, by, s=720,
                           edgecolors=PALETA["aviso"],
                           facecolors="none", linewidths=3, zorder=4)
                ax.text(bx, by + 6, "MANO", color=PALETA["aviso"],
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold", zorder=6, family=FUENTE)

        _leyenda(ax, [
            (PALETA["aliado"], "Equipo A", "o"),
            (PALETA["rival"],  "Equipo B", "o"),
            (COLOR_BALON,      "Balón",    "o"),
        ], loc="upper right")

    print(f"  [INFO] Renderizando {len(frames_unicos)} frames para el GIF...")

    cuadros_png = []
    for num_frame in frames_unicos:
        dibujar_frame(num_frame)
        buf = io.BytesIO()
        # Mismo recorte y mismo dpi que usan los PNG estáticos
        fig.savefig(buf, format="png", dpi=DPI_EXPORTACION,
                    bbox_inches="tight", facecolor=PALETA["fondo"])
        buf.seek(0)
        cuadros_png.append(Image.open(buf).convert("RGB"))
    plt.close(fig)

    # El recorte "tight" puede variar uno o dos píxeles entre cuadros
    # (por ejemplo el texto "Frame 9" vs "Frame 10"); unificamos el
    # tamaño de lienzo para que el GIF no "salte" al reproducirse.
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
    cuadros[0].save(ruta_gif, save_all=True, append_images=cuadros[1:],
                    duration=int(1000 / FPS), loop=0)
    print(f"  [ÉXITO] GIF guardado en: {ruta_gif}")


# -------------------------------------------------------------
# PIPELINE PRINCIPAL
# -------------------------------------------------------------
def main():
    print("=" * 64)
    print("=== Copa FutBotMX · AZTEM — Generador unificado de vistas ===")
    print("=" * 64)
    print(f"[INFO] Carpeta de métricas:       {CARPETA_METRICAS}")
    print(f"[INFO] Carpeta de visualizaciones: {CARPETA_VISUALIZACIONES}\n")

    if not os.path.exists(ARCHIVO_TRACKING):
        print(f"[ERROR] No se encontró: {ARCHIVO_TRACKING}")
        return
    tracking = cargar_tracking(ARCHIVO_TRACKING)
    if tracking.empty:
        print("[ERROR] tracking_data.csv está vacío o sin datos válidos.")
        return

    aliados = tracking[tracking["tipo_ext"] == "robot_aliado"]
    rivales = tracking[tracking["tipo_ext"] == "robot_rival"]
    n_robots_total = _robots(tracking)["objeto_id"].nunique()
    tiene_balon = not _balon(tracking).empty
    print(f"[INFO] Tracking: {len(tracking)} registros · "
          f"{tracking['frame'].nunique()} frames · "
          f"{n_robots_total} robots"
          f"{'  ·  balón detectado' if tiene_balon else '  ·  sin balón en tracking'}")
    print(f"       Equipo A: {aliados['objeto_id'].nunique()} robots  |  "
          f"Equipo B: {rivales['objeto_id'].nunique()} robots")

    if os.path.exists(ARCHIVO_EVENTOS):
        eventos = cargar_eventos(ARCHIVO_EVENTOS)
        print(f"[INFO] Eventos: {len(eventos)} registros · "
              f"{eventos['evento'].nunique()} tipos\n")
    else:
        print(f"[WARN] No se encontró {ARCHIVO_EVENTOS}\n")
        eventos = pd.DataFrame(columns=["frame", "tiempo", "evento",
                                        "robot_id", "equipo", "detalles"])

    pasos = [
        ("Heatmap",            vista_heatmap),
        ("Trayectorias",       vista_trayectoria),
        ("Mapa de pases",      vista_pases),
        ("Mapa de tiros",      vista_tiros),
        ("Voronoi",            vista_voronoi),
        ("Posesión por zonas", vista_posesion),
        ("Red de pases",       vista_red_pases),
        ("Dashboard",          vista_dashboard),
        ("GIF de animación",   vista_gif),
    ]

    for nombre, func in pasos:
        print(f"\n── {nombre} ──")
        try:
            func(tracking, eventos)
        except Exception as e:
            import traceback
            print(f"  [WARN] {nombre} falló: {e}")
            traceback.print_exc()

    print("\n" + "=" * 64)
    print(f"[LISTO] Todas las vistas guardadas en '{CARPETA_VISUALIZACIONES}'")
    print("=" * 64)


if __name__ == "__main__":
    main()