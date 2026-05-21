"""
================================================================
 cancha_template.py
 MOTOR DE VISUALIZACIONES - COPA FUTBOT MX (AZTEM)
 Estilo: cientifico / tecnico (light theme, tipo paper academico)
================================================================
 Este archivo SOLO contiene funciones de dibujo. No tiene datos.
 Lo importan otros scripts (ej: generar_visualizaciones.py).

 Funciones publicas:
   dibujar_cancha()        -> base reutilizable
   heatmap()               -> mapa de calor
   trayectoria()           -> ruta con degradado temporal
   mapa_pases()            -> pases completados/fallidos
   mapa_tiros()            -> tiros (gol/atajado/fuera)
   voronoi_control()       -> control de espacio (Voronoi real)
   posesion_zonas()        -> dominio territorial por cuadricula
   red_pases()             -> grafo de interaccion entre robots
   dashboard()             -> resumen del partido en un panel

 Dependencias:
   pip install matplotlib numpy scipy
================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import Voronoi
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# -------------------------------------------------------------
# TEMA VISUAL Y PALETA (CIENTIFICO CLARO)
# -------------------------------------------------------------
PALETA = {
    "fondo":        "#ffffff",   # blanco (fondo de figura)
    "cancha":       "#f7f9fb",   # gris muy claro (superficie)
    "grid":         "#e3e8ee",   # lineas de cuadricula tenue
    "lineas":       "#2b3a4a",   # lineas de cancha (azul gris oscuro)
    "texto":        "#1a2530",   # texto principal (casi negro)
    "texto_tenue":  "#6b7785",   # texto secundario (gris)
    "acento":       "#0066cc",   # azul tecnico (color de marca)
    "aliado":       "#0066cc",   # robots aliados (azul)
    "rival":        "#e8553a",   # robots rivales (naranja-rojo)
    "balon":        "#2b3a4a",   # balon (oscuro)
    "exito":        "#1aa179",   # verde (exito)
    "fallo":        "#d64545",   # rojo (fallo)
    "aviso":        "#e0a020",   # ambar (neutro)
}

# Medidas de la cancha (cm). Ajustar a reglas reales de Copa FutBot.
CANCHA = {
    "largo": 240,
    "ancho": 160,
    "area_largo": 40,
    "area_ancho": 90,
    "porteria_ancho": 40,
    "circulo_radio": 25,
    "margen": 14,
    "grid_paso": 20,     # separacion de la cuadricula tecnica (cm)
}


def _fuente():
    """Fuente con aspecto cientifico (serif para titulos no; sans limpio)."""
    preferidas = ["DejaVu Sans", "Helvetica Neue", "Arial", "Liberation Sans"]
    disponibles = {f.name for f in fm.fontManager.ttflist}
    for f in preferidas:
        if f in disponibles:
            return f
    return "sans-serif"


FUENTE = _fuente()


def _titulo(ax, titulo, subtitulo=None):
    """Titulo tipo figura de paper: 'FIG.' + texto, alineado a la izquierda."""
    ax.text(0, CANCHA["ancho"] + CANCHA["margen"] + 5,
            titulo, ha="left", va="bottom", fontsize=14.5,
            fontweight="bold", color=PALETA["texto"], family=FUENTE)
    if subtitulo:
        ax.text(0, CANCHA["ancho"] + CANCHA["margen"] + 1,
                subtitulo, ha="left", va="bottom", fontsize=9.5,
                color=PALETA["texto_tenue"], family=FUENTE, style="italic")


def _firma(ax):
    ax.text(CANCHA["largo"], -CANCHA["margen"] + 3,
            "Copa FutBotMX · Equipo AZTEM", ha="right", va="bottom",
            fontsize=7.5, color=PALETA["texto_tenue"], family=FUENTE)


def _ejes_tecnicos(ax):
    """Dibuja marcas de escala (ticks) en los bordes, como un grafico cientifico."""
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
    ax.text(-11, ancho / 2, "y (cm)", ha="center", va="center",
            rotation=90, fontsize=7.5, color=PALETA["texto_tenue"], family=FUENTE)


# -------------------------------------------------------------
# DIBUJAR LA CANCHA (BASE)
# -------------------------------------------------------------
def dibujar_cancha(ax=None, figsize=(13, 9), con_grid=True, con_ejes=True):
    """Dibuja la cancha con estilo cientifico claro. Devuelve (fig, ax)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    fig.patch.set_facecolor(PALETA["fondo"])
    ax.set_facecolor(PALETA["fondo"])

    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    lin = PALETA["lineas"]
    lw = 1.6

    # Superficie de la cancha
    ax.add_patch(patches.Rectangle((0, 0), largo, ancho,
                 facecolor=PALETA["cancha"], edgecolor="none", zorder=0))

    # Cuadricula tecnica tenue
    if con_grid:
        paso = CANCHA["grid_paso"]
        for x in range(paso, largo, paso):
            ax.plot([x, x], [0, ancho], color=PALETA["grid"], lw=0.7, zorder=1)
        for y in range(paso, ancho, paso):
            ax.plot([0, largo], [y, y], color=PALETA["grid"], lw=0.7, zorder=1)

    # Borde, medio campo, circulo
    ax.add_patch(patches.Rectangle((0, 0), largo, ancho, fill=False,
                 edgecolor=lin, linewidth=lw, zorder=2))
    ax.plot([largo / 2, largo / 2], [0, ancho], color=lin, linewidth=lw, zorder=2)
    ax.add_patch(patches.Circle((largo / 2, ancho / 2), CANCHA["circulo_radio"],
                 fill=False, edgecolor=lin, linewidth=lw, zorder=2))
    ax.add_patch(patches.Circle((largo / 2, ancho / 2), 1.6, color=lin, zorder=2))

    # Areas
    area_y = (ancho - CANCHA["area_ancho"]) / 2
    for x0 in (0, largo - CANCHA["area_largo"]):
        ax.add_patch(patches.Rectangle((x0, area_y), CANCHA["area_largo"],
                     CANCHA["area_ancho"], fill=False, edgecolor=lin,
                     linewidth=lw, zorder=2))

    # Porterias (color de acento)
    port_y = (ancho - CANCHA["porteria_ancho"]) / 2
    for x0 in (0, largo):
        ax.plot([x0, x0], [port_y, port_y + CANCHA["porteria_ancho"]],
                color=PALETA["acento"], linewidth=4.5, zorder=3,
                solid_capstyle="round")

    if con_ejes:
        _ejes_tecnicos(ax)

    m = CANCHA["margen"]
    ax.set_xlim(-m, largo + m)
    ax.set_ylim(-m, ancho + m + 16)
    ax.set_aspect("equal")
    ax.axis("off")
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


# -------------------------------------------------------------
# HEATMAP
# -------------------------------------------------------------
def heatmap(xs, ys, titulo="Mapa de calor", subtitulo=None, bins=40):
    fig, ax = dibujar_cancha(con_grid=False)
    heat, _, _ = np.histogram2d(xs, ys, bins=bins,
        range=[[0, CANCHA["largo"]], [0, CANCHA["ancho"]]])
    if SCIPY_OK:
        heat = gaussian_filter(heat, sigma=1.4)
    # Colormap cientifico: blanco -> azul -> morado -> rojo (tipo "viridis"/"jet" suave)
    cmap = LinearSegmentedColormap.from_list("calor", [
        (0.00, (1, 1, 1, 0)),
        (0.20, (0.30, 0.55, 0.85, 0.45)),
        (0.50, (0.45, 0.35, 0.75, 0.70)),
        (0.78, (0.90, 0.45, 0.30, 0.85)),
        (1.00, (0.80, 0.10, 0.15, 0.95))])
    im = ax.imshow(heat.T, extent=[0, CANCHA["largo"], 0, CANCHA["ancho"]],
                   origin="lower", cmap=cmap, interpolation="bilinear",
                   zorder=1, aspect="auto")
    # Barra de color (escala cientifica)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Densidad de presencia", fontsize=8,
                   color=PALETA["texto_tenue"])
    cbar.ax.tick_params(labelsize=6, colors=PALETA["texto_tenue"])
    _titulo(ax, titulo, subtitulo)
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# TRAYECTORIA
# -------------------------------------------------------------
def trayectoria(xs, ys, titulo="Trayectoria", subtitulo=None, color=None):
    fig, ax = dibujar_cancha()
    xs, ys = np.asarray(xs), np.asarray(ys)
    puntos = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segmentos = np.concatenate([puntos[:-1], puntos[1:]], axis=1)
    col_fin = color or PALETA["acento"]
    cmap = LinearSegmentedColormap.from_list("ruta", ["#cfe0f2", col_fin])
    lc = LineCollection(segmentos, cmap=cmap, linewidth=2.4, zorder=3)
    lc.set_array(np.linspace(0, 1, len(segmentos)))
    ax.add_collection(lc)
    ax.scatter(xs[0], ys[0], s=120, c="#ffffff", edgecolor=col_fin,
               linewidth=2, zorder=4)
    ax.scatter(xs[-1], ys[-1], s=160, c=col_fin, edgecolor="#ffffff",
               linewidth=2, zorder=4)
    ax.text(xs[0], ys[0] - 7, "inicio", ha="center", fontsize=7.5,
            color=PALETA["texto_tenue"], family=FUENTE)
    ax.text(xs[-1], ys[-1] + 6, "fin", ha="center", fontsize=7.5,
            color=col_fin, family=FUENTE, fontweight="bold")
    _titulo(ax, titulo, subtitulo)
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# MAPA DE PASES
# -------------------------------------------------------------
def mapa_pases(pases, titulo="Mapa de pases", subtitulo=None):
    """pases: lista de (x1, y1, x2, y2, completado_bool)"""
    fig, ax = dibujar_cancha()
    completados = sum(1 for *_, ok in pases if ok)
    total = len(pases)
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
    _titulo(ax, titulo, sub)
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# MAPA DE TIROS
# -------------------------------------------------------------
def mapa_tiros(tiros, titulo="Mapa de tiros", subtitulo=None):
    """tiros: lista de (x, y, resultado) con resultado gol/atajado/fuera"""
    fig, ax = dibujar_cancha()
    estilos = {
        "gol":     {"c": PALETA["exito"], "mk": "*", "s": 420},
        "atajado": {"c": PALETA["aviso"], "mk": "o", "s": 190},
        "fuera":   {"c": PALETA["fallo"], "mk": "X", "s": 190}}
    for x, y, res in tiros:
        e = estilos.get(res, estilos["fuera"])
        ax.scatter(x, y, c=e["c"], marker=e["mk"], s=e["s"],
                   edgecolor="#ffffff", linewidth=1.5, zorder=4)
    _leyenda(ax, [(estilos["gol"]["c"], "Gol", "*"),
                  (estilos["atajado"]["c"], "Atajado", "o"),
                  (estilos["fuera"]["c"], "Fuera", "X")])
    goles = sum(1 for *_, r in tiros if r == "gol")
    sub = subtitulo or f"Goles: {goles} · Tiros: {len(tiros)}"
    _titulo(ax, titulo, sub)
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# VORONOI (CONTROL DE ESPACIO) - algoritmo real con poligonos
# -------------------------------------------------------------
def _recortar_poligono(poligono, largo, ancho):
    """Recorta un poligono al rectangulo de la cancha (Sutherland-Hodgman)."""
    def dentro(p, borde):
        x, y = p
        if borde == "izq":   return x >= 0
        if borde == "der":   return x <= largo
        if borde == "abajo": return y >= 0
        if borde == "arriba":return y <= ancho

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
        if not salida:
            break
        entrada = salida
        salida = []
        for i in range(len(entrada)):
            actual = entrada[i]
            previo = entrada[i - 1]
            if dentro(actual, borde):
                if not dentro(previo, borde):
                    salida.append(interseccion(previo, actual, borde))
                salida.append(actual)
            elif dentro(previo, borde):
                salida.append(interseccion(previo, actual, borde))
    return salida


def voronoi_control(aliados, rivales, titulo="Control de espacio",
                    subtitulo=None):
    """
    aliados, rivales: listas de (x, y) con posiciones de robots de cada equipo
    en un instante. Dibuja el diagrama de Voronoi REAL (poligonos nitidos)
    coloreando cada region segun el equipo del robot que la domina.
    """
    fig, ax = dibujar_cancha(con_grid=False)
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]

    puntos = [(x, y) for x, y in aliados] + [(x, y) for x, y in rivales]
    equipo_de = [0] * len(aliados) + [1] * len(rivales)
    n_real = len(puntos)

    if n_real < 2 or not SCIPY_OK:
        for x, y in aliados:
            ax.scatter(x, y, s=170, c=PALETA["aliado"],
                       edgecolor="#ffffff", linewidth=2, zorder=4)
        for x, y in rivales:
            ax.scatter(x, y, s=170, c=PALETA["rival"],
                       edgecolor="#ffffff", linewidth=2, zorder=4)
        _titulo(ax, titulo, subtitulo); _firma(ax)
        return fig, ax

    extra = [(-largo * 3, -ancho * 3), (largo * 4, -ancho * 3),
             (-largo * 3, ancho * 4), (largo * 4, ancho * 4)]
    vor = Voronoi(np.array(puntos + extra))

    color_eq = {0: PALETA["aliado"], 1: PALETA["rival"]}
    area_aliado = 0.0
    area_total = largo * ancho

    for idx_punto in range(n_real):
        region_idx = vor.point_region[idx_punto]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        poligono = [tuple(vor.vertices[v]) for v in region]
        recortado = _recortar_poligono(poligono, largo, ancho)
        if len(recortado) < 3:
            continue
        eq = equipo_de[idx_punto]
        ax.add_patch(patches.Polygon(
            recortado, closed=True, facecolor=color_eq[eq], alpha=0.22,
            edgecolor=color_eq[eq], linewidth=1.4, zorder=1))
        xs = [p[0] for p in recortado]; ys = [p[1] for p in recortado]
        area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                             xs[(i + 1) % len(xs)] * ys[i]
                             for i in range(len(xs))))
        if eq == 0:
            area_aliado += area

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
    _titulo(ax, titulo, sub)
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# POSESION POR ZONAS
# -------------------------------------------------------------
def posesion_zonas(aliado_xy, rival_xy, titulo="Posesion por zonas",
                   subtitulo=None, cols=6, filas=4):
    """
    aliado_xy, rival_xy: listas de (x, y). Divide la cancha en cuadricula y
    colorea cada zona segun quien la domino (gradiente azul-rojo).
    """
    fig, ax = dibujar_cancha(con_grid=False)
    largo, ancho = CANCHA["largo"], CANCHA["ancho"]
    cw, ch = largo / cols, ancho / filas

    a = np.array(aliado_xy) if len(aliado_xy) else np.empty((0, 2))
    r = np.array(rival_xy) if len(rival_xy) else np.empty((0, 2))

    # Colormap divergente cientifico (rojo - blanco - azul)
    cmap = LinearSegmentedColormap.from_list("divergente", [
        PALETA["rival"], "#f2f2f2", PALETA["aliado"]])

    for i in range(cols):
        for j in range(filas):
            x0, y0 = i * cw, j * ch
            na = np.sum((a[:, 0] >= x0) & (a[:, 0] < x0 + cw) &
                        (a[:, 1] >= y0) & (a[:, 1] < y0 + ch)) if len(a) else 0
            nr = np.sum((r[:, 0] >= x0) & (r[:, 0] < x0 + cw) &
                        (r[:, 1] >= y0) & (r[:, 1] < y0 + ch)) if len(r) else 0
            total = na + nr
            if total == 0:
                continue
            frac = na / total
            ax.add_patch(patches.Rectangle(
                (x0, y0), cw, ch, facecolor=cmap(frac), alpha=0.78,
                edgecolor="#ffffff", linewidth=1.0, zorder=1))
            ax.text(x0 + cw / 2, y0 + ch / 2, f"{frac * 100:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color=PALETA["texto"], family=FUENTE, zorder=2,
                    fontweight="bold")

    _leyenda(ax, [(PALETA["aliado"], "Dominio aliado", "s"),
                  (PALETA["rival"], "Dominio rival", "s")])
    _titulo(ax, titulo, subtitulo or "% de posesion del balon por zona (aliados)")
    _firma(ax)
    return fig, ax


# -------------------------------------------------------------
# RED DE PASES (GRAFO DE INTERACCION)
# -------------------------------------------------------------
def red_pases(posiciones, conexiones, titulo="Red de pases", subtitulo=None):
    """
    posiciones: dict {robot_id: (x, y)}
    conexiones: lista de (id_origen, id_destino, cantidad_pases)
    """
    fig, ax = dibujar_cancha()
    max_pases = max((c for *_, c in conexiones), default=1)

    for o, d, cant in conexiones:
        if o not in posiciones or d not in posiciones:
            continue
        x1, y1 = posiciones[o]
        x2, y2 = posiciones[d]
        grosor = 1 + 6 * (cant / max_pases)
        ax.plot([x1, x2], [y1, y2], color=PALETA["acento"],
                linewidth=grosor, alpha=0.45, zorder=3, solid_capstyle="round")
        # Etiqueta con la cantidad en el punto medio
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


# -------------------------------------------------------------
# DASHBOARD RESUMEN
# -------------------------------------------------------------
def dashboard(metricas, titulo="Resumen del partido"):
    """
    metricas: dict {nombre: valor}. Genera un panel claro con tarjetas.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(PALETA["fondo"])
    ax.set_facecolor(PALETA["fondo"])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Encabezado con linea de acento
    ax.text(0.6, 7.4, titulo, ha="left", fontsize=20, fontweight="bold",
            color=PALETA["texto"], family=FUENTE)
    ax.text(0.6, 6.95, "Copa FutBotMX · Equipo AZTEM", ha="left", fontsize=10,
            color=PALETA["acento"], family=FUENTE)
    ax.plot([0.6, 11.4], [6.75, 6.75], color=PALETA["acento"], lw=2)

    items = list(metricas.items())
    cols = 3
    cw, ch = 3.5, 2.1
    x0, y0 = 0.6, 4.2
    gap_x, gap_y = 0.35, 0.4
    for idx, (clave, valor) in enumerate(items):
        col = idx % cols
        fila = idx // cols
        x = x0 + col * (cw + gap_x)
        y = y0 - fila * (ch + gap_y)
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), cw, ch, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=PALETA["cancha"], edgecolor=PALETA["grid"], linewidth=1.3))
        # Franja de acento arriba de la tarjeta
        ax.add_patch(patches.Rectangle((x, y + ch - 0.12), cw, 0.12,
                     facecolor=PALETA["acento"], edgecolor="none"))
        ax.text(x + cw / 2, y + ch * 0.58, str(valor), ha="center",
                fontsize=27, fontweight="bold", color=PALETA["acento"],
                family=FUENTE)
        ax.text(x + cw / 2, y + ch * 0.20, clave, ha="center", fontsize=10.5,
                color=PALETA["texto_tenue"], family=FUENTE)

    return fig, ax


# -------------------------------------------------------------
# UTILIDAD PARA GUARDAR
# -------------------------------------------------------------
def guardar(nombre, dpi=170):
    plt.savefig(nombre, dpi=dpi, bbox_inches="tight", facecolor=PALETA["fondo"])
    plt.close()