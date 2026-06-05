"""
================================================================
 generar_visualizaciones.py
 LEE tracking.csv (Y eventos.csv si existe) Y GENERA TODAS LAS
 VISUALIZACIONES POSIBLES, RESCATANDO EL MAXIMO DE DATOS.
================================================================
 Diseñado para tolerar:
   - falta de balon -> usa heatmap de robots y trayectoria individual
   - falta de eventos -> detecta colisiones del tracking,
                         genera red de pases por proximidad
   - extras (area, intervencion_humana) -> graficas nuevas
================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cancha_template as ct


# ---------- helpers ----------
def distancia_total(df_obj):
    dx = df_obj["x"].diff(); dy = df_obj["y"].diff()
    return np.sqrt(dx ** 2 + dy ** 2).sum()


def velocidad_maxima(df_obj):
    dt = df_obj["tiempo"].diff()
    dx = df_obj["x"].diff(); dy = df_obj["y"].diff()
    v = np.sqrt(dx ** 2 + dy ** 2) / dt
    return v.replace([np.inf, -np.inf], np.nan).max()


def detectar_colisiones(tracking, umbral_cm=15, min_frames=2):
    """Detecta cuando dos robots de equipos opuestos estan muy cerca
    durante varios frames seguidos."""
    cols = []
    aliados = tracking[tracking["tipo"] == "robot_aliado"]
    rivales = tracking[tracking["tipo"] == "robot_rival"]
    if len(aliados) == 0 or len(rivales) == 0:
        return cols
    frames = sorted(tracking["frame"].unique())
    cercanias = {}  # (id_a, id_r) -> [frames consecutivos]
    for f in frames:
        a = aliados[aliados["frame"] == f]
        r = rivales[rivales["frame"] == f]
        for _, ra in a.iterrows():
            for _, rr in r.iterrows():
                d = np.hypot(ra["x"] - rr["x"], ra["y"] - rr["y"])
                if d < umbral_cm:
                    key = (int(ra["objeto_id"]), int(rr["objeto_id"]))
                    cercanias.setdefault(key, []).append(
                        (f, (ra["x"] + rr["x"]) / 2, (ra["y"] + rr["y"]) / 2))
    # Filtrar las que duraron al menos min_frames
    for key, lista in cercanias.items():
        if len(lista) >= min_frames:
            # Punto medio de la "colision"
            xs = np.mean([p[1] for p in lista])
            ys = np.mean([p[2] for p in lista])
            cols.append((xs, ys, key[0], key[1], len(lista)))
    return cols


# ---------- visualizaciones extra (no estaban en cancha_template) ----------
def grafica_autonomia(tracking, archivo="10_autonomia.png"):
    """Bar chart: por cada robot, % de frames autonomos vs con intervencion."""
    if "intervencion_humana" not in tracking.columns:
        return False
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(ct.PALETA["fondo"])
    ax.set_facecolor(ct.PALETA["fondo"])

    datos = (tracking.groupby(["objeto_id", "intervencion_humana"])
             .size().unstack(fill_value=0))
    if "no" not in datos.columns: datos["no"] = 0
    if "si" not in datos.columns: datos["si"] = 0
    datos["total"] = datos["no"] + datos["si"]
    datos["pct_autonomo"] = datos["no"] / datos["total"] * 100
    datos = datos.sort_values("pct_autonomo", ascending=True)

    ids = [f"Robot {i}" for i in datos.index]
    autonomo = datos["pct_autonomo"].values
    asistido = 100 - autonomo

    y = np.arange(len(ids))
    ax.barh(y, autonomo, color=ct.PALETA["exito"], label="Autonomo",
            edgecolor="white", linewidth=1)
    ax.barh(y, asistido, left=autonomo, color=ct.PALETA["fallo"],
            label="Con intervencion", edgecolor="white", linewidth=1)
    for i, (a, s) in enumerate(zip(autonomo, asistido)):
        ax.text(a / 2, i, f"{a:.0f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
        if s > 5:
            ax.text(a + s / 2, i, f"{s:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)

    ax.set_yticks(y); ax.set_yticklabels(ids, color=ct.PALETA["texto"])
    ax.set_xlabel("% de frames", color=ct.PALETA["texto_tenue"], fontsize=10)
    ax.set_xlim(0, 100)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ct.PALETA["grid"])
    ax.spines["bottom"].set_color(ct.PALETA["grid"])
    ax.tick_params(colors=ct.PALETA["texto_tenue"])
    ax.legend(loc="lower right", frameon=True, fontsize=9,
              labelcolor=ct.PALETA["texto"])
    fig.suptitle("Autonomia de los robots",
                 x=0.06, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color=ct.PALETA["texto"])
    ax.set_title("% de frames operando sin intervencion humana",
                 loc="left", fontsize=10, style="italic",
                 color=ct.PALETA["texto_tenue"], pad=12)
    ct.guardar(archivo)
    return True


def grafica_colisiones(tracking, archivo="11_colisiones.png"):
    """Mapa de colisiones detectadas entre aliados y rivales."""
    cols = detectar_colisiones(tracking)
    fig, ax = ct.dibujar_cancha()
    if cols:
        for x, y, _, _, dur in cols:
            tamano = 100 + dur * 30
            ax.scatter(x, y, s=tamano, c=ct.PALETA["aviso"],
                       edgecolor=ct.PALETA["texto"], linewidth=1.5,
                       alpha=0.75, zorder=4, marker="X")
        ct._titulo(ax, "Mapa de colisiones",
                   f"{len(cols)} colisiones detectadas (>15cm de proximidad)")
    else:
        ct._titulo(ax, "Mapa de colisiones", "No se detectaron colisiones")
    ct._firma(ax)
    ct.guardar(archivo)
    return True


def grafica_distancia_acumulada(tracking, archivo="12_distancia.png"):
    """Bar chart: distancia recorrida por cada robot."""
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(ct.PALETA["fondo"])
    ax.set_facecolor(ct.PALETA["fondo"])

    robots = tracking[tracking["tipo"].str.startswith("robot")]
    datos = []
    for rid, g in robots.groupby("objeto_id"):
        g = g.sort_values("tiempo")
        d = distancia_total(g) / 100  # a metros
        tipo = g["tipo"].iloc[0]
        datos.append((int(rid), d, tipo))
    datos.sort(key=lambda x: x[1])

    ids = [f"Robot {d[0]}" for d in datos]
    distancias = [d[1] for d in datos]
    colores = [ct.PALETA["aliado"] if d[2] == "robot_aliado"
               else ct.PALETA["rival"] for d in datos]

    y = np.arange(len(ids))
    ax.barh(y, distancias, color=colores, edgecolor="white", linewidth=1)
    for i, d in enumerate(distancias):
        ax.text(d + max(distancias) * 0.01, i, f"{d:.1f} m",
                va="center", color=ct.PALETA["texto"], fontsize=10,
                fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(ids, color=ct.PALETA["texto"])
    ax.set_xlabel("Distancia (metros)", color=ct.PALETA["texto_tenue"], fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ct.PALETA["grid"])
    ax.spines["bottom"].set_color(ct.PALETA["grid"])
    ax.tick_params(colors=ct.PALETA["texto_tenue"])
    fig.suptitle("Distancia recorrida por robot",
                 x=0.06, y=0.96, ha="left", fontsize=15,
                 fontweight="bold", color=ct.PALETA["texto"])
    ax.set_title("Esfuerzo total de cada robot durante el partido",
                 loc="left", fontsize=10, style="italic",
                 color=ct.PALETA["texto_tenue"], pad=12)
    ct.guardar(archivo)
    return True


# ---------- main ----------
def main():
    tracking = pd.read_csv("tracking.csv")
    eventos = pd.read_csv("eventos.csv") if os.path.exists("eventos.csv") else None

    print(f"Tracking: {len(tracking)} filas, "
          f"{tracking['frame'].nunique()} frames, "
          f"{tracking['objeto_id'].nunique()} objetos")

    generadas = []

    # 01 Cancha base
    fig, ax = ct.dibujar_cancha()
    ct._titulo(ax, "Cancha base", "Plantilla de visualizaciones")
    ct._firma(ax); ct.guardar("01_cancha.png")
    generadas.append("01_cancha.png")

    # 02 Heatmap: si hay balon usa balon, si no usa robots
    balon = tracking[tracking["tipo"] == "balon"]
    if len(balon) > 0:
        ct.heatmap(balon["x"], balon["y"], "Mapa de calor del balon",
                   "Posesion territorial - partido completo")
    else:
        robots = tracking[tracking["tipo"].str.startswith("robot")]
        ct.heatmap(robots["x"], robots["y"], "Mapa de calor - actividad",
                   f"Presencia de los {robots['objeto_id'].nunique()} robots")
    ct.guardar("02_heatmap.png"); generadas.append("02_heatmap.png")

    # 03 Trayectoria: del balon o del robot mas activo
    if len(balon) > 0:
        ct.trayectoria(balon["x"].values, balon["y"].values,
                       "Trayectoria del balon", "Recorrido completo")
    else:
        # Robot con mayor cantidad de frames (mas activo)
        robots = tracking[tracking["tipo"].str.startswith("robot")]
        rid = int(robots["objeto_id"].value_counts().index[0])
        df = robots[robots["objeto_id"] == rid].sort_values("tiempo")
        ct.trayectoria(df["x"].values, df["y"].values,
                       f"Trayectoria robot {rid}",
                       f"Recorrido del robot mas activo ({len(df)} frames)")
    ct.guardar("03_trayectoria.png"); generadas.append("03_trayectoria.png")

    # 04 Mapa de pases (solo si hay eventos)
    if eventos is not None and "pase" in eventos["tipo"].values:
        pdf = eventos[eventos["tipo"] == "pase"]
        pases = [(r.x_origen, r.y_origen, r.x_destino, r.y_destino,
                  r.resultado == "completado") for r in pdf.itertuples()]
        ct.mapa_pases(pases, "Mapa de pases")
        ct.guardar("04_pases.png"); generadas.append("04_pases.png")

    # 05 Mapa de tiros (solo si hay eventos)
    if eventos is not None and "tiro" in eventos["tipo"].values:
        tdf = eventos[eventos["tipo"] == "tiro"]
        tiros = [(r.x_origen, r.y_origen, r.resultado) for r in tdf.itertuples()]
        ct.mapa_tiros(tiros, "Mapa de tiros")
        ct.guardar("05_tiros.png"); generadas.append("05_tiros.png")

    # 06 Voronoi: snapshot a la mitad
    frame_medio = int(tracking["frame"].median())
    snap = tracking[tracking["frame"] == frame_medio]
    aliados_xy = [(r.x, r.y) for r in snap[snap["tipo"] == "robot_aliado"].itertuples()]
    rivales_xy = [(r.x, r.y) for r in snap[snap["tipo"] == "robot_rival"].itertuples()]
    if len(aliados_xy) + len(rivales_xy) >= 2:
        ct.voronoi_control(aliados_xy, rivales_xy, "Control de espacio",
                           f"Instante: frame {frame_medio}, t={snap['tiempo'].iloc[0]:.1f}s")
        ct.guardar("06_voronoi.png"); generadas.append("06_voronoi.png")

    # 07 Posesion por zonas: con balon o con dominio territorial de robots
    if len(balon) > 0:
        bp = balon[["x", "y"]].values
        a = [tuple(p) for p in bp if p[0] < ct.CANCHA["largo"] / 2]
        r = [tuple(p) for p in bp if p[0] >= ct.CANCHA["largo"] / 2]
        ct.posesion_zonas(a, r, "Posesion por zonas (balon)")
    else:
        # Usamos posiciones de los robots como proxy de "dominio territorial"
        ali = tracking[tracking["tipo"] == "robot_aliado"][["x", "y"]].values
        riv = tracking[tracking["tipo"] == "robot_rival"][["x", "y"]].values
        ct.posesion_zonas(
            [tuple(p) for p in ali], [tuple(p) for p in riv],
            "Dominio territorial por zonas",
            "% de presencia de robots aliados por zona")
    ct.guardar("07_posesion.png"); generadas.append("07_posesion.png")

    # 08 Red de pases: si hay eventos, si no demo basica con posiciones medias
    aliados_track = tracking[tracking["tipo"] == "robot_aliado"]
    if len(aliados_track) > 0:
        posiciones = {int(rid): (g["x"].mean(), g["y"].mean())
                      for rid, g in aliados_track.groupby("objeto_id")}
        # Sin eventos: conectamos todos los pares con peso = veces que estuvieron cercanos
        conexiones = []
        ids = list(posiciones.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = aliados_track[aliados_track["objeto_id"] == ids[i]].sort_values("frame")
                b = aliados_track[aliados_track["objeto_id"] == ids[j]].sort_values("frame")
                merged = pd.merge(a, b, on="frame", suffixes=("_a", "_b"))
                if len(merged):
                    d = np.hypot(merged["x_a"] - merged["x_b"],
                                 merged["y_a"] - merged["y_b"])
                    cercanos = int((d < 40).sum())  # frames a < 40cm
                    if cercanos > 0:
                        conexiones.append((ids[i], ids[j], cercanos))
        ct.red_pases(posiciones, conexiones, "Red de proximidad (aliados)",
                     "Aliados que pasaron mas tiempo cerca entre si")
        ct.guardar("08_red_pases.png"); generadas.append("08_red_pases.png")

    # 09 Dashboard
    n_frames = int(tracking["frame"].nunique())
    n_robots = int(tracking[tracking["tipo"].str.startswith("robot")]["objeto_id"].nunique())
    duracion = tracking["tiempo"].max()
    metricas = {
        "Robots detectados": str(n_robots),
        "Frames": str(n_frames),
        "Duracion": f"{duracion:.1f} s",
    }
    if "intervencion_humana" in tracking.columns:
        autonomo = (tracking["intervencion_humana"] == "no").mean() * 100
        metricas["Autonomia global"] = f"{autonomo:.0f}%"
    # Distancia total (suma de todos los robots)
    robots = tracking[tracking["tipo"].str.startswith("robot")]
    dist_total = sum(distancia_total(g.sort_values("tiempo"))
                     for _, g in robots.groupby("objeto_id")) / 100
    metricas["Dist. total"] = f"{dist_total:.0f} m"
    # Velocidad maxima
    vmaxs = [velocidad_maxima(g.sort_values("tiempo"))
             for _, g in robots.groupby("objeto_id")]
    vmax = np.nanmax(vmaxs) / 100 if vmaxs else 0
    metricas["Vel. max"] = f"{vmax:.1f} m/s"
    if eventos is not None:
        if "pase" in eventos["tipo"].values:
            pdf = eventos[eventos["tipo"] == "pase"]
            comp = (pdf["resultado"] == "completado").sum()
            metricas["Precision pases"] = f"{comp / len(pdf) * 100:.0f}%"
        if "tiro" in eventos["tipo"].values:
            tdf = eventos[eventos["tipo"] == "tiro"]
            metricas["Goles"] = f"{(tdf['resultado'] == 'gol').sum()}"
    ct.dashboard(metricas, "Resumen del partido")
    ct.guardar("09_dashboard.png"); generadas.append("09_dashboard.png")

    # 10 Autonomia (solo si hay esa columna)
    if grafica_autonomia(tracking):
        generadas.append("10_autonomia.png")

    # 11 Colisiones detectadas del tracking
    if grafica_colisiones(tracking):
        generadas.append("11_colisiones.png")

    # 12 Distancia recorrida por robot
    if grafica_distancia_acumulada(tracking):
        generadas.append("12_distancia.png")

    print(f"\nGeneradas {len(generadas)} visualizaciones:")
    for g in generadas:
        print(f"   {g}")


if __name__ == "__main__":
    main()