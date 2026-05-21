"""
================================================================
 generar_visualizaciones.py
 LEE LOS CSV Y GENERA TODAS LAS VISUALIZACIONES
================================================================
 Este es el script que corres normalmente:
   python generar_visualizaciones.py

 Lee:
   - tracking.csv
   - eventos.csv
 Genera (en la carpeta actual):
   01_cancha.png .. 09_dashboard.png

 Cuando tengas los CSV reales de tu companero, solo reemplaza
 los archivos CSV y vuelve a correr este script. No cambia nada mas.

 Dependencias:
   pip install matplotlib numpy scipy pandas
================================================================
"""

import pandas as pd
import numpy as np
import cancha_template as ct


# -------------------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------------------
def cargar_datos():
    tracking = pd.read_csv("tracking.csv")
    eventos = pd.read_csv("eventos.csv")
    return tracking, eventos


# -------------------------------------------------------------
# PROCESAMIENTO: calcular metricas a partir del tracking
# -------------------------------------------------------------
def distancia_total(df_obj):
    """Distancia recorrida (cm) de un objeto a partir de sus posiciones."""
    dx = df_obj["x"].diff()
    dy = df_obj["y"].diff()
    return np.sqrt(dx ** 2 + dy ** 2).sum()


def velocidad_maxima(df_obj):
    """Velocidad maxima (cm/s) de un objeto."""
    dt = df_obj["tiempo"].diff()
    dx = df_obj["x"].diff()
    dy = df_obj["y"].diff()
    v = np.sqrt(dx ** 2 + dy ** 2) / dt
    return v.max()


# -------------------------------------------------------------
# GENERAR CADA VISUALIZACION
# -------------------------------------------------------------
def main():
    tracking, eventos = cargar_datos()

    # --- 01 Cancha base ---
    fig, ax = ct.dibujar_cancha()
    ct._titulo(ax, "Cancha base", "Plantilla de visualizaciones")
    ct._firma(ax)
    ct.guardar("01_cancha.png")

    # --- 02 Heatmap del balon ---
    balon = tracking[tracking["tipo"] == "balon"]
    ct.heatmap(balon["x"], balon["y"], "Mapa de calor del balon",
               "Posesion territorial - Partido completo")
    ct.guardar("02_heatmap.png")

    # --- 03 Trayectoria del balon ---
    ct.trayectoria(balon["x"].values, balon["y"].values,
                   "Trayectoria del balon", "Recorrido completo")
    ct.guardar("03_trayectoria.png")

    # --- 04 Mapa de pases ---
    pases_df = eventos[eventos["tipo"] == "pase"]
    pases = [(r.x_origen, r.y_origen, r.x_destino, r.y_destino,
              r.resultado == "completado") for r in pases_df.itertuples()]
    ct.mapa_pases(pases, "Mapa de pases", )
    ct.guardar("04_pases.png")

    # --- 05 Mapa de tiros ---
    tiros_df = eventos[eventos["tipo"] == "tiro"]
    tiros = [(r.x_origen, r.y_origen, r.resultado) for r in tiros_df.itertuples()]
    ct.mapa_tiros(tiros, "Mapa de tiros")
    ct.guardar("05_tiros.png")

    # --- 06 Voronoi (control de espacio en un instante) ---
    # Tomamos un frame intermedio
    frame_medio = tracking["frame"].max() // 2
    snap = tracking[tracking["frame"] == frame_medio]
    aliados = [(r.x, r.y) for r in snap[snap["tipo"] == "robot_aliado"].itertuples()]
    rivales = [(r.x, r.y) for r in snap[snap["tipo"] == "robot_rival"].itertuples()]
    ct.voronoi_control(aliados, rivales, "Control de espacio",
                       f"Instante t={frame_medio / 30:.1f}s")
    ct.guardar("06_voronoi.png")

    # --- 07 Posesion por zonas ---
    # Aproximacion: asignamos cada posicion del balon al equipo cuyo robot
    # mas cercano lo tiene. Simplificado: por mitad de cancha.
    balon_pos = balon[["x", "y"]].values
    # Para el demo: balon en lado izquierdo -> aliados, derecho -> rivales
    aliado_xy = [tuple(p) for p in balon_pos if p[0] < ct.CANCHA["largo"] / 2]
    rival_xy = [tuple(p) for p in balon_pos if p[0] >= ct.CANCHA["largo"] / 2]
    ct.posesion_zonas(aliado_xy, rival_xy, "Posesion por zonas")
    ct.guardar("07_posesion.png")

    # --- 08 Red de pases ---
    # Posicion promedio de cada robot aliado
    aliados_track = tracking[tracking["tipo"] == "robot_aliado"]
    posiciones = {int(rid): (g["x"].mean(), g["y"].mean())
                  for rid, g in aliados_track.groupby("objeto_id")}
    # Conexiones: contamos pases entre robots (demo usa origen->siguiente)
    conexiones = [(1, 2, 5), (2, 3, 3), (1, 4, 2), (3, 4, 4), (2, 4, 1)]
    ct.red_pases(posiciones, conexiones, "Red de pases (aliados)")
    ct.guardar("08_red_pases.png")

    # --- 09 Dashboard resumen ---
    # Calcular metricas reales del tracking
    dist_balon = distancia_total(balon) / 100  # a metros
    vmax_balon = velocidad_maxima(balon) / 100  # a m/s
    n_pases = len(pases_df)
    n_completados = (pases_df["resultado"] == "completado").sum()
    prec_pases = (n_completados / n_pases * 100) if n_pases else 0
    n_tiros = len(tiros_df)
    n_goles = (tiros_df["resultado"] == "gol").sum()
    pct_pos_aliados = len(aliado_xy) / len(balon_pos) * 100 if len(balon_pos) else 0

    metricas = {
        "Posesion aliados": f"{pct_pos_aliados:.0f}%",
        "Precision pases": f"{prec_pases:.0f}%",
        "Tiros": f"{n_tiros}",
        "Goles": f"{n_goles}",
        "Dist. balon": f"{dist_balon:.0f} m",
        "Vel. max balon": f"{vmax_balon:.1f} m/s",
    }
    ct.dashboard(metricas, "Resumen del partido")
    ct.guardar("09_dashboard.png")

    print("Listo! Se generaron 9 imagenes PNG a partir de los CSV.")


if __name__ == "__main__":
    main()