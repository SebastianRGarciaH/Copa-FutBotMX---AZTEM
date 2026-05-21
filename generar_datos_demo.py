"""
================================================================
 generar_datos_demo.py
 GENERADOR DE DATOS DE EJEMPLO (TEMPORAL)
================================================================
 Crea dos archivos CSV con datos SIMULADOS que imitan el formato
 que entregara el equipo de vision (SAM 3):

   - tracking.csv : posiciones frame a frame
   - eventos.csv  : pases, tiros y colisiones

 Cuando tengas los CSV reales de tu companero, BORRA este archivo
 y reemplaza los CSV. El resto del pipeline no cambia.

 Formato tracking.csv:
   frame, tiempo, objeto_id, tipo, x, y
   tipo in {robot_aliado, robot_rival, balon}

 Formato eventos.csv:
   tiempo, tipo, objeto_id, x_origen, y_origen, x_destino, y_destino, resultado
   tipo in {pase, tiro, colision, intercepcion}
   resultado: pases->completado/fallido ; tiros->gol/atajado/fuera
================================================================
"""

import csv
import numpy as np

LARGO, ANCHO = 240, 160
np.random.seed(42)


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def generar_tracking(n_frames=400, fps=30):
    """Simula 4 aliados, 4 rivales y 1 balon moviendose por la cancha."""
    filas = []
    # Posiciones iniciales
    aliados = {i: np.array([np.random.uniform(20, LARGO / 2),
                            np.random.uniform(20, ANCHO - 20)])
               for i in range(1, 5)}
    rivales = {i: np.array([np.random.uniform(LARGO / 2, LARGO - 20),
                            np.random.uniform(20, ANCHO - 20)])
               for i in range(5, 9)}
    balon = np.array([LARGO / 2, ANCHO / 2])

    for f in range(n_frames):
        t = round(f / fps, 3)
        # Mover aliados (tienden ligeramente al ataque, lado derecho)
        for i, p in aliados.items():
            p += np.random.normal([0.6, 0], [2.5, 2.5])
            p[0] = _clip(p[0], 5, LARGO - 5)
            p[1] = _clip(p[1], 5, ANCHO - 5)
            filas.append([f, t, i, "robot_aliado", round(p[0], 1), round(p[1], 1)])
        # Mover rivales
        for i, p in rivales.items():
            p += np.random.normal([-0.4, 0], [2.5, 2.5])
            p[0] = _clip(p[0], 5, LARGO - 5)
            p[1] = _clip(p[1], 5, ANCHO - 5)
            filas.append([f, t, i, "robot_rival", round(p[0], 1), round(p[1], 1)])
        # Mover balon (sigue una trayectoria sinusoidal con ruido)
        balon[0] = _clip(LARGO / 2 + 70 * np.sin(f / 40) +
                         np.random.normal(0, 3), 5, LARGO - 5)
        balon[1] = _clip(ANCHO / 2 + 45 * np.cos(f / 25) +
                         np.random.normal(0, 3), 5, ANCHO - 5)
        filas.append([f, t, 0, "balon", round(balon[0], 1), round(balon[1], 1)])

    with open("tracking.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "tiempo", "objeto_id", "tipo", "x", "y"])
        w.writerows(filas)
    print(f"tracking.csv generado ({len(filas)} filas)")


def generar_eventos():
    """Simula eventos del partido."""
    pases = [
        (2.5, "pase", 1, 30, 80, 100, 90, "completado"),
        (3.8, "pase", 2, 100, 90, 180, 70, "completado"),
        (5.1, "pase", 3, 180, 70, 220, 80, "fallido"),
        (7.0, "pase", 1, 50, 40, 120, 50, "completado"),
        (9.2, "pase", 4, 120, 50, 200, 120, "fallido"),
        (11.5, "pase", 2, 60, 120, 140, 100, "completado"),
        (13.1, "pase", 3, 140, 100, 205, 95, "completado"),
        (15.4, "pase", 1, 90, 70, 160, 60, "completado"),
        (17.8, "pase", 4, 160, 60, 210, 110, "completado"),
    ]
    tiros = [
        (19.0, "tiro", 3, 210, 80, "", "", "gol"),
        (22.3, "tiro", 1, 200, 60, "", "", "atajado"),
        (25.6, "tiro", 4, 215, 100, "", "", "fuera"),
        (28.1, "tiro", 3, 190, 90, "", "", "gol"),
        (30.5, "tiro", 2, 205, 70, "", "", "atajado"),
        (33.0, "tiro", 1, 218, 85, "", "", "fuera"),
    ]
    colisiones = [
        (12.0, "colision", 2, 150, 90, "", "", ""),
        (20.5, "colision", 3, 195, 75, "", "", ""),
    ]

    with open("eventos.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tiempo", "tipo", "objeto_id", "x_origen", "y_origen",
                    "x_destino", "y_destino", "resultado"])
        w.writerows(pases + tiros + colisiones)
    print(f"eventos.csv generado ({len(pases) + len(tiros) + len(colisiones)} filas)")


if __name__ == "__main__":
    generar_tracking()
    generar_eventos()
    print("Datos de ejemplo listos. Ahora corre: python generar_visualizaciones.py")