import pandas as pd
import numpy as np
import os
from collections import deque

def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO B1: MOTOR CINEMÁTICO DE EVENTOS Y JUGADAS DE JUEGO ===")
    print("==================================================================")

    ruta_entrada = "results/metrics/tracking_data.csv"
    ruta_salida  = "results/metrics/game_events.csv"

    if not os.path.exists(ruta_entrada):
        print(f"[!] No se encontró el archivo de tracking estable: {ruta_entrada}")
        return

    df = pd.read_csv(ruta_entrada)
    if df.empty:
        print("[!] El archivo de telemetría está vacío.")
        return

    MAPEO_EQUIPOS = {
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    # ⚙️ UMBRALES DE CALIBRACIÓN GEOMÉTRICA
    UMBRAL_POSESION_CM  = 14.5   # Radio de vecindad para considerar que un robot toca el balón
    UMBRAL_VEL_TIRO_CMS = 60.0   # Velocidad mínima (cm/s) para registrar un tiro o despeje

    # Anti-vibración: cuántos frames consecutivos debe mantenerse la posesión
    # para registrar un evento. Evita pases/controles fantasma de 1-2 frames.
    FRAMES_POSESION_MINIMA = 3

    # Ventana de frames para promediar la velocidad del balón (anti-spike)
    VENTANA_VEL_FRAMES = 3

    # FIX 2: Gap máximo de frames permitido dentro de la ventana de velocidad.
    # Si el balón no se detectó durante más de este número de frames y reaparece,
    # la ventana se vacía para no calcular velocidades falsas sobre ese salto.
    GAP_MAX_VEL_FRAMES = 5

    # FIX 3: Tiempo mínimo (en segundos) que debe transcurrir sin posesión para
    # que el mismo robot genere un nuevo "Control de Balón" al recuperar el balón.
    # Evita duplicados cuando el balón se ocluye unos frames entre el robot y la cámara.
    TIEMPO_MIN_RESET_POSEEDOR_S = 3.0

    # ── ESTADO DE LA MÁQUINA DE EVENTOS ─────────────────────────────────────
    ultimo_robot_poseedor  = None
    ultimo_equipo_poseedor = None
    ultimo_tiempo_poseedor = None   # FIX 3: timestamp del último frame con posesión confirmada
    ultimo_robot_liberado  = None   # FIX 3: ID del robot que soltó el balón más recientemente
    pos_balon_anterior     = None
    ultimo_frame_balon     = None   # FIX 2: frame donde el balón fue visto por última vez

    # Candidato actual de posesión y cuántos frames lleva consecutivos
    robot_candidato        = None
    equipo_candidato       = None
    frames_candidato       = 0

    # Ventana deslizante de posiciones del balón para calcular velocidad promedio
    ventana_posiciones_balon = deque(maxlen=VENTANA_VEL_FRAMES + 1)

    registro_eventos = []
    lista_frames = sorted(df['frame'].unique())

    print("[INFO] Analizando interacciones cinemáticas entre robots y balón...")

    for frame in lista_frames:
        df_frame   = df[df['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]

        row_balon   = df_frame[df_frame['id_objeto'] == 0]
        rows_robots = df_frame[df_frame['id_objeto'] != 0]

        if row_balon.empty or row_balon['area'].iloc[0] == 0:
            # Sin balón activo: resetear candidato para no acumular frames en falso
            robot_candidato  = None
            equipo_candidato = None
            frames_candidato = 0
            continue

        xb, yb           = row_balon['x'].iloc[0], row_balon['y'].iloc[0]
        pos_balon_actual  = (xb, yb)

        # FIX 2: Si hubo un gap grande desde la última detección del balón,
        # vaciar la ventana de velocidad para no calcular vel sobre ese salto.
        if ultimo_frame_balon is not None and (frame - ultimo_frame_balon) > GAP_MAX_VEL_FRAMES:
            ventana_posiciones_balon.clear()
        ultimo_frame_balon = frame

        # Guardar posición en ventana deslizante para velocidad promedio
        ventana_posiciones_balon.append(pos_balon_actual)

        # ── 1. VELOCIDAD PROMEDIO DEL BALÓN (anti-spike por vibración) ───────
        # Usamos el desplazamiento entre el punto más viejo y el más nuevo
        # de la ventana, dividido por el tiempo total de esa ventana.
        vel_balon = 0.0
        if len(ventana_posiciones_balon) >= 2:
            pos_vieja = ventana_posiciones_balon[0]
            pos_nueva = ventana_posiciones_balon[-1]
            n_intervalos = len(ventana_posiciones_balon) - 1
            dt_total  = n_intervalos * 0.04   # 25 FPS → 0.04s por frame
            vel_balon = calcular_distancia(pos_nueva, pos_vieja) / dt_total

        # ── 2. DETECTAR ROBOT MÁS CERCANO AL BALÓN ESTE FRAME ───────────────
        robot_poseedor_frame = None
        distancia_minima     = float('inf')

        for _, robot in rows_robots.iterrows():
            if robot['area'] == 0:
                continue
            xr, yr   = robot['x'], robot['y']
            dist_rb  = calcular_distancia((xb, yb), (xr, yr))

            if dist_rb < distancia_minima and dist_rb <= UMBRAL_POSESION_CM:
                distancia_minima     = dist_rb
                robot_poseedor_frame = int(robot['id_objeto'])

        # ── 3. VENTANA DE CONFIRMACIÓN DE POSESIÓN (anti-vibración) ─────────
        # Un robot debe estar en contacto N frames seguidos para confirmar posesión.
        if robot_poseedor_frame is not None:
            if robot_poseedor_frame == robot_candidato:
                # Mismo robot que el frame anterior → acumular
                frames_candidato += 1
            else:
                # Cambió el robot candidato → reiniciar contador
                robot_candidato  = robot_poseedor_frame
                equipo_candidato = MAPEO_EQUIPOS[robot_poseedor_frame]
                frames_candidato = 1

            # Solo actuar si el candidato lleva suficientes frames consecutivos
            if frames_candidato >= FRAMES_POSESION_MINIMA:
                robot_poseedor_confirmado = robot_candidato
                equipo_confirmado         = equipo_candidato

                # ── MÁQUINA DE ESTADOS: EVALUAR TRANSICIÓN ──────────────────
                if robot_poseedor_confirmado != ultimo_robot_poseedor:
                    if ultimo_robot_poseedor is not None:
                        if equipo_confirmado == ultimo_equipo_poseedor:
                            registro_eventos.append({
                                "frame": int(frame), "tiempo": float(tiempo_seg),
                                "evento": "Pase Completado",
                                "robot_implicado": robot_poseedor_confirmado,
                                "equipo": equipo_confirmado,
                                "detalles": f"Pase exitoso recibido del Robot {ultimo_robot_poseedor}"
                            })
                        else:
                            registro_eventos.append({
                                "frame": int(frame), "tiempo": float(tiempo_seg),
                                "evento": "Intercepcion",
                                "robot_implicado": robot_poseedor_confirmado,
                                "equipo": equipo_confirmado,
                                "detalles": f"Balón robado al {ultimo_equipo_poseedor} (Robot {ultimo_robot_poseedor})"
                            })
                    else:
                        # FIX 3: Solo registrar "Control de Balón" si el robot que lo toma
                        # es distinto al que lo soltó recientemente, O si pasó suficiente
                        # tiempo. Esto evita duplicados por oclusión breve del mismo robot.
                        tiempo_sin_posesion = (tiempo_seg - ultimo_tiempo_poseedor) if ultimo_tiempo_poseedor is not None else float('inf')
                        robot_diferente = (robot_poseedor_confirmado != ultimo_robot_liberado)
                        if robot_diferente or tiempo_sin_posesion >= TIEMPO_MIN_RESET_POSEEDOR_S:
                            registro_eventos.append({
                                "frame": int(frame), "tiempo": float(tiempo_seg),
                                "evento": "Control de Balon",
                                "robot_implicado": robot_poseedor_confirmado,
                                "equipo": equipo_confirmado,
                                "detalles": f"Robot {robot_poseedor_confirmado} dominó la pelota libre"
                            })

                    ultimo_robot_poseedor  = robot_poseedor_confirmado
                    ultimo_equipo_poseedor = equipo_confirmado
                    ultimo_tiempo_poseedor = tiempo_seg

        else:
            # Nadie toca el balón este frame → resetear candidato
            robot_candidato  = None
            equipo_candidato = None
            frames_candidato = 0

            # CASO B: El balón acaba de soltarse → evaluar tiro/despeje
            if ultimo_robot_poseedor is not None:
                if vel_balon >= UMBRAL_VEL_TIRO_CMS:
                    registro_eventos.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg),
                        "evento": "Tiro a Gol / Despeje",
                        "robot_implicado": ultimo_robot_poseedor,
                        "equipo": ultimo_equipo_poseedor,
                        "detalles": f"Disparo potente del Robot {ultimo_robot_poseedor} a {vel_balon:.1f} cm/s"
                    })

                # FIX 3: Guardar el robot que soltó el balón y el momento en que lo hizo.
                # No se resetea a None: así sabemos quién tuvo la posesión por última vez
                # y podemos distinguir oclusión breve (mismo robot recupera) de balón libre real.
                ultimo_tiempo_poseedor = tiempo_seg
                ultimo_robot_liberado  = ultimo_robot_poseedor
                ultimo_robot_poseedor  = None
                ultimo_equipo_poseedor = None

        pos_balon_anterior = pos_balon_actual

    # ── GUARDAR RESULTADOS ───────────────────────────────────────────────────
    df_eventos = pd.DataFrame(registro_eventos)
    if not df_eventos.empty:
        columnas_fijas = ["frame", "tiempo", "evento", "robot_implicado", "equipo", "detalles"]
        df_eventos = df_eventos[columnas_fijas].sort_values(by="frame").reset_index(drop=True)
        df_eventos.to_csv(ruta_salida, index=False)
        print(f"[OK] Módulo B1 exitoso. Archivo de eventos tácticos grabado en: {ruta_salida}\n")
        print("=== CRONOLOGÍA DE EVENTOS DEL PARTIDO ===")
        print(df_eventos.to_string(index=False))
    else:
        pd.DataFrame(columns=["frame", "tiempo", "evento", "robot_implicado", "equipo", "detalles"]).to_csv(ruta_salida, index=False)
        print("[!] No se detectaron interacciones o cambios de posesión en el partido.")
    print("==================================================================")

if __name__ == "__main__":
    main()