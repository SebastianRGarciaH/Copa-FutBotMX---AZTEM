import pandas as pd
import numpy as np
import os

def calcular_distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO B1: MOTOR CINEMÁTICO DE EVENTOS Y JUGADAS DE JUEGO ===")
    print("==================================================================")
    
    ruta_entrada = "results/metrics/tracking_data.csv"
    ruta_salida = "results/metrics/game_events.csv"
    
    if not os.path.exists(ruta_entrada):
        print(f"[!] No se encontró el archivo de tracking estable: {ruta_entrada}")
        return

    df = pd.read_csv(ruta_entrada)
    if df.empty:
        print("[!] El archivo de telemetría está vacío.")
        return

    # Contrato oficial de asignación de bandos para la evaluación de pases e intercepciones
    MAPEO_EQUIPOS = {
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    # ⚙️ UMBRALES DE CALIBRACIÓN GEOMÉTRICA
    UMBRAL_POSESION_CM = 14.5   # Radio físico de vecindad de posesión (Robot + Balón + Tolerancia)
    UMBRAL_VEL_TIRO_CMS = 60.0  # Velocidad mínima del balón para considerarse disparo o despeje

    # Variables de estado de la máquina de eventos
    ultimo_robot_poseedor = None
    ultimo_equipo_poseedor = None
    pos_balon_anterior = None
    
    registro_eventos = []
    lista_frames = sorted(df['frame'].unique())

    print("[INFO] Analizando interacciones cinemáticas entre robots y balón...")
    for frame in lista_frames:
        df_frame = df[df['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        
        # Extraer coordenadas del cuadro actual
        row_balon = df_frame[df_frame['id_objeto'] == 0]
        rows_robots = df_frame[df_frame['id_objeto'] != 0]
        
        # Si el balón no está activo o se perdió el cuadro, saltamos la iteración
        if row_balon.empty or row_balon['area'].iloc[0] == 0:
            continue
            
        xb, yb = row_balon['x'].iloc[0], row_balon['y'].iloc[0]
        pos_balon_actual = (xb, yb)
        
        # 1. Calcular la velocidad instantánea del balón (cm/s) para detectar tiros
        vel_balon = 0.0
        if pos_balon_anterior is not None:
            # Nuestro pipeline corre a 25 FPS fijos -> dt = 0.04s por cuadro
            dt = 0.04
            dist_balon = calcular_distancia(pos_balon_actual, pos_balon_anterior)
            vel_balon = dist_balon / dt
            
        # 2. Determinar si algún robot tiene control físico de la pelota en este frame
        robot_poseedor_actual = None
        distancia_minima = float('inf')
        
        for _, robot in rows_robots.iterrows():
            if robot['area'] == 0:  # Ignorar robots en oclusión
                continue
            xr, yr = robot['x'], robot['y']
            dist_rb = calcular_distancia((xb, yb), (xr, yr))
            
            if dist_rb < distancia_minima and dist_rb <= UMBRAL_POSESION_CM:
                distancia_minima = dist_rb
                robot_poseedor_actual = int(robot['id_objeto'])

        # 3. MÁQUINA DE ESTADOS REGLAMENTARIA (EVALUACIÓN DE TRANSICIONES)
        if robot_poseedor_actual is not None:
            equipo_actual = MAPEO_EQUIPOS[robot_poseedor_actual]
            
            # CASO A: Cambió el robot que tiene la posesión de la pelota
            if robot_poseedor_actual != ultimo_robot_poseedor:
                if ultimo_robot_poseedor is not None:
                    # Si el dueño anterior era del mismo equipo -> PASE COMPLETADO
                    if equipo_actual == ultimo_equipo_poseedor:
                        registro_eventos.append({
                            "frame": int(frame), "tiempo": float(tiempo_seg),
                            "evento": "Pase Completado", "robot_implicado": robot_poseedor_actual,
                            "equipo": equipo_actual, "detalles": f"Pase exitoso recibido del Robot {ultimo_robot_poseedor}"
                        })
                    # Si el dueño anterior era rival -> INTERCEPCIÓN / ROBO
                    else:
                        registro_eventos.append({
                            "frame": int(frame), "tiempo": float(tiempo_seg),
                            "evento": "Intercepcion", "robot_implicado": robot_poseedor_actual,
                            "equipo": equipo_actual, "detalles": f"Balón robado al {ultimo_equipo_poseedor} (Robot {ultimo_robot_poseedor})"
                        })
                else:
                    # El balón venía libre y un robot lo controló
                    registro_eventos.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg),
                        "evento": "Control de Balon", "robot_implicado": robot_poseedor_actual,
                        "equipo": equipo_actual, "detalles": f"Robot {robot_poseedor_actual} dominó la pelota libre"
                    })
                
                # Actualizar el estado de la posesión activa
                ultimo_robot_poseedor = robot_poseedor_actual
                ultimo_equipo_poseedor = equipo_actual
                
        else:
            # CASO B: La pelota está suelta en la lona (Nadie la toca en este frame)
            if ultimo_robot_poseedor is not None:
                # Si el balón se acaba de soltar y lleva una velocidad alta -> TIRO / DESPEJE
                if vel_balon >= UMBRAL_VEL_TIRO_CMS:
                    # Determinar dirección táctica del disparo (Lona X: 0 a 170)
                    # Si el Equipo A (izquierdo) dispara hacia la derecha (X creciente) es un ataque
                    registro_eventos.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg),
                        "evento": "Tiro a Gol / Despeje", "robot_implicado": ultimo_robot_poseedor,
                        "equipo": ultimo_equipo_poseedor, 
                        "detalles": f"Disparo potente del Robot {ultimo_robot_poseedor} a {vel_balon:.1f} cm/s"
                    })
                
                # Liberar la posesión en la máquina de estados
                ultimo_robot_poseedor = None
                ultimo_equipo_poseedor = None

        # Guardar historial cinemático para la siguiente iteración
        pos_balon_anterior = pos_balon_actual

    # Consolidar y guardar la matriz de eventos cronológicos
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