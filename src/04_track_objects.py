import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment

def calcular_distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO A2: RASTREADOR CINEMÁTICO DE ENTORNO COMPLETO ========")
    print("==================================================================")

    ruta_entrada = "results/metrics/raw_centroids.csv"
    ruta_salida  = "results/metrics/tracking_data.csv"

    if not os.path.exists(ruta_entrada) or os.path.getsize(ruta_entrada) == 0:
        print("[!] El archivo raw_centroids.csv no existe o está vacío.")
        return

    df_raw = pd.read_csv(ruta_entrada)

    MAPEO_EQUIPOS = {
        0: "balon",
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    # ── PARÁMETROS DE CONTROL ────────────────────────────────────────────────
    ALPHA_ROBOT      = 0.20       # Suavizado exponencial robots (más bajo = más estable ante vibración)
    ALPHA_BALON      = 0.45       # Más alto para seguir pases y rebotes reales
    UMBRAL_MATCH_MAX = 35.0       # Distancia máxima para asociar detección a robot conocido (cm)

    # Anti-vibración robots: si el robot "se mueve" menos de esto, ignoramos el desplazamiento
    ZONA_MUERTA_ROBOT_CM = 2.5

    # Anti-fantasma: un robot nuevo debe aparecer N frames seguidos antes de ser aceptado
    FRAMES_CONFIRMACION_ROBOT = 3

    # Anti-vibración balón: si el balón salta más de esto en un frame, descartamos la detección
    SALTO_MAX_BALON_CM = 80.0     # ~2 metros reales; ajustar si hay tiros muy rápidos
    # ────────────────────────────────────────────────────────────────────────

    historial_robots  = {1: None, 2: None, 3: None, 4: None}
    historial_balon   = None   # (x, y) suavizado del balón

    # Contador de frames consecutivos para robots candidatos (anti-fantasma)
    # Estructura: { id_robot: frames_confirmados_seguidos }
    conteo_confirmacion = {1: 0, 2: 0, 3: 0, 4: 0}
    ROBOTS_CONFIRMADOS  = set()   # IDs que ya superaron la ventana de confirmación

    datos_rastreados = []
    lista_frames = sorted(df_raw['frame'].unique())

    for frame in lista_frames:
        df_frame  = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]

        det_robots = df_frame[df_frame['tipo'] == 'robot'].to_dict('records')
        det_balon  = df_frame[df_frame['tipo'] == 'balon'].to_dict('records')

        ids_activos_robots = set()
        mapeo_robot_a_id   = {}

        # ── SECCIÓN A: SEGUIMIENTO DE ROBOTS (ALGORITMO HÚNGARO) ────────────
        ids_historicos = [i for i in [1, 2, 3, 4] if historial_robots[i] is not None]

        if ids_historicos and det_robots:
            pos_ref       = [historial_robots[i]['pos_smooth'] for i in ids_historicos]
            matriz_costos = np.zeros((len(ids_historicos), len(det_robots)))

            for i, p_ref in enumerate(pos_ref):
                for j, det in enumerate(det_robots):
                    matriz_costos[i, j] = calcular_distancia_euclidiana(p_ref, (det['x'], det['y']))

            filas, colas = linear_sum_assignment(matriz_costos)
            for f, c in zip(filas, colas):
                if matriz_costos[f, c] <= UMBRAL_MATCH_MAX:
                    r_id = ids_historicos[f]
                    mapeo_robot_a_id[c] = r_id
                    ids_activos_robots.add(r_id)

        # Inicialización de robots nuevos con ventana de confirmación anti-fantasma
        for idx_det, det in enumerate(det_robots):
            if idx_det not in mapeo_robot_a_id:
                if det['y'] < 85.0:
                    ids_preferidos = [1, 2]
                else:
                    ids_preferidos = [3, 4]

                ids_libres = [i for i in ids_preferidos if historial_robots[i] is None]
                if ids_libres:
                    r_id = ids_libres[0]
                    conteo_confirmacion[r_id] += 1

                    if conteo_confirmacion[r_id] >= FRAMES_CONFIRMACION_ROBOT:
                        # Ya apareció suficientes frames seguidos → lo aceptamos
                        ROBOTS_CONFIRMADOS.add(r_id)
                        mapeo_robot_a_id[idx_det] = r_id
                        ids_activos_robots.add(r_id)
                        if historial_robots[r_id] is None:
                            historial_robots[r_id] = {'pos_smooth': (det['x'], det['y'])}
                    # Si no llegó al mínimo, no lo registramos este frame
            else:
                # Robot ya conocido: resetear su contador (no es fantasma)
                r_id = mapeo_robot_a_id[idx_det]
                conteo_confirmacion[r_id] = FRAMES_CONFIRMACION_ROBOT  # Marcado como confirmado

        # Resetear contadores de robots que NO aparecieron este frame
        for r_id in [1, 2, 3, 4]:
            if r_id not in ids_activos_robots and historial_robots[r_id] is None:
                # Candidato que desapareció antes de confirmarse → reiniciar contador
                conteo_confirmacion[r_id] = 0

        # ── SECCIÓN B: SEGUIMIENTO DEL BALÓN ────────────────────────────────
        balon_detectado_este_frame = False
        pos_balon_raw  = (0.0, 0.0)
        area_balon_out = 0

        if det_balon:
            if historial_balon is not None:
                distancias_balon = [calcular_distancia_euclidiana(historial_balon, (d['x'], d['y'])) for d in det_balon]
                idx_optimo       = np.argmin(distancias_balon)
                best_det_balon   = det_balon[idx_optimo]
                dist_mejor       = distancias_balon[idx_optimo]

                # Anti-vibración balón: descartamos saltos imposibles
                if dist_mejor <= SALTO_MAX_BALON_CM:
                    pos_balon_raw  = (best_det_balon['x'], best_det_balon['y'])
                    area_balon_out = int(best_det_balon['area'])
                    balon_detectado_este_frame = True
                # Si el salto es mayor, ignoramos la detección y usamos la posición anterior
            else:
                # Primera vez que vemos el balón → aceptamos sin filtro
                best_det_balon = det_balon[0]
                pos_balon_raw  = (best_det_balon['x'], best_det_balon['y'])
                area_balon_out = int(best_det_balon['area'])
                balon_detectado_este_frame = True

        # Suavizado y persistencia temporal del balón
        if balon_detectado_este_frame:
            if historial_balon is not None:
                bx_smooth      = ALPHA_BALON * pos_balon_raw[0] + (1.0 - ALPHA_BALON) * historial_balon[0]
                by_smooth      = ALPHA_BALON * pos_balon_raw[1] + (1.0 - ALPHA_BALON) * historial_balon[1]
                historial_balon = (bx_smooth, by_smooth)
            else:
                historial_balon = pos_balon_raw
            xb_out, yb_out = historial_balon[0], historial_balon[1]
        else:
            if historial_balon is not None:
                xb_out, yb_out = historial_balon[0], historial_balon[1]
                area_balon_out = 0
            else:
                xb_out, yb_out, area_balon_out = 0.0, 0.0, 0

        # ── ESCRITURA DE LOS 4 ROBOTS ────────────────────────────────────────
        for r_id in [1, 2, 3, 4]:
            if r_id in ids_activos_robots:
                idx_o   = [k for k, v in mapeo_robot_a_id.items() if v == r_id][0]
                det     = det_robots[idx_o]
                pos_prev = historial_robots[r_id]['pos_smooth']

                # Anti-vibración robots: zona muerta — si el movimiento es ruido, no actualizamos
                desplazamiento = calcular_distancia_euclidiana(pos_prev, (det['x'], det['y']))
                if desplazamiento > ZONA_MUERTA_ROBOT_CM:
                    xr_s = ALPHA_ROBOT * det['x'] + (1.0 - ALPHA_ROBOT) * pos_prev[0]
                    yr_s = ALPHA_ROBOT * det['y'] + (1.0 - ALPHA_ROBOT) * pos_prev[1]
                else:
                    # Movimiento menor a la zona muerta → congelamos (es vibración)
                    xr_s, yr_s = pos_prev[0], pos_prev[1]

                historial_robots[r_id]['pos_smooth'] = (xr_s, yr_s)

                datos_rastreados.append({
                    "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                    "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                    "x": round(xr_s, 2), "y": round(yr_s, 2), "area": int(det['area']),
                    "intervencion_humana": "no"
                })
            else:
                if historial_robots[r_id] is not None:
                    congelado = historial_robots[r_id]['pos_smooth']
                    datos_rastreados.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                        "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                        "x": round(congelado[0], 2), "y": round(congelado[1], 2), "area": 0,
                        "intervencion_humana": "no"
                    })
                else:
                    datos_rastreados.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                        "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                        "x": 0.0, "y": 0.0, "area": 0,
                        "intervencion_humana": "no"
                    })

        # Inyección de la fila del balón (ID 0) en cada cuadro del partido
        datos_rastreados.append({
            "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": 0,
            "tipo": "balon", "equipo": "balon",
            "x": round(xb_out, 2), "y": round(yb_out, 2), "area": area_balon_out,
            "intervencion_humana": "no"
        })

    df_final = pd.DataFrame(datos_rastreados)
    df_final.to_csv(ruta_salida, index=False)
    print(f"[OK] Módulo A2 completado. Trazados robots e ID 0 del balón de forma paralela.")
    print("==================================================================")

if __name__ == "__main__":
    main()