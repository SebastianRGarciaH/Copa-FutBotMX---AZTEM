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
    ruta_salida = "results/metrics/tracking_data.csv"
    
    if not os.path.exists(ruta_entrada) or os.path.getsize(ruta_entrada) == 0:
        print("[!] El archivo raw_centroids.csv no existe o está vacío.")
        return

    df_raw = pd.read_csv(ruta_entrada)
    
    MAPEO_EQUIPOS = {
        0: "balon",
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    # Parámetros de control diferenciados
    ALPHA_ROBOT = 0.25
    ALPHA_BALON = 0.45          # Más alto para seguir la aceleración de pases y rebotes
    UMBRAL_MATCH_MAX = 35.0
    
    historial_robots = {1: None, 2: None, 3: None, 4: None}
    historial_balon = None      # Registro persistente del esférico (ID 0)
    
    datos_rastreados = []
    lista_frames = sorted(df_raw['frame'].unique())

    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        
        # Separación inmediata de canales de datos
        det_robots = df_frame[df_frame['tipo'] == 'robot'].to_dict('records')
        det_balon = df_frame[df_frame['tipo'] == 'balon'].to_dict('records')
        
        ids_activos_robots = set()
        mapeo_robot_a_id = {}
        
        # 📋 SECCIÓN A: SEGUIMIENTO DE ROBOTS (ALGORITMO HÚNGARO)
        ids_historicos = [i for i in [1, 2, 3, 4] if historial_robots[i] is not None]
        if ids_historicos and det_robots:
            pos_ref = [historial_robots[i]['pos_smooth'] for i in ids_historicos]
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

        # Inicialización de robots nuevos en zonas vírgenes
        for idx_det, det in enumerate(det_robots):
            if idx_det not in mapeo_robot_a_id:
                if det['y'] < 85.0: ids_preferidos = [1, 2]
                else: ids_preferidos = [3, 4]
                
                ids_libres = [i for i in ids_preferidos if historial_robots[i] is None]
                if ids_libres:
                    r_id = ids_libres[0]
                    mapeo_robot_a_id[idx_det] = r_id
                    ids_activos_robots.add(r_id)
                    historial_robots[r_id] = {'pos_smooth': (det['x'], det['y'])}

        # 📋 SECCIÓN B: SEGUIMIENTO DE LA PELOTA (ID 0)
        balon_detectado_este_frame = False
        pos_balon_raw = (0.0, 0.0)
        area_balon_out = 0
        
        if det_balon:
            # Si hay detecciones, elegimos la más cercana a la posición previa del balón
            if historial_balon is not None:
                distancias_balon = [calcular_distancia_euclidiana(historial_balon, (d['x'], d['y'])) for d in det_balon]
                idx_optimo = np.argmin(distancias_balon)
                best_det_balon = det_balon[idx_optimo]
            else:
                best_det_balon = det_balon[0]
                
            pos_balon_raw = (best_det_balon['x'], best_det_balon['y'])
            area_balon_out = int(best_det_balon['area'])
            balon_detectado_este_frame = True

        # Suavizado y persistencia temporal del balón
        if balon_detectado_este_frame:
            if historial_balon is not None:
                bx_smooth = ALPHA_BALON * pos_balon_raw[0] + (1.0 - ALPHA_BALON) * historial_balon[0]
                by_smooth = ALPHA_BALON * pos_balon_raw[1] + (1.0 - ALPHA_BALON) * historial_balon[1]
                historial_balon = (bx_smooth, by_smooth)
            else:
                historial_balon = pos_balon_raw
            xb_out, yb_out = historial_balon[1], historial_balon[0]
        else:
            if historial_balon is not None:
                xb_out, yb_out = historial_balon[1], historial_balon[0]
                area_balon_out = 0
            else:
                xb_out, yb_out, area_balon_out = 0.0, 0.0, 0

        # Escritura indexada de los 4 robots
        for r_id in [1, 2, 3, 4]:
            if r_id in ids_activos_robots:
                idx_o = [k for k, v in mapeo_robot_a_id.items() if v == r_id][0]
                det = det_robots[idx_o]
                pos_prev = historial_robots[r_id]['pos_smooth']
                
                xr_s = ALPHA_ROBOT * det['x'] + (1.0 - ALPHA_ROBOT) * pos_prev[0]
                yr_s = ALPHA_ROBOT * det['y'] + (1.0 - ALPHA_ROBOT) * pos_prev[1]
                historial_robots[r_id]['pos_smooth'] = (xr_s, yr_s)
                
                datos_rastreados.append({
                    "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                    "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                    "x": round(yr_s, 2), "y": round(xr_s, 2), "area": int(det['area']), "intervencion_humana": "no"
                })
            else:
                if historial_robots[r_id] is not None:
                    congelado = historial_robots[r_id]['pos_smooth']
                    datos_rastreados.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                        "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                        "x": round(congelado[1], 2), "y": round(congelado[0], 2), "area": 0, "intervencion_humana": "no"
                    })
                else:
                    datos_rastreados.append({
                        "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                        "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                        "x": 0.0, "y": 0.0, "area": 0, "intervencion_humana": "no"
                    })
                    
        # Inyección de la fila del balón (ID 0) en cada cuadro del partido
        datos_rastreados.append({
            "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": 0,
            "tipo": "balon", "equipo": "balon",
            "x": round(xb_out, 2), "y": round(yb_out, 2), "area": area_balon_out, "intervencion_humana": "no"
        })

    df_final = pd.DataFrame(datos_rastreados)
    df_final.to_csv(ruta_salida, index=False)
    print(f"[OK] Módulo A2 completado. Trazados robots e ID 0 del balón de forma paralela.")
    print("==================================================================")

if __name__ == "__main__":
    main()
