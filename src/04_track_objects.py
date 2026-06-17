import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment

def calcular_distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO A2: TRACKING CON CANDADO DE IDENTIDAD INMUTABLE =======")
    print("==================================================================")
    
    ruta_entrada = "results/metrics/raw_centroids.csv"
    ruta_salida = "results/metrics/tracking_data.csv"
    
    if not os.path.exists(ruta_entrada) or os.path.getsize(ruta_entrada) == 0:
        print("[!] El archivo raw_centroids.csv no existe o está vacío.")
        return

    df_raw = pd.read_csv(ruta_entrada)
    if df_raw.empty:
        print("[!] El archivo raw_centroids.csv no tiene datos.")
        return

    MAPEO_EQUIPOS = {
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    # ⚙️ CONFIGURACIÓN DE FILTRADO CINEMÁTICO ESTRICTO
    ALPHA_SUAVIZADO = 0.20       # Filtro pesado (0.20): Absorbe por completo el jitter y latigazos
    UMBRAL_MATCH_MAX = 35.0      # Radio máximo de desplazamiento lógico por cuadro (cm)
    
    # Estructura de memoria persistente para el ciclo de vida de los 4 robots
    historial_robots = {1: None, 2: None, 3: None, 4: None}
    datos_rastreados = []
    lista_frames = sorted(df_raw['frame'].unique())

    print("[INFO] Rastreando trayectorias con bloqueo de robo de identidad...")
    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        detecciones_actuales = df_frame[['x', 'y', 'area']].to_dict('records')
        
        ids_activos_este_frame = set()
        mapeo_deteccion_a_id = {}
        
        # 📋 PASO 1: ASOCIACIÓN GLOBAL EXCLUSIVA (ALGORITMO HÚNGARO)
        ids_historicos_activos = [i for i in [1, 2, 3, 4] if historial_robots[i] is not None]
        
        if ids_historicos_activos and detecciones_actuales:
            posiciones_referencia = [historial_robots[i]['pos_smooth'] for i in ids_historicos_activos]
            matriz_costos = np.zeros((len(ids_historicos_activos), len(detecciones_actuales)))
            
            for i, pos_ref in enumerate(posiciones_referencia):
                for j, det in enumerate(detecciones_actuales):
                    matriz_costos[i, j] = calcular_distancia_euclidiana(pos_ref, (det['x'], det['y']))
            
            filas_ind, col_ind = linear_sum_assignment(matriz_costos)
            for f, c in zip(filas_ind, col_ind):
                # Si el costo es menor al umbral físico, el robot se movió a esa posición
                if matriz_costos[f, c] <= UMBRAL_MATCH_MAX:
                    r_id = ids_historicos_activos[f]
                    mapeo_deteccion_a_id[c] = r_id
                    ids_activos_este_frame.add(r_id)

        # 📋 PASO 2: INICIALIZACIÓN BAJO DEMANDA (SÓLO EN ESPACIOS VÍRGENES)
        for idx_det, det in enumerate(detecciones_actuales):
            if idx_det not in mapeo_deteccion_a_id:
                landscape_x_test = det['y']  # Filtro longitudinal
                
                if landscape_x_test < 85.0:
                    ids_preferidos = [1, 2]
                else:
                    ids_preferidos = [3, 4]
                
                # CANDADO CRÍTICO: Sólo se puede usar el ID si nunca antes se ha inicializado (es None)
                # Esto impide por completo que un robot le robe el ID a otro que está coasting.
                ids_disponibles = [i for i in ids_preferidos if historial_robots[i] is None]
                
                if ids_disponibles:
                    r_id = ids_disponibles[0]
                    mapeo_deteccion_a_id[idx_det] = r_id
                    ids_activos_este_frame.add(r_id)
                    
                    historial_robots[r_id] = {
                        'pos_smooth': (det['x'], det['y']),
                        'pos_raw': (det['x'], det['y'])
                    }

        # 📋 PASO 3: FILTRADO TEMPORAL SUAVIZADO Y RENDERIZADO DE LAS 4 FILAS
        for r_id in [1, 2, 3, 4]:
            intervencion = "no"
            
            if r_id in ids_activos_este_frame:
                idx_origen = [k for k, v in mapeo_deteccion_a_id.items() if v == r_id][0]
                det = detecciones_actuales[idx_origen]
                pos_raw_nueva = (det['x'], det['y'])
                
                estado_previo = historial_robots[r_id]
                pos_smooth_ant = estado_previo['pos_smooth']
                
                # Aplicación del Filtro Exponencial Pasabajas
                x_smooth = ALPHA_SUAVIZADO * pos_raw_nueva[0] + (1.0 - ALPHA_SUAVIZADO) * pos_smooth_ant[0]
                y_smooth = ALPHA_SUAVIZADO * pos_raw_nueva[1] + (1.0 - ALPHA_SUAVIZADO) * pos_smooth_ant[1]
                pos_smooth_nueva = (x_smooth, y_smooth)
                
                historial_robots[r_id] = {
                    'pos_smooth': pos_smooth_nueva,
                    'pos_raw': pos_raw_nueva
                }
                
                area_out = int(det['area'])
                x_out = float(pos_smooth_nueva[1])  # Mapeo a Landscape X (Longitudinal 170cm)
                y_out = float(pos_smooth_nueva[0])  # Mapeo a Landscape Y (Lateral 130cm)
                
            else:
                # 🛡️ PROTECCIÓN MEMORIA PERSISTENTE
                # Si el robot no fue detectado, mantiene congelada su última posición exacta calculada
                if historial_robots[r_id] is not None:
                    estado_congelado = historial_robots[r_id]['pos_smooth']
                    x_out = float(estado_congelado[1])
                    y_out = float(estado_congelado[0])
                    area_out = 0
                else:
                    # El robot aún no se ha presentado en la lona
                    x_out = 0.0
                    y_out = 0.0
                    area_out = 0
            
            datos_rastreados.append({
                "frame": int(frame), "tiempo": float(tiempo_seg), "id_objeto": int(r_id),
                "tipo": "robot", "equipo": MAPEO_EQUIPOS[r_id],
                "x": round(x_out, 2), "y": round(y_out, 2), "area": area_out,
                "intervencion_humana": intervencion
            })

    # Guardado seguro de la matriz limpia
    df_final = pd.DataFrame(datos_rastreados)
    columnas_exactas = ["frame", "tiempo", "id_objeto", "tipo", "equipo", "x", "y", "area", "intervencion_humana"]
    df_final = df_final[columnas_exactas].sort_values(by=["frame", "id_objeto"]).reset_index(drop=True)
    df_final.to_csv(ruta_salida, index=False)
    print(f"[OK] Módulo A2 exitoso. Saltos instantáneos e intercambios eliminados de la lona.")
    print("==================================================================")

if __name__ == "__main__":
    main()
