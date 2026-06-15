import numpy as np
import pandas as pd
import os

def calcular_distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO A2: TRACKING POR VELOCIDAD PREDICTIVA (FILTRO DE VEL) ===")
    print("==================================================================")
    
    ruta_entrada = "results/metrics/raw_centroids.csv"
    ruta_salida = "results/metrics/tracking_data.csv"
    
    if not os.path.exists(ruta_entrada) or os.path.getsize(ruta_entrada) == 0:
        print("[!] Error con el archivo de entrada raw_centroids.csv")
        return

    df_raw = pd.read_csv(ruta_entrada)

    MAPEO_EQUIPOS = {
        1: "Equipo A", 2: "Equipo A",
        3: "Equipo B", 4: "Equipo B"
    }

    historial_robots = {}
    vectores_velocidad = {} # {id: (vx, vy)}
    proximo_id_global = 1
    datos_rastreados = []
    
    UMBRAL_INTERVENCION = 25  
    lista_frames = sorted(df_raw['frame'].unique())

    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        
        nuevos_robots_frame = {}
        historial_disponible = historial_robots.copy()
        
        for _, row in df_frame.iterrows():
            pos_actual = (row['x'], row['y'])
            id_asignado = None
            intervencion = "no"
            
            if historial_disponible:
                # PREDICCIÓN: Calculamos dónde DEBERÍA estar el robot basándonos en su velocidad previa
                dict_predicciones = {}
                for k, v in historial_disponible.items():
                    vx, vy = vectores_velocidad.get(k, (0, 0))
                    dict_predicciones[k] = (v[0] + vx, v[1] + vy)
                
                # Buscamos el candidato más cercano a la PREDICCIÓN, no a la posición estática
                id_mas_cercano = min(
                    historial_disponible.keys(),
                    key=lambda k: calcular_distancia_euclidiana(pos_actual, dict_predicciones[k])
                )
                distancia_real = calcular_distancia_euclidiana(pos_actual, historial_disponible[id_mas_cercano])
                
                if distancia_real <= UMBRAL_INTERVENCION:
                    id_asignado = id_mas_cercano
                    # Actualizar vector de velocidad real del robot (atenuado para suavizar ruido)
                    pos_ant = historial_disponible[id_mas_cercano]
                    vectores_velocidad[id_asignado] = (pos_actual[0] - pos_ant[0], pos_actual[1] - pos_ant[1])
                else:
                    id_asignado = id_mas_cercano
                    intervencion = "si"
                    vectores_velocidad[id_asignado] = (0, 0) # Frenado por intervención humana
                
                del historial_disponible[id_asignado]
            
            if id_asignado is None:
                id_asignado = proximo_id_global
                proximo_id_global += 1
                if proximo_id_global > 4:
                    proximo_id_global = 1
                vectores_velocidad[id_asignado] = (0, 0)
            
            nuevos_robots_frame[id_asignado] = pos_actual
            
            x_landscape = float(row['y'])
            y_landscape = float(row['x'])
            
            equipo_real = MAPEO_EQUIPOS.get(id_asignado, f"Equipo_{id_asignado}")
            
            datos_rastreados.append({
                "frame": int(frame),
                "tiempo": float(tiempo_seg),
                "id_objeto": int(id_asignado),
                "tipo": "robot",
                "equipo": equipo_real,
                "x": x_landscape,
                "y": y_landscape,
                "area": int(row['area']),
                "intervencion_humana": intervencion
            })
            
        historial_robots = nuevos_robots_frame

    df_final = pd.DataFrame(datos_rastreados)
    df_final['equipo'] = df_final['id_objeto'].map(MAPEO_EQUIPOS).fillna(df_final['equipo'])

    columnas_exactas = ["frame", "tiempo", "id_objeto", "tipo", "equipo", "x", "y", "area", "intervencion_humana"]
    df_final = df_final[columnas_exactas]
    df_final = df_final.sort_values(by=["frame", "id_objeto"]).reset_index(drop=True)

    if os.path.exists(ruta_salida):
        try: os.remove(ruta_salida)
        except Exception: pass
            
    df_final.to_csv(ruta_salida, index=False)
    print(f"[OK] TRACKING PREDICTIVO PROCESADO CON ÉXITO.")
    print(df_final.head(4).to_string(index=False))

if __name__ == "__main__":
    main()