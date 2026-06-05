import numpy as np
import pandas as pd
import os

def calcular_distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("=== MÓDULO A2: MOTOR DE TRACKING GLOBAL MULTI-OBJETO ===")
    
    ruta_entrada = "results/metrics/raw_centroids.csv"
    ruta_salida = "results/metrics/tracking_data.csv"
    
    if not os.path.exists(ruta_entrada):
        print(f"[!] Archivo no encontrado: {ruta_entrada}")
        return

    df_raw = pd.read_csv(ruta_entrada)
    if df_raw.empty:
        return

    # Historial de posiciones del frame anterior: { id_objeto: (x, y) }
    historial_robots = {}
    proximo_id_global = 1
    datos_rastreados = []
    
    UMBRAL_INTERVENCION = 70  # Umbral de píxeles para detectar saltos por manos humanas
    lista_frames = sorted(df_raw['frame'].unique())

    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        
        nuevos_robots_frame = {}
        historial_disponible = historial_robots.copy()
        
        # Procesar los robots detectados en este cuadro
        for _, row in df_frame.iterrows():
            pos_actual = (row['x'], row['y'])
            id_asignado = None
            intervencion = "no"
            
            if historial_disponible:
                # Buscamos la coincidencia más cercana en todo el historial disponible
                id_mas_cercano = min(
                    historial_disponible.keys(),
                    key=lambda k: calcular_distancia_euclidiana(pos_actual, historial_disponible[k])
                )
                distancia = calcular_distancia_euclidiana(pos_actual, historial_disponible[id_mas_cercano])
                
                # Si está a una distancia lógica de su posición anterior
                if distancia <= UMBRAL_INTERVENCION:
                    id_asignado = id_mas_cercano
                else:
                    # Si el salto es muy grande pero es el único candidato coherente, es una intervención humana
                    id_asignado = id_mas_cercano
                    intervencion = "si"
                
                # Consumimos el ID para que otro contorno no lo tome en este frame
                del historial_disponible[id_asignado]
            
            # Si es el primer frame o aparece un robot nuevo en escena
            if id_asignado is None:
                id_asignado = proximo_id_global
                proximo_id_global += 1
                # Toque de seguridad: la copa solo permite un máximo de 4 robots en juego
                if proximo_id_global > 4:
                    proximo_id_global = 1
            
            nuevos_robots_frame[id_asignado] = pos_actual
            
            datos_rastreados.append({
                "frame": frame,
                "tiempo": tiempo_seg,
                "id_objeto": id_asignado,
                "tipo": "robot",
                "equipo": f"Robot_{id_asignado}", # Identidad persistente única
                "x": int(row['x']),
                "y": int(row['y']),
                "area": int(row['area']),
                "intervencion_humana": intervencion
            })
            
        # Actualizamos la memoria del tracker para el siguiente frame
        historial_robots = nuevos_robots_frame

    df_final = pd.DataFrame(datos_rastreados)
    df_final = df_final.sort_values(by=["frame", "id_objeto"]).reset_index(drop=True)
    df_final.to_csv(ruta_salida, index=False)
    
    print("==================================================================")
    print(f"[OK] Tracking global fijado. Archivo generado en: {ruta_salida}")
    print("==================================================================\n")
    print(df_final.head(15).to_string(index=False))

if __name__ == "__main__":
    main()