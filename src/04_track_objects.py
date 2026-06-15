import numpy as np
import pandas as pd
import os
from scipy.optimize import linear_sum_assignment

def calcular_distancia_euclidiana(p1, p2):
    """Calcula la distancia geométrica real en la cancha (escala 170x130)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    print("==================================================================")
    print("=== MÓDULO A2: TRACKING GLOBAL CON INICIO LONGITUDINAL VISUAL ===")
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

    # CONTRATO SAGRADO DE EQUIPOS: Fijado por ID desde el inicio de los tiempos
    MAPEO_EQUIPOS = {
        1: "Equipo A", 2: "Equipo A",  # IDs 1 y 2 siempre serán Equipo A
        3: "Equipo B", 4: "Equipo B"   # IDs 3 y 4 siempre serán Equipo B
    }

    # Estructura de memoria persistente para los 4 robots oficiales
    historial_robots = {1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0)}
    primer_frame_inicializado = False
    
    datos_rastreados = []
    UMBRAL_INTERVENCION = 35.0  

    lista_frames = sorted(df_raw['frame'].unique())

    print("[INFO] Ejecutando ordenamiento por visualización de porterías (X Landscape) y rastreo continuo...")
    for frame in lista_frames:
        df_frame = df_raw[df_raw['frame'] == frame]
        tiempo_seg = df_frame['tiempo'].iloc[0]
        # Extraemos detecciones crudas
        detecciones_actuales = df_frame[['x', 'y', 'area']].to_dict('records')
        
        ids_emparejados_este_frame = set()
        mapeo_deteccion_a_id = {}
        
        # --------------------------------------------------------------
        # 📋 FASE 1: DETERMINACIÓN AUTOMÁTICA EN EL PRIMER FRAME (KICKOFF)
        # --------------------------------------------------------------
        # NUEVA LÓGICA: Ordenar DESPUÉS de voltear las coordenadas a Landscape.
        if not primer_frame_inicializado:
            
            detecciones_con_landscape = []
            for i, det in enumerate(detecciones_actuales):
                # Volteamos las coordenadas crudas de OpenCV a formato Landscape (A3)
                # Raw Y -> Landscape X (Eje Largo 0-170, Goal-to-Goal)
                # Raw X -> Landscape Y (Eje Corto 0-130, Lateral)
                landscape_x = det['y']
                landscape_y = det['x']
                
                # Guardamos la versión volteada junto con su índice crudo original
                detecciones_con_landscape.append({
                    'original_index': i,
                    'lx': landscape_x, # Coordenada visual longitudinal
                    'det_cruda': det
                })
                
            # ORDENAMOS basándonos en 'lx' (Eje Largo Visual)
            # Esto pone a los robots en orden secuencial de portería A a portería B.
            detecciones_landscape_ordenadas = sorted(detecciones_con_landscape, key=lambda d: d['lx'])
            
            # ASIGNACIÓN DE IDs PERMANENTE
            for idx in range(4):
                r_id = idx + 1 # IDs: 1, 2, 3, 4
                if idx < len(detecciones_landscape_ordenadas):
                    # Robot detectado físicamente
                    det_flipped = detecciones_landscape_ordenadas[idx]
                    det_raw = det_flipped['det_cruda']
                    
                    # Actualizamos la memoria con las coordenadas CRUDAS (X, Y de cámara)
                    historial_robots[r_id] = (det_raw['x'], det_raw['y'])
                    
                    # Mapeamos el índice crudo de OpenCV al ID permanente
                    ids_emparejados_este_frame.add(r_id)
                    mapeo_deteccion_a_id[det_flipped['original_index']] = r_id
                else:
                    # Respaldo si el frame 1 no tuviera los 4 robots completos (Mano en medio)
                    historial_robots[r_id] = (0.0, 0.0)
                    
            primer_frame_inicializado = True
            
        # --------------------------------------------------------------
        # 📋 FASE 2: ASOCIACIÓN GLOBAL PERMANENTE (ALGORITMO HÚNGARO)
        # --------------------------------------------------------------
        elif detecciones_actuales:
            lista_ids = [1, 2, 3, 4]
            posiciones_viejas = [historial_robots[i] for i in lista_ids]
            
            # Matriz de costos (4 tracking histórico x N detecciones actuales crudas)
            matriz_costos = np.zeros((4, len(detecciones_actuales)))
            for i, pos_vieja in enumerate(posiciones_viejas):
                for j, det_actual in enumerate(detecciones_actuales):
                    matriz_costos[i, j] = calcular_distancia_euclidiana(pos_vieja, (det_actual['x'], det_actual['y']))
            
            # Asociación global óptima para minimizar el movimiento del grupo
            filas_ind, col_ind = linear_sum_assignment(matriz_costos)
            
            for f, c in zip(filas_ind, col_ind):
                r_id = lista_ids[f]
                # Filtro de seguridad (no emparejar si el salto es ridículo >75cm)
                if matriz_costos[f, c] > 75.0:
                    continue
                mapeo_deteccion_a_id[c] = r_id
                ids_emparejados_este_frame.add(r_id)

        # --------------------------------------------------------------
        # 📋 FASE 3: CONSOLIDACIÓN Y GARANTÍA DE LAS 4 FILAS OBLIGATORIAS
        # --------------------------------------------------------------
        for r_id in [1, 2, 3, 4]:
            intervencion = "no"
            
            if r_id in ids_emparejados_este_frame:
                # Extraer valores reales si el robot fue detectado
                idx_det = [k for k, v in mapeo_deteccion_a_id.items() if v == r_id][0]
                det = detecciones_actuales[idx_det]
                
                pos_anterior = historial_robots[r_id]
                # Validar brinco de intervención humana
                if pos_anterior != (0.0, 0.0) and calcular_distancia_euclidiana((det['x'], det['y']), pos_anterior) > UMBRAL_INTERVENCION:
                    intervencion = "si"
                
                # Actualizar memoria persistente con coordenadas CRUDAS
                historial_robots[r_id] = (det['x'], det['y'])
                area_salida = int(det['area'])
            else:
                # Si el robot se ocluye, mantiene su posición previa intacta en memoria (coasting)
                area_salida = 0
            
            # 🔄 VOLTEO FINAL DE PERSPECTIVA OBLIGATORIO A LANDSCAPE (A3)
            # Volvemos a aplicar la lógica: Raw Y -> Landscape X, Raw X -> Landscape Y
            pos_final_cruda = historial_robots[r_id]
            x_landscape = float(pos_final_cruda[1]) # Eje Largo Goal-to-Goal
            y_landscape = float(pos_final_cruda[0]) # Eje Corto Lateral
            
            datos_rastreados.append({
                "frame": int(frame),
                "tiempo": float(tiempo_seg),
                "id_objeto": int(r_id),
                "tipo": "robot",
                "equipo": MAPEO_EQUIPOS[r_id], # El equipo está amarrado al ID eternamente
                "x": x_landscape,
                "y": y_landscape,
                "area": area_salida,
                "intervencion_humana": intervencion
            })

    # Construcción y guardado del DataFrame sanitizado final
    df_final = pd.DataFrame(datos_rastreados)
    columnas_exactas = ["frame", "tiempo", "id_objeto", "tipo", "equipo", "x", "y", "area", "intervencion_humana"]
    df_final = df_final[columnas_exactas].sort_values(by=["frame", "id_objeto"]).reset_index(drop=True)
    df_final.to_csv(ruta_salida, index=False)
    
    print(f"[OK] Módulo A2 exitoso. Registradas {len(df_final)} filas totales (4 por frame estrictas).")
    print("==================================================================")
    print(df_final.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
