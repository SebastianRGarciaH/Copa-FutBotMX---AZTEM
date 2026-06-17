import cv2
import numpy as np
import pandas as pd
import os
import re

def ordenar_naturalmente(lista_archivos):
    return sorted(lista_archivos, key=lambda arch: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', arch)])

def main():
    print("==================================================================")
    print("=== MÓDULO A1: EXTRACTOR HÍBRIDO DE ROBOTS Y BALÓN =============")
    print("==================================================================")
    
    carpeta_mascaras = "results/masks"
    ruta_csv_salida = "results/metrics/raw_centroids.csv"
    os.makedirs("results/metrics", exist_ok=True)
    
    columnas_tabla = ["frame", "tiempo", "id_objeto", "tipo", "equipo", "x", "y", "area"]

    if not os.path.exists(carpeta_mascaras) or not os.listdir(carpeta_mascaras):
        print(f"[!] No hay máscaras en {carpeta_mascaras}")
        pd.DataFrame(columns=columnas_tabla).to_csv(ruta_csv_salida, index=False)
        return

    archivos_mascaras = [f for f in os.listdir(carpeta_mascaras) if f.endswith(('.png', '.jpg', '.jpeg'))]
    archivos_mascaras = ordenar_naturalmente(archivos_mascaras)
    
    ANCHO_REAL_X = 130.0  
    LARGO_REAL_Y = 170.0  
    FPS_VIDEO = 25  
    datos_totales = []

    print(f"[INFO] Extrayendo centros con discriminación de escala para el balón...")

    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = mascara_frame.shape
        
        _, mascara_frame = cv2.threshold(mascara_frame, 180, 255, cv2.THRESH_BINARY)
        es_perspectiva = w > h
        
        if es_perspectiva:
            pts_origen = np.float32([
                [int(w * 0.15), int(h * 0.35)],
                [int(w * 0.85), int(h * 0.35)],
                [int(w * 0.98), int(h * 0.95)],
                [int(w * 0.02), int(h * 0.95)]
            ])
            w_plano, h_plano = 680, 520
            pts_destino = np.float32([[0, 0], [w_plano, 0], [w_plano, h_plano], [0, h_plano]])
            
            M_homografia = cv2.getPerspectiveTransform(pts_origen, pts_destino)
            mascara_frame = cv2.warpPerspective(mascara_frame, M_homografia, (w_plano, h_plano))
            
            mascara_frame[:, 0:int(w_plano * 0.06)] = 0        
            mascara_frame[:, int(w_plano * 0.94):] = 0        
            
            factor_x = ANCHO_REAL_X / h_plano                  
            factor_y = LARGO_REAL_Y / w_plano                  
            
            # ⚽ VENTANAS DE COMPACTACIÓN (PERSPECTIVA)
            BALL_MIN_AREA = 8
            BALL_MAX_AREA = 45
            ROBOT_MIN_AREA = 50
        else:
            mascara_frame[int(h * 0.78):, :int(w * 0.45)] = 0
            kernel = np.ones((5, 5), np.uint8)
            mascara_frame = cv2.erode(mascara_frame, kernel, iterations=1)
            
            factor_x = ANCHO_REAL_X / w
            factor_y = LARGO_REAL_Y / h
            
            # ⚽ VENTANAS DE COMPACTACIÓN (CENITAL)
            BALL_MIN_AREA = 100
            BALL_MAX_AREA = 900
            ROBOT_MIN_AREA = 1200

        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contornos:
            area_pixel = cv2.contourArea(c)
            
            # Clasificación analítica por densidad de pixeles
            if BALL_MIN_AREA <= area_pixel < BALL_MAX_AREA:
                tipo_objeto = "balon"
                equipo_objeto = "balon"
            elif area_pixel >= ROBOT_MIN_AREA:
                tipo_objeto = "robot"
                equipo_objeto = "robot_cancha"
            else:
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0: continue
                
            cX_pixel = int(M["m10"] / M["m00"])
            cY_pixel = int(M["m01"] / M["m00"])
            
            if es_perspectiva:
                x_real = round(cY_pixel * factor_x, 2)
                y_real = round(cX_pixel * factor_y, 2)
            else:
                x_real = round(cX_pixel * factor_x, 2)
                y_real = round(cY_pixel * factor_y, 2)
            
            numeros_en_nombre = re.findall(r'(\d+)', nombre_archivo)
            frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else (idx_archivo + 1)
            tiempo_seg = round(frame_id / FPS_VIDEO, 2)
            
            datos_totales.append({
                "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                "tipo": tipo_objeto, "equipo": equipo_objeto,
                "x": x_real, "y": y_real, "area": int(area_pixel)
            })

    df = pd.DataFrame(datos_totales, columns=columnas_tabla)
    df = df.sort_values(by=["frame"]).reset_index(drop=True)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"[OK] Extracción mixta terminada. Detectadas {len(df)} entidades totales en bruto.")

if __name__ == "__main__":
    main()
