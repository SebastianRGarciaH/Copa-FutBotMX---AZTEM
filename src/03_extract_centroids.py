import cv2
import numpy as np
import pandas as pd
import os
import re

def ordenar_naturalmente(lista_archivos):
    return sorted(lista_archivos, key=lambda arch: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', arch)])

def main():
    print("==================================================================")
    print("=== MÓDULO A1: EXTRACTOR CORREGIDO (ORIENTACIÓN LANDSCAPE REAL) ===")
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

    print(f"[INFO] Procesando con alineación estricta de porterías horizontales...")

    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = mascara_frame.shape
        
        # 1. Binarización para remover el piso gris
        _, mascara_frame = cv2.threshold(mascara_frame, 180, 255, cv2.THRESH_BINARY)
        
        es_perspectiva = w > h
        
        if es_perspectiva:
            # Puntos del trapecio de la lona inclinada
            pts_origen = np.float32([
                [int(w * 0.15), int(h * 0.35)],  # Esquina Superior Izquierda
                [int(w * 0.85), int(h * 0.35)],  # Esquina Superior Derecha
                [int(w * 0.98), int(h * 0.95)],  # Esquina Inferior Derecha
                [int(w * 0.02), int(h * 0.95)]   # Esquina Inferior Izquierda
            ])
            
            # NUEVA GEOMETRÍA: Lienzo completamente horizontal (680x520)
            w_plano, h_plano = 680, 520
            pts_destino = np.float32([[0, 0], [w_plano, 0], [w_plano, h_plano], [0, h_plano]])
            
            M_homografia = cv2.getPerspectiveTransform(pts_origen, pts_destino)
            mascara_frame = cv2.warpPerspective(mascara_frame, M_homografia, (w_plano, h_plano))
            
            # 🛡️ ROI CORREGIDO: Las porterías ahora están a la IZQUIERDA y DERECHA del lienzo wide
            mascara_frame[:, 0:int(w_plano * 0.06)] = 0        # Portería Izquierda
            mascara_frame[:, int(w_plano * 0.94):] = 0        # Portería Derecha
            mascara_frame[0:int(h_plano * 0.02), :] = 0        # Lateral Superior
            mascara_frame[int(h_plano * 0.98):, :] = 0        # Lateral Inferior
            
            # Mapeo directo y calibrado de factores de conversión
            factor_x = ANCHO_REAL_X / h_plano                  # Eje Corto (130cm)
            factor_y = LARGO_REAL_Y / w_plano                  # Eje Largo (170cm)
            ROBOT_MIN_AREA = 80   
        else:
            # Ojo de Águila tradicional (Vertical)
            mascara_frame[int(h * 0.78):, :int(w * 0.45)] = 0
            kernel = np.ones((5, 5), np.uint8)
            mascara_frame = cv2.erode(mascara_frame, kernel, iterations=1)
            
            factor_x = ANCHO_REAL_X / w
            factor_y = LARGO_REAL_Y / h
            ROBOT_MIN_AREA = 1500   

        # Encontrar contornos sobre la imagen rectificada
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        robots_detectados_frame = 0
        for c in contornos:
            area_pixel = cv2.contourArea(c)
            if area_pixel < ROBOT_MIN_AREA:
                continue
                
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
                
            cX_pixel = int(M["m10"] / M["m00"])
            cY_pixel = int(M["m01"] / M["m00"])
            
            # Asignación correcta de coordenadas espaciales homologadas para el Script 04
            if es_perspectiva:
                x_real = round(cY_pixel * factor_x, 2)  # El eje vertical plano se vuelve el ancho
                y_real = round(cX_pixel * factor_y, 2)  # El eje horizontal plano se vuelve el largo
            else:
                x_real = round(cX_pixel * factor_x, 2)
                y_real = round(cY_pixel * factor_y, 2)
            
            # Filtro de proximidad para evitar duplicados residuales
            duplicado = False
            for r_det in datos_totales:
                if r_det["frame"] == (idx_archivo + 1):
                    dist = np.sqrt((r_det["x"] - x_real)**2 + (r_det["y"] - y_real)**2)
                    if dist < 12.0:
                        duplicado = True
                        break
            
            if not duplicado and robots_detectados_frame < 4:
                numeros_en_nombre = re.findall(r'(\d+)', nombre_archivo)
                frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else (idx_archivo + 1)
                tiempo_seg = round(frame_id / FPS_VIDEO, 2)
                
                datos_totales.append({
                    "frame": int(frame_id),
                    "tiempo": float(tiempo_seg),
                    "id_objeto": 0,
                    "tipo": "robot",
                    "equipo": "robot_cancha",
                    "x": x_real,
                    "y": y_real,
                    "area": int(area_pixel)
                })
                robots_detectados_frame += 1

    df = pd.DataFrame(datos_totales, columns=columnas_tabla)
    df = df.sort_values(by=["frame"]).reset_index(drop=True)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"[OK] Extracción de homografía horizontal completada con {len(df)} centros reales alineados.")

if __name__ == "__main__":
    main()
