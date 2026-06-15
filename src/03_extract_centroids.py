import cv2
import numpy as np
import pandas as pd
import os
import re
from scipy.cluster.vq import kmeans2

def ordenar_naturalmente(lista_archivos):
    return sorted(lista_archivos, key=lambda arch: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', arch)])

def main():
    print("==================================================================")
    print("=== MÓDULO A1: EXTRACTOR ADAPTATIVO POR K-MEANS INTERNATIVO =====")
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
    
    # Rango ultra-ancho para evitar que las fusiones se eliminen del mapa
    ROBOT_MIN_AREA = 1500   
    ROBOT_MAX_AREA = 30000   
    AREA_PROMEDIO_ROBOT = 6300.0 # Tamaño real detectado en tus imágenes
    
    ANCHO_REAL_X = 130.0  
    LARGO_REAL_Y = 170.0  
    FPS_VIDEO = 25  
    datos_totales = []

    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = mascara_frame.shape
        
        # 1. Limpieza absoluta del ruido blanco de la esquina inferior izquierda
        mascara_frame[int(h * 0.78):, :int(w * 0.45)] = 0
        
        factor_x = ANCHO_REAL_X / w
        factor_y = LARGO_REAL_Y / h
        
        numeros_en_nombre = re.findall(r'(\d+)', nombre_archivo)
        frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else (idx_archivo + 1)
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)
        
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        robots_detectados_frame = 0
        
        for c in contornos:
            area_pixel = cv2.contourArea(c)
            
            # Filtro base para ignorar ruidos pequeños
            if area_pixel < ROBOT_MIN_AREA:
                continue
                
            # Calcular cuántos robots estimamos que viven dentro de este bloque
            num_robots_fused = int(np.round(area_pixel / AREA_PROMEDIO_ROBOT))
            if num_robots_fused < 1:
                num_robots_fused = 1
                
            # Crear máscara del contorno bajo análisis
            mascara_individual = np.zeros_like(mascara_frame)
            cv2.drawContours(mascara_individual, [c], -1, 255, -1)
            
            # CASO 1: Un solo robot limpio y aislado
            if num_robots_fused == 1 and area_pixel <= ROBOT_MAX_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    if not (cY > (h * 0.96) or cY < (h * 0.02)):
                        datos_totales.append({
                            "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                            "tipo": "robot", "equipo": "robot_cancha",
                            "x": round(cX * factor_x, 2), "y": round(cY * factor_y, 2), "area": int(area_pixel)
                        })
                        robots_detectados_frame += 1
                        
            # CASO 2: FUSIÓN DETECTADA (2 o más robots encimados)
            elif num_robots_fused > 1 and area_pixel <= ROBOT_MAX_AREA:
                # Extraemos los índices de todos los píxeles blancos del bloque
                puntos_blancos = np.argwhere(mascara_individual == 255)
                # Convertimos a coordenadas [X, Y] cartesianas
                puntos_x_y = puntos_blancos[:, [1, 0]].astype(float)
                
                try:
                    # K-Means dividirá la nube de píxeles en el número exacto de centros requeridos
                    centros_calculados, _ = kmeans2(puntos_x_y, num_robots_fused, minit='points')
                    area_proporcional = int(area_pixel / num_robots_fused)
                    
                    for centro in centros_calculados:
                        cX, cY = int(centro[0]), int(centro[1])
                        
                        if not (cY > (h * 0.96) or cY < (h * 0.02)):
                            datos_totales.append({
                                "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                                "tipo": "robot", "equipo": "robot_cancha",
                                "x": round(cX * factor_x, 2), "y": round(cY * factor_y, 2), "area": area_proporcional
                            })
                            robots_detectados_frame += 1
                except Exception:
                    # Respaldo geométrico si K-Means llega a quedar indeterminado
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        datos_totales.append({
                            "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                            "tipo": "robot", "equipo": "robot_cancha",
                            "x": round(cX * factor_x, 2), "y": round(cY * factor_y, 2), "area": int(area_pixel)
                        })

    df = pd.DataFrame(datos_totales, columns=columnas_tabla)
    df = df.sort_values(by=["frame"]).reset_index(drop=True)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"[OK] Extracción adaptativa terminada con {len(df)} coordenadas generadas con éxito.")

if __name__ == "__main__":
    main()