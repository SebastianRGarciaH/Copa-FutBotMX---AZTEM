import cv2
import numpy as np
import pandas as pd
import os
import re

def extraer_centroide(imagen_mascara):
    M = cv2.moments(imagen_mascara)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = int(M["m00"])
        return cX, cY, area
    return None

def ordenar_naturalmente(lista_archivos):
    convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
    return sorted(lista_archivos, key=lambda arch: [convertir(c) for c in re.split(r'(\d+)', arch)])

def main():
    print("=== MÓDULO A1: EXTRACCIÓN GEOMÉTRICA (VISTA DE ÁGUILA) ===")
    
    carpeta_mascaras = "results/masks"
    ruta_csv_salida = "results/metrics/raw_centroids.csv"
    os.makedirs("results/metrics", exist_ok=True)
    
    if not os.path.exists(carpeta_mascaras) or not os.listdir(carpeta_mascaras):
        print(f"[!] No hay máscaras en {carpeta_mascaras}")
        return

    archivos_mascaras = [f for f in os.listdir(carpeta_mascaras) if f.endswith(('.png', '.jpg', '.jpeg'))]
    archivos_mascaras = ordenar_naturalmente(archivos_mascaras)
    
    # Rango de tamaño de los robots en tus imágenes de WhatsApp
    ROBOT_MIN_AREA = 1201   
    ROBOT_MAX_AREA = 6000   
    
    FPS_VIDEO = 25  
    datos_totales = []

    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = mascara_frame.shape
        
        frame_id = idx_archivo + 1 if "WhatsApp" in nombre_archivo else int(re.findall(r'frame_(\d+)', nombre_archivo)[0])
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)
        
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contornos_con_area = []
        for c in contornos:
            M = cv2.moments(c)
            if M["m00"] > 0:
                contornos_con_area.append((c, M["m00"]))
        
        contornos_con_area = sorted(contornos_con_area, key=lambda x: x[1], reverse=True)
        robots_detectados = 0
        
        for contorno, area in contornos_con_area:
            mascara_individual = np.zeros_like(mascara_frame)
            cv2.drawContours(mascara_individual, [contorno], -1, 255, -1)
            resultado = extraer_centroide(mascara_individual)
            
            if resultado:
                cX, cY, area = resultado
                
                # Filtro adaptativo para ignorar el ruido de la esquina inferior izquierda
                if cY > (h * 0.85) and cX < (w * 0.35):
                    continue
                
                # Filtrar por tamaño: máximo los 4 robots reales de la copa
                if ROBOT_MIN_AREA <= area <= ROBOT_MAX_AREA and robots_detectados < 4:
                    datos_totales.append({
                        "frame": frame_id,
                        "tiempo": tiempo_seg,
                        "id_objeto": 0,
                        "tipo": "robot",
                        "equipo": "robot_cancha",  # Identidad genérica inicial
                        "x": cX,
                        "y": cY,
                        "area": area
                    })
                    robots_detectados += 1
                    
        print(f"[OK] Frame {frame_id}: {robots_detectados} robots localizados.")

    df = pd.DataFrame(datos_totales)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"\n[ÉXITO] {len(df)} posiciones crudas exportadas a '{ruta_csv_salida}'")

if __name__ == "__main__":
    main()