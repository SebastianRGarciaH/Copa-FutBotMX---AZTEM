import cv2
import numpy as np
import pandas as pd
import os
import re

def extraer_centroide(imagen_mascara):
    """Calcula el centroide (x, y) y el área usando momentos de OpenCV."""
    M = cv2.moments(imagen_mascara)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = int(M["m00"])
        return cX, cY, area
    return None

def ordenar_naturalmente(lista_archivos):
    """Asegura el ordenamiento numérico de los archivos de imagen."""
    convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
    return sorted(lista_archivos, key=lambda arch: [convertir(c) for c in re.split(r'(\d+)', arch)])

def main():
    print("==================================================================")
    print("=== MÓDULO A1: EXTRACCIÓN POR MÁSCARAS (ESCALA REAL 170 x 130) ===")
    print("==================================================================")
    
    carpeta_mascaras = "results/masks"
    ruta_csv_salida = "results/metrics/raw_centroids.csv"
    os.makedirs("results/metrics", exist_ok=True)
    
    if not os.path.exists(carpeta_mascaras) or not os.listdir(carpeta_mascaras):
        print(f"[!] No hay máscaras en {carpeta_mascaras}")
        return

    archivos_mascaras = [f for f in os.listdir(carpeta_mascaras) if f.endswith(('.png', '.jpg', '.jpeg'))]
    archivos_mascaras = ordenar_naturalmente(archivos_mascaras)
    
    # Filtro de tamaño: Ajustado para ignorar dedos/brazos humanos y aislar chasis
    ROBOT_MIN_AREA = 1201   
    ROBOT_MAX_AREA = 6000   
    
    # Dimensiones físicas de la lona (Toma vertical de la cámara)
    ANCHO_REAL_X = 130.0  # El ancho corto mapea a X
    LARGO_REAL_Y = 170.0  # El largo largo mapea a Y
    FPS_VIDEO = 25  
    datos_totales = []

    print(f"[INFO] Procesando {len(archivos_mascaras)} máscaras de segmentación...")
    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = mascara_frame.shape
        
        # Factores de conversión dinámicos (cm por píxel)
        factor_x = ANCHO_REAL_X / w
        factor_y = LARGO_REAL_Y / h
        
        # Extraer el número de frame de forma segura del nombre del archivo
        numeros_en_nombre = re.findall(r'(\d+)', nombre_archivo)
        frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else (idx_archivo + 1)
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)
        
        # Encontrar contornos externos de las manchas blancas
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contornos_con_area = []
        for c in contornos:
            M = cv2.moments(c)
            if M["m00"] > 0:
                contornos_con_area.append((c, M["m00"]))
        
        # Ordenar de mayor a menor tamaño para priorizar los objetos grandes (robots)
        contornos_con_area = sorted(contornos_con_area, key=lambda x: x[1], reverse=True)
        robots_detectados = 0
        
        for contorno, area in contornos_con_area:
            mascara_individual = np.zeros_like(mascara_frame)
            cv2.drawContours(mascara_individual, [contorno], -1, 255, -1)
            resultado = extraer_centroide(mascara_individual)
            
            if resultado:
                cX, cY, area = resultado
                
                # Filtro adaptativo para ignorar el ruido de los operadores en los bordes extremos
                if cY > (h * 0.92) or cY < (h * 0.05):
                    continue
                
                # Guardar la posición si pasa el filtro de tamaño de chasis
                if ROBOT_MIN_AREA <= area <= ROBOT_MAX_AREA and robots_detectados < 4:
                    datos_totales.append({
                        "frame": int(frame_id),
                        "tiempo": float(tiempo_seg),
                        "id_objeto": 0,
                        "tipo": "robot",
                        "equipo": "robot_cancha",
                        "x": round(cX * factor_x, 2),
                        "y": round(cY * factor_y, 2),
                        "area": int(area)
                    })
                    robots_detectados += 1

    df = pd.DataFrame(datos_totales)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"\n[ÉXITO] {len(df)} coordenadas base exportadas correctamente a '{ruta_csv_salida}'")

if __name__ == "__main__":
    main()