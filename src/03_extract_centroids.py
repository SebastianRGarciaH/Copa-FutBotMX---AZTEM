import cv2
import numpy as np
import pandas as pd
import os
import re

def extraer_centroide(imagen_mascara):
    """Calcula el centroide (x, y) y el área de un objeto usando momentos."""
    M = cv2.moments(imagen_mascara)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = int(M["m00"])
        return cX, cY, area
    return None

def ordenar_naturalmente(lista):
    """Ordena los archivos de forma numérica (ej: frame_2 antes de frame_10)."""
    convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
    clave_orden = lambda nombre_archivo: [convertir(c) for c in re.split('([0-8]+)', nombre_archivo)]
    return sorted(lista, key=clave_orden)

def main():
    print("--- Extracción de Centroides desde Máscaras ---")
    
    # Rutas asignadas en el plan de trabajo
    carpeta_mascaras = "results/masks"
    ruta_csv_salida = "results/metrics/tracking_data.csv"
    
    # Crear carpetas de salida si no existen
    os.makedirs("results/metrics", exist_ok=True)
    
    # Verificar si la carpeta de máscaras existe y tiene archivos
    if not os.path.exists(carpeta_mascaras) or not os.listdir(carpeta_mascaras):
        print(f"\n[!] Alerta: La carpeta '{carpeta_mascaras}' está vacía o no existe.")
        return

    # Leer y ordenar los archivos de la carpeta
    archivos_mascaras = [f for f in os.listdir(carpeta_mascaras) if f.endswith(('.png', '.jpg', '.jpeg'))]
    archivos_mascaras = ordenar_naturalmente(archivos_mascaras)
    
    datos_totales = []
    FPS_VIDEO = 25  # Ajusta este valor según los FPS del video original del torneo
    
    # Procesar cada archivo de máscara encontrado
    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        
        # 1. Leer la máscara real en escala de grises
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        
        # Intentar extraer el número de frame directamente del nombre del archivo
        numeros_en_nombre = re.findall(r'\d+', nombre_archivo)
        frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else idx_archivo + 1
        
        # Calcular el segundo exacto del partido
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)
        
        # 2. Encontrar todos los contornos (objetos blancos) en esta máscara
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        robots_equipo_a = 0
        robots_equipo_b = 0
        
        for idx_objeto, contorno in enumerate(contornos):
            # Aislar el contorno en una máscara individual
            mascara_individual = np.zeros_like(mascara_frame)
            cv2.drawContours(mascara_individual, [contorno], -1, 255, -1)
            
            resultado = extraer_centroide(mascara_individual)
            
            if resultado:
                cX, cY, area = resultado
                
                # Clasificación por tamaño (ajustar umbral según resolución final del video)
                if area < 60000:
                    tipo_objeto = "balon"
                    equipo_asignado = "ninguno"
                    id_objeto = 0
                else:
                    tipo_objeto = "robot"
                    # Segmentación espacial temporal (mitad izquierda vs mitad derecha)
                    if cX < (mascara_frame.shape[1] / 2):
                        robots_equipo_a += 1
                        equipo_asignado = "Equipo_A"
                        id_objeto = robots_equipo_a
                    else:
                        robots_equipo_b += 1
                        equipo_asignado = "Equipo_B"
                        id_objeto = robots_equipo_b
                
                # Insertar en la lista con tu formato en español
                datos_totales.append({
                    "frame": frame_id,
                    "segundo": tiempo_seg,
                    "id_objeto": id_objeto,
                    "tipo_objeto": tipo_objeto,
                    "equipo": equipo_asignado,
                    "x": cX,
                    "y": cY,
                    "area": area,
                    "confianza": 1.0
                })
                
        print(f"-> Procesado archivo: {nombre_archivo} (Frame {frame_id} | Detectados: {len(contornos)} objetos)")

    # 3. Guardar todo el histórico en el CSV final
    df = pd.DataFrame(datos_totales)
    df = df.sort_values(by=["frame", "tipo_objeto", "equipo"]).reset_index(drop=True)
    df.to_csv(ruta_csv_salida, index=False)
    
    print(f"\n[OK] Procesamiento completo. Archivo guardado en: {ruta_csv_salida}")

if __name__ == "__main__":
    main()