import cv2
import numpy as np
import pandas as pd
import os
import re

def extraer_centroide(imagen_mascara):
    """
    Calcula el centroide matemático (x, y) y el área de un objeto 
    en una máscara binaria utilizando momentos de imagen de OpenCV.
    """
    M = cv2.moments(imagen_mascara)
    if M["m00"] != 0:
        # Fórmulas físicas para el centro de masa
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = int(M["m00"])
        return cX, cY, area
    return None

def ordenar_naturalmente(lista_archivos):
    """
    Ordena los nombres de archivos de forma numérica natural 
    (ej: asegura que 'frame_2.png' vaya antes que 'frame_10.png').
    """
    convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
    clave_orden = lambda nombre_archivo: [convertir(c) for c in re.split('(\d+)', nombre_archivo)]
    return sorted(lista_archivos, key=clave_orden)

def main():
    print("==================================================================")
    print(" Copa FutBotMX 2026 - Módulo A2: Extracción de Centroides Real ")
    print("==================================================================")
    
    # Configuración de rutas según el plan de trabajo estructurado
    carpeta_mascaras = "results/masks"
    ruta_csv_salida = "results/metrics/tracking_data.csv"
    
    os.makedirs("results/metrics", exist_ok=True)
    
    # 1. VERIFICACIÓN DE ENTRADAS DEL PIPELINE
    if not os.path.exists(carpeta_mascaras) or not os.listdir(carpeta_mascaras):
        print(f"\n[!] Alerta: La carpeta '{carpeta_mascaras}' está vacía o no existe.")
        print("    A1 aún no ha subido máscaras reales. No hay datos para procesar.")
        return

    archivos_mascaras = [f for f in os.listdir(carpeta_mascaras) if f.endswith(('.png', '.jpg', '.jpeg'))]
    archivos_mascaras = ordenar_naturalmente(archivos_mascaras)
    
    # 2. PARÁMETROS FILTROS ENTORNO REAL (Evitan registrar manos o ruidos)
    # Región de Interés (ROI): Coordenadas límite de las bandas de madera negras de la cancha
    X_MIN, X_MAX = 15, 625
    Y_MIN, Y_MAX = 35, 455

    # Umbrales de área en píxeles (considerando multiplicación x255 de OpenCV)
    BALON_MIN_AREA = 1000
    BALON_MAX_AREA = 15000       # Si excede esto, es que una mano humana está tocando el balón
    
    ROBOT_MIN_AREA = 35000
    ROBOT_MAX_AREA = 600000      # Si excede esto, es un bloque de ruido o cables colgantes

    FPS_VIDEO = 25  # Frecuencia de muestreo temporal del video oficial de la copa
    datos_totales = []

    # 3. PROCESAMIENTO SECUENCIAL (MÁSCARA POR MÁSCARA)
    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        
        # Leer máscara en escala de grises
        mascara_frame = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        
        # Extraer el índice numérico del frame desde el nombre del archivo
        numeros_en_nombre = re.findall(r'\d+', nombre_archivo)
        frame_id = int(numeros_en_nombre[0]) if numeros_en_nombre else idx_archivo + 1
        
        # Sincronización de tiempo discreto (segundos del partido)
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)
        
        # Aislar contornos independientes en el frame actual
        contornos, _ = cv2.findContours(mascara_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        robots_equipo_a = 0
        robots_equipo_b = 0
        
        for contorno in contornos:
            # Reconstruir máscara local para el contorno individual examinado
            mascara_individual = np.zeros_like(mascara_frame)
            cv2.drawContours(mascara_individual, [contorno], -1, 255, -1)
            
            resultado = extraer_centroide(mascara_individual)
            
            if resultado:
                cX, cY, area = resultado
                
                # FILTRO ESPACIAL: Ignorar elementos fuera de los límites de juego (público)
                if not (X_MIN <= cX <= X_MAX and Y_MIN <= cY <= Y_MAX):
                    continue
                
                # FILTRO DE TAMAÑO Y CLASIFICACIÓN LOGICA
                if BALON_MIN_AREA <= area <= BALON_MAX_AREA:
                    tipo_objeto = "balon"
                    equipo_asignado = "ninguno"
                    id_objeto = 0
                elif ROBOT_MIN_AREA <= area <= ROBOT_MAX_AREA:
                    tipo_objeto = "robot"
                    # Segmentación espacial de equipos (Izquierda vs Derecha de la cancha)
                    if cX < (mascara_frame.shape[1] / 2):
                        robots_equipo_a += 1
                        equipo_asignado = "Equipo_A"
                        id_objeto = robots_equipo_a
                    else:
                        robots_equipo_b += 1
                        equipo_asignado = "Equipo_B"
                        id_objeto = robots_equipo_b
                else:
                    # El objeto es ruido (ej: el brazo del árbitro metiendo la pelota) -> Se descarta
                    continue
                
                # ESTRUCTURACIÓN DE COLUMNAS SOLICITADAS POR EL EQUIPO
                datos_totales.append({
                    "frame": frame_id,
                    "tiempo": tiempo_seg,
                    "id_objeto": id_objeto,
                    "tipo": tipo_objeto,
                    "equipo": equipo_asignado,
                    "x": cX,
                    "y": cY,
                    "area": area,
                    "intervencion_humana": "no"  # Bandera lógica para el suavizado de la Semana 2/3
                })
                
        print(f"[PROCESANDO] -> {nombre_archivo} | Objetos válidos registrados: {robots_equipo_a + robots_equipo_b + (1 if id_objeto==0 else 0)}")

    # 4. EXPORTACIÓN MATRICIAL A ARCHIVO PLANO CSV
    df = pd.DataFrame(datos_totales)
    
    if not df.empty:
        # Ordenar para que el CSV sea totalmente legible (Frame por frame, Balón arriba, luego Robots)
        df = df.sort_values(by=["frame", "tipo", "equipo"]).reset_index(drop=True)
        df.to_csv(ruta_csv_salida, index=False)
        print("\n==================================================================")
        print(f"[OK] Pipeline de la Semana 1 completado exitosamente.")
        print(f"Archivo guardado en: {ruta_csv_salida}")
        print("==================================================================\n")
        print(df.head(10).to_string(index=False)) # Muestra los primeros 10 registros en consola
    else:
        print("\n[!] Alerta: Se procesaron las imágenes pero ningún objeto superó los filtros de la cancha.")

if __name__ == "__main__":
    main()