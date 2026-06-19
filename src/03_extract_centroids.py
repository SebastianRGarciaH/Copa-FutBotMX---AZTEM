import cv2
import numpy as np
import pandas as pd
import os
import re

def ordenar_naturalmente(lista_archivos):
    return sorted(lista_archivos, key=lambda arch: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', arch)])

def procesar_contornos_brillantes(mascara_brillante, es_perspectiva, factor_x, factor_y, BALL_MIN_AREA, BALL_MAX_AREA):
    """
    Procesa la banda brillante (~220-255): puede contener balón o mano.
    El balón es pequeño y redondo. La mano es grande y alargada.
    Retorna lista de detecciones de balón válidas como (cX, cY, area).
    """
    # Cierre suave para unir píxeles del balón que vibración separó
    kernel = np.ones((3, 3), np.uint8)
    mascara_brillante = cv2.morphologyEx(mascara_brillante, cv2.MORPH_CLOSE, kernel, iterations=1)

    contornos, _ = cv2.findContours(mascara_brillante, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []

    for c in contornos:
        area_pixel = cv2.contourArea(c)

        # Solo interesa el rango de tamaño del balón
        if not (BALL_MIN_AREA <= area_pixel < BALL_MAX_AREA):
            continue

        # Filtro de forma: el balón es redondo, la mano es alargada e irregular
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidad = (4 * np.pi * area_pixel) / (perimetro ** 2)

        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(c)
        lado_mayor = max(w_bb, h_bb)
        lado_menor = min(w_bb, h_bb)
        relacion_aspecto = lado_mayor / lado_menor if lado_menor > 0 else 999

        # Mano: baja circularidad Y relación de aspecto alta → descartar
        if circularidad < 0.35 or relacion_aspecto > 2.5:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        detecciones.append((cX, cY, int(area_pixel)))

    return detecciones


def procesar_contornos_robots(mascara_robots, es_perspectiva, factor_x, factor_y, ROBOT_MIN_AREA):
    """
    Procesa la banda gris (~100-210): contiene robots.
    Los robots son blobs compactos con huecos internos.
    Aplica morfología agresiva para unir fragmentos y cerrar huecos.
    Retorna lista de detecciones de robot válidas como (cX, cY, area).
    """
    # CLOSE grande para cerrar los huecos negros internos del robot
    kernel_cierre = np.ones((9, 9), np.uint8)
    mascara_robots = cv2.morphologyEx(mascara_robots, cv2.MORPH_CLOSE, kernel_cierre, iterations=2)

    # OPEN suave para eliminar fragmentos sueltos por vibración,
    # pero sin borrar el robot principal
    kernel_apertura = np.ones((5, 5), np.uint8)
    mascara_robots = cv2.morphologyEx(mascara_robots, cv2.MORPH_OPEN, kernel_apertura, iterations=1)

    contornos, _ = cv2.findContours(mascara_robots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []

    for c in contornos:
        area_pixel = cv2.contourArea(c)

        if area_pixel < ROBOT_MIN_AREA:
            continue

        # Una mano que se cuele en esta banda también sería grande y alargada
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidad = (4 * np.pi * area_pixel) / (perimetro ** 2)

        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(c)
        lado_mayor = max(w_bb, h_bb)
        lado_menor = min(w_bb, h_bb)
        relacion_aspecto = lado_mayor / lado_menor if lado_menor > 0 else 999

        # Robots son compactos; descartamos blobs muy alargados
        if relacion_aspecto > 3.5 or circularidad < 0.15:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        detecciones.append((cX, cY, int(area_pixel)))

    return detecciones


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

    # --- UMBRALES DE BANDA DE BRILLO ---
    # Robots: gris medio. Balón y manos: muy brillante (casi blanco).
    UMBRAL_GRIS_MIN   = 100   # Por debajo → fondo negro, ignorar
    UMBRAL_GRIS_MAX   = 210   # Banda de robots
    UMBRAL_BRILLANTE  = 220   # Por encima → balón o mano

    print(f"[INFO] Extrayendo centros con separación por banda de brillo...")

    for idx_archivo, nombre_archivo in enumerate(archivos_mascaras):
        ruta_completa = os.path.join(carpeta_mascaras, nombre_archivo)
        imagen_raw = cv2.imread(ruta_completa, cv2.IMREAD_GRAYSCALE)
        h, w = imagen_raw.shape

        es_perspectiva = w > h

        # ── SEPARACIÓN POR BANDA DE BRILLO ──────────────────────────────────
        # Banda robots: píxeles en rango gris medio
        _, banda_inf = cv2.threshold(imagen_raw, UMBRAL_GRIS_MIN - 1, 255, cv2.THRESH_BINARY)
        _, banda_sup = cv2.threshold(imagen_raw, UMBRAL_GRIS_MAX,     255, cv2.THRESH_BINARY)
        mascara_robots = cv2.bitwise_and(banda_inf, cv2.bitwise_not(banda_sup))

        # Banda brillante: píxeles muy claros (balón o mano)
        _, mascara_brillante = cv2.threshold(imagen_raw, UMBRAL_BRILLANTE, 255, cv2.THRESH_BINARY)
        # ────────────────────────────────────────────────────────────────────

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

            mascara_robots    = cv2.warpPerspective(mascara_robots,    M_homografia, (w_plano, h_plano))
            mascara_brillante = cv2.warpPerspective(mascara_brillante, M_homografia, (w_plano, h_plano))

            # Recortar bordes de perspectiva en ambas bandas
            mascara_robots[:,    0:int(w_plano * 0.06)] = 0
            mascara_robots[:,    int(w_plano * 0.94):]  = 0
            mascara_brillante[:, 0:int(w_plano * 0.06)] = 0
            mascara_brillante[:, int(w_plano * 0.94):]  = 0

            factor_x = ANCHO_REAL_X / h_plano
            factor_y = LARGO_REAL_Y / w_plano

            BALL_MIN_AREA  = 8
            BALL_MAX_AREA  = 45
            ROBOT_MIN_AREA = 50
        else:
            # Recortar zona conflictiva en vista cenital
            mascara_robots[int(h * 0.78):,    :int(w * 0.45)] = 0
            mascara_brillante[int(h * 0.78):, :int(w * 0.45)] = 0

            factor_x = ANCHO_REAL_X / w
            factor_y = LARGO_REAL_Y / h

            BALL_MIN_AREA  = 100
            BALL_MAX_AREA  = 1400   # Balón real ~950-970 px (35x34); margen para variación entre frames
            ROBOT_MIN_AREA = 1200

        # Extraer número de frame y tiempo
        numeros_en_nombre = re.findall(r'(\d+)', nombre_archivo)
        frame_id   = int(numeros_en_nombre[0]) if numeros_en_nombre else (idx_archivo + 1)
        tiempo_seg = round(frame_id / FPS_VIDEO, 2)

        # ── PROCESAR BALÓN (banda brillante) ────────────────────────────────
        for (cX, cY, area) in procesar_contornos_brillantes(
                mascara_brillante, es_perspectiva, factor_x, factor_y,
                BALL_MIN_AREA, BALL_MAX_AREA):
            if es_perspectiva:
                x_real = round(cY * factor_x, 2)
                y_real = round(cX * factor_y, 2)
            else:
                x_real = round(cX * factor_x, 2)
                y_real = round(cY * factor_y, 2)
            datos_totales.append({
                "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                "tipo": "balon", "equipo": "balon",
                "x": x_real, "y": y_real, "area": area
            })

        # ── PROCESAR ROBOTS (banda gris) ─────────────────────────────────────
        for (cX, cY, area) in procesar_contornos_robots(
                mascara_robots, es_perspectiva, factor_x, factor_y,
                ROBOT_MIN_AREA):
            if es_perspectiva:
                x_real = round(cY * factor_x, 2)
                y_real = round(cX * factor_y, 2)
            else:
                x_real = round(cX * factor_x, 2)
                y_real = round(cY * factor_y, 2)
            datos_totales.append({
                "frame": int(frame_id), "tiempo": float(tiempo_seg), "id_objeto": 0,
                "tipo": "robot", "equipo": "robot_cancha",
                "x": x_real, "y": y_real, "area": area
            })

    df = pd.DataFrame(datos_totales, columns=columnas_tabla)
    df = df.sort_values(by=["frame"]).reset_index(drop=True)
    df.to_csv(ruta_csv_salida, index=False)
    print(f"[OK] Extracción mixta terminada. Detectadas {len(df)} entidades totales en bruto.")

if __name__ == "__main__":
    main()