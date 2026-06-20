import os
import cv2
import numpy as np

# =====================================================================
# COLORES PREMIUM (Formato BGR para OpenCV)
# =====================================================================
COLOR_BG_DARK = (18, 18, 20)      # Gris casi negro
COLOR_ACCENT = (255, 132, 10)     # Azul Apple (0A84FF en RGB -> BGR)
COLOR_TEXT_MAIN = (250, 250, 250) # Blanco puro suave
COLOR_TEXT_SUB = (160, 160, 160)  # Gris claro para subtítulos
COLOR_BORDER = (50, 50, 50)       # Gris oscuro para bordes sutiles

# =====================================================================
# FUNCIONES DE DISEÑO UI PREMIUM
# =====================================================================

def letterbox_image(img, ancho_total, alto_total, bg_color=COLOR_BG_DARK):
    """Ajusta una imagen al tamaño deseado manteniendo su proporción intacta."""
    if img is None:
        return np.full((alto_total, ancho_total, 3), bg_color, dtype=np.uint8)
    
    h, w = img.shape[:2]
    scale = min(ancho_total / w, alto_total / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((alto_total, ancho_total, 3), bg_color, dtype=np.uint8)
    x_off = (ancho_total - new_w) // 2
    y_off = (alto_total - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def poner_texto_centrado(img, texto, y, fuente, escala, color, grosor):
    tamano_texto = cv2.getTextSize(texto, fuente, escala, grosor)[0]
    x = (img.shape[1] - tamano_texto[0]) // 2
    cv2.putText(img, texto, (x, y), fuente, escala, color, grosor, cv2.LINE_AA)

def crear_frame_texto(texto_principal, subtitulo, ancho, alto):
    """Pantalla de título estilo diapositiva minimalista."""
    frame = np.full((alto, ancho, 3), COLOR_BG_DARK, dtype=np.uint8)
    fuente_titulo = cv2.FONT_HERSHEY_TRIPLEX
    fuente_sub = cv2.FONT_HERSHEY_DUPLEX
    
    # Textos
    poner_texto_centrado(frame, texto_principal, alto // 2 - 20, fuente_titulo, 1.6, COLOR_TEXT_MAIN, 2)
    poner_texto_centrado(frame, subtitulo, alto // 2 + 60, fuente_sub, 1.0, COLOR_TEXT_SUB, 1)
    
    # Línea de acento separadora
    tam_titulo = cv2.getTextSize(texto_principal, fuente_titulo, 1.6, 2)[0]
    x_linea_inicio = (ancho - tam_titulo[0]) // 2
    x_linea_fin = x_linea_inicio + tam_titulo[0]
    y_linea = alto // 2 + 15
    cv2.line(frame, (x_linea_inicio, y_linea), (x_linea_fin, y_linea), COLOR_ACCENT, 2, cv2.LINE_AA)
    
    return frame

def poner_subtitulo(frame, texto):
    """Barra superior elegante con acento inferior estilo dashboard."""
    alto, ancho = frame.shape[:2]
    overlay = frame.copy()
    alto_barra = 90
    
    # Fondo translúcido oscuro
    cv2.rectangle(overlay, (0, 0), (ancho, alto_barra), (10, 10, 12), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    
    # Línea de acento azul en la parte inferior de la barra
    cv2.line(frame, (0, alto_barra), (ancho, alto_barra), COLOR_ACCENT, 2, cv2.LINE_AA)
    
    fuente = cv2.FONT_HERSHEY_DUPLEX
    escala = 1.0
    grosor = 1
    tamano_texto = cv2.getTextSize(texto, fuente, escala, grosor)[0]
    x_texto = (ancho - tamano_texto[0]) // 2
    y_texto = (alto_barra + tamano_texto[1]) // 2
    
    cv2.putText(frame, texto, (x_texto, y_texto), fuente, escala, COLOR_TEXT_MAIN, grosor, cv2.LINE_AA)
    return frame

def poner_etiqueta_esquina(frame, texto, posicion='izq'):
    """Etiqueta moderna de estado (UI Box con indicador luminoso)."""
    alto, ancho = frame.shape[:2]
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.65
    grosor = 1
    tam_texto = cv2.getTextSize(texto, fuente, escala, grosor)[0]
    
    pad_x, pad_y = 25, 12
    radio_punto = 5
    espacio_punto = 15
    
    w_box = tam_texto[0] + pad_x*2 + radio_punto*2 + espacio_punto
    h_box = tam_texto[1] + pad_y*2
    
    y_box = alto - h_box - 25
    x_box = 25 if posicion == 'izq' else ancho - w_box - 25

    # Fondo oscuro de la etiqueta y borde fino
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_box, y_box), (x_box + w_box, y_box + h_box), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), COLOR_BORDER, 1, cv2.LINE_AA)
    
    # Punto luminoso azul
    cx_punto = x_box + pad_x + radio_punto
    cy_punto = y_box + h_box // 2
    cv2.circle(frame, (cx_punto, cy_punto), radio_punto, COLOR_ACCENT, -1, cv2.LINE_AA)
    
    # Texto de la etiqueta
    x_texto = cx_punto + radio_punto + espacio_punto
    y_texto = y_box + h_box - pad_y - 2
    cv2.putText(frame, texto, (x_texto, y_texto), fuente, escala, COLOR_TEXT_MAIN, grosor, cv2.LINE_AA)
    
    return frame

def dibujar_transicion_minimalista(frame):
    """Badge central minimalista: Transición SAM 3."""
    alto, ancho = frame.shape[:2]
    cx, cy = ancho // 2, alto // 2
    radio = 45
    
    # Círculo oscuro de base con borde azul fino
    cv2.circle(frame, (cx, cy), radio, (15, 15, 15), -1)
    cv2.circle(frame, (cx, cy), radio, COLOR_ACCENT, 2, cv2.LINE_AA)
    
    # Flecha blanca pequeña y limpia
    cv2.arrowedLine(frame, (cx - 20, cy + 8), (cx + 20, cy + 8), COLOR_TEXT_MAIN, 2, tipLength=0.3, line_type=cv2.LINE_AA)
    
    # Texto
    texto = "SAM 3"
    tam_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(frame, texto, (cx - tam_texto[0]//2, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    return frame

# =====================================================================
# LOGICA DE ENSAMBLAJE Y CUADRICULAS
# =====================================================================

def crear_split_screen(f_izq, f_der, ancho, alto):
    f_izq_adj = letterbox_image(f_izq, ancho // 2, alto)
    f_der_adj = letterbox_image(f_der, ancho // 2, alto)
    
    lienzo = np.hstack((f_izq_adj, f_der_adj))
    # Línea separadora central delgada
    cv2.line(lienzo, (ancho // 2, 0), (ancho // 2, alto), COLOR_BORDER, 1, cv2.LINE_AA)
    return lienzo

def crear_grid_4_imagenes(imagenes, ancho, alto):
    """Grid optimizado numerado que MANTIENE LA PROPORCIÓN (diseño limpio)."""
    lienzo = np.full((alto, ancho, 3), COLOR_BG_DARK, dtype=np.uint8)
    celda_w, celda_h = ancho // 2, alto // 2
    
    for i, img in enumerate(imagenes):
        if img is None or i >= 4: continue
        row, col = i // 2, i % 2
        
        # Letterboxing interno para no deformar
        h, w = img.shape[:2]
        scale = min(celda_w / w, celda_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        y_off = (row * celda_h) + ((celda_h - new_h) // 2)
        x_off = (col * celda_w) + ((celda_w - new_w) // 2)
        
        lienzo[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        # Etiqueta de numeración UI (Círculo sutil en esquina de la celda)
        cx, cy = (col * celda_w) + 35, (row * celda_h) + 35
        cv2.circle(lienzo, (cx, cy), 16, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(lienzo, (cx, cy), 16, COLOR_BORDER, 1, cv2.LINE_AA)
        
        texto_num = str(i + 1)
        tam_txt = cv2.getTextSize(texto_num, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(lienzo, texto_num, (cx - tam_txt[0]//2, cy + tam_txt[1]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # Líneas separadoras de la cuadrícula sutiles
    cv2.line(lienzo, (celda_w, 0), (celda_w, alto), COLOR_BORDER, 1, cv2.LINE_AA)
    cv2.line(lienzo, (0, celda_h), (ancho, celda_h), COLOR_BORDER, 1, cv2.LINE_AA)
    
    return lienzo

def generar_video_demo():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir) 
    
    # Rutas
    vid_path = os.path.join(root_dir, 'data', 'partido.mp4')
    gif_path = os.path.join(root_dir, 'results', 'visualizations', 'tracking_visualization.gif')
    out_path = os.path.join(root_dir, 'results', 'demo_final.mp4')
    
    ANCHO, ALTO = 1920, 1080
    FPS = 30
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (ANCHO, ALTO))
    
    print("Iniciando renderizado del video demo (Diseño UI Premium) en Full HD...")

    # Cargar las 7 visualizaciones
    vis_data = [
        ('01_heatmap.png', "Mapas de calor: Zonas de mayor actividad global."),
        ('02_trayectorias.png', "Trayectorias: Seguimiento historico del desplazamiento."),
        ('03_pases.png', "Deteccion de Pases: Interacciones completadas entre robots."),
        ('04_tiros.png', "Tiros a Porteria: Identificacion de jugadas ofensivas."),
        ('05_voronoi.png', "Diagramas de Voronoi: Analisis de control y dominio espacial."),
        ('06_posesion.png', "Posesion: Distribucion del control del balon."),
        ('07_red_pases.png', "Red de Pases: Conectividad y asociaciones del equipo.")
    ]
    
    visualizaciones_cargadas = []
    for filename, subtitulo in vis_data:
        p = os.path.join(root_dir, 'results', 'visualizations', filename)
        img = cv2.imread(p)
        if img is not None:
            visualizaciones_cargadas.append((img, subtitulo))

    # Cargar máscaras
    # Nota: la vista angular (partido_angle.mp4) es un clip sintético de
    # prueba de rutas (circulo en fondo negro), no metraje real de cancha,
    # por lo que el demo final se construye solo con la vista cenital.
    lista_masks = [cv2.imread(os.path.join(root_dir, 'results', 'masks', f'frame_{i:05d}_mask.png')) for i in range(1, 5)]

    grid_masks = crear_grid_4_imagenes(lista_masks, ANCHO, ALTO)

    cap_orig = cv2.VideoCapture(vid_path)
    cap_gif = cv2.VideoCapture(gif_path)

    def read_frame_safe(cap):
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, f = cap.read()
        return f

    # --- FASE 1: Presentación (5s) [0:00 - 0:05] ---
    print("Renderizando Fase 1 (5s)...")
    frame_intro = crear_frame_texto("Copa FutBotMX - Equipo AZTEM", "Herramienta de Analisis", ANCHO, ALTO)
    for _ in range(5 * FPS): out.write(frame_intro)

    # --- FASE 2: Segmentación (30s) [0:05 - 0:35] ---
    print("Renderizando Fase 2 (30s)...")
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(30 * FPS):
        f_orig = read_frame_safe(cap_orig)
        frame_split = crear_split_screen(f_orig, grid_masks, ANCHO, ALTO)
        
        frame_split = poner_etiqueta_esquina(frame_split, "Vista Aerea - Original", 'izq')
        frame_split = poner_etiqueta_esquina(frame_split, "Segmentacion 2x2", 'der')
        frame_split = dibujar_transicion_minimalista(frame_split)
        frame_split = poner_subtitulo(frame_split, "Segmentacion automatica de elementos en cancha.")
        out.write(frame_split)

    # --- FASE 3: Tracking (45s) [0:35 - 1:20] ---
    print("Renderizando Fase 3 (45s)...")
    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    for _ in range(45 * FPS):
        f_orig_clean = read_frame_safe(cap_orig)
        f_gif = read_frame_safe(cap_gif)
        frame_split = crear_split_screen(f_orig_clean, f_gif, ANCHO, ALTO)
        
        frame_split = poner_etiqueta_esquina(frame_split, "Vista Original", 'izq')
        frame_split = poner_etiqueta_esquina(frame_split, "Tracking Digital", 'der')
        frame_split = poner_subtitulo(frame_split, "Seguimiento preciso mediante calculo de centroides.")
        out.write(frame_split)

    # --- FASE 4: Visualizaciones (35s) [1:20 - 1:55] ---
    print("Renderizando Fase 4 (35s)...")
    if not visualizaciones_cargadas:
        frame_err = crear_frame_texto("Error de Archivos", "No se encontraron imagenes en results/visualizations", ANCHO, ALTO)
        for _ in range(35 * FPS): out.write(frame_err)
    else:
        frames_por_imagen = (35 * FPS) // len(visualizaciones_cargadas)
        for img, texto_sub in visualizaciones_cargadas:
            frame_vis = letterbox_image(img, ANCHO, ALTO, bg_color=COLOR_BG_DARK)
            frame_vis = poner_subtitulo(frame_vis, texto_sub)
            for _ in range(frames_por_imagen): out.write(frame_vis)
            
        frames_restantes = (35 * FPS) - (frames_por_imagen * len(visualizaciones_cargadas))
        if frames_restantes > 0 and visualizaciones_cargadas:
            for _ in range(frames_restantes): out.write(frame_vis)

    # --- FASE 5: Cierre (5s) [1:55 - 2:00] ---
    print("Renderizando Fase 5 (5s)...")
    frame_cierre = crear_frame_texto("Gracias", "Repositorio disponible en GitHub", ANCHO, ALTO)
    for _ in range(5 * FPS): out.write(frame_cierre)

    cap_orig.release()
    cap_gif.release()
    out.release()
    print("\n[OK] Video Demo Full HD Generado Exitosamente en:", out_path)

if __name__ == "__main__":
    generar_video_demo()
