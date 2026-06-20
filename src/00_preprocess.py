import os
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Directorios
# ---------------------------------------------------------------------------
FRAMES_DIR = "data/frames"
OUTPUT_DIR = "data/frames_processed"
DEBUG_DIR  = "data/preprocess_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR,  exist_ok=True)

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
OUTPUT_W = 900
OUTPUT_H = 1200

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

DARK_THRESH = 90
CLEAN_DILATE = 25 

# Detección automática de esquinas activada por defecto (corta solo donde
# hay verde). Si falla o se ve mal en el preview, descomenta la línea de
# abajo para volver a las esquinas fijas que ya tenías:
MANUAL_CORNERS = None
MANUAL_CORNERS_FALLBACK = np.array([
    [7,310 ], #ad
    [829,229 ],    #ai
    [1002,1619],  # Abajo-Derecha
    [0,1794],  # Abajo-Izquierda
], dtype=np.float32)
# MANUAL_CORNERS = MANUAL_CORNERS_FALLBACK   # <- descomenta esta línea para revertir

# Rango HSV de "verde cancha". Amplio a propósito para tolerar variación de
# luz; si el recorte agarra de más (pared/graderías verdosas) prueba subir
# GREEN_HSV_LOW un poco; si pierde pasto en sombras, bájalo.
GREEN_HSV_LOW  = np.array([35, 50, 50])
GREEN_HSV_HIGH = np.array([85, 255, 255])

# Ajuste fino del cuadrilátero detectado, en píxeles, antes de la homografía.
# Positivo = encoge el recorte hacia el centro (si te quedó una franja de
# fondo/pared visible). Negativo = lo expande hacia afuera (si te está
# comiendo pasto real en los bordes). Empieza en 0 y ajusta de 5 en 5.
CORNER_MARGIN_PX = 0


# ===========================================================================
# DETECCIÓN DE ESQUINAS
# ===========================================================================

def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    idx_y  = np.argsort(pts[:, 1])
    top    = pts[idx_y[:2]]; top    = top[np.argsort(top[:,0])]
    bottom = pts[idx_y[2:]]; bottom = bottom[np.argsort(bottom[:,0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

def apply_margin(corners, margin_px):
    """Mueve cada esquina hacia (margin_px>0) o desde (margin_px<0) el
    centroide del cuadrilátero, a lo largo de la línea esquina-centroide.
    Sirve para afinar el recorte sin volver a tocar el HSV ni los kernels."""
    if margin_px == 0:
        return corners
    centro = corners.mean(axis=0)
    ajustadas = []
    for p in corners:
        direccion = centro - p
        norma = np.linalg.norm(direccion)
        if norma < 1e-6:
            ajustadas.append(p)
            continue
        ajustadas.append(p + (direccion / norma) * margin_px)
    return np.array(ajustadas, dtype=np.float32)

def detect_field_corners(image, debug_path=None):
    h, w  = image.shape[:2]
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    print(f"  [INFO] Píxeles 'verde cancha' detectados: "
          f"{100 * cv2.countNonZero(green) / (h * w):.1f}% del frame.")
    best_corners = None

    for close_k, erode_k in [(80, 20), (60, 15), (100, 25), (50, 10)]:
        closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
        # FIX: antes esto era solo MORPH_ERODE, que encoge el blob de forma
        # PERMANENTE (nunca se vuelve a crecer) -- el contorno final salía
        # sistemáticamente más chico que el verde real por ~erode_k píxeles
        # en cada borde, comiéndose pasto real en el recorte. MORPH_OPEN
        # (erosiona y luego dilata con el mismo kernel) limpia el ruido
        # pequeño igual, pero el blob principal recupera su tamaño real.
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((erode_k, erode_k), np.uint8))
        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: continue
        field_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(field_cnt) < h * w * 0.25: continue

        hull = cv2.convexHull(field_cnt)
        for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
            epsilon = eps * cv2.arcLength(hull, True)
            approx  = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
            if len(approx) == 4:
                best_corners = order_corners(approx)
                break

        if best_corners is None:
            rect    = cv2.minAreaRect(field_cnt)
            box_pts = cv2.boxPoints(rect)
            best_corners = order_corners(box_pts.astype(np.float32))

        if best_corners is not None:
            break

    if best_corners is not None and CORNER_MARGIN_PX != 0:
        best_corners = apply_margin(best_corners, CORNER_MARGIN_PX)

    if debug_path and best_corners is not None:
        dbg = image.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for j, (px, py) in enumerate(best_corners.astype(int)):
            cv2.circle(dbg, (px, py), 18, (0, 255, 0), -1)
            cv2.putText(dbg, labels[j], (px+8, py+8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        pts_int = best_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(dbg, [pts_int], True, (0, 255, 0), 3)
        green_overlay = image.copy()
        green_overlay[green > 0] = (0, 200, 0)
        dbg = cv2.addWeighted(dbg, 0.7, green_overlay, 0.3, 0)
        cv2.imwrite(debug_path, dbg)

    return best_corners

# ===========================================================================
# FUNCIONES DE LIMPIEZA CLÁSICA (MANTENIDAS POR SI ACASO, PERO DESACTIVADAS)
# ===========================================================================

def clean_non_field(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    dark_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([DARK_THRESH]*3))
    
    r1     = cv2.inRange(hsv, np.array([  0, 60, 60]), np.array([ 15, 255, 255]))
    r2     = cv2.inRange(hsv, np.array([161, 60, 60]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([  8, 80, 80]), np.array([ 22, 255, 255]))
    ball_mask = cv2.bitwise_or(cv2.bitwise_or(r1, r2), orange)

    keep = cv2.bitwise_or(green_mask, dark_mask)
    keep = cv2.bitwise_or(keep, ball_mask)
    keep = cv2.dilate(keep, np.ones((CLEAN_DILATE, CLEAN_DILATE), np.uint8))

    green_pixels = image[green_mask > 0]
    if len(green_pixels) > 0:
        avg_green = np.median(green_pixels, axis=0).astype(np.uint8)
    else:
        avg_green = np.array([60, 140, 60], dtype=np.uint8)

    result = image.copy()
    non_field = (keep == 0)
    result[non_field] = avg_green
    return result

def apply_clahe(image, clip=CLAHE_CLIP, grid=CLAHE_GRID):
    lab    = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l_eq   = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ===========================================================================
# MAIN
# ===========================================================================

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
if not frames:
    raise ValueError(f"No se encontraron frames en {FRAMES_DIR}")

print(f"Frames a preprocesar: {len(frames)}")
print(f"Salida: {OUTPUT_DIR}\n")

# ── Paso 1: Detectar esquinas ──────────────────────────────────────────────
print("=== DETECCIÓN DE ESQUINAS (frame 0) ===")
frame_0 = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))
h0, w0  = frame_0.shape[:2]

if MANUAL_CORNERS is not None:
    corners = MANUAL_CORNERS
    print("  Usando esquinas MANUALES.")
else:
    debug_path = os.path.join(DEBUG_DIR, "corners_debug.jpg")
    corners    = detect_field_corners(frame_0, debug_path=debug_path)
    if corners is None:
        print("  No se detectaron esquinas automáticamente.")
        print("  Opción rápida: en el script, pon")
        print("    MANUAL_CORNERS = MANUAL_CORNERS_FALLBACK")
        print("  o ajusta GREEN_HSV_LOW/HIGH y vuelve a correr.")
        exit(1)
    print(f"  Esquinas detectadas:")
    labels = ["Arriba-Izq", "Arriba-Der", "Abajo-Der", "Abajo-Izq"]
    for lbl, (px, py) in zip(labels, corners.astype(int)):
        print(f"    {lbl}: ({px}, {py})")
    print(f"  → Debug en: {debug_path}")

# ── Paso 2: Homografía ────────────────────────────────────────────────────
dst = np.array([
    [0,         0        ],
    [OUTPUT_W-1,0        ],
    [OUTPUT_W-1,OUTPUT_H-1],
    [0,         OUTPUT_H-1],
], dtype=np.float32)
H, _ = cv2.findHomography(corners, dst)
print(f"\n  Resolución de salida: {OUTPUT_W}x{OUTPUT_H}")

# ── Paso 3: Preview del frame 0 ───────────────────────────────────────────
proc0 = cv2.warpPerspective(frame_0, H, (OUTPUT_W, OUTPUT_H))
# proc0 = clean_non_field(proc0)  <-- DESACTIVADO
# proc0 = apply_clahe(proc0)      <-- DESACTIVADO
preview_path = os.path.join(DEBUG_DIR, "frame_0_processed.jpg")
cv2.imwrite(preview_path, proc0)
print(f"  Preview guardado en: {preview_path}")
print(f"\n  REVISA {preview_path} antes de continuar.")
print("  El campo debe verse rectangular.\n")

resp = input("¿Se ve correcto? (s/n): ").strip().lower() if (sys.stdin.isatty() and os.environ.get("AUTO_CONFIRM") != "1") else "s"
if not (sys.stdin.isatty() and os.environ.get("AUTO_CONFIRM") != "1"):
    print("[INFO] Ejecución no interactiva detectada (AUTO_CONFIRM o sin TTY) -- "
          "se continúa asumiendo que MANUAL_CORNERS ya fue validado.")
if resp != 's':
    print("\nAjusta MANUAL_CORNERS o el rango HSV y vuelve a correr.")
    exit(0)

# ── Paso 4: Procesar todos los frames ─────────────────────────────────────
print(f"\nProcesando {len(frames)} frames...")
for i, fname in enumerate(frames):
    img = cv2.imread(os.path.join(FRAMES_DIR, fname))
    if img is None:
        continue

    # 1. Corrección de perspectiva → solo el campo interior
    proc = cv2.warpPerspective(img, H, (OUTPUT_W, OUTPUT_H))

    # 2. Funciones clásicas DESACTIVADAS para no romper a SAM3 
    # proc = clean_non_field(proc) 
    # proc = apply_clahe(proc)     

    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), proc)

    if i % 200 == 0:
        print(f"  [{i+1:05d}/{len(frames)}] procesados")

print(f"\n¡Listo! Frames preprocesados en: {OUTPUT_DIR}")
print(f"Siguiente paso: python src/02_segment_tracking.py")
