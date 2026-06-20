"""
02_segment_tracking.py
Copa FutBotMX — Capitulo Vision por Computadora (Categoria Amateur)

Segmentacion y tracking de robots y pelota sobre el video ya
preprocesado (vista cenital, sin limpieza de color). La deteccion
inicial de candidatos se hace por color (HSV) y forma; SAM3 se usa
solo para refinar esos candidatos en una mascara mas limpia. Si SAM3
no da un resultado aceptable, se cae a un circulo de respaldo
calculado de forma clasica.

Nota sobre el uso de SAM3: las cajas que se le mandan al modelo vienen
de la deteccion HSV, no de un prompt de texto ni geometria fija. El
circulo de respaldo es 100% clasico, sin SAM3 de por medio.

Como leer el log:
  Cada 10 frames se imprime algo asi:
    [0011/19401] SAM3(3) tracked:3[SAM3-chassis] | pelota:(x,y)
  En los primeros 3 frames se imprime detalle por candidato.
  Si aparece "reset_all_prompts() no disponible", el script ya cambio
  automaticamente al modo de re-codificar por candidato (mas lento,
  pero sigue funcionando).
"""

import os

# Reduce fragmentacion de memoria en GPUs chicas (recomendado por PyTorch
# cuando aparece CUDA out of memory). Debe ir antes de importar torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
import cv2
import traceback
from contextlib import nullcontext
from PIL import Image

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------------------------------------------------------------------
# Directorios
# ---------------------------------------------------------------------------
FRAMES_DIR = "data/frames_processed"
MASKS_DIR  = "results/masks"
os.makedirs(MASKS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Filtros estructurales
# ---------------------------------------------------------------------------
# Margen para descartar blobs que tocan el borde del frame (manos/brazos
# entrando desde fuera de camara para acomodar robots).
BORDER_MARGIN        = 70
MIN_CANDIDATE_SEP    = 120

# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------
SAM3_INTERVAL   = 1
SAM3_MIN_SCORE  = 0.35
SAM3_MIN_AREA   = 0.0015
SAM3_MAX_AREA   = 0.08
SAM3_MAX_ASPECT = 2.5
# Solidez minima de la mascara cruda de SAM3. Mascaras con forma de
# medialuna (agarraron solo una parte saliente del chasis) tienen
# solidez baja y se descartan; van a circulo de respaldo en su lugar.
SOLIDITY_MIN    = 0.65

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
MAX_ROBOTS      = 4
MIN_CENTER_DIST = 40
MAX_LOST        = 20
MAX_MATCH_DIST  = 60

# ---------------------------------------------------------------------------
# Pelota
# ---------------------------------------------------------------------------
BALL_BORDER_MARGIN = 40

# ---------------------------------------------------------------------------
# Dispositivo
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    autocast_ctx = torch.autocast("mps", dtype=torch.float16)
else:
    device = torch.device("cpu")
    autocast_ctx = nullcontext()
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# SAM3 - inicializacion
# ---------------------------------------------------------------------------
BPE_PATH = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
LOCAL_WEIGHTS_PATH = r"C:\Users\Sebastian\Documents\Copa-FutBotMX---AZTEM\sam3.pt"

print("Construyendo el esqueleto completo de SAM3 (Video)...")
model = build_sam3_video_model(
    bpe_path=BPE_PATH,
    checkpoint_path=None,
)

print(f"Inyectando tensores originales de Meta desde: {LOCAL_WEIGHTS_PATH}")
if not os.path.exists(LOCAL_WEIGHTS_PATH):
    raise RuntimeError(f"¡Fallo critico! Mueve el archivo sam3.pt a esta carpeta. No se encontro en: {LOCAL_WEIGHTS_PATH}")

checkpoint = torch.load(LOCAL_WEIGHTS_PATH, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)

model.load_state_dict(state_dict)
model.eval()

model = model.to(device)
print("Modelo de video sincronizado y listo.\n")

# El modelo de video es un wrapper detector+tracker; el submodulo de
# imagen real (el que tiene .backbone, que es lo que necesita
# Sam3Processor para set_image/add_geometric_prompt) vive dentro de
# model.detector en este checkpoint.
if hasattr(model, "detector") and hasattr(model.detector, "backbone"):
    IMAGE_MODEL = model.detector
    print("[OK] Usando 'model.detector' como submodulo de imagen para Sam3Processor.\n")
elif hasattr(model, "backbone"):
    IMAGE_MODEL = model
    print("[OK] El modelo expone '.backbone' directamente, se usa tal cual.\n")
else:
    hijos = [n for n, _ in model.named_children()]
    raise RuntimeError(
        "No se encontro un submodulo con atributo '.backbone' ni en 'model' "
        f"ni en 'model.detector'. Atributos de primer nivel disponibles: {hijos}. "
        "Revisa cual de estos contiene el backbone (ej. model.tracker.image_model)."
    )

# ===========================================================================
# UTILIDADES
# ===========================================================================

def extract_masks_scores(output):
    if output is None: return [], []
    _get = (lambda k: output.get(k)) if isinstance(output, dict) else (lambda k: getattr(output, k, None))
    raw, sc = _get("masks"), _get("scores")
    if raw is None: return [], []

    if isinstance(raw, torch.Tensor): raw = raw.float().cpu().numpy()
    elif not isinstance(raw, np.ndarray):
        try: raw = np.array(raw)
        except Exception: return [], []

    if raw.ndim == 0 or raw.size == 0: return [], []
    if raw.ndim == 4: raw = raw[:, 0, :, :]
    if raw.ndim == 2: raw = raw[np.newaxis]
    if isinstance(sc, torch.Tensor): sc = sc.float().cpu().numpy()

    masks, scores = [], []
    for i, m in enumerate(raw):
        bm = (m > 0.5).astype(bool) if m.dtype in (np.float32, np.float64) else m.astype(bool)
        s  = float(sc[i]) if (sc is not None and i < len(sc)) else 1.0
        masks.append(bm)
        scores.append(s)
    return masks, scores

def mask_solidity(m):
    contours, _ = cv2.findContours(m.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 1.0
    cnt       = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return float(area) / float(hull_area) if hull_area > 0 else 1.0

def masks_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def deduplicate(detections, min_dist=MIN_CENTER_DIST, iou_thresh=0.10):
    result = []
    for d in sorted(detections, key=lambda x: -x.get('score', 0)):
        dup = any(
            ((d['cx']-r['cx'])**2 + (d['cy']-r['cy'])**2)**0.5 < min_dist
            or masks_iou(d['mask'], r['mask']) > iou_thresh
            for r in result
        )
        if not dup: result.append(d)
        if len(result) >= MAX_ROBOTS: break
    return result

def _add_prompt(proc, state, box_norm, label, autocast_ctx):
    try:
        with torch.no_grad(), autocast_ctx:
            new_state = proc.add_geometric_prompt(box_norm, label, state=state)
        if new_state is None:
            new_state = state
        return new_state, True
    except Exception as e:
        print(f"\n[!!! SAM3 ERROR EN _add_prompt !!!]")
        traceback.print_exc()
        print(f"-----------------------------------------\n")
        return state, False


# ===========================================================================
# HSV — LOCALIZACION DE CANDIDATOS
# ===========================================================================

def hsv_find_candidates(image, img_h, img_w):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)

    # El chasis se distingue del pasto por SATURACION baja, no por ser
    # oscuro ni por ser verde (el borde verde del PCB cae casi en el
    # mismo rango de color que el pasto). S<150 y V<180 separa bien los
    # robots del resto del campo.
    chassis = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 150, 180]))
    chassis = cv2.morphologyEx(chassis, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    chassis = cv2.morphologyEx(chassis, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    raw = []
    for mask_bin, tipo in [(chassis, 'chassis')]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)
        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]

            # Rango de area pensado para ignorar sombras grandes y
            # reflejos pequeños.
            if not (400 < area < 20000): continue

            bw_c = stats[lbl, cv2.CC_STAT_WIDTH]
            bh_c = stats[lbl, cv2.CC_STAT_HEIGHT]
            if max(bw_c, bh_c) / max(min(bw_c, bh_c), 1) > 3.0: continue

            blob_mask = (labels == lbl).astype(np.uint8)

            # Filtro de circularidad: los robots son redondos, brazos y
            # sombras suelen salir alargados o irregulares.
            cnts_blob, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_blob: continue
            c_blob = max(cnts_blob, key=cv2.contourArea)
            contour_area = cv2.contourArea(c_blob)
            perim = cv2.arcLength(c_blob, True)
            circularity = (4 * np.pi * contour_area / (perim ** 2)) if perim > 0 else 0
            if circularity < 0.20: continue

            blob_edges = cv2.bitwise_and(edges, edges, mask=blob_mask)
            edge_density = cv2.countNonZero(blob_edges) / float(area)
            if edge_density < 0.015: continue

            # Centroide geometrico del blob completo (no ponderado por
            # densidad de bordes, que se sesga hacia cables/antenas).
            cx = int(centroids[lbl, 0])
            cy = int(centroids[lbl, 1])

            if (cx < BORDER_MARGIN or cx > img_w - BORDER_MARGIN or
                    cy < BORDER_MARGIN or cy > img_h - BORDER_MARGIN): continue

            raw.append({'cx': cx, 'cy': cy, 'area': area, 'tipo': tipo, 'blob_mask': blob_mask.astype(bool), 'score': edge_density})

    raw.sort(key=lambda x: -x['score'])
    filtered = []
    for c in raw:
        if not any(((c['cx']-f['cx'])**2 + (c['cy']-f['cy'])**2)**0.5 < MIN_CANDIDATE_SEP for f in filtered):
            filtered.append(c)
        if len(filtered) >= MAX_ROBOTS * 2: break
    return filtered


# ===========================================================================
# SAM3 — REFINAMIENTO
# ===========================================================================

_SUPPORTS_RESET_PROMPTS = None  # None=sin probar, True/False una vez determinado

def sam3_refine_candidates(image_pil, img_h, img_w, candidates, autocast_ctx, verbose=False):
    global _SUPPORTS_RESET_PROMPTS

    if not candidates: return []

    fa = img_h * img_w
    all_detections = []

    if verbose:
        print(f"  [DEBUG] {len(candidates)} candidatos HSV de entrada: " +
              ", ".join(f"({c['cx']},{c['cy']})" for c in candidates))

    def _circle_fallback(blob_mask, cx, cy, prompt_suffix):
        cnts_fall, _ = cv2.findContours(blob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_fall:
            c_max = max(cnts_fall, key=cv2.contourArea)
            (x_c, y_c), radius = cv2.minEnclosingCircle(c_max)
            radius = min(max(int(radius), 35), 55)
            solid_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.circle(solid_mask, (int(x_c), int(y_c)), radius, 255, -1)
            final_mask = solid_mask.astype(bool)
        else:
            final_mask = blob_mask.astype(bool)
        return {'cx': cx, 'cy': cy, 'mask': final_mask, 'score': 0.05, 'prompt': prompt_suffix}

    def _process_one_candidate(cand, proc, base_state, reencode_per_candidate):
        cx, cy    = cand['cx'], cand['cy']
        blob_mask = cand['blob_mask']

        # Caja de 150x150 px alrededor del candidato (el robot mide
        # ~140-150 px de diametro). add_geometric_prompt espera
        # [center_x, center_y, width, height] normalizado en [0,1].
        R_HALF = 75
        x1_clean = max(0, cx - R_HALF)
        y1_clean = max(0, cy - R_HALF)
        x2_clean = min(img_w, cx + R_HALF)
        y2_clean = min(img_h, cy + R_HALF)
        box_w  = x2_clean - x1_clean
        box_h  = y2_clean - y1_clean
        box_cx = x1_clean + box_w / 2.0
        box_cy = y1_clean + box_h / 2.0
        box_norm = [
            float(box_cx) / img_w, float(box_cy) / img_h,
            float(box_w)  / img_w, float(box_h)  / img_h,
        ]

        # Reinicia los prompts sobre la misma imagen ya codificada, para
        # que cada candidato se evalue de forma independiente.
        state_for_prompt = base_state
        if not reencode_per_candidate:
            try:
                with torch.no_grad(), autocast_ctx:
                    state_for_prompt = proc.reset_all_prompts(base_state)
                if state_for_prompt is None:
                    state_for_prompt = base_state
            except Exception as e:
                return None, f"reset_all_prompts fallo: {e}"

        try:
            state, ok = _add_prompt(proc, state_for_prompt, box_norm, True, autocast_ctx)
            if not ok: raise RuntimeError("add_geometric_prompt fallo internamente")
            if isinstance(state, (torch.Tensor, np.ndarray)):
                m_arr = state.float().cpu().numpy() if isinstance(state, torch.Tensor) else state
                masks = [(m_arr > 0.5).astype(bool) if m_arr.dtype in (np.float32, np.float64) else m_arr.astype(bool)]
                scores = [1.0]
            else:
                masks, scores = extract_masks_scores(state)
        except Exception as e:
            print(f"\n[!!! ERROR AL EXTRAER MASCARAS !!!]")
            traceback.print_exc()
            print(f"---------------------------------\n")
            masks, scores = [], []

        best = None
        for m, sc in zip(masks, scores):
            if sc < SAM3_MIN_SCORE: continue
            area = int(m.sum())
            if not (fa * SAM3_MIN_AREA < area < fa * SAM3_MAX_AREA): continue
            rows, cols = np.any(m, axis=1), np.any(m, axis=0)
            if not rows.any() or not cols.any(): continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            if max(rmax-rmin+1, cmax-cmin+1) / max(min(rmax-rmin+1, cmax-cmin+1), 1) > SAM3_MAX_ASPECT: continue
            if mask_solidity(m) < SOLIDITY_MIN: continue
            if best is None or sc > best[1]: best = (m, sc)

        if best is None:
            return _circle_fallback(blob_mask, cx, cy, f"Circ-{cand['tipo']}"), None

        m_best, sc_best = best
        m_uint = m_best.astype(np.uint8) * 255
        m_solid = cv2.morphologyEx(m_uint, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        k = max(5, int(np.sqrt(m_solid.sum() / 255.0 / np.pi) * 2) // 6)
        m_clean = cv2.morphologyEx(m_solid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
        if m_clean.sum() < int(fa * SAM3_MIN_AREA): m_clean = m_best.astype(bool)

        # Si el OPEN partio un cuello delgado de la mascara cruda, se
        # quedan dos o mas piezas sueltas. Nos quedamos solo con el
        # componente conectado de mayor area.
        n_cc_clean, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(m_clean.astype(np.uint8))
        if n_cc_clean > 2:
            biggest_lbl = 1 + int(np.argmax(stats_cc[1:, cv2.CC_STAT_AREA]))
            m_clean = (labels_cc == biggest_lbl)

        m_clean[:, :15], m_clean[:, -15:] = False, False
        m_clean[:15, :], m_clean[-15:] = False, False

        rows_c, cols_c = np.any(m_clean, axis=1), np.any(m_clean, axis=0)
        if not (rows_c.any() and cols_c.any()):
            return _circle_fallback(blob_mask, cx, cy, f"Circ-{cand['tipo']}"), None

        final_cx = int(np.where(cols_c)[0].mean())
        final_cy = int(np.where(rows_c)[0].mean())

        # Si la mascara de SAM3 queda muy lejos del candidato HSV
        # original, probablemente agarro algo conectado (la sombra del
        # robot, por ejemplo) en vez del robot. Se descarta y se cae a
        # circulo de respaldo.
        drift = ((final_cx - cx) ** 2 + (final_cy - cy) ** 2) ** 0.5
        MAX_DRIFT = R_HALF * 0.6
        if drift > MAX_DRIFT:
            if verbose:
                print(f"  [DERIVA] candidato HSV=({cx},{cy}) -> SAM3 dio ({final_cx},{final_cy}), "
                      f"deriva={drift:.0f}px > {MAX_DRIFT:.0f}px. Se descarta el mask, va a circulo.")
            return _circle_fallback(blob_mask, cx, cy, f"Circ-{cand['tipo']}(deriva)"), None

        return {'cx': final_cx, 'cy': final_cy, 'mask': m_clean, 'score': sc_best,
                'prompt': f"SAM3-{cand['tipo']}"}, None

    # Se codifica la imagen una sola vez por frame con set_image(), y
    # reset_all_prompts() se usa entre candidatos para evaluar cada caja
    # de forma independiente sobre la misma imagen codificada. Si la
    # libreria instalada no tiene reset_all_prompts(), se detecta una
    # vez y se cae al metodo de re-codificar por candidato (mas lento,
    # pero funcional).
    proc = Sam3Processor(IMAGE_MODEL, confidence_threshold=0.0)
    try:
        with torch.no_grad(), autocast_ctx:
            base_state = proc.set_image(image_pil)
        frame_encoded_ok = True
    except Exception as e:
        print(f"  [SAM3 set_image error -- todo el frame cae a circulo]: {e}")
        frame_encoded_ok = False
        base_state = None

    if not frame_encoded_ok:
        for cand in candidates:
            all_detections.append(_circle_fallback(cand['blob_mask'], cand['cx'], cand['cy'], f"Circ-{cand['tipo']}(oom)"))
        del proc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        final = deduplicate(all_detections)
        if verbose:
            print(f"  [DEBUG] {len(all_detections)} detecciones antes de dedup: " +
                  ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})sc={d['score']:.2f}" for d in all_detections))
            print(f"  [DEBUG] {len(final)} detecciones DESPUES de dedup: " +
                  ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})" for d in final))
        return final

    for idx, cand in enumerate(candidates):
        # Se revisa la bandera en cada candidato porque puede haber
        # cambiado en una iteracion anterior de este mismo frame.
        reencode_now = (_SUPPORTS_RESET_PROMPTS is False) and idx > 0

        if reencode_now:
            try:
                with torch.no_grad(), autocast_ctx:
                    this_base_state = proc.set_image(image_pil)
            except Exception as e:
                print(f"  [SAM3 set_image error candidato {idx}]: {e}")
                all_detections.append(_circle_fallback(cand['blob_mask'], cand['cx'], cand['cy'], f"Circ-{cand['tipo']}(oom)"))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
        else:
            this_base_state = base_state

        det, err = _process_one_candidate(cand, proc, this_base_state, reencode_now)

        if err is not None and _SUPPORTS_RESET_PROMPTS is None:
            # Primer candidato del script: reset_all_prompts no existe o
            # fallo. Se marca para el resto del video y se reintenta
            # este candidato re-codificando.
            _SUPPORTS_RESET_PROMPTS = False
            print(f"  [INFO] reset_all_prompts() no disponible ({err}). "
                  f"Usando re-codificacion por candidato el resto del video (mas lento, pero funcional).")
            try:
                with torch.no_grad(), autocast_ctx:
                    this_base_state = proc.set_image(image_pil)
                det, err = _process_one_candidate(cand, proc, this_base_state, True)
            except Exception as e:
                det = _circle_fallback(cand['blob_mask'], cand['cx'], cand['cy'], f"Circ-{cand['tipo']}(oom)")
                err = None
        elif err is None and _SUPPORTS_RESET_PROMPTS is None:
            _SUPPORTS_RESET_PROMPTS = True

        if det is not None:
            all_detections.append(det)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    del proc, base_state

    final = deduplicate(all_detections)
    if verbose:
        print(f"  [DEBUG] {len(all_detections)} detecciones antes de dedup: " +
              ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})sc={d['score']:.2f}" for d in all_detections))
        print(f"  [DEBUG] {len(final)} detecciones DESPUES de dedup: " +
              ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})" for d in final))
    return final


# ===========================================================================
# PELOTA
# ===========================================================================

def detect_ball(image, border_margin=BALL_BORDER_MARGIN):
    h, w  = image.shape[:2]
    fa    = h * w
    hsv   = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    r1     = cv2.inRange(hsv, np.array([  0, 120,  60]), np.array([ 14, 255, 255]))
    r2     = cv2.inRange(hsv, np.array([161, 120,  60]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([  8, 120,  80]), np.array([ 22, 255, 255]))

    ball_c = cv2.bitwise_or(cv2.bitwise_or(r1, r2), orange)
    ball_c = cv2.morphologyEx(ball_c, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ball_c = cv2.dilate(ball_c, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(ball_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if not (fa * 0.0001 < a < fa * 0.02): continue
        p = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (p**2) if p > 0 else 0

        if circ < 0.75: continue

        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        if (cx < border_margin or cx > w - border_margin or cy < border_margin or cy > h - border_margin): continue

        if best is None or a > best[3]: best = (cx, cy, c, a)

    if best is None: return None, None, None

    cx, cy, contour, _ = best
    msk = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(msk, [contour], -1, 255, -1)
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8))
    return cx, cy, msk.astype(bool)


# ===========================================================================
# TRACKER
# ===========================================================================

class RobotTracker:
    def __init__(self):
        self.tracked = {}
        self.next_id = 2

    def update(self, detections):
        if not self.tracked:
            for det in detections[:MAX_ROBOTS]: self._register(det)
            return {tid: t['mask'] for tid, t in self.tracked.items() if t['lost'] == 0}

        det_assigned = [False] * len(detections)
        matches = {}
        for tid in list(self.tracked.keys()):
            tcx, tcy = self.tracked[tid]['cx'], self.tracked[tid]['cy']
            best_dist, best_idx = float('inf'), -1
            for j, det in enumerate(detections):
                dist = ((det['cx']-tcx)**2 + (det['cy']-tcy)**2)**0.5
                if dist < best_dist:
                    best_dist, best_idx = dist, j
            if best_idx >= 0 and best_dist < MAX_MATCH_DIST:
                matches[tid] = (best_idx, best_dist)

        det_winner = {}
        for tid, (idx, dist) in matches.items():
            if idx not in det_winner or dist < det_winner[idx][1]:
                det_winner[idx] = (tid, dist)

        assigned_tids = set()
        for idx, (tid, _) in det_winner.items():
            det = detections[idx]
            vx, vy = det['cx'] - self.tracked[tid]['cx'], det['cy'] - self.tracked[tid]['cy']
            self.tracked[tid].update({
                'cx': det['cx'], 'cy': det['cy'], 'mask': det['mask'], 'lost': 0,
                'prompt': det.get('prompt', '?'), 'vx': vx, 'vy': vy,
            })
            det_assigned[idx] = True
            assigned_tids.add(tid)

        for tid in list(self.tracked.keys()):
            if tid not in assigned_tids: self.tracked[tid]['lost'] += 1
            if self.tracked[tid]['lost'] >= MAX_LOST: del self.tracked[tid]

        for j, det in enumerate(detections):
            if not det_assigned[j]:
                cx, cy = det['cx'], det['cy']
                activos = len([t for t in self.tracked.values() if t['lost']==0])
                if activos < MAX_ROBOTS and not any(
                    abs(cx-t['cx']) < MIN_CENTER_DIST and abs(cy-t['cy']) < MIN_CENTER_DIST
                    for t in self.tracked.values()
                ):
                    self._register(det)

        return {tid: t['mask'] for tid, t in self.tracked.items() if t['lost'] == 0}

    def extrapolate(self, img_h, img_w):
        result = {}
        for tid, t in self.tracked.items():
            if t['lost'] != 0: continue
            dx, dy = int(round(t.get('vx', 0) / SAM3_INTERVAL)), int(round(t.get('vy', 0) / SAM3_INTERVAL))
            if dx == 0 and dy == 0:
                result[tid] = t['mask']
                continue
            shifted = np.roll(np.roll(t['mask'], dy, axis=0), dx, axis=1)
            t['cx'], t['cy'] = int(np.clip(t['cx'] + dx, 0, img_w - 1)), int(np.clip(t['cy'] + dy, 0, img_h - 1))
            t['mask'] = shifted
            result[tid] = shifted
        return result

    def _register(self, det):
        if sum(1 for t in self.tracked.values() if t['lost']==0) >= MAX_ROBOTS: return
        self.tracked[self.next_id] = {
            'cx': det['cx'], 'cy': det['cy'], 'mask': det['mask'], 'lost': 0,
            'prompt': det.get('prompt', '?'), 'vx': 0, 'vy': 0,
        }
        self.next_id += 1

    @property
    def n_active(self):
        return sum(1 for t in self.tracked.values() if t['lost']==0)

    @property
    def prompts_active(self):
        return [t['prompt'] for t in self.tracked.values() if t['lost']==0]


# ===========================================================================
# LOOP PRINCIPAL
# ===========================================================================

# Mismo criterio (sorted + filtro .jpg) que 00_preprocess.py, ya que ese
# script reescribe los frames procesados con el mismo nombre que tenian
# originalmente, para conservar el orden temporal.
frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))

if not frames:
    raise RuntimeError(
        f"No se encontraron frames en '{FRAMES_DIR}'. "
        f"¿Ya corriste 'python src/00_preprocess.py'?"
    )
print(f"Frames encontrados: {len(frames)} en '{FRAMES_DIR}'\n")

tracker   = RobotTracker()

for i, fname in enumerate(frames):
    image = cv2.imread(os.path.join(FRAMES_DIR, fname))
    if image is None: continue

    h, w = image.shape[:2]

    if i % SAM3_INTERVAL == 0:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        candidates  = hsv_find_candidates(image, h, w)
        detections  = sam3_refine_candidates(image_pil, h, w, candidates, autocast_ctx, verbose=(i < 3))
        det_src     = f"SAM3({len(detections)})"
        robot_masks = tracker.update(detections)
    else:
        robot_masks = tracker.extrapolate(h, w)
        det_src     = "extrap"

    # Limpieza periodica de cache CUDA para ayudar a desfragmentar
    # memoria cada ciertos frames, sin esperar a que el allocator
    # decida por su cuenta.
    if device.type == "cuda" and i % 50 == 0:
        torch.cuda.empty_cache()

    ball_cx, ball_cy, ball_mask = detect_ball(image)

    combined = np.zeros((h, w), dtype=np.uint8)
    overlay  = image.copy()
    for rm in robot_masks.values():
        combined[rm] = 200
        overlay[rm]  = (255, 80, 0)
    if ball_mask is not None:
        combined[ball_mask] = 255
        overlay[ball_mask]  = (0, 0, 255)

    stem = fname.replace(".jpg", "")
    cv2.imwrite(os.path.join(MASKS_DIR, f"{stem}_mask.png"), combined)
    cv2.imwrite(os.path.join(MASKS_DIR, f"{stem}_overlay.jpg"), cv2.addWeighted(image, 0.4, overlay, 0.6, 0))

    if i % 10 == 0:
        ball_str = f"({ball_cx},{ball_cy})" if ball_cx is not None else "no"
        prompts  = "/".join(set(tracker.prompts_active)) or "-"
        print(f"  [{i+1:04d}/{len(frames)}] {det_src} tracked:{tracker.n_active}[{prompts}] | pelota:{ball_str}")

print(f"\nListo. Mascaras en '{MASKS_DIR}'")
