import os
import time
import torch
import numpy as np
import cv2
import traceback
from contextlib import nullcontext
from PIL import Image

from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_image_processor import Sam3Processor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FRAMES_DIR = "data/frames_processed"
MASKS_DIR  = "results/masks"
os.makedirs(MASKS_DIR, exist_ok=True)

BORDER_MARGIN        = 70     
MIN_CANDIDATE_SEP    = 90

SAM3_INTERVAL   = 8       
SAM3_MIN_SCORE  = 0.35     
SAM3_MIN_AREA   = 0.0015   
SAM3_MAX_AREA   = 0.08
SAM3_MAX_ASPECT = 2.5
SOLIDITY_MIN    = 0.65

MAX_ROBOTS      = 4
MIN_CENTER_DIST = 40       
MAX_LOST        = 20
MAX_MATCH_DIST  = 60       

BALL_BORDER_MARGIN = 40

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major >= 8:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
        print(f"[INFO] GPU con compute capability {cc_major}.x (<8, pre-Ampere): usando float16 en vez de bfloat16 para evitar cómputo lento.")
    autocast_ctx = torch.autocast("cuda", dtype=autocast_dtype)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    autocast_ctx = torch.autocast("mps", dtype=torch.float16)
else:
    device = torch.device("cpu")
    autocast_ctx = nullcontext()
print(f"Dispositivo: {device}")

BPE_PATH = "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
LOCAL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "sam3.pt")

print("Construyendo el esqueleto completo de SAM3 (Video)...")
model = build_sam3_video_model(
    bpe_path=BPE_PATH,
    checkpoint_path=None, 
)

print(f"Inyectando tensores originales de Meta desde: {LOCAL_WEIGHTS_PATH}")
if not os.path.exists(LOCAL_WEIGHTS_PATH):
    raise RuntimeError(f"¡Fallo crítico! Mueve el archivo sam3.pt a esta carpeta. No se encontró en: {LOCAL_WEIGHTS_PATH}")

checkpoint = torch.load(LOCAL_WEIGHTS_PATH, map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict)
model.eval()

model = model.to(device)
print("Modelo de video sincronizado y listo.\n")

if hasattr(model, "detector") and hasattr(model.detector, "backbone"):
    IMAGE_MODEL = model.detector
    print("[OK] Usando 'model.detector' como submódulo de imagen para Sam3Processor.\n")
elif hasattr(model, "backbone"):
    IMAGE_MODEL = model
    print("[OK] El modelo expone '.backbone' directamente, se usa tal cual.\n")
else:
    hijos = [n for n, _ in model.named_children()]
    raise RuntimeError(
        "No se encontró un submódulo con atributo '.backbone' ni en 'model' "
        f"ni en 'model.detector'. Atributos de primer nivel disponibles: {hijos}. "
        "Revisa cuál de estos contiene el backbone."
    )

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
        print(f"\n[!!! SAM3 FATAL ERROR EN _add_prompt !!!]")
        traceback.print_exc()
        print(f"-----------------------------------------\n")
        return state, False

def hsv_find_candidates(image, img_h, img_w, field_mask=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)

    chassis = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 150, 180]))
    chassis = cv2.morphologyEx(chassis, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    chassis = cv2.morphologyEx(chassis, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    raw = []
    for mask_bin, tipo in [(chassis, 'chassis')]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)
        for lbl in range(1, n):
            area = stats[lbl, cv2.CC_STAT_AREA]
            
            if not (2500 < area < 20000): continue 
            
            bw_c = stats[lbl, cv2.CC_STAT_WIDTH]
            bh_c = stats[lbl, cv2.CC_STAT_HEIGHT]
            if max(bw_c, bh_c) / max(min(bw_c, bh_c), 1) > 2.0: continue 
            
            blob_mask = (labels == lbl).astype(np.uint8)

            cnts_blob, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_blob: continue
            c_blob = max(cnts_blob, key=cv2.contourArea)
            contour_area = cv2.contourArea(c_blob)
            perim = cv2.arcLength(c_blob, True)
            circularity = (4 * np.pi * contour_area / (perim ** 2)) if perim > 0 else 0
            if circularity < 0.20: continue

            blob_edges = cv2.bitwise_and(edges, edges, mask=blob_mask)
            edge_density = cv2.countNonZero(blob_edges) / float(area)
            if edge_density < 0.15: continue  

            cx = int(centroids[lbl, 0])
            cy = int(centroids[lbl, 1])

            if (cx < BORDER_MARGIN or cx > img_w - BORDER_MARGIN or
                    cy < BORDER_MARGIN or cy > img_h - BORDER_MARGIN): continue

            if field_mask is not None and not field_mask[cy, cx]:
                continue

            raw.append({'cx': cx, 'cy': cy, 'area': area, 'tipo': tipo, 'blob_mask': blob_mask.astype(bool), 'score': edge_density})

    raw.sort(key=lambda x: -x['score'])
    filtered = []
    for c in raw:
        if not any(((c['cx']-f['cx'])**2 + (c['cy']-f['cy'])**2)**0.5 < MIN_CANDIDATE_SEP for f in filtered):
            filtered.append(c)
        if len(filtered) >= MAX_ROBOTS * 2: break
    return filtered

FIELD_HSV_LOW  = np.array([70, 100,  60])
FIELD_HSV_HIGH = np.array([95, 255, 255])
FIELD_MIN_FRACTION = 0.15
FIELD_MASK_VALUE = 50

def detect_field_mask(image, img_h, img_w):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, FIELD_HSV_LOW, FIELD_HSV_HIGH)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c_max = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c_max) < FIELD_MIN_FRACTION * img_h * img_w:
        return None
    field_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(field_mask, [c_max], -1, 255, -1)
    return field_mask.astype(bool)

def sam3_refine_candidates(image_pil, img_h, img_w, candidates, autocast_ctx, verbose=False):
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

    def _process_one_candidate(cand, proc, base_state):
        cx, cy    = cand['cx'], cand['cy']
        blob_mask = cand['blob_mask']

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

        try:
            state, ok = _add_prompt(proc, base_state, box_norm, True, autocast_ctx)
            if not ok: raise RuntimeError("add_geometric_prompt falló internamente")
            if isinstance(state, (torch.Tensor, np.ndarray)):
                m_arr = state.float().cpu().numpy() if isinstance(state, torch.Tensor) else state
                masks = [(m_arr > 0.5).astype(bool) if m_arr.dtype in (np.float32, np.float64) else m_arr.astype(bool)]
                scores = [1.0]
            else:
                masks, scores = extract_masks_scores(state)
        except Exception as e:
            print(f"\n[!!! EXTRACCIÓN FATAL ERROR !!!]")
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

        drift = ((final_cx - cx) ** 2 + (final_cy - cy) ** 2) ** 0.5
        MAX_DRIFT = R_HALF * 0.6
        if drift > MAX_DRIFT:
            if verbose:
                print(f"  [DERIVA] candidato HSV=({cx},{cy}) -> SAM3 dio ({final_cx},{final_cy}), "
                      f"deriva={drift:.0f}px > {MAX_DRIFT:.0f}px. Se descarta el mask, va a círculo.")
            return _circle_fallback(blob_mask, cx, cy, f"Circ-{cand['tipo']}(deriva)"), None

        return {'cx': final_cx, 'cy': final_cy, 'mask': m_clean, 'score': sc_best,
                'prompt': f"SAM3-{cand['tipo']}"}, None

    for idx, cand in enumerate(candidates):
        try:
            proc = Sam3Processor(IMAGE_MODEL, confidence_threshold=0.0)
            with torch.no_grad(), autocast_ctx:
                base_state = proc.set_image(image_pil)
        except Exception as e:
            print(f"  [SAM3 set_image error candidato {idx}]: {e}")
            all_detections.append(_circle_fallback(cand['blob_mask'], cand['cx'], cand['cy'], f"Circ-{cand['tipo']}(oom)"))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        det, _ = _process_one_candidate(cand, proc, base_state)
        if det is not None:
            all_detections.append(det)

        del proc, base_state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final = deduplicate(all_detections)
    if verbose:
        print(f"  [DEBUG] {len(all_detections)} detecciones antes de dedup: " +
              ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})sc={d['score']:.2f}" for d in all_detections))
        print(f"  [DEBUG] {len(final)} detecciones DESPUÉS de dedup: " +
              ", ".join(f"{d['prompt']}@({d['cx']},{d['cy']})" for d in final))
    return final

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


frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
if not frames:
    raise RuntimeError(
        f"No se encontraron frames en '{FRAMES_DIR}'. "
        f"¿Ya corriste 'python src/00_preprocess.py'?"
    )
print(f"Frames encontrados: {len(frames)} en '{FRAMES_DIR}'\n")

tracker   = RobotTracker()

for i, fname in enumerate(frames):
    stem = fname.replace(".jpg", "")

    if os.path.exists(os.path.join(MASKS_DIR, f"{stem}_mask.png")):
        continue

    t_frame_start = time.time()

    image = cv2.imread(os.path.join(FRAMES_DIR, fname))
    if image is None: continue
    t_read = time.time()

    h, w = image.shape[:2]

    field_mask = detect_field_mask(image, h, w)

    if i % SAM3_INTERVAL == 0:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        t_pil = time.time()
        candidates  = hsv_find_candidates(image, h, w, field_mask=field_mask)
        t_hsv = time.time()
        detections  = sam3_refine_candidates(image_pil, h, w, candidates, autocast_ctx, verbose=(i < 3))
        t_sam3 = time.time()
        det_src     = f"SAM3({len(detections)})"
        robot_masks = tracker.update(detections)
        t_track = time.time()
    else:
        robot_masks = tracker.extrapolate(h, w)
        det_src     = "extrap"
        t_pil = t_hsv = t_sam3 = t_track = time.time()

    if device.type == "cuda" and i % 50 == 0:
        torch.cuda.empty_cache()
    t_cache = time.time()

    ball_cx, ball_cy, ball_mask = detect_ball(image)
    t_ball = time.time()

    combined = np.zeros((h, w), dtype=np.uint8)
    overlay  = image.copy()
    if field_mask is not None:
        combined[field_mask] = FIELD_MASK_VALUE
    for rm in robot_masks.values():
        combined[rm] = 200
        overlay[rm]  = (255, 80, 0)
    if ball_mask is not None:
        combined[ball_mask] = 255
        overlay[ball_mask]  = (0, 0, 255)

    cv2.imwrite(os.path.join(MASKS_DIR, f"{stem}_mask.png"), combined)
    cv2.imwrite(os.path.join(MASKS_DIR, f"{stem}_overlay.jpg"), cv2.addWeighted(image, 0.4, overlay, 0.6, 0))
    t_disk = time.time()

    if i % 10 == 0:
        print(f"  [TIEMPOS frame {i}] lectura={t_read-t_frame_start:.2f}s "
              f"hsv={t_hsv-t_pil:.2f}s sam3={t_sam3-t_hsv:.2f}s "
              f"track={t_track-t_sam3:.2f}s cache={t_cache-t_track:.2f}s "
              f"pelota={t_ball-t_cache:.2f}s disco={t_disk-t_ball:.2f}s "
              f"TOTAL={t_disk-t_frame_start:.2f}s")
        ball_str = f"({ball_cx},{ball_cy})" if ball_cx is not None else "no"
        prompts  = "/".join(set(tracker.prompts_active)) or "-"
        print(f"  [{i+1:04d}/{len(frames)}] {det_src} tracked:{tracker.n_active}[{prompts}] | pelota:{ball_str}")

print(f"\nListo. Máscaras en '{MASKS_DIR}'")
