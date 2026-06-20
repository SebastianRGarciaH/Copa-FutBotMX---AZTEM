import cv2
import os

VIDEO_PATH = "data/videos/video2.mp4"
OUTPUT_DIR = "data/frames"
FRAME_INTERVAL = 1  # 1 = guarda todos los frames, 2 = uno de cada dos, etc.

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames a {fps:.1f} fps")

saved = 0
frame_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_n % FRAME_INTERVAL == 0:
        path = os.path.join(OUTPUT_DIR, f"frame_{frame_n:05d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
    frame_n += 1

cap.release()
print(f"Listo. {saved} frames guardados en '{OUTPUT_DIR}'")
