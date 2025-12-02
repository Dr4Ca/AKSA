import cv2
import time
from ultralytics import YOLO
import torch

# ==========================
# SETTINGS
# ==========================
model_path = "my_model_ver3.pt"       # path ke model YOLOv11 kamu
source = 0                   # 0 = webcam | "foto.jpg" | "video.mp4"
conf_threshold = 0.5
imgsz = 416
fps_limit = 12               # batasi FPS agar CPU adem
# ==========================

# Pastikan CPU (cocok untuk AMD)
torch.set_num_threads(6)
model = YOLO(model_path)
model.to("cpu")

# Mulai kamera / video / gambar
cap = cv2.VideoCapture(source)
if isinstance(source, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
print("🚀 AKSA Detection started — tekan Q untuk berhenti.")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Sumber video tidak bisa dibaca.")
            break

        now = time.time()
        if now - prev_time < 1 / fps_limit:
            continue
        prev_time = now

        # Deteksi
        results = model(frame, imgsz=imgsz, verbose=False)
        detections = results[0].boxes

        for det in detections:
            if det.conf < conf_threshold:
                continue

            # koordinat bbox
            x1, y1, x2, y2 = map(int, det.xyxy[0])

            cls_id = int(det.cls[0])
            label = model.names[cls_id].lower()

            # warna kotak
            if label == "rapi":
                color = (0, 255, 0)   # hijau
            else:
                color = (0, 0, 255)   # merah

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("AKSA — Smart Uniform Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
