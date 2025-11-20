import cv2
import time
from ultralytics import YOLO

# 1. Load model hasil training
model = YOLO("my_model_ver2.pt")   # ganti jika namanya lain
model.to("cpu")

# 2. Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka.")
    exit()

# Target resolusi kecil supaya cepat
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("AKSA YOLOv11 Realtime (Optimized CPU)\nTekan 'q' untuk keluar...")

# Frame skipping
skip_rate = 2     # deteksi tiap 2 frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize frame biar YOLO kerjanya ringan
    small_frame = cv2.resize(frame, (480, 360))

    # Skip frame untuk ringankan CPU
    if frame_count % skip_rate == 0:
        results = model(
            small_frame,
            imgsz=256,       # model input lebih kecil → jauh lebih cepat
            verbose=False
        )
        annotated = results[0].plot()
    else:
        # gunakan annotated sebelumnya agar tidak blank
        try:
            annotated
        except:
            annotated = small_frame

    # Tampilkan
    cv2.imshow("AKSA YOLOv11 (Optimized)", annotated)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
