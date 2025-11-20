from ultralytics import YOLO
import cv2

# 1. Load model hasil training
model = YOLO("my_model.pt")  # ganti sesuai nama file hasil trainingmu

# 2. Path gambar yang mau kamu deteksi
img_path = "C:/AKSA/val/train3.jpeg"

# 3. Jalankan prediksi
results = model(img_path)

# 4. Visualisasikan hasil
annotated = results[0].plot()

# 5. Tampilkan
cv2.imshow("Hasil Deteksi", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
