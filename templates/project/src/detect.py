"""Boilerplate de deteccion — adaptar segun el proyecto."""

import torch
from ultralytics import YOLO

# Verificar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

# Cargar modelo
model = YOLO("yolov8n.pt")

# Inferencia
# results = model("path/to/image.jpg", device=device)
# results[0].show()
