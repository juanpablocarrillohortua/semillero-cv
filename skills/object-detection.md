# Skill: Object Detection

## Cuando Usar
Detectar objetos en imagenes/video, bounding boxes, YOLO, RT-DETR, contar objetos, filtrar por clase/confidence, exportar modelos, inferencia batch/video/webcam.

## Prerequisitos
```python
import torch, cv2, numpy as np
from ultralytics import YOLO
import supervision as sv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## Modelos Disponibles

### YOLO26 (ultimo -- enero 2026, recomendado)
End-to-end nativo (sin NMS). Hasta 43% mas rapido en CPU que YOLO11.

| Modelo | mAP@50-95 | Params | GFLOPs | GPU (ms) | CPU (ms) | Caso de uso |
|--------|-----------|--------|--------|----------|----------|-------------|
| `yolo26n.pt` | 40.9 | 2.4M | 5.4 | 1.7 | 38.9 | Prototipado, edge, webcam |
| `yolo26s.pt` | 48.6 | 9.5M | 20.7 | 2.5 | 87.2 | Balance velocidad/precision |
| `yolo26m.pt` | 53.1 | 20.4M | 68.2 | 4.7 | 220.0 | Produccion, analisis deportivo |
| `yolo26l.pt` | 55.0 | 24.8M | 86.4 | 6.2 | 286.2 | Alta precision |
| `yolo26x.pt` | 57.5 | 55.7M | 193.9 | 11.8 | 525.8 | Maxima precision, investigacion |

### YOLO11 (estable -- septiembre 2024)
Arquitectura clasica con NMS. Amplia comunidad y tutoriales.

| Modelo | mAP@50-95 | Params | GFLOPs | Caso de uso |
|--------|-----------|--------|--------|-------------|
| `yolo11n.pt` | 39.5 | 2.6M | 6.5 | Edge, prototipos |
| `yolo11s.pt` | 47.0 | 9.4M | 21.5 | Balance general |
| `yolo11m.pt` | 51.5 | 20.1M | 68.0 | Produccion |
| `yolo11l.pt` | 53.4 | 25.3M | 86.9 | Precision alta |
| `yolo11x.pt` | 54.7 | 56.9M | 194.9 | Maxima precision |

### RT-DETR (Vision Transformer)
Sin NMS nativo. Mejor en objetos grandes y escenas complejas.

| Modelo | mAP@50-95 | FPS (T4) | Caso de uso |
|--------|-----------|----------|-------------|
| `rtdetr-l.pt` | 53.0 | 114 | Precision + velocidad |
| `rtdetr-x.pt` | 54.8 | 74 | Maxima precision transformer |

> mAP en COCO val2017 @ 640px. GPU speed: TensorRT FP16 en T4.

**Regla rapida:** Prototipo --> `yolo26n` | Produccion --> `yolo26m` | Max precision --> `yolo26x` | Objetos grandes --> `rtdetr-l` | Reproducir papers --> `yolo11m`

## Workflow: Deteccion en Imagen
```python
from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("yolo26m.pt")

# Inferencia
result = model.predict("imagen.jpg", device="cuda", conf=0.5, imgsz=640, verbose=False)[0]

# Acceder a detecciones
for box in result.boxes:
    cls_name = model.names[int(box.cls[0])]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{cls_name}: {conf:.2f} @ [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Visualizacion con Supervision
image = cv2.imread("imagen.jpg")
detections = sv.Detections.from_ultralytics(result)
labels = [f"{model.names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]

annotated = sv.BoxAnnotator(thickness=2).annotate(scene=image.copy(), detections=detections)
annotated = sv.LabelAnnotator(text_scale=0.5).annotate(scene=annotated, detections=detections, labels=labels)
cv2.imwrite("resultado.jpg", annotated)
```

## Workflow: Deteccion en Video
```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo26m.pt")
video_path = "partido.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)

box_ann = sv.BoxAnnotator(thickness=2)
label_ann = sv.LabelAnnotator(text_scale=0.5)

with sv.VideoSink(target_path="salida.mp4", video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(source_path=video_path):
        result = model.predict(frame, device="cuda", conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [f"{model.names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]

        annotated = box_ann.annotate(scene=frame.copy(), detections=detections)
        annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)
        sink.write_frame(annotated)
```

**Alternativa simple (stream nativo):** `model.predict("video.mp4", device="cuda", stream=True, save=True)`

## Workflow: Deteccion en Webcam
```python
import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolo26n.pt")  # nano para real-time
box_ann, label_ann = sv.BoxAnnotator(thickness=2), sv.LabelAnnotator(text_scale=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    result = model.predict(frame, device="cuda", conf=0.5, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [f"{model.names[c]} {conf:.2f}" for c, conf in zip(detections.class_id, detections.confidence)]
    annotated = box_ann.annotate(scene=frame.copy(), detections=detections)
    annotated = label_ann.annotate(scene=annotated, detections=detections, labels=labels)
    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
```

## Personalizacion
```python
# Filtrar por clases COCO en inferencia
results = model.predict("img.jpg", device="cuda", classes=[0, 32])  # 0=person, 32=sports ball

# Filtrar post-inferencia con Supervision
detections = sv.Detections.from_ultralytics(result)
detections = detections[detections.confidence > 0.7]       # por confidence
detections = detections[detections.class_id == 0]           # solo personas
detections = detections[detections.box_area > 500]          # eliminar detecciones pequenas
```

Clases COCO deportes: `0` person | `32` sports ball | `33` kite | `35` skateboard | `37` tennis racket. Lista completa: `print(model.names)`.

## Integracion con Supervision
```python
# Annotators principales
sv.BoxAnnotator(thickness=2)                                      # bounding boxes clasicos
sv.RoundBoxAnnotator(thickness=2, roundness=0.3)                  # esquinas redondeadas
sv.LabelAnnotator(text_scale=0.5, text_padding=5)                # labels con clase+conf
sv.DotAnnotator(radius=5)                                         # punto central
sv.TriangleAnnotator(base=20, height=15)                          # triangulo (util en deportes)
sv.TraceAnnotator(thickness=2, trace_length=50)                   # trail de movimiento (con tracker)

# Todos siguen el mismo patron
annotated = annotator.annotate(scene=image.copy(), detections=detections)
```

## Gestion de Modelos
```python
# Cambiar directorio de pesos (persistente)
from ultralytics import settings
settings.update({"weights_dir": "models/"})

# Variable de entorno alternativa
# export YOLO_CONFIG_DIR="/ruta/config/"
```
Patron proyecto: pesos en `models/` (gitignored). NUNCA commitear `.pt`, `.onnx`, `.engine`.

## Export de Modelos
```python
model = YOLO("yolo26m.pt")
model.export(format="onnx", imgsz=640, simplify=True)                    # ONNX (CPU deploy)
model.export(format="engine", imgsz=640, half=True, device="cuda")       # TensorRT (GPU NVIDIA)
model.export(format="coreml", imgsz=640)                                 # CoreML (iOS/macOS)

# Usar modelo exportado (misma API)
model_trt = YOLO("yolo26m.engine")
results = model_trt.predict("imagen.jpg", device="cuda", conf=0.5)
```

## Parametros Clave

| Parametro | Default | Descripcion | Rango recomendado |
|-----------|---------|-------------|-------------------|
| `conf` | 0.25 | Umbral de confianza minimo | 0.3-0.7 (deportes: 0.5) |
| `iou` | 0.7 | Umbral IoU para NMS | 0.5-0.8 |
| `imgsz` | 640 | Tamano de imagen entrada | 320, 640, 1280 (multiplo de 32) |
| `classes` | None | Filtrar por IDs de clase | `[0]` personas, `[0,32]` +pelotas |
| `device` | None | Dispositivo | `"cuda"`, `"cpu"` |
| `batch` | 1 | Tamano del batch | 1-16 segun VRAM |
| `half` | False | FP16 (mitad VRAM) | True en GPU con Tensor Cores |
| `stream` | False | Modo generador (ahorra RAM) | True para videos largos |
| `vid_stride` | 1 | Procesar 1 de cada N frames | 2-5 para acelerar |
| `max_det` | 300 | Max detecciones por imagen | 100-1000 |
| `verbose` | True | Imprimir logs | False en loops |
| `save` | False | Guardar imagenes anotadas | True para debug |

## Errores Comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| `CUDA out of memory` | VRAM insuficiente | Reducir `batch`, modelo mas pequeno, `half=True` |
| Detecciones vacias | `conf` muy alto o clase no existe | Bajar `conf` a 0.25, verificar `model.names` |
| Video sin audio en salida | OpenCV/Supervision no copian audio | Mergear con `ffmpeg` despues |
| Modelo se descarga cada vez | Ruta relativa cambia | Usar ruta absoluta o `models/` |
| Inferencia lenta en GPU | Modelo no en CUDA | Verificar `device="cuda"` |
| `Results not subscriptable` | Acceso incorrecto | Usar `results[0]` y luego `.boxes` |

## Tips de Experto

1. **Warm-up:** La primera inferencia es lenta. Hacer dummy predict antes del loop: `model.predict(np.zeros((640,640,3), dtype=np.uint8), device="cuda", verbose=False)`
2. **FP16 gratis:** `half=True` en RTX A5000 duplica throughput con perdida imperceptible.
3. **TensorRT:** Exportar a `.engine` da 2-3x speedup. El archivo es especifico a la GPU donde se exporto.
4. **vid_stride=3:** Triplica velocidad en video sin perder info relevante en deportes.
5. **imgsz=1280:** Para objetos pequenos (pelota lejana). Cuadruplica compute pero mejora recall.
6. **Batch optimo A5000:** yolo26n=32 | yolo26s=16 | yolo26m=8 | yolo26l=4 | yolo26x=2
7. **YOLO26 vs YOLO11:** YOLO26 gana en precision y velocidad. YOLO11 tiene mas comunidad. Para investigacion nueva, YOLO26.
8. **RT-DETR brilla con objetos grandes.** En escenas con muchos objetos pequenos y oclusiones, YOLO gana.
9. **Limpiar cache:** `torch.cuda.empty_cache()` entre experimentos pesados.
10. **stream=True obligatorio** para videos de partidos completos (90+ min), evita acumular resultados en RAM.

## Siguiente Paso

| Quiero... | Skill |
|-----------|-------|
| Seguir objetos entre frames | `skills/tracking.md` |
| Mascaras por pixel, segmentar | `skills/segmentation.md` |
| Detectar pose, keypoints | `skills/pose-estimation.md` |
| Pipeline de video completo | `skills/video-pipeline.md` |
| Entrenar con dataset propio | `skills/training.md` |

> **Combo deportivo:** Detection --> Tracking --> Pose. Detectar jugadores, trackearlos, y estimar postura.
