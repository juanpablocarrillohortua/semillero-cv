# Skill: Entrenamiento y Fine-Tuning

Workflow completo para entrenar y hacer fine-tuning de modelos YOLO con datasets propios usando Ultralytics.

## Cuando Usar

- Entrenar modelo con datos propios (jugadores, pelotas, gestos, etc.)
- Adaptar modelo pre-entrenado a dominio especifico (deportes, campo de juego)
- Exportar modelo entrenado para despliegue (ONNX, TensorRT)

## Prerequisitos

```python
import torch
assert torch.cuda.is_available(), "GPU no disponible — no entrenar en CPU"
print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

## Formatos de Dataset

### Formato YOLO (recomendado)

Cada imagen tiene un `.txt` con el mismo nombre. Una linea por objeto: `<class_id> <x_center> <y_center> <width> <height>` (coordenadas normalizadas 0.0-1.0).

```
dataset/
├── images/
│   ├── train/          # 80% de imagenes
│   │   └── img_001.jpg
│   └── val/            # 20% de imagenes
│       └── img_100.jpg
├── labels/
│   ├── train/
│   │   └── img_001.txt    # "0 0.260 0.278 0.063 0.259"
│   └── val/
│       └── img_100.txt
└── data.yaml
```

> CRITICO: `images/` y `labels/` al mismo nivel con subdirectorios identicos. `.txt` vacio = imagen sin objetos (background negativo).

### Roboflow (anotacion)

Para anotar datasets: subir a [roboflow.com](https://roboflow.com), anotar, exportar formato "YOLOv8". Genera estructura + `data.yaml` automaticamente.

```python
from roboflow import Roboflow
rf = Roboflow(api_key="KEY")
dataset = rf.workspace("ws").project("proj").version(1).download("yolov8")
```

### COCO JSON a YOLO

```python
from ultralytics.data.converter import convert_coco
convert_coco(labels_dir="path/to/coco/annotations/", use_segments=False)
```

## Estructura de data.yaml

```yaml
path: /home/usuario/datasets/mi-dataset    # Ruta ABSOLUTA al directorio raiz
train: images/train                         # Relativa a 'path'
val: images/val                             # Relativa a 'path'
test: images/test                           # Opcional
nc: 4                                       # Numero de clases
names:
  0: jugador
  1: pelota
  2: arbitro
  3: portero
```

## Workflow: Fine-Tuning YOLO con Dataset Propio

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # nano(n) | small(s) | medium(m) | large(l)
results = model.train(
    data="path/to/data.yaml",
    epochs=150,              # Epocas (patience detiene antes si no mejora)
    imgsz=640,               # Tamano imagen (640 o 1280)
    batch=32,                # Batch size (ajustar segun VRAM — ver tabla abajo)
    device="cuda",
    patience=30,             # Early stopping
    optimizer="auto",        # auto | SGD | Adam | AdamW
    lr0=0.01,                # Learning rate inicial
    lrf=0.01,                # LR final = lr0 * lrf
    cos_lr=True,             # Cosine LR scheduler
    warmup_epochs=3.0,
    close_mosaic=15,         # Desactivar mosaic ultimas N epocas
    workers=8,               # Workers para data loading
    plots=True,              # Generar graficas
    project="runs/detect",
    name="mi_experimento",
)
# Resultado: runs/detect/mi_experimento/weights/best.pt (mejor modelo por mAP)
```

## Workflow: Transfer Learning

| Tamano dataset | Estrategia | Codigo |
|----------------|-----------|--------|
| < 200 imgs | Congelar backbone | `model.train(data=..., freeze=10, lr0=0.001)` |
| 200-1000 imgs | Fine-tune completo | `model.train(data=..., lr0=0.005)` |
| > 1000 imgs | Fine-tune completo | `model.train(data=..., lr0=0.01)` |

**Descongelamiento progresivo** (datasets pequenos):
```python
# Fase 1: Solo cabeza (20 epocas)
model = YOLO("yolo11m.pt")
model.train(data="data.yaml", epochs=20, freeze=10, lr0=0.001)
# Fase 2: Fine-tune completo
model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data.yaml", epochs=80, lr0=0.0005)
```

## Workflow: Validacion y Metricas

```python
model = YOLO("runs/detect/mi_experimento/weights/best.pt")
metrics = model.val(data="data.yaml", imgsz=640, device="cuda")

print(f"mAP@50:    {metrics.box.map50:.4f}")   # Mean AP al IoU 50%
print(f"mAP@50-95: {metrics.box.map:.4f}")      # Mean AP (metrica principal)
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

# AP por clase
for i, name in enumerate(model.names.values()):
    print(f"  {name}: mAP50={metrics.box.ap50()[i]:.4f}")
```

| Metrica | Bueno | Excelente | Mide |
|---------|-------|-----------|------|
| mAP@50 | > 0.70 | > 0.85 | Precision general |
| mAP@50-95 | > 0.45 | > 0.60 | Precision estricta |
| Precision | > 0.80 | > 0.90 | Confiabilidad de detecciones |
| Recall | > 0.70 | > 0.85 | Cobertura de objetos reales |

## Workflow: Export de Modelo

```python
model = YOLO("runs/detect/train/weights/best.pt")

# ONNX (interoperabilidad, CPU)
model.export(format="onnx", imgsz=640, simplify=True)

# TensorRT FP16 (maximo rendimiento GPU NVIDIA — 3-5x speedup)
model.export(format="engine", imgsz=640, half=True, device="cuda", workspace=4)

# TensorRT INT8 (maximo FPS — requiere datos para calibracion)
model.export(format="engine", imgsz=640, int8=True, data="data.yaml", workspace=4)

# OpenVINO (CPU Intel optimizado)
model.export(format="openvino", imgsz=640, half=True)
```

## Hyperparametros Clave

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `epochs` | 100 | Numero de epocas (usar patience para early stop) |
| `batch` | 16 | Batch size (ver tabla VRAM) |
| `imgsz` | 640 | Tamano imagen (640 o 1280) |
| `lr0` | 0.01 | Learning rate inicial (SGD 0.01, Adam 0.001) |
| `lrf` | 0.01 | LR final = lr0 * lrf |
| `optimizer` | auto | SGD (grande), AdamW (pequeno) |
| `patience` | 100 | Epocas sin mejora para early stop |
| `cos_lr` | False | Cosine LR scheduler (recomendado: True) |
| `freeze` | None | Capas a congelar (0=nada, 10=backbone) |
| `dropout` | 0.0 | Regularizacion (0.1-0.3 si overfitting) |
| `weight_decay` | 0.0005 | Regularizacion L2 |
| `close_mosaic` | 10 | Desactivar mosaic ultimas N epocas |

## Augmentation

Augmentaciones nativas (se pasan a `model.train()`):

| Parametro | Default | Efecto |
|-----------|---------|--------|
| `hsv_h/s/v` | 0.015/0.7/0.4 | Variacion de color |
| `degrees` | 0.0 | Rotacion (+/- grados) |
| `translate` | 0.1 | Translacion (fraccion) |
| `scale` | 0.5 | Escalado aleatorio |
| `fliplr` | 0.5 | Volteo horizontal |
| `flipud` | 0.0 | Volteo vertical |
| `mosaic` | 1.0 | Mezcla de 4 imagenes |
| `mixup` | 0.0 | Mezcla de 2 imagenes |
| `erasing` | 0.4 | Random erasing |

**Albumentations custom** (reemplaza defaults de Albumentations, augmentaciones nativas YOLO siguen activas):
```python
import albumentations as A
model.train(data="data.yaml", augmentations=[
    A.CLAHE(p=0.3), A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(blur_limit=7, p=0.2), A.GaussNoise(p=0.3),
])
```

## Monitoreo: TensorBoard

```bash
tensorboard --logdir runs/detect/mi_experimento  # http://localhost:6006
```

| Patron de loss | Diagnostico | Accion |
|----------------|-------------|--------|
| Train baja, val baja | Normal | Continuar |
| Train baja, val sube | **Overfitting** | Mas datos, mas augmentation, dropout |
| Ambas planas | LR bajo o saturacion | Subir lr0, modelo mas grande |
| Loss oscila | LR alto o batch chico | Bajar lr0, subir batch |

## Parametros para RTX A5000 (24 GB)

| Modelo | batch (imgsz=640) | batch (imgsz=1280) |
|--------|-------------------|---------------------|
| YOLO11n/YOLO26n | 64 | 16 |
| YOLO11s/YOLO26s | 48 | 12 |
| YOLO11m/YOLO26m | 32 | 8 |
| YOLO11l | 16 | 4 |
| YOLO11x | 8 | 2 |

> TIP: `batch=-1` para autodeteccion de batch optimo segun VRAM disponible.

## Errores Comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| `CUDA out of memory` | Batch o imgsz muy grande | Reducir batch a la mitad, verificar `nvidia-smi` |
| `images not found` | Paths mal en data.yaml | `path` ABSOLUTO, `train/val` RELATIVOS a path |
| `Labels class X exceeds nc` | class_id >= nc en labels | IDs validos: 0 a nc-1 |
| Train loss baja, val sube | Overfitting | Mas datos, dropout=0.1, mas augmentation |
| Entrenamiento lento | CPU o sin AMP | `device="cuda"`, `amp=True`, `workers=8` |

## Tips de Experto

1. **Calidad > cantidad.** 500 imagenes bien anotadas > 5000 mal anotadas
2. **Siempre early stopping.** `patience=20-50` + epochs alto (200-300)
3. **Empezar chico.** Prototipar con YOLO11n, luego escalar a m/l
4. **Split 80/20 estratificado.** Todas las clases representadas en train y val
5. **Verificar labels visualmente** antes de entrenar (Roboflow o `model.val(plots=True)`)
6. **Background negativo.** 5-10% de imagenes SIN objetos reduce falsos positivos
7. **Cosine LR.** `cos_lr=True` da mejor convergencia que linear decay
8. **close_mosaic > 0.** Desactivar mosaic las ultimas epocas mejora predicciones finales
9. **640 basta.** Solo usar 1280 si los objetos son muy pequenos en el frame
10. **Log todo.** Usar `project` y `name` descriptivos. Mantener historial de experimentos

## Siguiente Paso

Despues de entrenar, usar el modelo con:
- `skills/object-detection.md` — Deteccion en imagenes y video
- `skills/tracking.md` — Tracking multi-objeto
- `skills/video-pipeline.md` — Pipeline de procesamiento de video
