# Skill: Segmentation

## Cuando Usar

- El usuario quiere **separar objetos del fondo** (background removal)
- Necesita **mascaras pixeladas** de objetos (no solo bounding boxes)
- Quiere **aislar jugadores** u objetos de la cancha para analisis
- Requiere **segmentacion interactiva** (clic en un punto y obtener la mascara)
- Necesita **contornos precisos** para medir area, forma o silueta de objetos
- Quiere combinar segmentacion con tracking o pose estimation

---

## Prerequisitos

```python
# Stack minimo
from ultralytics import YOLO, SAM
import supervision as sv
import numpy as np
import cv2

# Verificar GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Tipos de Segmentacion

| Tipo | Que hace | Ejemplo | Modelo tipico |
|------|----------|---------|---------------|
| **Instance** | Mascara individual por cada objeto detectado | Cada jugador tiene su mascara separada | YOLO11-Seg |
| **Semantic** | Clasifica cada pixel en una categoria (sin distinguir instancias) | Todos los jugadores = 1 color, cancha = otro | DeepLabV3, SegFormer |
| **Panoptic** | Combina instance + semantic: cada pixel tiene clase + instancia | Jugadores individuales + cancha + cielo como regiones | Mask2Former, DETR Panoptic |
| **Interactive** | El usuario indica puntos/cajas y el modelo segmenta ese objeto | Clic en un jugador -> mascara precisa | SAM 2, SAM 3 |

**Para el semillero usamos principalmente:**
- **Instance segmentation** (YOLO-Seg) para deteccion automatica con mascaras
- **Interactive segmentation** (SAM) para segmentacion guiada por prompts

---

## Modelos Disponibles

### YOLO11-Seg (Instance Segmentation)

| Modelo | Params | FLOPs | mAP Box | mAP Mask | GPU T4 (ms) | VRAM estimada |
|--------|--------|-------|---------|----------|-------------|---------------|
| `yolo11n-seg.pt` | 2.9M | 10.4B | 38.9 | 32.0 | 1.8 | ~2 GB |
| `yolo11s-seg.pt` | 10.1M | 35.5B | 46.6 | 37.8 | 2.9 | ~3 GB |
| `yolo11m-seg.pt` | 22.4M | 123.3B | 51.5 | 41.5 | 6.3 | ~5 GB |
| `yolo11l-seg.pt` | 27.6M | 142.2B | 53.4 | 42.9 | 7.8 | ~6 GB |
| `yolo11x-seg.pt` | 62.1M | 319.0B | 54.7 | 43.8 | 15.8 | ~10 GB |

> Con la RTX A5000 (24 GB), cualquier variante corre sin problema. Para tiempo real, usar `n` o `s`. Para maxima precision, `l` o `x`.

### SAM 2 (Segment Anything Model 2 — Interactive)

| Modelo | Encoder | Params aprox. | VRAM estimada | Uso |
|--------|---------|---------------|---------------|-----|
| `sam2_t.pt` | Hiera Tiny | ~39M | ~4 GB | Rapido, interactivo |
| `sam2_s.pt` | Hiera Small | ~46M | ~5 GB | Balance velocidad/precision |
| `sam2_b.pt` | Hiera Base+ | ~81M | ~8 GB | Buena precision |
| `sam2_l.pt` | Hiera Large | ~224M | ~12 GB | Maxima calidad |

### SAM 3 (Segment Anything Model 3 — Noviembre 2025)

| Modelo | Params aprox. | VRAM estimada | Capacidades |
|--------|---------------|---------------|-------------|
| `sam3.pt` | ~840M | ~14-16 GB | Text prompts, image exemplars, video tracking |

> **SAM 3** es la version mas reciente (nov 2025). Agrega **Promptable Concept Segmentation**: puedes describir lo que quieres segmentar con texto ("jugadores con camiseta roja") o con una imagen de ejemplo. Requiere descargar pesos desde HuggingFace (acceso con solicitud). SAM 2 sigue siendo la opcion mas practica para prompts de punto/caja.

---

## Workflow: Segmentacion con YOLO-Seg

### Inferencia basica

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")

# Inferencia en imagen
results = model("cancha.jpg", device="cuda", conf=0.5)

# Visualizar resultado con mascaras superpuestas
results[0].show()

# Guardar imagen anotada
results[0].save("resultado_segmentado.jpg")
```

### Acceder a mascaras, clases y confianza

```python
result = results[0]

# Mascaras binarias: tensor (num_objetos, H, W)
masks = result.masks.data            # torch.Tensor en GPU
masks_np = result.masks.data.cpu().numpy()  # numpy array

# Poligonos (contornos) de cada mascara
polygons = result.masks.xy           # lista de arrays (x, y) en pixeles
polygons_norm = result.masks.xyn     # normalizado [0, 1]

# Clases y confianza
classes = result.boxes.cls.cpu().numpy().astype(int)
confidences = result.boxes.conf.cpu().numpy()
names = result.names  # dict {0: 'person', 1: 'bicycle', ...}

# Iterar por cada objeto detectado
for i, (mask, cls, conf) in enumerate(zip(masks_np, classes, confidences)):
    print(f"Objeto {i}: {names[cls]} ({conf:.2f}) — mascara shape: {mask.shape}")
```

### Filtrar solo personas

```python
# Clase 0 en COCO = 'person'
person_indices = [i for i, cls in enumerate(classes) if cls == 0]

for idx in person_indices:
    person_mask = masks_np[idx]  # mascara binaria (H, W), valores 0 o 1
    conf = confidences[idx]
    print(f"Persona detectada con {conf:.1%} confianza")
```

### Segmentacion en video (streaming)

```python
model = YOLO("yolo11s-seg.pt")

for result in model("partido.mp4", device="cuda", conf=0.5, stream=True):
    frame = result.plot()  # frame con mascaras dibujadas
    cv2.imshow("Segmentacion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
```

---

## Workflow: Segmentacion con SAM 2

SAM permite segmentacion **interactiva**: el usuario indica que segmentar mediante puntos o cajas.

### Con point prompts

```python
from ultralytics import SAM

model = SAM("sam2_b.pt")

# Un punto: (x, y), label=1 (foreground)
results = model("jugador.jpg", points=[450, 300], labels=[1])
results[0].show()

# Multiples puntos (foreground + background)
# label=1: "incluir esto", label=0: "excluir esto"
results = model(
    "jugador.jpg",
    points=[[[450, 300], [500, 100]]],  # puntos agrupados por objeto
    labels=[[1, 0]]                       # foreground, background
)
```

### Con box prompts (bounding box)

```python
# Dar una caja [x1, y1, x2, y2] y SAM segmenta el objeto dentro
results = model("jugador.jpg", bboxes=[100, 50, 400, 500])
results[0].show()

# Multiples cajas
results = model("equipo.jpg", bboxes=[[100, 50, 400, 500], [500, 80, 750, 480]])
```

### Pipeline YOLO + SAM (lo mejor de ambos)

YOLO detecta automaticamente -> SAM refina las mascaras con mayor precision:

```python
from ultralytics import YOLO, SAM

# Paso 1: YOLO detecta personas
detector = YOLO("yolo11n.pt")  # modelo de deteccion (sin seg)
det_results = detector("cancha.jpg", device="cuda", conf=0.5, classes=[0])  # solo personas

# Paso 2: extraer bounding boxes
boxes = det_results[0].boxes.xyxy.cpu().numpy()  # [[x1,y1,x2,y2], ...]

# Paso 3: SAM segmenta con precision usando esas cajas
sam = SAM("sam2_b.pt")
sam_results = sam("cancha.jpg", bboxes=boxes.tolist())
sam_results[0].show()
```

### SAMPredictor (inferencia eficiente en multiples prompts)

```python
from ultralytics.models.sam import Predictor as SAMPredictor

overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
predictor = SAMPredictor(overrides=overrides)

# Setear imagen UNA vez (encoder corre 1 sola vez)
predictor.set_image("cancha.jpg")

# Multiples queries sin re-encodear la imagen
masks_1 = predictor(points=[450, 300], labels=[1])
masks_2 = predictor(bboxes=[100, 50, 400, 500])
masks_3 = predictor(points=[[200, 150], [600, 400]], labels=[1, 1])

# Resetear cuando cambies de imagen
predictor.reset_image()
```

---

## Workflow: Segmentacion de Personas (Background Removal)

Caso comun en deportes: aislar jugadores del fondo para analisis visual.

```python
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO("yolo11m-seg.pt")
image = cv2.imread("cancha.jpg")

results = model(image, device="cuda", conf=0.5, classes=[0])  # solo personas
result = results[0]

if result.masks is not None:
    # Combinar todas las mascaras de personas en una sola
    all_masks = result.masks.data.cpu().numpy()  # (N, H, W)
    combined = np.any(all_masks > 0.5, axis=0).astype(np.uint8)  # union de mascaras

    # Redimensionar mascara al tamano original de la imagen
    h, w = image.shape[:2]
    mask_resized = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

    # Opcion A: fondo negro
    bg_removed = image.copy()
    bg_removed[mask_resized == 0] = 0
    cv2.imwrite("jugadores_aislados.jpg", bg_removed)

    # Opcion B: fondo transparente (PNG con canal alpha)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_resized * 255
    cv2.imwrite("jugadores_transparente.png", rgba)

    # Opcion C: fondo personalizado (verde chroma)
    bg_color = np.full_like(image, (0, 255, 0))  # verde
    chroma = np.where(mask_resized[:, :, None] == 1, image, bg_color)
    cv2.imwrite("jugadores_chroma.jpg", chroma)
```

---

## Manipulacion de Mascaras

### Extraer mascara binaria individual

```python
# masks_np shape: (num_objetos, H_mask, W_mask)
# NOTA: H_mask y W_mask pueden diferir de la imagen original
mask_i = masks_np[0]  # primera mascara, valores float [0, 1]
binary = (mask_i > 0.5).astype(np.uint8)  # binarizar

# Redimensionar al tamano original
h_orig, w_orig = result.orig_img.shape[:2]
binary_full = cv2.resize(binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
```

### Calcular area de la mascara

```python
area_pixels = np.sum(binary_full)
area_pct = area_pixels / (h_orig * w_orig) * 100
print(f"Area: {area_pixels} px ({area_pct:.1f}% de la imagen)")
```

### Extraer bounding box de la mascara

```python
coords = np.argwhere(binary_full > 0)  # (row, col)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
crop = result.orig_img[y_min:y_max+1, x_min:x_max+1]
cv2.imwrite(f"crop_objeto.jpg", crop)
```

### Combinar mascaras selectivamente

```python
# Unir mascaras de objetos especificos
indices_seleccionados = [0, 2, 5]
combined = np.zeros((h_orig, w_orig), dtype=np.uint8)

for idx in indices_seleccionados:
    m = (masks_np[idx] > 0.5).astype(np.uint8)
    m_resized = cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    combined = np.maximum(combined, m_resized)  # union
```

### Guardar mascaras individuales como imagenes

```python
import os
os.makedirs("mascaras", exist_ok=True)

for i, mask in enumerate(masks_np):
    binary = (mask > 0.5).astype(np.uint8) * 255
    binary_full = cv2.resize(binary, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"mascaras/mask_{i:03d}.png", binary_full)
```

---

## Integracion con Supervision

### Detecciones con mascaras + MaskAnnotator

```python
import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
image = cv2.imread("cancha.jpg")
results = model(image, device="cuda")[0]

# Convertir a sv.Detections (incluye mascaras automaticamente)
detections = sv.Detections.from_ultralytics(results)

# Anotar con mascaras semitransparentes
mask_annotator = sv.MaskAnnotator(opacity=0.4)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

# Generar labels
labels = [
    f"{results.names[cls]} {conf:.0%}"
    for cls, conf in zip(detections.class_id, detections.confidence)
]

annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

cv2.imwrite("anotado_mascaras.jpg", annotated)
```

### Filtrar detecciones por clase

```python
# Solo personas (class_id == 0 en COCO)
persons = detections[detections.class_id == 0]

# Solo objetos con confianza > 0.7
high_conf = detections[detections.confidence > 0.7]

# Anotar solo lo filtrado
annotated = mask_annotator.annotate(scene=image.copy(), detections=persons)
```

### Acceder a mascaras desde sv.Detections

```python
# detections.mask -> numpy array (N, H, W), dtype bool
if detections.mask is not None:
    for i, mask in enumerate(detections.mask):
        # mask es bool array (H, W)
        area = mask.sum()
        print(f"Objeto {i}: {area} pixeles")
```

---

## Parametros Clave

### YOLO-Seg

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `conf` | 0.25 | Umbral de confianza minimo |
| `iou` | 0.7 | Umbral IoU para NMS (Non-Max Suppression) |
| `classes` | None | Filtrar clases: `classes=[0]` solo personas |
| `imgsz` | 640 | Tamano de entrada. Mas grande = mas preciso pero mas lento |
| `retina_masks` | False | `True` genera mascaras a resolucion completa (mas RAM) |
| `device` | auto | `"cuda"`, `"cpu"`, o `"cuda:0"` |
| `stream` | False | `True` para video (procesa frame a frame, ahorra RAM) |

### SAM 2/3

| Parametro | Descripcion |
|-----------|-------------|
| `points` | Coordenadas (x, y) de puntos prompt |
| `labels` | 1 = foreground, 0 = background (para cada punto) |
| `bboxes` | Bounding boxes [x1, y1, x2, y2] como region de interes |
| `imgsz` | 1024 recomendado para SAM (resolucion del encoder) |

---

## Errores Comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| `result.masks is None` | Usaste modelo de deteccion (`yolo11n.pt`) en vez de segmentacion | Usar `yolo11n-seg.pt` |
| Mascara con tamano diferente a la imagen | Las mascaras salen a resolucion del modelo (ej: 160x160) | Redimensionar con `cv2.resize(..., interpolation=cv2.INTER_NEAREST)` |
| Mascaras borrosas o con bordes irregulares | `retina_masks=False` (default) | Usar `retina_masks=True` para mascaras de alta resolucion |
| CUDA out of memory con SAM | SAM usa mucha VRAM por el encoder | Usar `sam2_t.pt` (Tiny) o reducir `imgsz` |
| SAM no detecta nada automaticamente | SAM necesita prompts (puntos o cajas) | Usar YOLO para deteccion automatica y SAM solo para refinar |
| `sam3.pt` no se descarga | SAM 3 requiere acceso aprobado en HuggingFace | Solicitar acceso en la pagina del modelo y descargar manualmente |
| Mascaras de YOLO-Seg son tipo float | `.masks.data` contiene probabilidades [0, 1] | Binarizar con `(mask > 0.5).astype(np.uint8)` |

---

## Tips de Experto

### Cuando usar YOLO-Seg vs SAM

| Escenario | Modelo recomendado | Razon |
|-----------|--------------------|-------|
| Deteccion automatica + mascaras en video | **YOLO11-Seg** | Rapido, end-to-end, funciona en tiempo real |
| Segmentar un objeto especifico manualmente | **SAM 2** | Point/box prompts, mascara precisa |
| Maxima precision en mascaras (offline) | **YOLO + SAM pipeline** | YOLO detecta, SAM refina |
| Segmentar por descripcion textual | **SAM 3** | "jugadores con camiseta azul" |
| Video largo, muchos frames | **YOLO11n-seg** con `stream=True` | Minimo uso de RAM, velocidad maxima |
| Dataset annotation / labeling | **SAM 2 con SAMPredictor** | Set image 1 vez, multiples prompts rapidos |

### Optimizacion de memoria

```python
# Para videos largos, SIEMPRE usar stream=True
for result in model("video_largo.mp4", stream=True, device="cuda"):
    # Procesar frame a frame (no carga todo en RAM)
    masks = result.masks.data.cpu().numpy()  # mover a CPU inmediatamente
    # ... procesar ...

# Liberar memoria GPU manualmente si es necesario
import torch
torch.cuda.empty_cache()
```

### Post-procesamiento de mascaras

```python
# Suavizar bordes de mascara con operaciones morfologicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.morphologyEx(binary_full, cv2.MORPH_CLOSE, kernel)  # cerrar huecos
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)    # remover ruido

# Obtener contorno suave
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Dibujar contorno sobre la imagen
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

### Combinar segmentacion con tracking

```python
# YOLO-Seg + ByteTrack para tracking con mascaras
model = YOLO("yolo11s-seg.pt")

for result in model.track("partido.mp4", tracker="bytetrack.yaml",
                           device="cuda", stream=True, persist=True):
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        for tid, mask in zip(track_ids, masks):
            print(f"Jugador #{tid}: area mascara = {(mask > 0.5).sum()} px")
```

---

## Siguiente Paso

- **Quieres detectar objetos sin mascaras?** -> `skills/object-detection.md`
- **Quieres estimar la pose de los jugadores segmentados?** -> `skills/pose-estimation.md`
- **Quieres seguir jugadores segmentados a lo largo del video?** -> `skills/tracking.md`
- **Quieres procesar un video completo frame a frame?** -> `skills/video-pipeline.md`
- **Quieres entrenar YOLO-Seg con tu propio dataset?** -> `skills/training.md`
