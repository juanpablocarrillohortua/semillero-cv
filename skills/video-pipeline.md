# Skill: Video Pipeline

## Cuando Usar
- Procesar videos frame-a-frame (deteccion, tracking, segmentacion, anotacion)
- Leer videos de archivo o webcam en tiempo real
- Escribir videos anotados con resultados de inferencia
- Batch processing de multiples videos en un directorio
- Extraer metadata de video (fps, resolucion, total de frames)

## Prerequisitos
```bash
conda activate semillero-cv   # o el env del proyecto
pip install opencv-python-headless supervision tqdm
# Para soporte GPU (CUDA 12.2 + RTX A5000):
pip install opencv-contrib-python  # incluye modulos extra con CUDA
```

**Imports estandar:**
```python
import cv2
import supervision as sv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import copy
```

---

## Workflow: Lectura de Video con OpenCV

Patron clasico con `cv2.VideoCapture`. Funciona con archivos y URLs RTSP/HTTP.

```python
cap = cv2.VideoCapture("input.mp4")

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

# Metadata
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps:.1f} FPS — {total_frames} frames")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame es np.ndarray (H, W, 3) BGR
    frame_count += 1

cap.release()  # SIEMPRE liberar
print(f"Frames leidos: {frame_count}")
```

**CRITICO:** Siempre llamar `cap.release()` al terminar. Usar `try/finally` en produccion.

---

## Workflow: Lectura con Supervision

Supervision envuelve OpenCV con una API mas limpia y Pythonica.

```python
import supervision as sv

VIDEO_PATH = "input.mp4"

# Metadata del video
video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)
print(f"Resolucion: {video_info.width}x{video_info.height}")
print(f"FPS: {video_info.fps}")
print(f"Total frames: {video_info.total_frames}")
print(f"Duracion: {video_info.total_frames / video_info.fps:.1f}s")

# Generador de frames (lazy — no carga todo en memoria)
for frame in sv.get_video_frames_generator(source_path=VIDEO_PATH):
    # frame: np.ndarray (H, W, 3) BGR
    pass

# Con stride (saltar frames — util para videos largos)
# stride=2 lee 1 de cada 2 frames (50% de frames)
for frame in sv.get_video_frames_generator(source_path=VIDEO_PATH, stride=2):
    pass
```

**Ventaja sobre OpenCV puro:** `get_video_frames_generator` es un generador Python nativo — no necesitas manejar `ret, frame` ni el loop `while True`.

---

## Workflow: Pipeline Completo (Leer → Procesar → Escribir)

Pipeline end-to-end: leer video, ejecutar modelo de deteccion, anotar y escribir resultado.

```python
import cv2
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# --- Configuracion ---
SOURCE_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_detections.mp4"
MODEL_PATH = "yolov8n.pt"  # o ruta a modelo custom

# --- Cargar modelo y metadata ---
model = YOLO(MODEL_PATH)
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO)
frames_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO)

# --- Anotadores ---
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# --- Pipeline ---
with sv.VideoSink(target_path=OUTPUT_VIDEO, video_info=video_info) as sink:
    for frame in tqdm(frames_generator, total=video_info.total_frames, desc="Procesando"):
        # 1. Inferencia
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. Filtrar por confianza
        detections = detections[detections.confidence > 0.5]

        # 3. Anotar
        labels = [
            f"{model.names[cls_id]} {conf:.2f}"
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        # 4. Escribir frame anotado
        sink.write_frame(frame=annotated)

print(f"Video guardado en: {OUTPUT_VIDEO}")
```

**Notas:**
- `frame.copy()` evita modificar el frame original (importante si reutilizas el frame)
- `verbose=False` en YOLO silencia logs por frame
- VideoSink maneja automaticamente codec, fps y resolucion desde `video_info`

---

## Workflow: Procesamiento con Progreso

Para videos largos, `tqdm` da feedback visual del progreso.

```python
import supervision as sv
from tqdm import tqdm

VIDEO_PATH = "video_largo.mp4"
video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_PATH)

# tqdm necesita total para mostrar porcentaje y ETA
frames_gen = sv.get_video_frames_generator(source_path=VIDEO_PATH)
for frame in tqdm(frames_gen, total=video_info.total_frames, desc="Frames"):
    # procesar frame...
    pass
```

**Con stride** (procesas menos frames pero tqdm muestra progreso correcto):
```python
STRIDE = 3
total_effective = video_info.total_frames // STRIDE

for frame in tqdm(
    sv.get_video_frames_generator(source_path=VIDEO_PATH, stride=STRIDE),
    total=total_effective,
    desc="Procesando (stride=3)"
):
    pass
```

**Con OpenCV puro y tqdm:**
```python
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

try:
    for _ in tqdm(range(total), desc="Procesando"):
        ret, frame = cap.read()
        if not ret:
            break
        # procesar frame...
finally:
    cap.release()
```

---

## Workflow: Webcam en Tiempo Real

Procesamiento en vivo desde webcam con visualizacion.

```python
import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(0)  # 0 = webcam por defecto
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)

        cv2.imshow("Webcam — Deteccion en Vivo", annotated)

        # 'q' para salir — waitKey(1) = ~1000 FPS max
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

**Nota:** En servidores sin display (SSH), omitir `imshow` y `waitKey`. Usar VideoSink para guardar output.

---

## Codecs y Formatos

| Codec (FourCC) | Extension | Compatibilidad | Calidad | Notas |
|-----------------|-----------|----------------|---------|-------|
| `mp4v` | `.mp4` | Alta (universal) | Buena | **Recomendado por defecto.** MPEG-4 Part 2 |
| `avc1` / `H264` | `.mp4` | Alta | Excelente | H.264 — requiere ffmpeg con libx264 |
| `XVID` | `.avi` | Alta (legacy) | Buena | Alternativa .avi clasica |
| `MJPG` | `.avi` | Muy alta | Baja (grande) | Motion JPEG — sin compresion temporal |
| `HEVC` | `.mp4` | Media | Excelente | H.265 — archivos mas pequenos, encoding lento |
| `FFV1` | `.avi` | Baja | Lossless | Sin perdida — archivos muy grandes |

**Uso con OpenCV:**
```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 4 caracteres
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
```

**Uso con Supervision:**
```python
# VideoSink usa "mp4v" por defecto para .mp4
with sv.VideoSink(target_path="output.mp4", video_info=video_info, codec="mp4v") as sink:
    sink.write_frame(frame=frame)
```

---

## Supervision VideoSink

`VideoSink` es un context manager que gestiona apertura, escritura y cierre del video automaticamente.

```python
video_info = sv.VideoInfo.from_video_path("input.mp4")

# Uso basico — hereda fps y resolucion del video original
with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator("input.mp4"):
        processed = mi_funcion_de_procesamiento(frame)
        sink.write_frame(frame=processed)
# Al salir del `with`, el video se cierra y finaliza correctamente

# Cambiar resolucion del output (ej: mitad de tamano)
from copy import copy
small_info = copy(video_info)
small_info.width = video_info.width // 2
small_info.height = video_info.height // 2

with sv.VideoSink(target_path="output_small.mp4", video_info=small_info) as sink:
    for frame in sv.get_video_frames_generator("input.mp4"):
        resized = cv2.resize(frame, (small_info.width, small_info.height))
        sink.write_frame(frame=resized)
```

**CRITICO:** Si cambias la resolucion en `video_info`, DEBES hacer `cv2.resize()` en cada frame al tamano correspondiente. Si el tamano del frame no coincide con `video_info`, el video se genera vacio o corrupto.

---

## Batch Processing

Procesar todos los videos de un directorio.

```python
import cv2
import supervision as sv
from pathlib import Path
from tqdm import tqdm
import gc

INPUT_DIR = Path("videos/raw")
OUTPUT_DIR = Path("videos/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

video_files = sorted([
    f for f in INPUT_DIR.iterdir()
    if f.suffix.lower() in VIDEO_EXTENSIONS
])
print(f"Videos encontrados: {len(video_files)}")

for video_path in video_files:
    print(f"\n--- Procesando: {video_path.name} ---")
    output_path = OUTPUT_DIR / f"{video_path.stem}_processed.mp4"

    video_info = sv.VideoInfo.from_video_path(video_path=str(video_path))
    frames_gen = sv.get_video_frames_generator(source_path=str(video_path))

    with sv.VideoSink(target_path=str(output_path), video_info=video_info) as sink:
        for frame in tqdm(frames_gen, total=video_info.total_frames, desc=video_path.name):
            processed = mi_funcion_de_procesamiento(frame)
            sink.write_frame(frame=processed)

    # Limpieza de memoria entre videos
    del frames_gen, video_info
    gc.collect()
    print(f"   Guardado: {output_path}")

print(f"\nBatch completo. {len(video_files)} videos procesados.")
```

---

## Parametros Clave

| Parametro | Tipo | Descripcion |
|-----------|------|-------------|
| `source_path` | `str` | Ruta al video (archivo o URL RTSP) |
| `stride` | `int` | Leer 1 de cada N frames (default: 1) |
| `target_path` | `str` | Ruta de salida para VideoSink |
| `video_info` | `VideoInfo` | Metadata (fps, resolucion) para escritura |
| `codec` | `str` | FourCC como string: `"mp4v"`, `"XVID"`, etc. |
| `CAP_PROP_FPS` | `int` | Propiedad OpenCV para obtener FPS |
| `CAP_PROP_FRAME_WIDTH` | `int` | Ancho del frame en OpenCV |
| `CAP_PROP_FRAME_HEIGHT` | `int` | Alto del frame en OpenCV |
| `CAP_PROP_FRAME_COUNT` | `int` | Total de frames en OpenCV |

---

## Errores Comunes

### 1. Video de salida vacio o corrupto
```
Causa: Tamano del frame no coincide con el VideoWriter/VideoSink.
```
```python
# MAL — frame de 1920x1080, writer espera 640x480
writer = cv2.VideoWriter("out.mp4", fourcc, 30, (640, 480))
writer.write(frame_1920x1080)  # escribe pero frame queda corrupto

# BIEN — siempre verificar o hacer resize
h, w = frame.shape[:2]
writer = cv2.VideoWriter("out.mp4", fourcc, 30, (w, h))  # (width, height), NO (height, width)
```

### 2. Memory leak en loops largos
```
Causa: No liberar VideoCapture, acumular frames en lista.
```
```python
# MAL — acumula TODOS los frames en RAM
frames = []
for frame in sv.get_video_frames_generator("video_4k.mp4"):
    frames.append(frame)  # 4K x 30fps x 60s = ~22 GB en RAM

# BIEN — procesar frame a frame sin acumular
for frame in sv.get_video_frames_generator("video_4k.mp4"):
    resultado = procesar(frame)  # frame anterior se libera automaticamente
```

### 3. Codec no disponible
```
Causa: Falta ffmpeg o codecs del sistema.
```
```python
# Verificar codecs disponibles:
# Si "mp4v" falla, probar "XVID" con extension .avi
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output.avi", fourcc, fps, (w, h))
if not writer.isOpened():
    print("ERROR: Codec no soportado. Instalar ffmpeg.")
```

### 4. FPS incorrecto en output
```
Causa: Usar stride en lectura pero no ajustar FPS de escritura.
```
```python
# Con stride=2, el video tiene la mitad de frames
# Si mantienes FPS original, el video sale al doble de velocidad
STRIDE = 2
adjusted_info = copy(video_info)
adjusted_info.fps = video_info.fps / STRIDE  # Mantener velocidad real

with sv.VideoSink("output.mp4", adjusted_info) as sink:
    for frame in sv.get_video_frames_generator("input.mp4", stride=STRIDE):
        sink.write_frame(frame)
```

### 5. (width, height) vs (height, width)
```
Causa: OpenCV usa (W, H) para VideoWriter pero NumPy shape es (H, W, C).
```
```python
h, w, c = frame.shape        # NumPy: (height, width, channels)
writer_size = (w, h)          # OpenCV VideoWriter: (width, height)
# NUNCA pasar frame.shape[:2] directo a VideoWriter — esta invertido
```

---

## Tips de Experto

1. **Skip frames para velocidad:** Usa `stride=N` para procesar N veces mas rapido. Ideal para videos donde no necesitas cada frame (ej: contar personas cada 5 frames).

2. **Reutilizar buffer de anotacion:** Usar `frame.copy()` solo cuando necesites el frame original intacto. Si no, anotar in-place ahorra memoria.

3. **Control de resolucion para inferencia rapida:**
   ```python
   # Procesar a baja resolucion, escribir a resolucion original
   small = cv2.resize(frame, (640, 360))
   results = model(small, verbose=False)
   # Escalar detecciones al tamano original antes de anotar
   ```

4. **Garbage collection en batch:** Llamar `gc.collect()` despues de procesar cada video en batch para liberar memoria del modelo y frames intermedios.

5. **Evitar `imshow` en servidores:** En VPS/SSH sin display, `cv2.imshow` lanza error. Usar solo VideoSink para guardar output.

6. **Verificar video antes de procesar:**
   ```python
   cap = cv2.VideoCapture("input.mp4")
   if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
       print("Video invalido o vacio")
   cap.release()
   ```

7. **try/finally para limpieza segura:**
   ```python
   cap = cv2.VideoCapture("input.mp4")
   try:
       while cap.isOpened():
           ret, frame = cap.read()
           if not ret:
               break
           # procesar...
   finally:
       cap.release()  # se ejecuta SIEMPRE, incluso con excepciones
   ```

8. **Hardware:** RTX A5000 (24 GB VRAM) permite modelos YOLO grandes (yolov8x) en frames 4K sin problemas. El bottleneck suele ser I/O de disco, no GPU.

---

## Siguiente Paso
- **Deteccion:** Ver `skills/object-detection.md` para configurar modelos YOLO
- **Tracking:** Ver `skills/tracking.md` para ByteTrack/BoT-SORT sobre video
- **Segmentacion:** Ver `skills/segmentation.md` para mascaras por frame
