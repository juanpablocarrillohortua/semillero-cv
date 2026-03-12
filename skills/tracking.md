# Skill: Multi-Object Tracking (MOT)

## Cuando Usar

- Seguir objetos individuales a traves de frames de video (jugadores, vehiculos, personas)
- Contar objetos que cruzan una linea o entran a una zona
- Generar trayectorias, heatmaps, o estadisticas por objeto
- Analisis deportivo: distancia recorrida, velocidad, posesion zonal

## Prerequisitos

```bash
pip install ultralytics supervision
# GPU: RTX A5000 24GB, CUDA 12.2
# ultralytics >= 8.3 (soporta YOLO11/YOLO26)
# supervision >= 0.25
```

## Trackers Disponibles

| Caracteristica | ByteTrack | BoT-SORT |
|----------------|-----------|----------|
| Asociacion | Solo IoU | IoU + apariencia (ReID) + CMC |
| Velocidad | Mas rapido | ~10-15% mas lento |
| ID Switches | Mas en oclusiones | Menos (usa ReID) |
| Configuracion | Sencilla | Requiere modelo ReID |
| Mejor para | Escenas simples, camaras fijas | Deportes, multitud, oclusiones |
| Default en Ultralytics | No (hay que especificar) | Si (default) |

**Regla:** Para deportes o escenas con oclusiones frecuentes, usar **BoT-SORT**. Para conteo simple o trafico, **ByteTrack** es suficiente.

## Workflow: Tracking con Ultralytics

`model.track()` integra deteccion + tracking en una sola llamada.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # o yolo11s.pt para mejor precision

# --- Opcion 1: ByteTrack ---
results = model.track(
    source="partido.mp4",
    tracker="bytetrack.yaml",  # archivo de config
    persist=True,              # mantener IDs entre frames
    conf=0.3,                  # umbral de confianza
    iou=0.5,                   # umbral IoU para NMS
    show=True                  # visualizar en tiempo real
)

# --- Opcion 2: BoT-SORT (default, mejor para deportes) ---
results = model.track(
    source="partido.mp4",
    tracker="botsort.yaml",
    persist=True,
    conf=0.3,
    stream=True   # modo streaming para videos largos (generator)
)

# Acceder a track IDs
for result in results:
    boxes = result.boxes
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()     # [1, 3, 7, ...]
        classes = boxes.cls.int().cpu().tolist()       # [0, 0, 32, ...]
        confs = boxes.conf.cpu().tolist()              # [0.92, 0.87, ...]
        xywh = boxes.xywh.cpu().numpy()               # centroides + dims

        for tid, cls, conf in zip(track_ids, classes, confs):
            print(f"Track {tid}: clase={cls}, conf={conf:.2f}")
```

### Configuracion YAML Personalizada

Crear `custom_bytetrack.yaml`:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25   # umbral 1ra asociacion (subir = tracks mas limpios)
track_low_thresh: 0.1     # umbral 2da asociacion (detecciones debiles)
new_track_thresh: 0.25    # conf minima para iniciar track nuevo
track_buffer: 30          # frames que un track "perdido" sobrevive
match_thresh: 0.8         # similitud minima para asociar (IoU)
fuse_score: True          # fusionar score de deteccion con IoU
```

Crear `custom_botsort.yaml`:

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: True
# --- Exclusivo BoT-SORT ---
gmc_method: sparseOptFlow  # compensacion de movimiento de camara
proximity_thresh: 0.5      # umbral de proximidad espacial
appearance_thresh: 0.25    # umbral de similitud de apariencia (ReID)
with_reid: False           # True para usar modelo ReID externo
```

Uso: `model.track(source="video.mp4", tracker="custom_botsort.yaml")`

## Workflow: Tracking con Supervision

Supervision da control total sobre el pipeline: detectar, trackear, anotar, escribir.

```python
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,   # conf para activar track
    lost_track_buffer=30,              # frames sin deteccion antes de borrar
    minimum_matching_threshold=0.8,    # similitud minima (IoU)
    frame_rate=30,                     # FPS del video
    minimum_consecutive_frames=3       # frames consecutivos para confirmar track
)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)
trace_annotator = sv.TraceAnnotator(
    position=sv.Position.CENTER,
    trace_length=60,    # longitud de la estela (en frames)
    thickness=2
)

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {class_name}"
        for class_name, tracker_id
        in zip(detections.data["class_name"], detections.tracker_id)
    ]

    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    annotated = trace_annotator.annotate(annotated, detections)
    return annotated

sv.process_video(
    source_path="partido.mp4",
    target_path="resultado_tracking.mp4",
    callback=callback
)
```

### Pipeline Manual con VideoSink (mas control)

```python
video_info = sv.VideoInfo.from_video_path("partido.mp4")
frame_generator = sv.get_video_frames_generator("partido.mp4")

with sv.VideoSink("resultado.mp4", video_info) as sink:
    for frame in frame_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Filtrar solo personas (class_id=0 en COCO)
        detections = detections[detections.class_id == 0]

        annotated = trace_annotator.annotate(frame.copy(), detections)
        sink.write_frame(annotated)
```

## Workflow: Conteo por Zona

### Conteo con PolygonZone (area cerrada)

```python
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()

# Definir poligono (coordenadas del area de interes)
polygon = np.array([
    [200, 400],   # punto 1
    [800, 400],   # punto 2
    [800, 700],   # punto 3
    [200, 700]    # punto 4
])

zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=[sv.Position.BOTTOM_CENTER]
)
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.RED,
    thickness=4
)

video_info = sv.VideoInfo.from_video_path("video.mp4")
frames = sv.get_video_frames_generator("video.mp4")

with sv.VideoSink("conteo_zona.mp4", video_info) as sink:
    for frame in frames:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        is_in_zone = zone.trigger(detections)
        count = zone.current_count

        annotated = zone_annotator.annotate(frame.copy())
        sv.draw_text(
            scene=annotated,
            text=f"En zona: {count}",
            text_anchor=sv.Point(x=50, y=50),
            text_scale=1.5,
            text_color=sv.Color.WHITE
        )
        sink.write_frame(annotated)
```

### Conteo con LineZone (linea de cruce IN/OUT)

```python
import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()

# Definir linea de conteo (punto inicio, punto fin)
line_start = sv.Point(x=0, y=500)
line_end = sv.Point(x=1920, y=500)

line_zone = sv.LineZone(
    start=line_start,
    end=line_end,
    triggering_anchors=[sv.Position.BOTTOM_CENTER],
    minimum_crossing_threshold=2   # frames para confirmar cruce (anti-jitter)
)
line_annotator = sv.LineZoneAnnotator(thickness=4)

video_info = sv.VideoInfo.from_video_path("video.mp4")
frames = sv.get_video_frames_generator("video.mp4")

with sv.VideoSink("conteo_linea.mp4", video_info) as sink:
    for frame in frames:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        crossed_in, crossed_out = line_zone.trigger(detections)

        annotated = line_annotator.annotate(frame.copy(), line_counter=line_zone)
        print(f"IN: {line_zone.in_count} | OUT: {line_zone.out_count}")
        sink.write_frame(annotated)
```

## Workflow: Trayectorias y Heatmaps

### Heatmap de densidad

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()

heatmap_annotator = sv.HeatMapAnnotator(
    position=sv.Position.BOTTOM_CENTER,
    opacity=0.6,
    radius=40,
    kernel_size=25
)

video_info = sv.VideoInfo.from_video_path("partido.mp4")
frames = sv.get_video_frames_generator("partido.mp4")

with sv.VideoSink("heatmap.mp4", video_info) as sink:
    for frame in frames:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        # Solo personas
        detections = detections[detections.class_id == 0]

        annotated = heatmap_annotator.annotate(frame.copy(), detections)
        sink.write_frame(annotated)
```

### Extraer trayectorias para analisis

```python
from collections import defaultdict
import numpy as np

track_history = defaultdict(list)  # {track_id: [(x, y, frame_num), ...]}

video_info = sv.VideoInfo.from_video_path("partido.mp4")
frames = sv.get_video_frames_generator("partido.mp4")

for frame_num, frame in enumerate(frames):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is not None:
        for tid, xyxy in zip(detections.tracker_id, detections.xyxy):
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            track_history[tid].append((cx, cy, frame_num))

# Calcular distancia recorrida por cada track (en pixeles)
for tid, points in track_history.items():
    coords = np.array([(p[0], p[1]) for p in points])
    if len(coords) > 1:
        diffs = np.diff(coords, axis=0)
        dist = np.sum(np.linalg.norm(diffs, axis=1))
        duration = (points[-1][2] - points[0][2]) / video_info.fps
        speed = dist / max(duration, 1e-6)  # px/seg
        print(f"Track {tid}: {dist:.0f}px, {duration:.1f}s, vel={speed:.1f}px/s")
```

Para convertir pixeles a metros, usar homografia:

```python
import cv2

# 4 puntos conocidos en imagen -> coordenadas reales (metros)
pts_img = np.float32([[100, 200], [800, 200], [800, 600], [100, 600]])
pts_real = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])  # cancha futbol

H, _ = cv2.findHomography(pts_img, pts_real)

# Transformar punto de pixeles a metros
def px_to_meters(x, y, H):
    pt = np.array([x, y, 1.0])
    result = H @ pt
    return result[0] / result[2], result[1] / result[2]
```

## Metricas de Tracking

| Metrica | Que mide | Rango | Ideal |
|---------|----------|-------|-------|
| **MOTA** | Precision global: penaliza FP, FN, ID switches | -inf a 1.0 | > 0.6 |
| **MOTP** | Precision de localizacion (IoU promedio de TP) | 0 a 1.0 | > 0.7 |
| **IDF1** | Consistencia de identidad (F1 de asociacion) | 0 a 1.0 | > 0.5 |
| **HOTA** | Metrica unificada (deteccion + asociacion) | 0 a 1.0 | > 0.5 |
| **ID Sw** | Cambios de identidad (menor = mejor) | 0 a inf | < 100 |

### Calcular metricas

```bash
pip install motmetrics
```

```python
import motmetrics as mm

acc = mm.MOTAccumulator(auto_id=True)

# Por cada frame: ground truth vs predicciones
# gt_ids: list[int], pred_ids: list[int], distances: np.array (IoU matrix)
for frame_idx in range(num_frames):
    gt_ids = [...]     # IDs ground truth presentes
    pred_ids = [...]   # IDs predichos presentes
    # Matriz de distancia (1 - IoU) entre gt y pred
    dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
    acc.update(gt_ids, pred_ids, dists)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=["mota", "motp", "idf1", "num_switches"])
print(mm.io.render_summary(summary))
```

## Parametros Clave

### Supervision sv.ByteTrack

| Parametro | Default | Efecto |
|-----------|---------|--------|
| `track_activation_threshold` | 0.25 | Conf minima para iniciar track. Subir = menos falsos positivos |
| `lost_track_buffer` | 30 | Frames sin deteccion antes de eliminar track. Subir = tolera oclusiones |
| `minimum_matching_threshold` | 0.8 | Similitud IoU para asociar. Bajar = mas flexible, mas ID switches |
| `frame_rate` | 30 | FPS del video. Afecta calculo interno de buffers |
| `minimum_consecutive_frames` | 1 | Frames consecutivos para confirmar track. Subir = menos falsos tracks |

### Ultralytics bytetrack.yaml / botsort.yaml

| Parametro | Default | Efecto |
|-----------|---------|--------|
| `track_high_thresh` | 0.25 | 1ra ronda de asociacion. Subir = tracks mas limpios |
| `track_low_thresh` | 0.1 | 2da ronda (detecciones debiles). Recupera objetos parcialmente ocultos |
| `new_track_thresh` | 0.25 | Crear track nuevo. Subir = menos tracks espurios |
| `track_buffer` | 30 | Frames de supervivencia sin deteccion |
| `match_thresh` | 0.8 | Umbral IoU de asociacion |
| `gmc_method` | sparseOptFlow | Solo BoT-SORT: compensacion movimiento camara |
| `appearance_thresh` | 0.25 | Solo BoT-SORT: similitud de apariencia (ReID) |

## Errores Comunes

### ID Switches frecuentes

**Causa:** Objetos se cruzan, oclusiones, detecciones intermitentes.

```yaml
# Solucion: aumentar buffer y bajar match_thresh
track_buffer: 60          # era 30
match_thresh: 0.7         # era 0.8 (mas permisivo)
```

En Supervision:
```python
tracker = sv.ByteTrack(
    lost_track_buffer=60,
    minimum_matching_threshold=0.7,
    minimum_consecutive_frames=3  # filtra tracks espurios
)
```

### Tracks perdidos (objetos desaparecen)

**Causa:** `conf` muy alto, o `track_activation_threshold` muy alto.

```python
# Solucion: bajar umbrales
model.track(source="video.mp4", conf=0.15, tracker="bytetrack.yaml")
# O en supervision:
tracker = sv.ByteTrack(track_activation_threshold=0.15)
```

### Tracks fantasma (IDs en zonas vacias)

**Causa:** `minimum_consecutive_frames` muy bajo, detecciones espurias.

```python
tracker = sv.ByteTrack(
    minimum_consecutive_frames=5,    # necesita 5 frames seguidos para confirmar
    track_activation_threshold=0.4   # subir umbral
)
```

### IDs que no persisten entre frames (Ultralytics)

**Causa:** Falta `persist=True`.

```python
# INCORRECTO:
results = model.track(source="video.mp4")

# CORRECTO:
results = model.track(source="video.mp4", persist=True)
```

### Conteo doble en LineZone

**Causa:** Jitter en detecciones cerca de la linea.

```python
line_zone = sv.LineZone(
    start=start, end=end,
    minimum_crossing_threshold=3  # requiere 3 frames para confirmar cruce
)
```

## Tips de Experto

### Deportes: Reducir ID Switches

1. **Usar BoT-SORT con ReID** para partidos con contacto fisico:
   ```yaml
   tracker_type: botsort
   with_reid: True
   appearance_thresh: 0.3   # ajustar segun deporte
   track_buffer: 90         # 3 segundos a 30fps
   ```

2. **No saltar frames** (`vid_stride=1`). El tracker necesita continuidad:
   ```python
   model.track(source="video.mp4", vid_stride=1, persist=True)
   ```

3. **Modelo mas grande = mejores detecciones = mejor tracking:**
   ```python
   # yolo11n -> ~8 ID switches/min en futbol
   # yolo11m -> ~3 ID switches/min en futbol
   # yolo11l -> ~1.5 ID switches/min en futbol
   model = YOLO("yolo11m.pt")  # sweet spot precision/velocidad
   ```

4. **Pre-filtrar clases** para reducir ruido:
   ```python
   results = model.track(source="video.mp4", classes=[0], persist=True)
   # classes=[0] = solo personas en COCO
   ```

### Optimizacion de Velocidad

- `yolo11n` + ByteTrack: ~120 FPS en RTX A5000
- `yolo11m` + BoT-SORT: ~45 FPS en RTX A5000
- Para video offline, usar batch mayor no ayuda (tracking es secuencial)
- `half=True` para FP16:
  ```python
  model.track(source="video.mp4", half=True, persist=True)
  ```

### Multiples Camaras

Para tracking cross-camera, necesitas ReID externo:

```python
# BoT-SORT con modelo ReID
# botsort.yaml:
#   with_reid: True
#   model: "osnet_x0_25_msmt17.pt"  # modelo ReID ligero
```

### Homografia para Metricas Reales

En analisis deportivo, siempre calibrar la camara:

1. Marcar 4+ puntos de referencia en el campo (esquinas, circulos)
2. Calcular homografia con `cv2.findHomography()`
3. Transformar todas las trayectorias a coordenadas reales
4. Calcular distancia en metros, velocidad en km/h

### Supervision + Ultralytics: Cuando Usar Cada Uno

| Escenario | Recomendacion |
|-----------|---------------|
| Prototipo rapido, visualizar | `model.track()` de Ultralytics |
| Conteo por zonas/lineas | Supervision (PolygonZone, LineZone) |
| Anotaciones personalizadas | Supervision (multiples annotators) |
| Heatmaps, trayectorias | Supervision (HeatMapAnnotator, TraceAnnotator) |
| Streaming en produccion | `model.track(stream=True)` |
| Pipeline complejo multi-paso | Supervision (detect -> filter -> track -> annotate) |

## Siguiente Paso

- **Pose + Tracking:** Combinar `yolo11n-pose.pt` con tracking para analisis biomecanico
- **Segmentacion + Tracking:** `yolo11n-seg.pt` para tracks con mascaras
- **Fine-tune detector:** Entrenar en tu dominio (skill `training.md`) antes de trackear
- **Dashboard:** Exportar trayectorias a CSV/JSON para visualizar en Streamlit o Plotly
