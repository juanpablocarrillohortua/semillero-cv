# Skill: Pose Estimation

## Cuando Usar
- Detectar la postura corporal de personas en imagenes o video
- Analisis biomecanico: angulos articulares (codo, rodilla, cadera)
- Seguimiento de movimiento en deportes (correccion de tecnica, rep counting)
- Fisioterapia y rehabilitacion: medir rangos de movimiento
- Fitness: contar repeticiones, validar forma de ejercicios
- Interaccion humano-computador basada en gestos corporales

## Prerequisitos
```bash
pip install mediapipe ultralytics supervision opencv-python numpy
```

Hardware recomendado: RTX A5000 24GB, CUDA 12.2. MediaPipe funciona bien en CPU.

## Modelos Disponibles

### Ultralytics YOLO Pose (Multi-persona)

| Modelo | Params | mAP50-95 | Velocidad CPU | GPU | Uso |
|--------|--------|----------|---------------|-----|-----|
| `yolo11n-pose.pt` | 2.6M | 50.0 | ~56ms | ~2ms | Prototipado rapido, edge |
| `yolo11s-pose.pt` | 9.9M | 56.8 | ~98ms | ~3ms | Balance velocidad/precision |
| `yolo11m-pose.pt` | 20.9M | 62.8 | ~195ms | ~5ms | Produccion general |
| `yolo11l-pose.pt` | 26.2M | 64.2 | ~297ms | ~6ms | Alta precision |
| `yolo11x-pose.pt` | 58.8M | 65.3 | ~570ms | ~10ms | Maxima precision |
| `yolo26n-pose.pt` | ~3M | 50.0 | ~39ms | ~2ms | Ultima generacion, edge-first |
| `yolo26s-pose.pt` | ~10M | 57.5 | ~65ms | ~3ms | YOLO26 balance |
| `yolo26m-pose.pt` | ~21M | 63.5 | ~140ms | ~5ms | YOLO26 produccion |

> YOLO26 (2026) integra RLE (Residual Log-Likelihood Estimation) para mayor precision en keypoints.
> Ambas generaciones usan formato COCO: 17 keypoints, shape `[N, 17, 3]` (x, y, conf).

### MediaPipe Pose (Single-persona, 33 landmarks)

| Complejidad | Parametro | Precision | Velocidad | Uso |
|-------------|-----------|-----------|-----------|-----|
| Lite | `model_complexity=0` | Baja | Muy rapida | Mobile, real-time |
| Full | `model_complexity=1` | Media | Rapida | Default, webcam |
| Heavy | `model_complexity=2` | Alta | Moderada | Precision maxima, imagenes estaticas |

> MediaPipe detecta 33 landmarks 3D (x, y, z + visibility). Incluye dedos de manos/pies.
> Limitacion: solo UNA persona por frame. Para multi-persona usar YOLO Pose.

---

## Workflow: Pose con MediaPipe

```python
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Imagen estatica ---
image = cv2.imread("persona.jpg")

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5
) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        h, w, _ = image.shape

        # Acceder landmarks individuales
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        l_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        l_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Coordenadas en pixeles
        print(f"Nariz: ({nose.x * w:.0f}, {nose.y * h:.0f}), vis: {nose.visibility:.2f}")

        # World landmarks (en metros, centrado en cadera)
        if results.pose_world_landmarks:
            nose_w = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            print(f"Nariz 3D: x={nose_w.x:.3f}m, y={nose_w.y:.3f}m, z={nose_w.z:.3f}m")

        # Dibujar landmarks
        annotated = image.copy()
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        cv2.imwrite("pose_resultado.jpg", annotated)

        # Mascara de segmentacion (separar persona del fondo)
        if results.segmentation_mask is not None:
            mask = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg = np.zeros(image.shape, dtype=np.uint8)
            bg[:] = (40, 40, 40)
            segmented = np.where(mask, annotated, bg)
            cv2.imwrite("pose_segmentado.jpg", segmented)
```

### Extraer TODOS los landmarks como array NumPy
```python
def landmarks_to_array(pose_landmarks, image_shape):
    """Convierte landmarks de MediaPipe a array (33, 4): x, y, z, visibility."""
    h, w, _ = image_shape
    landmarks = []
    for lm in pose_landmarks.landmark:
        landmarks.append([lm.x * w, lm.y * h, lm.z, lm.visibility])
    return np.array(landmarks)

# Uso
pts = landmarks_to_array(results.pose_landmarks, image.shape)
# pts[11] = left_shoulder [x, y, z, vis]
# pts[13] = left_elbow    [x, y, z, vis]
```

---

## Workflow: Pose con Ultralytics

```python
from ultralytics import YOLO
import cv2
import torch

# Seleccionar device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo (descarga automatica la primera vez)
model = YOLO("yolo11m-pose.pt")  # o "yolo26n-pose.pt" para ultima generacion

# --- Inferencia en imagen ---
results = model("personas.jpg", device=device)

for result in results:
    # Keypoints: tensor shape [N_personas, 17, 3] (x, y, conf)
    keypoints = result.keypoints

    # Coordenadas en pixeles
    xy = keypoints.xy           # shape [N, 17, 2]
    xyn = keypoints.xyn         # shape [N, 17, 2] normalizado [0,1]
    conf = keypoints.conf       # shape [N, 17] confianza por keypoint
    data = keypoints.data       # shape [N, 17, 3] (x, y, conf)

    # Iterar por persona detectada
    for i in range(xy.shape[0]):
        persona_kpts = xy[i].cpu().numpy()   # (17, 2)
        persona_conf = conf[i].cpu().numpy() # (17,)

        # Acceder puntos especificos (indices COCO)
        nariz = persona_kpts[0]          # index 0: nose
        hombro_izq = persona_kpts[5]     # index 5: left_shoulder
        codo_izq = persona_kpts[7]       # index 7: left_elbow
        muneca_izq = persona_kpts[9]     # index 9: left_wrist
        rodilla_izq = persona_kpts[13]   # index 13: left_knee

        print(f"Persona {i}: nariz={nariz}, conf_nariz={persona_conf[0]:.2f}")

    # Visualizar con anotaciones built-in
    annotated = result.plot()
    cv2.imwrite("yolo_pose.jpg", annotated)
```

### Inferencia por lotes
```python
# Multiples imagenes
results = model(["img1.jpg", "img2.jpg", "img3.jpg"], device=device)

# Video completo (genera video anotado)
results = model("video.mp4", device=device, save=True, show=False)
```

---

## Workflow: Calculo de Angulos Articulares

Funcion central para biomecanica: calcular el angulo entre 3 puntos articulares.

```python
import numpy as np

def calcular_angulo(p1, p2, p3):
    """
    Calcula el angulo en el punto p2 formado por los segmentos p1-p2 y p2-p3.

    Args:
        p1, p2, p3: arrays (x, y) o (x, y, z). p2 es el vertice del angulo.

    Returns:
        Angulo en grados [0, 180].

    Ejemplo: angulo del codo = calcular_angulo(hombro, codo, muneca)
    """
    p1, p2, p3 = np.array(p1[:2]), np.array(p2[:2]), np.array(p3[:2])

    v1 = p1 - p2  # vector p2 -> p1
    v2 = p3 - p2  # vector p2 -> p3

    # Producto punto y magnitudes
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # evitar errores numericos

    angle = np.degrees(np.arccos(cos_angle))
    return angle


def calcular_angulo_3d(p1, p2, p3):
    """Version 3D para world landmarks de MediaPipe (coordenadas en metros)."""
    p1, p2, p3 = np.array(p1[:3]), np.array(p2[:3]), np.array(p3[:3])

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))
```

### Angulos articulares comunes (indices COCO para YOLO)

```python
# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

def angulos_biomecanicos(kpts, min_conf=0.5, conf=None):
    """
    Calcula angulos articulares principales.

    Args:
        kpts: array (17, 2) o (17, 3) con coordenadas de keypoints COCO.
        min_conf: confianza minima para considerar un keypoint valido.
        conf: array (17,) con confianza por keypoint (opcional).

    Returns:
        dict con angulos en grados. None si keypoints insuficientes.
    """
    def _valido(*indices):
        if conf is None:
            return True
        return all(conf[i] >= min_conf for i in indices)

    angulos = {}

    # Codo izquierdo: hombro-codo-muneca
    if _valido(L_SHOULDER, L_ELBOW, L_WRIST):
        angulos["codo_izq"] = calcular_angulo(kpts[L_SHOULDER], kpts[L_ELBOW], kpts[L_WRIST])

    # Codo derecho
    if _valido(R_SHOULDER, R_ELBOW, R_WRIST):
        angulos["codo_der"] = calcular_angulo(kpts[R_SHOULDER], kpts[R_ELBOW], kpts[R_WRIST])

    # Rodilla izquierda: cadera-rodilla-tobillo
    if _valido(L_HIP, L_KNEE, L_ANKLE):
        angulos["rodilla_izq"] = calcular_angulo(kpts[L_HIP], kpts[L_KNEE], kpts[L_ANKLE])

    # Rodilla derecha
    if _valido(R_HIP, R_KNEE, R_ANKLE):
        angulos["rodilla_der"] = calcular_angulo(kpts[R_HIP], kpts[R_KNEE], kpts[R_ANKLE])

    # Hombro izquierdo: codo-hombro-cadera
    if _valido(L_ELBOW, L_SHOULDER, L_HIP):
        angulos["hombro_izq"] = calcular_angulo(kpts[L_ELBOW], kpts[L_SHOULDER], kpts[L_HIP])

    # Hombro derecho
    if _valido(R_ELBOW, R_SHOULDER, R_HIP):
        angulos["hombro_der"] = calcular_angulo(kpts[R_ELBOW], kpts[R_SHOULDER], kpts[R_HIP])

    # Cadera izquierda: hombro-cadera-rodilla
    if _valido(L_SHOULDER, L_HIP, L_KNEE):
        angulos["cadera_izq"] = calcular_angulo(kpts[L_SHOULDER], kpts[L_HIP], kpts[L_KNEE])

    # Cadera derecha
    if _valido(R_SHOULDER, R_HIP, R_KNEE):
        angulos["cadera_der"] = calcular_angulo(kpts[R_SHOULDER], kpts[R_HIP], kpts[R_KNEE])

    # Inclinacion del torso: midpoint_hombros - midpoint_caderas vs vertical
    if _valido(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        mid_hombros = (kpts[L_SHOULDER][:2] + kpts[R_SHOULDER][:2]) / 2
        mid_caderas = (kpts[L_HIP][:2] + kpts[R_HIP][:2]) / 2
        torso_vec = mid_hombros - mid_caderas
        vertical = np.array([0, -1])  # y apunta hacia arriba en imagen invertida
        cos_a = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-8)
        angulos["inclinacion_torso"] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    return angulos
```

### Ejemplo completo: YOLO + angulos
```python
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11m-pose.pt")
results = model("sentadilla.jpg", device=device)

for r in results:
    for i in range(r.keypoints.xy.shape[0]):
        kpts = r.keypoints.xy[i].cpu().numpy()      # (17, 2)
        confs = r.keypoints.conf[i].cpu().numpy()    # (17,)

        angulos = angulos_biomecanicos(kpts, min_conf=0.5, conf=confs)
        print(f"Persona {i}:")
        for nombre, grados in angulos.items():
            print(f"  {nombre}: {grados:.1f} grados")
```

---

## Workflow: Pose en Video / Webcam

### Con YOLO (multi-persona, recomendado)
```python
from ultralytics import YOLO
import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11s-pose.pt")

cap = cv2.VideoCapture(0)  # 0 = webcam, o "video.mp4"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device, verbose=False)
    annotated = results[0].plot()

    # Overlay de angulos en tiempo real
    if results[0].keypoints is not None and results[0].keypoints.xy.shape[0] > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()
        angulos = angulos_biomecanicos(kpts, conf=confs)

        y_offset = 30
        for nombre, grados in angulos.items():
            texto = f"{nombre}: {grados:.0f} deg"
            cv2.putText(annotated, texto, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

    cv2.imshow("YOLO Pose", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Con MediaPipe (single-persona, menor latencia en CPU)
```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optimizacion: marcar como no-escribible durante procesamiento
        frame.flags.writeable = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        frame.flags.writeable = True

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.imshow("MediaPipe Pose", cv2.flip(frame, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
```

---

## Keypoints de Referencia

### Formato COCO (17 keypoints) -- YOLO Pose

| Index | Nombre | Parte del cuerpo |
|-------|--------|-----------------|
| 0 | nose | Cara |
| 1 | left_eye | Cara |
| 2 | right_eye | Cara |
| 3 | left_ear | Cara |
| 4 | right_ear | Cara |
| 5 | left_shoulder | Torso superior |
| 6 | right_shoulder | Torso superior |
| 7 | left_elbow | Brazo izquierdo |
| 8 | right_elbow | Brazo derecho |
| 9 | left_wrist | Brazo izquierdo |
| 10 | right_wrist | Brazo derecho |
| 11 | left_hip | Torso inferior |
| 12 | right_hip | Torso inferior |
| 13 | left_knee | Pierna izquierda |
| 14 | right_knee | Pierna derecha |
| 15 | left_ankle | Pierna izquierda |
| 16 | right_ankle | Pierna derecha |

> Cada keypoint tiene 3 valores: `(x, y, confidence)`. Confidence < 0.5 indica keypoint no visible.

### Formato MediaPipe (33 landmarks)

| Index | Nombre | Parte |
|-------|--------|-------|
| 0 | NOSE | Cara |
| 1-6 | LEFT/RIGHT_EYE (inner, center, outer) | Cara |
| 7-8 | LEFT/RIGHT_EAR | Cara |
| 9-10 | MOUTH_LEFT / MOUTH_RIGHT | Cara |
| 11-12 | LEFT/RIGHT_SHOULDER | Torso superior |
| 13-14 | LEFT/RIGHT_ELBOW | Brazos |
| 15-16 | LEFT/RIGHT_WRIST | Brazos |
| 17-18 | LEFT/RIGHT_PINKY | Manos |
| 19-20 | LEFT/RIGHT_INDEX | Manos |
| 21-22 | LEFT/RIGHT_THUMB | Manos |
| 23-24 | LEFT/RIGHT_HIP | Torso inferior |
| 25-26 | LEFT/RIGHT_KNEE | Piernas |
| 27-28 | LEFT/RIGHT_ANKLE | Piernas |
| 29-30 | LEFT/RIGHT_HEEL | Pies |
| 31-32 | LEFT/RIGHT_FOOT_INDEX | Pies |

> MediaPipe incluye `visibility` (0-1) y coordenada Z (profundidad relativa al centro de cadera).
> Indices 0-10: cara, 11-22: torso/brazos, 23-32: piernas/pies.

### Mapeo COCO -> MediaPipe (para conversion)
```python
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose -> NOSE
    1: 2,    # left_eye -> LEFT_EYE
    2: 5,    # right_eye -> RIGHT_EYE
    3: 7,    # left_ear -> LEFT_EAR
    4: 8,    # right_ear -> RIGHT_EAR
    5: 11,   # left_shoulder -> LEFT_SHOULDER
    6: 12,   # right_shoulder -> RIGHT_SHOULDER
    7: 13,   # left_elbow -> LEFT_ELBOW
    8: 14,   # right_elbow -> RIGHT_ELBOW
    9: 15,   # left_wrist -> LEFT_WRIST
    10: 16,  # right_wrist -> RIGHT_WRIST
    11: 23,  # left_hip -> LEFT_HIP
    12: 24,  # right_hip -> RIGHT_HIP
    13: 25,  # left_knee -> LEFT_KNEE
    14: 26,  # right_knee -> RIGHT_KNEE
    15: 27,  # left_ankle -> LEFT_ANKLE
    16: 28,  # right_ankle -> RIGHT_ANKLE
}
```

---

## Integracion con Supervision

```python
import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11m-pose.pt")
image = cv2.imread("personas.jpg")
results = model(image)[0]

# Convertir resultados a KeyPoints de Supervision
keypoints = sv.KeyPoints.from_ultralytics(results)

# Annotadores de keypoints
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex("#00FF00"),
    radius=6
)
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex("#FF6600"),
    thickness=2
)

# Anotar imagen
annotated = image.copy()
annotated = edge_annotator.annotate(scene=annotated, key_points=keypoints)
annotated = vertex_annotator.annotate(scene=annotated, key_points=keypoints)

cv2.imwrite("supervision_pose.jpg", annotated)
```

### Con etiquetas por vertex
```python
vertex_label_annotator = sv.VertexLabelAnnotator(
    color=sv.Color.from_hex("#FFFFFF"),
    text_color=sv.Color.from_hex("#000000"),
    border_radius=4
)

# Nombres COCO para labels
COCO_KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle"
]

# Generar labels: lista de listas (una por persona)
labels = [COCO_KEYPOINT_NAMES for _ in range(len(keypoints))]

annotated = vertex_label_annotator.annotate(
    scene=annotated,
    key_points=keypoints,
    labels=labels
)
```

### Tracking de keypoints en video
```python
tracker = sv.ByteTrack()

# En cada frame:
keypoints = sv.KeyPoints.from_ultralytics(results)
detections = keypoints.as_detections()          # convertir a Detections para tracker
tracked = tracker.update_with_detections(detections)
# tracked.tracker_id contiene IDs consistentes entre frames
```

---

## MediaPipe vs Ultralytics: Cuando Usar Cual

| Criterio | MediaPipe Pose | YOLO Pose (11/26) |
|----------|----------------|-------------------|
| **Personas** | 1 sola | Multi-persona |
| **Keypoints** | 33 landmarks (3D) | 17 COCO (2D + conf) |
| **GPU requerida** | No (optimizado CPU) | Recomendada (CUDA) |
| **Latencia CPU** | ~15-30ms | ~40-570ms segun modelo |
| **Latencia GPU** | N/A (no la aprovecha) | ~2-10ms |
| **Segmentacion** | Si (mascara de persona) | No incluida |
| **Coordenadas 3D** | Si (world landmarks en metros) | No (solo 2D) |
| **Detalle manos/pies** | Si (dedos, talon) | No (solo muneca/tobillo) |
| **Tracking integrado** | Si (detector-tracker interno) | Requiere ByteTrack/BoTSORT |
| **Precision** | Media-alta (single person) | Alta (especialmente YOLO26 con RLE) |
| **Mejor para** | Fitness apps, mobile, apps de una persona | Deportes, multi-persona, analisis de equipo |

### Regla practica
- **1 persona + CPU + app mobile/web** -> MediaPipe
- **Multiples personas + GPU + deportes/analisis** -> YOLO Pose
- **Necesitas coordenadas 3D o segmentacion** -> MediaPipe
- **Necesitas tracking multi-persona** -> YOLO Pose + Supervision ByteTrack

---

## Parametros Clave

### YOLO Pose
```python
results = model(
    source="imagen.jpg",
    device="cuda",          # "cuda" o "cpu"
    conf=0.25,              # confianza minima deteccion persona
    iou=0.7,                # threshold NMS
    imgsz=640,              # tamanio de entrada (multiplo de 32)
    half=True,              # FP16 en GPU (2x mas rapido, minima perdida)
    max_det=20,             # maximo personas a detectar
    verbose=False,          # silenciar logs en real-time
)
```

### MediaPipe Pose
```python
mp_pose.Pose(
    static_image_mode=False,         # True para imagenes independientes
    model_complexity=1,              # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,           # suavizado temporal (solo video)
    enable_segmentation=False,       # mascara de persona
    smooth_segmentation=True,        # suavizado de mascara
    min_detection_confidence=0.5,    # umbral deteccion inicial
    min_tracking_confidence=0.5,     # umbral re-deteccion entre frames
)
```

---

## Errores Comunes

| Error | Causa | Solucion |
|-------|-------|----------|
| `keypoints.xy` vacio | No se detecto ninguna persona | Verificar `results[0].keypoints is not None and results[0].keypoints.xy.shape[0] > 0` |
| Angulos de 0 o 180 siempre | Keypoints colineales o coordenadas iguales | Filtrar por `conf >= 0.5` antes de calcular |
| `RuntimeError: CUDA out of memory` | Modelo muy grande para el frame | Reducir `imgsz` o usar modelo mas pequenio (`n` o `s`) |
| MediaPipe detecta landmarks fantasma | Persona parcialmente visible, ocluida | Filtrar por `visibility > 0.5` |
| Angulos inestables en video | Ruido frame-a-frame en keypoints | Aplicar suavizado temporal (moving average o EMA) |
| `IndexError` al acceder keypoints | Persona no detectada en ese frame | Siempre verificar existencia antes de indexar |
| MediaPipe no usa GPU | Disenio intencional (optimizado para CPU) | No es un error; usar YOLO si necesitas GPU |

---

## Tips de Experto

### Biomecanica
1. **Angulos 2D vs 3D**: los angulos calculados desde video 2D son PROYECCIONES. Un codo doblado a 90 grados visto de frente parecera ~180. Siempre que sea posible usar world landmarks 3D de MediaPipe o multiples camaras.

2. **Plano de movimiento importa**: en biomecanica, los angulos se definen en planos anatomicos (sagital, frontal, transversal). Un angulo de rodilla en squats necesita vista lateral (plano sagital). Vista frontal mide valgo/varo.

3. **Rangos de referencia** (adulto sano, plano sagital):
   - Extension de rodilla: 0-5 grados (pierna recta)
   - Flexion de rodilla: 0-135 grados
   - Flexion de codo: 0-145 grados
   - Flexion de hombro (elevacion): 0-180 grados
   - Flexion de cadera: 0-120 grados

4. **No confundir angulo articular con angulo geometrico**: el angulo del vector es 0-180, pero en biomecanica, una rodilla completamente extendida es 180 grados geometricos = 0 grados de flexion. Ajustar: `flexion = 180 - angulo_geometrico`.

### Estabilidad y Precision
5. **Suavizado temporal EMA** (Exponential Moving Average) para video:
```python
class KeypointSmoother:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev = None

    def smooth(self, keypoints):
        if self.prev is None:
            self.prev = keypoints.copy()
            return keypoints
        smoothed = self.alpha * keypoints + (1 - self.alpha) * self.prev
        self.prev = smoothed.copy()
        return smoothed

# Uso
smoother = KeypointSmoother(alpha=0.7)  # alpha alto = mas reactivo, bajo = mas suave
kpts_smooth = smoother.smooth(kpts_raw)
```

6. **Keypoints ocluidos**: NUNCA calcular angulos con keypoints de baja confianza (< 0.5). Un keypoint ocluido puede estar en cualquier posicion y producira angulos sin sentido.

7. **Multi-persona**: YOLO Pose devuelve keypoints sin orden garantizado entre frames. Usar tracking (ByteTrack) para mantener la identidad de cada persona.

8. **Resolucion de entrada**: `imgsz=640` es el default y funciona bien para la mayoria de casos. Para personas lejanas (estadio, campo), subir a `imgsz=1280` mejora la deteccion de keypoints distantes.

9. **Calibracion de camara**: para mediciones metricas reales (distancias en cm, velocidades en m/s), se requiere calibracion intrinseca de la camara. Sin calibracion, solo se pueden calcular proporciones y angulos.

10. **FPS en produccion**: usar `half=True` (FP16) en GPU duplica el throughput con perdida de precision despreciable. Combinar con `imgsz=640` y modelo `s` o `n` para superar 60 FPS.

---

## Siguiente Paso
- **Tracking**: ver `skills/tracking.md` para seguimiento de personas entre frames
- **Deportes**: combinar pose + tracking para analisis de tecnica deportiva
- **Training custom**: entrenar modelo YOLO Pose en keypoints personalizados (ver `skills/training.md`)
- **Segmentacion**: ver `skills/segmentation.md` para aislar personas del fondo
- **Video pipeline**: ver `skills/video-pipeline.md` para procesar videos completos
