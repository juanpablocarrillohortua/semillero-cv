# Modelos

Directorio centralizado para almacenar modelos descargados. Esto evita que cada proyecto descargue su propia copia.

## Configuracion

Para que Ultralytics descargue modelos en este directorio:

```bash
# Agregar al inicio de la sesion (o en .env)
export YOLO_CONFIG_DIR="$(pwd)/models"
```

Esto hace que `YOLO("yolov8n.pt")` busque y descargue el modelo en `models/` en lugar de `~/.config/Ultralytics/`.

## Estructura esperada

```
models/
├── yolo11n.pt              # Deteccion nano (rapido)
├── yolo11s.pt              # Deteccion small
├── yolo11m.pt              # Deteccion medium
├── yolo11n-pose.pt         # Pose estimation nano
├── yolo11n-seg.pt          # Segmentacion nano
├── sam2.1_b.pt             # SAM 2.1 base
└── README.md               # Este archivo
```

## Modelos recomendados para empezar

| Modelo | Tarea | Tamano | Descarga |
|--------|-------|--------|----------|
| `yolo11n.pt` | Deteccion | ~6 MB | `YOLO("yolo11n.pt")` |
| `yolo11s.pt` | Deteccion | ~22 MB | `YOLO("yolo11s.pt")` |
| `yolo11n-pose.pt` | Pose | ~6 MB | `YOLO("yolo11n-pose.pt")` |
| `yolo11n-seg.pt` | Segmentacion | ~6 MB | `YOLO("yolo11n-seg.pt")` |

Los modelos se descargan automaticamente la primera vez que se usan. No es necesario descargarlos manualmente.

## IMPORTANTE

- Los archivos `.pt`, `.pth`, `.onnx` estan en `.gitignore` — NO se suben a git
- Cada PC del lab descarga sus modelos localmente
- Si la red esta lenta, pedir al profesor los modelos en USB
