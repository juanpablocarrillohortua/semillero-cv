# Workshop 01 — MediaPipe Explorer

**Semillero de Investigacion en Vision por Computador**
Universidad Externado de Colombia — 2026

---

## Que es esto?

Una app web que permite explorar 6 modelos de vision por computador de MediaPipe en tiempo real, directamente en el browser. No requiere instalacion de Python ni librerias — todo corre en WebAssembly.

## Modelos disponibles

| Modelo | Que detecta | Landmarks |
|--------|-------------|-----------|
| **Pose Estimation** | Cuerpo completo | 33 puntos 3D |
| **Hand Tracking** | Manos (hasta 2) | 21 puntos por mano |
| **Face Mesh** | Rostro | 478 puntos 3D |
| **Gesture Recognition** | Gestos de mano | Gesto + landmarks |
| **Object Detection** | 80 categorias COCO | Bounding boxes |
| **Segmentation** | Persona vs fondo | Mascara pixel a pixel |

## Como correrlo

### Opcion 1: Python (recomendado)

```bash
cd workshops/01-mediapipe-explorer
python -m http.server 8000
```

Abrir en el browser: `http://localhost:8000`

### Opcion 2: Node.js

```bash
npx serve workshops/01-mediapipe-explorer
```

### Opcion 3: VS Code

Instalar la extension "Live Server", click derecho en `index.html` > "Open with Live Server".

> **Importante:** Abrir via servidor local (no `file://`) porque los modulos ES6 requieren HTTP.

## Uso

1. Seleccionar un modelo del dropdown
2. Click en "Iniciar Webcam" (o cambiar a "Subir imagen")
3. Cambiar el modelo mientras la webcam esta activa — se recarga automaticamente
4. Ajustar el threshold de confianza con el slider
5. Observar las estadisticas: FPS, detecciones, tiempo de inferencia

## Requisitos

- Browser moderno (Chrome, Firefox, Edge)
- Webcam (opcional — se pueden subir imagenes)
- Conexion a internet (para cargar modelos la primera vez, ~5-30 MB por modelo)

## Stack tecnico

- **MediaPipe Tasks Vision JS** (`@mediapipe/tasks-vision`) — inferencia en el browser via WebAssembly
- **HTML/CSS/JS vanilla** — sin frameworks
- Los modelos se descargan de `storage.googleapis.com/mediapipe-models/` al primer uso

## Estructura

```
01-mediapipe-explorer/
├── index.html    # UI
├── style.css     # Estilos (dark theme)
├── app.js        # Logica: carga modelos, procesa frames, dibuja resultados
└── README.md     # Este archivo
```
