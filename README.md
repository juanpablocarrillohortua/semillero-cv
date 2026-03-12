# Semillero de Investigacion en Vision por Computador

**Universidad Externado de Colombia — Programa de Ciencia de Datos**

Repositorio del Semillero de Investigacion en Vision por Computador. Contiene el SDK de ORBIX Semillero, un asistente de IA especializado en Computer Vision que guia a investigadores en el desarrollo de aplicaciones de vision por computador.

## Inicio Rapido

```bash
# 1. Clonar el repositorio
git clone https://github.com/uexternado-cv/semillero-cv.git
cd semillero-cv

# 2. Verificar entorno
bash setup.sh

# 3. Abrir tu agente de IA preferido
claude          # Claude Code
codex           # OpenAI Codex CLI
gemini          # Google Gemini CLI
aider           # Aider
cursor .        # Cursor
```

El agente automaticamente se convierte en **ORBIX Semillero**, tu asistente de investigacion en CV.

## Que es ORBIX Semillero?

ORBIX Semillero (Vision Intelligence for Semillero Investigation and Orientation) es un agente de IA preconfigurado que:

- Conoce los frameworks de CV disponibles (PyTorch, Ultralytics, MediaPipe, Supervision, OpenCV)
- Sabe crear aplicaciones de deteccion, tracking, pose estimation, segmentacion
- Conoce el hardware del laboratorio (NVIDIA RTX A5000, 24 GB VRAM, CUDA 12.2)
- Te guia paso a paso en tus proyectos de investigacion

## Estructura del Repositorio

```
semillero-cv/
├── AGENTS.md              # Configuracion principal del agente (estandar AAIF)
├── CLAUDE.md              # Adapter para Claude Code
├── GEMINI.md              # Adapter para Gemini CLI
├── knowledge/             # Base de conocimiento del agente
│   ├── environment.md     # Hardware y librerias del lab
│   ├── frameworks.md      # Referencia de frameworks CV
│   └── research-lines.md  # Lineas de investigacion
├── skills/                # Skills de CV (workflows paso a paso)
│   ├── object-detection.md
│   ├── pose-estimation.md
│   ├── tracking.md
│   ├── segmentation.md
│   ├── video-pipeline.md
│   └── training.md
├── templates/             # Templates para proyectos
├── workshops/             # Talleres guiados
└── datasets/              # Info sobre datasets
```

## Lineas de Investigacion

| # | Linea | Frameworks |
|---|-------|------------|
| 1 | Deteccion y tracking de objetos en deportes | YOLO, Supervision, ByteTrack |
| 2 | Pose estimation y analisis biomecanico | MediaPipe, Ultralytics Pose |
| 3 | Re-identificacion de jugadores | torchreid, embeddings |
| 4 | Generacion y analisis de datos deportivos con IA | Synthetic data, video analysis |

## Credenciales (PCs Compartidos)

Las PCs del laboratorio son compartidas. **Nunca guardes credenciales en archivos.**

```bash
# Usa variables de entorno (mueren al cerrar terminal)
export GEMINI_API_KEY="tu-key"
export GH_TOKEN="tu-token"
```

## Contacto

- **Profesor:** Julian Zuluaga — julian.zuluaga2@uexternado.edu.co
- **Coordinacion:** Michell Uruena Esquivel — coordinacioncienciadedatos@uexternado.edu.co
- **GitHub:** [uexternado-cv](https://github.com/uexternado-cv)
