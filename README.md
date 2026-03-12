# Semillero de Investigacion en Vision por Computador

**Universidad Externado de Colombia — Programa de Ciencia de Datos**

Repositorio del Semillero de Investigacion en Vision por Computador. Al clonar este repo y abrir un agente de IA (Gemini CLI, Claude Code, Codex CLI, etc.), el agente se convierte en **ORBIX Semillero**, tu asistente de investigacion en Computer Vision.

## Inicio Rapido (PC nueva)

```bash
# 1. Clonar el repositorio
git clone https://github.com/uexternado-cv/semillero-cv.git
cd semillero-cv

# 2. Instalar todo (Python, librerias CV, agentes IA)
bash install.sh

# 3. Activar entorno
source .venv/bin/activate

# 4. Abrir tu agente de IA
gemini          # Google Gemini CLI (gratis con cuenta Google)
codex           # OpenAI Codex CLI
claude          # Claude Code
```

> **Primera vez con Gemini CLI:** ejecuta `gemini`, se abre el navegador para login con tu cuenta Google. Listo.

## Sesiones Siguientes

```bash
cd semillero-cv
source .venv/bin/activate
gemini
```

## Que es ORBIX Semillero?

ORBIX Semillero es un agente de IA preconfigurado que:

- Conoce los frameworks de CV (PyTorch, Ultralytics, MediaPipe, Supervision, OpenCV)
- Sabe crear aplicaciones de deteccion, tracking, pose estimation, segmentacion
- Conoce el hardware del laboratorio (NVIDIA RTX A5000, 24 GB VRAM)
- Te guia paso a paso en tus proyectos de investigacion

## Estructura del Repositorio

```
semillero-cv/
├── install.sh             # Instalador completo (correr una vez)
├── setup.sh               # Verificacion de entorno
├── requirements.txt       # Dependencias Python
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
├── templates/             # Templates para proyectos de equipo
├── workshops/             # Talleres guiados
├── models/                # Modelos descargados (gitignored)
└── datasets/              # Info sobre datasets publicos
```

## Entorno del Laboratorio

| Spec | Valor |
|------|-------|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) |
| Python | 3.12 (via uv) |
| PyTorch | 2.5.1 + CUDA 12.1 |
| OS | Ubuntu Linux |
| Package Manager | uv (10-100x mas rapido que pip) |

## Lineas de Investigacion

| # | Linea | Frameworks |
|---|-------|------------|
| 1 | Deteccion y tracking de objetos en deportes | YOLO, Supervision, ByteTrack |
| 2 | Pose estimation y analisis biomecanico | MediaPipe, Ultralytics Pose |
| 3 | Re-identificacion de jugadores | torchreid, embeddings |
| 4 | Generacion y analisis de datos deportivos con IA | Synthetic data, video analysis |

## Agentes de IA Compatibles

| Agente | Instalacion | Auth | Costo |
|--------|-------------|------|-------|
| **Gemini CLI** | `npm install -g @google/gemini-cli` | Login Google | Gratis |
| **Codex CLI** | `npm install -g @openai/codex` | Cuenta ChatGPT | Gratis (Plus/Pro) |
| **Claude Code** | `npm install -g @anthropic-ai/claude-code` | Cuenta Anthropic | Plan Max |
| **Aider** | `uv pip install aider-chat` | API key (cualquier LLM) | Segun modelo |
| **Cursor** | Descargar de cursor.com | Cuenta Cursor | Free tier |

## Credenciales (PCs Compartidos)

Las PCs del laboratorio son compartidas. **Nunca guardes credenciales en archivos.**

```bash
# Usa variables de entorno (mueren al cerrar terminal)
export GEMINI_API_KEY="tu-key"    # Solo si no usas login Google
export GH_TOKEN="tu-token"        # Para GitHub (opcional)
```

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| `nvidia-smi` no funciona | El driver NVIDIA no esta instalado. Pedir al profesor |
| `ModuleNotFoundError` | Activar el entorno: `source .venv/bin/activate` |
| PyTorch no ve GPU | Verificar: `python -c "import torch; print(torch.cuda.is_available())"` |
| `gemini` no encontrado | `npm install -g @google/gemini-cli` |
| Descarga lenta de modelos | Pedir modelos en USB al profesor |

## Contacto

- **Profesor:** Julian Zuluaga — julian.zuluaga2@uexternado.edu.co
- **Coordinacion:** Michell Uruena Esquivel — coordinacioncienciadedatos@uexternado.edu.co
- **GitHub:** [uexternado-cv](https://github.com/uexternado-cv)
