#!/bin/bash
# ============================================================
# ORBIX Semillero — Instalador de Entorno
# Semillero de Vision por Computador
# Universidad Externado de Colombia
#
# Uso: bash install.sh
# Tiempo estimado: 5-10 minutos (primera vez)
# ============================================================

# set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ORBIX Semillero — Instalador de Entorno                ║"
echo "║  Semillero de Vision por Computador                     ║"
echo "║  Universidad Externado de Colombia                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 1. Verificar GPU
# ============================================================
echo "=== [1/6] GPU ==="
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    ok "GPU: $GPU_NAME | Driver: $DRIVER"
else
    warn "nvidia-smi no encontrado — se instalara en modo CPU"
fi

# ============================================================
# 2. Dependencias del sistema
# ============================================================
echo ""
echo "=== [2/6] Dependencias del sistema ==="
sudo apt-get update -qq
sudo apt-get install -y -qq git curl wget build-essential \
     libglib2.0-0 libsm6 libxext6 libxrender-dev \
    > /dev/null 2>&1
ok "Dependencias del sistema instaladas"

# ============================================================
# 3. uv (gestor de paquetes Python ultrarapido)
# ============================================================
echo ""
echo "=== [3/6] uv (gestor de paquetes) ==="
if command -v uv &> /dev/null; then
    ok "uv ya instalado: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv instalado: $(uv --version)"
fi

# ============================================================
# 4. Node.js (para Gemini CLI y otros agentes)
# ============================================================
echo ""
echo "=== [4/6] Node.js ==="
if command -v node &> /dev/null; then
    NODE_VER=$(node --version)
    ok "Node.js ya instalado: $NODE_VER"
else
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - > /dev/null 2>&1
    sudo apt-get install -y -qq nodejs > /dev/null 2>&1
    ok "Node.js instalado: $(node --version)"
fi

# ============================================================
# 5. Entorno Python + librerias CV
# ============================================================
echo ""
echo "=== [5/6] Entorno Python + librerias CV ==="

# Crear venv con Python 3.12 (requerido por MediaPipe)
if [ ! -d ".venv" ]; then
    echo "    Creando entorno virtual (Python 3.12)..."
    uv venv --python 3.12 .venv
    ok "Entorno virtual creado en .venv/"
else
    ok "Entorno virtual ya existe en .venv/"
fi

# Activar venv
source .venv/bin/activate

# Instalar PyTorch con CUDA 12.1 (compatible con driver 535+)
echo "    Instalando PyTorch + CUDA (esto puede tomar unos minutos)..."
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    > /dev/null 2>&1
ok "PyTorch 2.5.1 + CUDA 12.1"

# Instalar stack de CV
echo "    Instalando librerias de Computer Vision..."
uv pip install \
    ultralytics \
    opencv-python \
    mediapipe \
    supervision \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    tqdm \
    Pillow \
    > /dev/null 2>&1
ok "Stack CV instalado (Ultralytics, OpenCV, MediaPipe, Supervision)"

# ============================================================
# 6. Agentes de IA (CLI)
# ============================================================
echo ""
echo "=== [6/6] Agentes de IA ==="

# Gemini CLI (gratis con cuenta Google)
if command -v gemini &> /dev/null; then
    ok "Gemini CLI ya instalado"
else
    echo "    Instalando Gemini CLI..."
    npm install -g @anthropic-ai/claude-code @google/gemini-cli @openai/codex 2>/dev/null || true
    command -v gemini &> /dev/null && ok "Gemini CLI instalado" || warn "Gemini CLI no se pudo instalar (instalar manual: npm install -g @google/gemini-cli)"
fi

command -v claude &> /dev/null && ok "Claude Code disponible" || echo "    [ ] Claude Code (npm install -g @anthropic-ai/claude-code)"
command -v codex &> /dev/null && ok "Codex CLI disponible" || echo "    [ ] Codex CLI (npm install -g @openai/codex)"

# ============================================================
# Verificacion final
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Verificacion Final                                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# GPU + PyTorch
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'[OK] PyTorch {torch.__version__} | GPU: {gpu} | VRAM: {vram:.0f} GB | CUDA: {torch.version.cuda}')
else:
    print(f'[!!] PyTorch {torch.__version__} | GPU: No disponible (modo CPU)')
" 2>/dev/null || echo "[!!] PyTorch no funciona"

# CV libs
python -c "import ultralytics; print(f'[OK] Ultralytics {ultralytics.__version__}')" 2>/dev/null || echo "[!!] Ultralytics"
python -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>/dev/null || echo "[!!] OpenCV"
python -c "import mediapipe; print(f'[OK] MediaPipe {mediapipe.__version__}')" 2>/dev/null || echo "[!!] MediaPipe"
python -c "import supervision; print(f'[OK] Supervision {supervision.__version__}')" 2>/dev/null || echo "[!!] Supervision"

echo ""
echo "============================================================"
echo ""
echo "  INSTALACION COMPLETA"
echo ""
echo "  Para activar el entorno en cada sesion:"
echo ""
echo "    source .venv/bin/activate"
echo ""
echo "  Para iniciar tu agente de IA:"
echo ""
echo "    gemini          # Google Gemini CLI (gratis)"
echo "    codex            # OpenAI Codex CLI"
echo "    claude           # Claude Code"
echo ""
echo "  Primera vez con Gemini CLI:"
echo "    1. Ejecuta: gemini"
echo "    2. Se abre el navegador para login con Google"
echo "    3. Listo — ORBIX Semillero cobra vida"
echo ""
echo "============================================================"
