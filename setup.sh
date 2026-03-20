#!/bin/bash
# ORBIX Semillero SDK — Verificacion de Entorno
# Semillero de Investigacion en Vision por Computador
# Universidad Externado de Colombia

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ORBIX Semillero — Verificacion de Entorno              ║"
echo "║  Universidad Externado de Colombia                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0

# Entorno virtual
echo "=== Entorno Virtual ==="
if [ -n "$VIRTUAL_ENV" ]; then
    echo "[OK] venv activo: $VIRTUAL_ENV"
elif [ -d ".venv" ]; then
    echo "[!!] venv existe pero NO esta activo"
    echo "     Ejecuta: source .venv/bin/activate"
    ERRORS=$((ERRORS + 1))
else
    echo "[!!] No hay entorno virtual"
    echo "     Ejecuta: bash install.sh"
    ERRORS=$((ERRORS + 1))
fi

# GPU
echo ""
echo "=== GPU ==="
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)
    echo "[OK] $GPU_INFO"
else
    echo "[!!] nvidia-smi no encontrado — modo CPU"
    ERRORS=$((ERRORS + 1))
fi

# Python
echo ""
echo "=== Python ==="
if command -v python3 &> /dev/null; then
    PYVER=$(python3 --version 2>&1)
    echo "[OK] $PYVER"
else
    echo "[!!] Python3 no encontrado — ejecuta: bash install.sh"
    ERRORS=$((ERRORS + 1))
fi

# Librerias CV
echo ""
echo "=== Librerias CV ==="
python3 -c "import torch; print(f'[OK] PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || { echo "[!!] PyTorch no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import ultralytics; print(f'[OK] Ultralytics {ultralytics.__version__}')" 2>/dev/null || { echo "[!!] Ultralytics no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>/dev/null || { echo "[!!] OpenCV no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import mediapipe; print(f'[OK] MediaPipe {mediapipe.__version__}')" 2>/dev/null || { echo "[!!] MediaPipe no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import supervision; print(f'[OK] Supervision {supervision.__version__}')" 2>/dev/null || { echo "[!!] Supervision no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import numpy; print(f'[OK] NumPy {numpy.__version__}')" 2>/dev/null || { echo "[!!] NumPy no instalado"; ERRORS=$((ERRORS + 1)); }
python3 -c "import matplotlib; print(f'[OK] Matplotlib {matplotlib.__version__}')" 2>/dev/null || { echo "[!!] Matplotlib no instalado"; ERRORS=$((ERRORS + 1)); }

# CLI Agents
echo ""
echo "=== Agentes IA ==="
command -v gemini &> /dev/null && echo "[OK] Gemini CLI" || echo "[ ] Gemini CLI — npm install -g @google/gemini-cli"
command -v codex &> /dev/null && echo "[OK] Codex CLI" || echo "[ ] Codex CLI — npm install -g @openai/codex"
command -v claude &> /dev/null && echo "[OK] Claude Code" || echo "[ ] Claude Code — npm install -g @anthropic-ai/claude-code"
command -v aider &> /dev/null && echo "[OK] Aider" || echo "[ ] Aider — uv pip install aider-chat"

# Resultado
echo ""
if [ $ERRORS -eq 0 ]; then
    echo "============================================================"
    echo "  TODO LISTO. Abre tu agente de IA:"
    echo ""
    echo "    gemini    (gratis con Google)"
    echo "    codex     (gratis con ChatGPT Plus)"
    echo "    claude    (requiere plan Max)"
    echo ""
    echo "  ORBIX Semillero esta listo para ayudarte."
    echo "============================================================"
else
    echo "============================================================"
    echo "  $ERRORS problemas detectados."
    echo "  Si no has instalado, ejecuta: bash install.sh"
    echo "============================================================"
fi
