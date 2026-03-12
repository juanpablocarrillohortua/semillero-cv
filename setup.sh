#!/bin/bash
# ORBIX Semillero SDK — Verificacion de Entorno
# Semillero de Investigacion en Vision por Computador
# Universidad Externado de Colombia

echo "╔══════════════════════════════════════════════════╗"
echo "║  ORBIX Semillero SDK — Semillero de Vision por Computador  ║"
echo "║  Universidad Externado de Colombia               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# GPU
echo "=== GPU ==="
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null)
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    echo "[OK] $GPU_INFO"
else
    echo "[!!] nvidia-smi no encontrado — modo CPU"
fi

# Python
echo ""
echo "=== Python ==="
if command -v python3 &> /dev/null; then
    echo "[OK] $(python3 --version)"
else
    echo "[!!] Python3 no encontrado"
fi

# Librerias CV
echo ""
echo "=== Librerias CV ==="
python3 -c "import torch; print(f'[OK] PyTorch {torch.__version__} | CUDA={torch.cuda.is_available()} | GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "[!!] PyTorch no instalado"
python3 -c "import ultralytics; print(f'[OK] Ultralytics {ultralytics.__version__}')" 2>/dev/null || echo "[!!] Ultralytics no instalado — pip install ultralytics"
python3 -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>/dev/null || echo "[!!] OpenCV no instalado — pip install opencv-python"
python3 -c "import mediapipe; print(f'[OK] MediaPipe {mediapipe.__version__}')" 2>/dev/null || echo "[!!] MediaPipe no instalado — pip install mediapipe"
python3 -c "import supervision; print(f'[OK] Supervision {supervision.__version__}')" 2>/dev/null || echo "[!!] Supervision no instalado — pip install supervision"
python3 -c "import numpy; print(f'[OK] NumPy {numpy.__version__}')" 2>/dev/null || echo "[!!] NumPy no instalado"
python3 -c "import matplotlib; print(f'[OK] Matplotlib {matplotlib.__version__}')" 2>/dev/null || echo "[!!] Matplotlib no instalado"

# CLI Agents
echo ""
echo "=== Agentes IA ==="
command -v claude &> /dev/null && echo "[OK] Claude Code" || echo "[ ] Claude Code"
command -v codex &> /dev/null && echo "[OK] OpenAI Codex CLI" || echo "[ ] OpenAI Codex CLI"
command -v gemini &> /dev/null && echo "[OK] Gemini CLI" || echo "[ ] Gemini CLI"
command -v aider &> /dev/null && echo "[OK] Aider" || echo "[ ] Aider"
command -v cursor &> /dev/null && echo "[OK] Cursor" || echo "[ ] Cursor"

echo ""
echo "Abre tu agente de IA preferido en este directorio."
echo "ORBIX Semillero esta listo para ayudarte."
