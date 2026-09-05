#!/usr/bin/env bash
# Provision the Hunyuan3D-2 shape-only engine for WSL2 + CUDA.
#
# Shape-only by design: this box is an 8GB RTX 3060 Ti and cannot run the
# texture stage (see docs/wsl2-port.md). That means we skip custom_rasterizer
# and DifferentiableRenderer entirely — the two components whose compilation
# breaks most Windows/WSL installs.
#
# Idempotent: every step inspects before it acts.
set -euo pipefail

ENGINE_REPO="${HY3D_ENGINE_REPO:-$HOME/git/repos/Hunyuan3D-2}"
ENGINE_GIT="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
ENGINE_REF="${HY3D_ENGINE_REF:-f8db63096c8282cb27354314d896feba5ba6ff8a}"
VENV="${HY3D_ENGINE_VENV:-$HOME/.hy3d/engine-venv}"
UV="${UV:-$HOME/.local/bin/uv}"

say() { printf '\n== %s\n' "$*"; }

say "engine repo -> $ENGINE_REPO"
if [ -d "$ENGINE_REPO/.git" ]; then
  echo "present, at $(git -C "$ENGINE_REPO" rev-parse --short HEAD)"
else
  git clone "$ENGINE_GIT" "$ENGINE_REPO"
  git -C "$ENGINE_REPO" checkout "$ENGINE_REF"
fi

say "venv -> $VENV (python 3.10)"
if [ -x "$VENV/bin/python" ]; then
  echo "present, $("$VENV/bin/python" -V)"
else
  "$UV" venv --python 3.10 "$VENV"
fi
PY="$VENV/bin/python"

say "torch 2.5.1+cu124 / torchvision 0.20.1"
# Both from the cu124 index in ONE command. Installing torchvision from PyPI
# afterwards silently pulls a different torch and discards the cu124 build.
if "$PY" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.5.1+cu124") else 1)' 2>/dev/null; then
  echo "present, $("$PY" -c 'import torch;print(torch.__version__)')"
else
  "$UV" pip install --python "$PY" \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124
fi

say "shape-only python deps"
# numpy<2: pymeshlab wheels of this era are built against the numpy 1.x ABI
# and fail at *import*, not at use.
# scikit-image: required by shapegen/models/autoencoders/surface_extractors.py
# for the default mc_algo='mc' path, but COMMENTED OUT in upstream
# requirements.txt. Omitting it is the first thing that breaks a shape run.
# Deliberately absent: xatlas + diso (texture UV / mc_algo='dmc'),
# gradio/fastapi/uvicorn (demo app), custom_rasterizer, DifferentiableRenderer.
"$UV" pip install --python "$PY" \
  "numpy<2" \
  transformers diffusers accelerate safetensors \
  einops omegaconf pyyaml tqdm \
  opencv-python pillow scipy \
  trimesh pygltflib scikit-image \
  pymeshlab \
  rembg onnxruntime

say "engine importable from $ENGINE_REPO"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "$ENGINE_REPO")
import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("pipeline import OK")
PYEOF

say "done"
