#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# install_tf_od_api.sh
# Automates the painful TF Object Detection API setup from the git submodule.
# Run this ONCE after cloning the repo (or it's auto-called by Docker).
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

MODELS_DIR="/app/tensorflow_models"
RESEARCH_DIR="${MODELS_DIR}/research"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " TF Object Detection API — Setup Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Check submodule is initialized ────────────────────────────────────────
if [ ! -d "${RESEARCH_DIR}" ]; then
  echo "❌  tensorflow_models submodule not found at ${MODELS_DIR}"
  echo "    Run: git submodule update --init --recursive"
  exit 1
fi
echo "✅  Submodule found: ${MODELS_DIR}"

# ── 2. Compile Protobuf definitions ──────────────────────────────────────────
echo ""
echo "🔧  Compiling protobuf definitions..."
cd "${RESEARCH_DIR}"
protoc object_detection/protos/*.proto --python_out=.
echo "✅  Protobufs compiled"

# ── 3. Install the OD API as a Python package ────────────────────────────────
echo ""
echo "📦  Installing object_detection package..."
cp object_detection/packages/tf2/setup.py .
pip install --no-cache-dir -q .
echo "✅  object_detection package installed"

# ── 4. Install tf-models-official from the submodule ─────────────────────────
echo ""
echo "📦  Installing tf-models-official..."
cd "${MODELS_DIR}"
pip install --no-cache-dir -q ".[tensorflow]" 2>/dev/null || \
pip install --no-cache-dir -q tf-models-official==2.11.3
echo "✅  tf-models-official installed"

# ── 5. Verify the installation ────────────────────────────────────────────────
echo ""
echo "🔍  Verifying installation..."
python -c "
import tensorflow as tf
print(f'   TensorFlow:  {tf.__version__}')
from object_detection.utils import label_map_util
print(f'   OD API:      OK')
from object_detection.utils import visualization_utils
print(f'   Viz utils:   OK')
"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " ✅  Setup complete! Open Jupyter at:"
echo "     http://localhost:8888"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
