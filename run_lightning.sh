#!/bin/bash
# ============================================================
# NCTNet — Lightning AI Run Script
# ============================================================
# This script runs the complete smoke test + 1% training on
# Lightning AI with NVIDIA GPU.
#
# Usage: bash run_lightning.sh
# ============================================================

set -e  # Exit on any error

echo "============================================================"
echo "  NCTNet — Lightning AI Setup & Smoke Test"
echo "============================================================"
echo ""

# --- Step 1: System Info ---
echo "[1/7] System Information"
echo "========================"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# --- Step 2: Install Dependencies ---
echo "[2/7] Installing Dependencies"
echo "=============================="
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet wandb
echo "✓ Dependencies installed"
echo ""

# --- Step 3: Download Tiny Real Dataset Sample ---
echo "[3/7] Downloading Real Dataset Sample"
echo "===================================="
mkdir -p data/real/bluetooth
curl -L --retry 3 --retry-delay 2 \
    -o data/real/bluetooth/sample_1_meter.csv \
    https://zenodo.org/api/records/17347695/files/sample_1_meter.csv/content
export REAL_DATASET_PATH="data/real/bluetooth/sample_1_meter.csv"
echo "✓ Real dataset sample downloaded"
echo ""

# --- Step 4: WandB Login ---
echo "[4/7] WandB Authentication"
echo "=========================="
read -sp "Enter your WandB API Key: " WANDB_API_KEY && export WANDB_API_KEY && echo ""
wandb login "$WANDB_API_KEY" 2>/dev/null || echo "WandB login via env var"
echo "✓ WandB configured"
echo ""

# --- Step 5: Run Smoke Test ---
echo "[5/7] Running Smoke Test (Pre-Flight Check)"
echo "============================================="
python3 smoke_test.py
SMOKE_EXIT=$?
if [ $SMOKE_EXIT -ne 0 ]; then
    echo ""
    echo "🔴 SMOKE TEST FAILED — Aborting to save credits!"
    echo "   Fix the errors above before running on GPU."
    exit 1
fi
echo ""

# --- Step 6: Run 1% Smoke Training ---
echo "[6/7] Running Real-Data Smoke Training (tiny, 1 epoch)"
echo "========================================================="
python3 -m src.train --config configs/smoke_test.yaml
echo ""
echo "✓ Smoke training completed successfully"
echo ""

# --- Step 7: GPU Memory Report ---
echo "[7/7] GPU Memory Report"
echo "======================="
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Total Memory: {props.total_mem / 1024**3:.1f} GB')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'SM Count: {props.multi_processor_count}')
else:
    print('No GPU available')
"
echo ""

# --- Step 8: Summary ---
echo "[8/8] Summary"
echo "============="
echo "✅ All checks passed. Safe to run full experiment."
echo ""
echo "To run the FULL experiment:"
echo "  python3 -m src.train --config configs/default.yaml"
echo ""
echo "============================================================"
