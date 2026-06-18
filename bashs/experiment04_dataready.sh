#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3 || true)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp010"
NN_DIR="$ROOT/NeuralNetworks"
SYNTH_CONFIG="$EXP_DIR/config.yaml"

NPROC=$(nproc 2>/dev/null || echo 4)
NUM_WORKERS=$(( NPROC < 8 ? NPROC : 8 ))

echo "=============================================="
echo "  Experiment04 (dataready) — osc_a"
echo "  Config  : $SYNTH_CONFIG"
echo "=============================================="

echo ""
echo "--- [1/3] Feature extraction ---"
cd "$NN_DIR"
$PYTHON feature_extraction_v2.py \
    --config "$SYNTH_CONFIG" \
    --save-png \
    --num-workers "$NUM_WORKERS"

echo ""
echo "--- [2/3] Training: osc_a ---"
$PYTHON sequenced_train_frames.py --config "$SYNTH_CONFIG" --frame osc_a

echo ""
echo "--- [3/3] Results ---"
EXP_ROOT="$EXP_DIR" $PYTHON - <<'PYEOF'
import json, os, sys
root = os.environ.get("EXP_ROOT", "")
log_path = os.path.join(root, "logs", "osc_a", "train_log.json")
if not os.path.exists(log_path):
    print("  train_log.json bulunamadi:", log_path); sys.exit(0)
with open(log_path) as f:
    log = json.load(f)
best_epoch   = log.get("best_epoch", 1)
epochs_run   = log.get("epochs_run", "?")
val_mse_list = log.get("val_mse", [])
best_mse     = val_mse_list[best_epoch - 1] if val_mse_list else float("nan")
random_baseline = 1.0 / 12.0
print(f"  Toplam epoch   : {epochs_run}")
print(f"  En iyi epoch   : {best_epoch}")
print(f"  En iyi val MSE : {best_mse:.6f}")
print(f"  Random baseline: {random_baseline:.6f}")
if best_mse < random_baseline:
    print(f"  Sonuc: model OGRENIYOR ({best_mse/random_baseline:.1%} of random baseline)")
else:
    print(f"  Sonuc: model OGRENMEDI")
print("\n  Val MSE seyri:")
for i, v in enumerate(val_mse_list):
    print(f"    Epoch {i+1:3d}: {v:.6f}")
PYEOF

echo ""
echo "  Checkpoints : $EXP_DIR/checkpoints/osc_a/"
echo "  Log         : $EXP_DIR/logs/osc_a/train_log.json"
echo ""
echo "=============================================="
echo "  Tamamlandi: $(date)"
echo "=============================================="
