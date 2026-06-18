#!/bin/bash
# experiment03_dataready.sh — osc_a sanity check, veri ve feature hazir
# S03D01 datasetini kullanir, cache'ten feature yukler, direkt egitir.
#   bash bashs/experiment03_dataready.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3.13 || command -v python3.10 || command -v python3)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp010"
NN_DIR="$ROOT/NeuralNetworks"
SYNTH_CONFIG="$EXP_DIR/config.yaml"
DS_NAME="S03D01"

echo "=============================================="
echo "  Experiment03 (dataready) — osc_a sanity check"
echo "  Dataset : $DS_NAME"
echo "  Config  : $SYNTH_CONFIG"
echo "=============================================="

# ── 1. Training — osc_a only ──────────────────────────────────────────────────
echo ""
echo "--- [1/2] Training: osc_a ---"
cd "$NN_DIR"
$PYTHON sequenced_train_frames.py --config "$SYNTH_CONFIG" --frame osc_a

# ── 2. Results summary ────────────────────────────────────────────────────────
echo ""
echo "--- [2/2] Results ---"
EXP_ROOT="$EXP_DIR" $PYTHON - <<'PYEOF'
import json, os, sys
root = os.environ.get("EXP_ROOT", "")
log_path = os.path.join(root, "logs", "osc_a", "train_log.json")
if not os.path.exists(log_path):
    print("  train_log.json bulunamadi:", log_path)
    sys.exit(0)
with open(log_path) as f:
    log = json.load(f)
best_epoch   = log.get("best_epoch", 1)
best_val     = log.get("best_val_loss", float("nan"))
epochs_run   = log.get("epochs_run", "?")
val_mse_list = log.get("val_mse", [])
best_mse     = val_mse_list[best_epoch - 1] if val_mse_list else float("nan")
random_baseline = 1.0 / 12.0
print(f"  Toplam epoch   : {epochs_run}")
print(f"  En iyi epoch   : {best_epoch}")
print(f"  En iyi val loss: {best_val:.6f}")
print(f"  En iyi val MSE : {best_mse:.6f}")
print(f"  Random baseline: {random_baseline:.6f}")
if best_mse < random_baseline:
    ratio = best_mse / random_baseline
    print(f"  Sonuc: model OGRENIYOR ({ratio:.1%} of random baseline)")
else:
    print(f"  Sonuc: model OGRENMEDI (val MSE >= random baseline)")
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
