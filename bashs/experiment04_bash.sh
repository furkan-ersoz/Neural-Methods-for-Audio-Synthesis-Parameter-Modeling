#!/bin/bash
# experiment04_bash.sh — osc_a sanity check, 20k samples, RunPod/remote
# Veri uretimi + feature extraction + egitim tek scriptte.
# Calistirmak icin (proje kök dizininden):
#   bash bashs/experiment04_bash.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3 || true)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp010"
NN_DIR="$ROOT/NeuralNetworks"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_exp03.yaml"
SYNTH_CONFIG="$EXP_DIR/config.yaml"
N=20000
SEED=77

# Worker sayisi: CPU core basina 1, max 8
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
NUM_WORKERS=$(( NPROC < 8 ? NPROC : 8 ))

echo "=============================================="
echo "  Experiment04 — osc_a sanity check (20k)"
echo "  Root       : $ROOT"
echo "  Python     : $PYTHON"
echo "  Workers    : $NUM_WORKERS"
echo "=============================================="

# ── DataGeneratorCLI binary bul ───────────────────────────────────────────────
CLI_BIN="$ROOT/cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"
if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN=$(find "$ROOT" -name "DataGeneratorCLI" -type f 2>/dev/null | grep -v "\.dSYM" | head -1)
fi
if [ -z "$CLI_BIN" ] || [ ! -f "$CLI_BIN" ]; then
    echo "ERROR: DataGeneratorCLI binary bulunamadi. Once build edin (initialization.sh)."
    exit 1
fi
echo "Binary: $CLI_BIN"

# ── 1. Data Generation ────────────────────────────────────────────────────────
echo ""
echo "--- [1/4] Data generation (n=$N, seed=$SEED) ---"
"$CLI_BIN" --config "$DATAGEN_CONFIG" --n $N --seed $SEED

LATEST_DS=$(ls -td "$ROOT/synths/SynthV3/datasets/S03D"*/ 2>/dev/null | head -1)
if [ -z "$LATEST_DS" ]; then
    echo "ERROR: Dataset olusturulamadi."
    exit 1
fi
DS_NAME=$(basename "$LATEST_DS")
echo "Dataset: $DS_NAME"

sed -i.bak "s|path: \"datasets/S03D[0-9]*\"|path: \"datasets/$DS_NAME\"|" "$SYNTH_CONFIG"
echo "Config guncellendi: dataset.path = datasets/$DS_NAME"

# ── 2. Feature extraction ─────────────────────────────────────────────────────
echo ""
echo "--- [2/4] Feature extraction ---"
cd "$NN_DIR"
$PYTHON feature_extraction_v2.py \
    --config "$SYNTH_CONFIG" \
    --save-png \
    --num-workers "$NUM_WORKERS"

# ── 3. Training ───────────────────────────────────────────────────────────────
echo ""
echo "--- [3/4] Training: osc_a ---"
$PYTHON sequenced_train_frames.py --config "$SYNTH_CONFIG" --frame osc_a

# ── 4. Results ────────────────────────────────────────────────────────────────
echo ""
echo "--- [4/4] Results ---"
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
print(f"  En iyi val MSE : {best_mse:.6f}")
print(f"  Random baseline: {random_baseline:.6f}")
if best_mse < random_baseline:
    ratio = best_mse / random_baseline
    print(f"  Sonuc: model OGRENIYOR ({ratio:.1%} of random baseline)")
else:
    print(f"  Sonuc: model OGRENMEDI")
print("\n  Val MSE seyri:")
for i, v in enumerate(val_mse_list):
    print(f"    Epoch {i+1:3d}: {v:.6f}")
PYEOF

echo ""
echo "  Dataset     : $ROOT/synths/SynthV3/datasets/$DS_NAME/"
echo "  Spectrograms: $ROOT/synths/SynthV3/datasets/$DS_NAME/spectrograms/"
echo "  Checkpoints : $EXP_DIR/checkpoints/osc_a/"
echo "  Log         : $EXP_DIR/logs/osc_a/train_log.json"
echo ""
echo "=============================================="
echo "  Experiment04 tamamlandi: $(date)"
echo "=============================================="
