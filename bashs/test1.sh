#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Test 1 Start: $(date) ==="

# ── Config ────────────────────────────────────────────────────────────────────
CLI_BIN="$ROOT/cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"

if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN=$(find "$ROOT/cmake-build-debug" -name "DataGeneratorCLI" -type f | head -1)
    if [ -z "$CLI_BIN" ]; then
        echo "ERROR: DataGeneratorCLI binary not found. Run initialization.sh first."
        exit 1
    fi
    echo "WARNING: Using debug binary: $CLI_BIN"
fi

SYNTH_CONFIG="$ROOT/synths/SynthV3/Experiments/Exp003/config.yaml"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"
N=20000
SEED=42

# ── 1. Data generation ────────────────────────────────────────────────────────
echo "--- [1/7] Data generation (n=$N, seed=$SEED) ---"
"$CLI_BIN" --config "$DATAGEN_CONFIG" --n $N --seed $SEED

LATEST_DS=$(ls -td "$ROOT/synths/SynthV3/datasets/S03D"*/ 2>/dev/null | head -1)
if [ -z "$LATEST_DS" ]; then
    echo "ERROR: No dataset found after generation."
    exit 1
fi
DS_NAME=$(basename "$LATEST_DS")
echo "Dataset: $DS_NAME"

sed -i.bak "s|path:.*|path: \"datasets/$DS_NAME\"|" "$SYNTH_CONFIG"
echo "Config updated: dataset.path = datasets/$DS_NAME"

# ── 2. Feature cache ──────────────────────────────────────────────────────────
echo "--- [2/7] Feature extraction ---"
cd "$ROOT/NeuralNetworks"
python3 feature_extraction.py --config "$SYNTH_CONFIG"

# 3. Door classifier
python3 sequenced_train_door.py --config "$SYNTH_CONFIG"

# 4. Frame predictors (paralel)
FRAMES="osc_a osc_b noise filter env1 env2 lfo1 lfo2 lfo3 lfo4 reverb delay"
for FRAME in $FRAMES; do
  python3 sequenced_train_frames.py \
    --config "$SYNTH_CONFIG" --frame "$FRAME" &
done
wait
echo "All frame predictors done"

# 5. Evaluate
python3 sequenced_evaluate.py --config "$SYNTH_CONFIG" --frame door
for FRAME in $FRAMES; do
  python3 sequenced_evaluate.py \
    --config "$SYNTH_CONFIG" --frame "$FRAME" || true
done

# 6. ONNX export (auto_export=true ile zaten oluyor,
#    ama eksik varsa elle çalıştır)
python3 sequenced_export_onnx.py --config "$SYNTH_CONFIG"

# 7. Git push
cd "$ROOT"
mkdir -p "$ROOT/synths/SynthV3/Experiments/Exp003/exports"
git add synths/SynthV3/Experiments/Exp003/logs/
git add synths/SynthV3/Experiments/Exp003/exports/ 2>/dev/null || true
git add synths/SynthV3/Experiments/Exp003/checkpoints/
cd "$ROOT/NeuralNetworks"
echo "--- Results summary ---"
python3 -c "
import json
with open('$ROOT/synths/SynthV3/Experiments/Exp003/logs/eval_results.json') as f:
    r = json.load(f)
door = r.get('door', {})
val  = door.get('val', door)
print('  bypass_acc: {:.1f}%'.format(val.get('overall_bypass_accuracy', 0) * 100))
print('  lfo_acc:    {:.1f}%'.format(val.get('lfo_count_accuracy', 0) * 100))
for fname, fdata in r.get('frames', {}).items():
    fval = fdata.get('val', fdata)
    print('  {}: mse={:.4f}  mae={:.4f}'.format(
        fname,
        fval.get('overall_mse', fval.get('mse', 0)),
        fval.get('overall_mae', fval.get('mae', 0)),
    ))
"
cd "$ROOT"
git commit -m "test1: results $(date +%Y-%m-%d)"
git push

echo "=== Test 1 Complete: $(date) ==="
