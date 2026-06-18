#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== test2 Start: $(date) ==="

# ── Config ────────────────────────────────────────────────────────────────────
SYNTH_CONFIG="$ROOT/synths/SynthV3/Experiments/Exp006/config.yaml"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"
N=20000
SEED=42

# ── Binary ────────────────────────────────────────────────────────────────────
CLI_BIN="$ROOT/cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"
if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN=$(find "$ROOT/cmake-build-debug" -name "DataGeneratorCLI" -type f | head -1)
    if [ -z "$CLI_BIN" ]; then
        echo "ERROR: DataGeneratorCLI binary not found. Build the project first."
        exit 1
    fi
    echo "WARNING: Using debug binary: $CLI_BIN"
fi

# ── 1. Data generation ────────────────────────────────────────────────────────
echo "--- [1/6] Data generation (n=$N, seed=$SEED, all cores) ---"
"$CLI_BIN" --config "$DATAGEN_CONFIG" --n $N --seed $SEED

LATEST_DS=$(ls -td "$ROOT/synths/SynthV3/datasets/S03D"*/ 2>/dev/null | head -1)
if [ -z "$LATEST_DS" ]; then
    echo "ERROR: No dataset found after generation."
    exit 1
fi
DS_NAME=$(basename "$LATEST_DS")
echo "Dataset: $DS_NAME"

# dataset.path'i config'e yaz
sed -i.bak "s|path:.*|path: \"datasets/$DS_NAME\"|" "$SYNTH_CONFIG"
echo "Config updated: dataset.path = datasets/$DS_NAME"

# ── 2. Feature extraction ─────────────────────────────────────────────────────
echo "--- [2/6] Feature extraction ---"
cd "$ROOT/NeuralNetworks"
python3 feature_extraction_v2.py --config "$SYNTH_CONFIG"

# ── 3. Door classifier ────────────────────────────────────────────────────────
echo "--- [3/6] Door classifier ---"
python3 sequenced_train_door.py --config "$SYNTH_CONFIG"

# ── 4. Frame predictors (paralel) ─────────────────────────────────────────────
echo "--- [4/6] Frame predictors (parallel) ---"
FRAMES="osc_a osc_b noise filter env1 env2 lfo1 lfo2 lfo3 lfo4 reverb delay"
FAILED=0
for FRAME in $FRAMES; do
    echo "  >> Training frame: $FRAME"
    if ! python3 sequenced_train_frames.py \
        --config "$SYNTH_CONFIG" --frame "$FRAME" \
        > "$ROOT/synths/SynthV3/Experiments/Exp006/logs/train_${FRAME}.log" 2>&1; then
        echo "  ERROR: $FRAME failed"
        FAILED=1
    else
        echo "  OK: $FRAME"
    fi
done
[ $FAILED -eq 1 ] && echo "WARNING: Some frame predictors failed — check logs/" || echo "All frame predictors done"

# ── 5. Evaluate + ONNX export ─────────────────────────────────────────────────
echo "--- [5/6] Evaluate + ONNX export ---"
python3 sequenced_evaluate.py --config "$SYNTH_CONFIG" --frame door
for FRAME in $FRAMES; do
    python3 sequenced_evaluate.py \
        --config "$SYNTH_CONFIG" --frame "$FRAME" || true
done
python3 sequenced_export_onnx.py --config "$SYNTH_CONFIG"

# ── 6. Git push — modeller dahil, WAV/dataset hariç ──────────────────────────
echo "--- [6/6] Git push ---"
cd "$ROOT"

# Sonuç özeti
echo ""
echo "=== Results summary ==="
python3 -c "
import json, os
results_path = '$ROOT/synths/SynthV3/Experiments/Exp006/logs/eval_results.json'
if not os.path.exists(results_path):
    print('  eval_results.json not found, check logs/ manually.')
    exit()
with open(results_path) as f:
    r = json.load(f)
door = r.get('door', {})
val  = door.get('val', door)
print('  Door — bypass_acc: {:.1f}%  lfo_acc: {:.1f}%'.format(
    val.get('overall_bypass_accuracy', 0) * 100,
    val.get('lfo_count_accuracy', 0) * 100,
))
for fname, fdata in r.get('frames', {}).items():
    fval = fdata.get('val', fdata)
    print('  {:<10} mse={:.4f}  mae={:.4f}'.format(
        fname + ':',
        fval.get('overall_mse', fval.get('mse', 0)),
        fval.get('overall_mae', fval.get('mae', 0)),
    ))
"

# Stage: checkpoints, exports, logs — WAV ve dataset klasörleri hariç
git add synths/SynthV3/Experiments/Exp006/checkpoints/
git add synths/SynthV3/Experiments/Exp006/exports/
git add synths/SynthV3/Experiments/Exp006/logs/
git add synths/SynthV3/Experiments/Exp006/config.yaml
# dataset klasörünü kesinlikle ekleme
git reset HEAD synths/SynthV3/datasets/ 2>/dev/null || true

git commit -m "test2: Exp006 results $(date +%Y-%m-%d) dataset=$DS_NAME"
git push

echo ""
echo "=== test2 Complete: $(date) ==="