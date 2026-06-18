#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Test 0 — Smoke Test: $(date) ==="

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

SYNTH_CONFIG="$ROOT/synths/SynthV3/Experiments/Exp002/config.yaml"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"
N=100
SEED=42
EPOCHS=5

# ── 1. Data generation ────────────────────────────────────────────────────────
echo "--- [1/5] Data generation (n=$N, seed=$SEED) ---"
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
echo "--- [2/5] Feature extraction ---"
cd "$ROOT/NeuralNetworks"
python3 feature_extraction.py --config "$SYNTH_CONFIG"

# ── 3. Door classifier ────────────────────────────────────────────────────────
echo "--- [3/5] Training door classifier ---"
python3 sequenced_train_door.py --config "$SYNTH_CONFIG" --epochs $EPOCHS

# ── 3b. Frame predictor smoke test ───────────────────────────────────────────
echo "--- [3b/5] Training all frame predictors (epochs=$EPOCHS) ---"
FRAMES="osc_a osc_b noise filter env1 env2 lfo1 lfo2 lfo3 lfo4 reverb delay"
for FRAME in $FRAMES; do
  echo "  Training frame: $FRAME"
  python3 sequenced_train_frames.py \
    --config "$SYNTH_CONFIG" --frame "$FRAME" --epochs $EPOCHS || true
done

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
echo "--- [4/5] Evaluate ---"
python3 sequenced_evaluate.py --config "$SYNTH_CONFIG" --frame door
for FRAME in $FRAMES; do
  python3 sequenced_evaluate.py \
    --config "$SYNTH_CONFIG" --frame "$FRAME" || true
done

# ── 5. Results summary ────────────────────────────────────────────────────────
echo "--- [5/5] Results summary ---"
python3 -c "
import json

with open('$ROOT/synths/SynthV3/Experiments/Exp002/logs/eval_results.json') as f:
    r = json.load(f)

door = r.get('door', {})
val  = door.get('val', door)   # fallback: flat structure
print('  bypass_acc: {:.1f}%'.format(val.get('overall_bypass_accuracy', 0) * 100))
print('  lfo_acc:    {:.1f}%'.format(val.get('lfo_count_accuracy', 0) * 100))
print('  val_loss:   {:.4f}'.format(door.get('val_loss', 0)))

frames_results = r.get('frames', {})
for fname, fdata in frames_results.items():
    fval = fdata.get('val', fdata)
    print('  {}: val_loss={:.4f}  mse={:.6f}  mae={:.6f}'.format(
        fname,
        fdata.get('val_loss', 0),
        fval.get('overall_mse', fval.get('mse', 0)),
        fval.get('overall_mae', fval.get('mae', 0)),
    ))

meta = r.get('meta', {})
if meta:
    print('  dataset:   ', meta.get('dataset_path', ''))
    print('  evaluated: ', meta.get('evaluated_at', ''))
"

# ── 6. Git push ───────────────────────────────────────────────────────────────
echo "--- [6/6] Git push ---"
cd "$ROOT"
mkdir -p "$ROOT/synths/SynthV3/Experiments/Exp002/exports"
git add synths/SynthV3/Experiments/Exp002/logs/
git add synths/SynthV3/Experiments/Exp002/checkpoints/
git add synths/SynthV3/Experiments/Exp002/exports/ 2>/dev/null || true
git add synths/SynthV3/Experiments/Exp002/config.yaml
git commit -m "test0: smoke test $(date +%Y-%m-%d) dataset=$DS_NAME" || echo "Nothing to commit"
git push

echo "=== Test 0 Complete: $(date) ==="
