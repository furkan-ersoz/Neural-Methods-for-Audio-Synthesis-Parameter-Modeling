#!/bin/bash
# experiment17_smoke.sh — Exp017 pipeline doğrulama (hızlı, ~5-10 dk)
# 500 sample, 3 epoch, door + osc_a + ONNX export.
# Amac: full_frame_hint kodu, feature extraction, egitim, eval ve ONNX yolunu
# uctan uca test etmek. Hata varsa burada ortaya cikar, 5 saatlik run'dan once.
#
# Calistir:
#   bash bashs/experiment17_smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3 || true)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp017"
NN_DIR="$ROOT/NeuralNetworks"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"
SYNTH_CONFIG="$EXP_DIR/config.yaml"

N=500
SEED=99
DS_NAME="S03D17_smoke"
DS_DIR="$ROOT/synths/SynthV3/datasets/$DS_NAME"
EPOCHS=3
PATIENCE=3

NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
NUM_WORKERS=$(( NPROC < 8 ? NPROC : 8 ))

echo "=============================================="
echo "  Experiment17 — SMOKE TEST"
echo "  Root     : $ROOT"
echo "  Dataset  : $DS_NAME (n=$N, seed=$SEED)"
echo "  Epochs   : $EPOCHS  |  Patience: $PATIENCE"
echo "=============================================="

# KRITIK: smoke ciktilarini (log/checkpoint/export) GERCEK Exp017'den izole et.
# Egitim scriptleri _exp_dir'i config'in bulundugu klasorden turetir; bu yuzden
# smoke config'i AYRI bir Exp017_smoke/ klasorune yaziyoruz. Boylece smoke,
# tam run'in Exp017/logs ve Exp017/checkpoints klasorlerine asla dokunmaz.
SMOKE_EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp017_smoke"
mkdir -p "$SMOKE_EXP_DIR"
SMOKE_CONFIG="$SMOKE_EXP_DIR/config.yaml"
python3 - <<PYEOF
import yaml
with open("$SYNTH_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["name"] = "Exp017_smoke"
cfg["dataset"]["path"] = "datasets/$DS_NAME"
cfg["experiment"]["notes"] = "SMOKE TEST — gecici, silinebilir. Gercek Exp017'den izole."
with open("$SMOKE_CONFIG", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)
print("Smoke config yazildi: $SMOKE_CONFIG")
PYEOF

# ── DataGeneratorCLI binary bul ───────────────────────────────────────────────
CLI_BIN="$ROOT/cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"
if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN=$(find "$ROOT" -name "DataGeneratorCLI" -type f 2>/dev/null | grep -v "\.dSYM" | head -1)
fi
if [ -z "${CLI_BIN:-}" ] || [ ! -f "$CLI_BIN" ]; then
    echo "ERROR: DataGeneratorCLI binary bulunamadi. Once build edin (initialization.sh)."
    exit 1
fi

# ── 1. Data Generation (500 sample) ───────────────────────────────────────────
echo ""
echo "--- [1] Data generation (smoke: $N sample) ---"
if [ -d "$DS_DIR/samples" ] && [ -n "$(ls -A "$DS_DIR/samples" 2>/dev/null)" ]; then
    echo "Smoke dataset zaten mevcut, generation atlandi."
else
    "$CLI_BIN" --config "$DATAGEN_CONFIG" --n "$N" --seed "$SEED"
    NEW_DS=$(ls -td "$ROOT/synths/SynthV3/datasets/S03D"*/ 2>/dev/null | head -1)
    NEW_DS="${NEW_DS%/}"
    if [ "$NEW_DS" != "$DS_DIR" ]; then
        mv "$NEW_DS" "$DS_DIR"
    fi
    echo "Dataset hazir: $DS_NAME"
fi

# ── 2. Feature extraction ─────────────────────────────────────────────────────
echo ""
echo "--- [2] Feature extraction ---"
cd "$NN_DIR"
$PYTHON feature_extraction_v2.py \
    --config "$SMOKE_CONFIG" \
    --num-workers "$NUM_WORKERS"

# ── 3. Door egitimi (3 epoch) ─────────────────────────────────────────────────
echo ""
echo "--- [3] Door egitimi (smoke) ---"
$PYTHON sequenced_train_door.py \
    --config "$SMOKE_CONFIG" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE"

# ── 4. osc_a egitimi (3 epoch) — full_frame_hint kodunu test eder ─────────────
echo ""
echo "--- [4] osc_a egitimi (smoke — full_frame_hint test) ---"
$PYTHON sequenced_train_frames.py \
    --config "$SMOKE_CONFIG" \
    --frame osc_a \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE"

# ── 5. ONNX export (door + osc_a) ─────────────────────────────────────────────
echo ""
echo "--- [5] ONNX export (door + osc_a) ---"
$PYTHON sequenced_export_onnx.py --config "$SMOKE_CONFIG" --frame door
$PYTHON sequenced_export_onnx.py --config "$SMOKE_CONFIG" --frame osc_a

echo ""
echo "=============================================="
echo "  SMOKE TEST GECTI"
echo "  ONNX: $SMOKE_EXP_DIR/exports/"
echo "  Simdi tam run icin:"
echo "    bash bashs/experiment17_bash.sh"
echo ""
echo "  Smoke artifaktlarini temizlemek icin:"
echo "    rm -rf $SMOKE_EXP_DIR $DS_DIR"
echo "=============================================="
