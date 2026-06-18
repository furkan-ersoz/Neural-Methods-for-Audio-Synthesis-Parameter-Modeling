#!/bin/bash
# experiment15_bash.sh — Exp015: FULL sequential training (door + 12 frames), 20k samples
# RunPod/remote. Tek scriptte: veri uretimi -> feature extraction -> sirali egitim -> eval.
#
# NOT (Exp010 hizalamasi): tum ogrenme hiperparametreleri Exp015/config.yaml'dan
#   okunur (bu script override BAYRAGI gecmez). Config artik 010'un bilinen-iyi
#   ayarlarini birebir tasir: normalize=true, dropout=0.2, dropout_p=0.0,
#   per_param_loss_weight=false, lr_scheduler=plateau, epochs=300, patience=40,
#   tum frame'lerde categorical_params={} (saf continuous regresyon).
#
# SIRALI EGITIM (kritik): once door (classification), sonra 12 frame TEK TEK (paralel DEGIL).
# RESUME: ayni scripti tekrar calistirinca kaldigi yerden devam eder
#   - dataset varsa  -> generation atlanir
#   - feature cache  -> mevcut .npy'ler atlanir
#   - bir asamanin train_log.json'i varsa -> o asama (door/frame) atlanir
#
# Calistir (proje kök dizininden):
#   bash bashs/experiment15_bash.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3 || true)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp015"
NN_DIR="$ROOT/NeuralNetworks"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"   # tum frameler acik
SYNTH_CONFIG="$EXP_DIR/config.yaml"

N=20000
SEED=15
DS_NAME="S03D15"
DS_DIR="$ROOT/synths/SynthV3/datasets/$DS_NAME"

# Sirali frame egitim sirasi (config'teki frame sirasi)
FRAMES=(osc_a osc_b noise filter env1 env2 lfo1 lfo2 lfo3 lfo4 reverb delay)

# Worker sayisi: feature extraction icin (CPU core basina 1, max 16)
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
NUM_WORKERS=$(( NPROC < 16 ? NPROC : 16 ))

echo "=============================================="
echo "  Experiment15 — FULL sequential (20k)"
echo "  Root     : $ROOT"
echo "  Python   : $PYTHON"
echo "  Dataset  : $DS_NAME (n=$N, seed=$SEED)"
echo "  Workers  : $NUM_WORKERS"
echo "=============================================="

# ── git yardimcisi: hata olsa bile egitimi durdurmaz ──────────────────────────
git_save() {
    local msg="$1"
    # subshell: caller'in cwd'sini degistirmez
    (
        cd "$ROOT"
        git add -A "$EXP_DIR" 2>/dev/null || true
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "$msg" || true
            git push || echo "  [WARN] git push basarisiz (egitim devam ediyor)"
        else
            echo "  [git] commit edilecek degisiklik yok."
        fi
    )
}

# ── DataGeneratorCLI binary bul ───────────────────────────────────────────────
CLI_BIN="$ROOT/cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"
if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN=$(find "$ROOT" -name "DataGeneratorCLI" -type f 2>/dev/null | grep -v "\.dSYM" | head -1)
fi
if [ -z "${CLI_BIN:-}" ] || [ ! -f "$CLI_BIN" ]; then
    echo "ERROR: DataGeneratorCLI binary bulunamadi. Once build edin (initialization.sh)."
    exit 1
fi
echo "Binary: $CLI_BIN"

# ── 1. Data Generation (resume: $DS_NAME varsa atla) ──────────────────────────
echo ""
echo "--- [1] Data generation ---"
if [ -d "$DS_DIR/samples" ] && [ -n "$(ls -A "$DS_DIR/samples" 2>/dev/null)" ]; then
    echo "Dataset $DS_NAME zaten mevcut, generation atlandi."
else
    "$CLI_BIN" --config "$DATAGEN_CONFIG" --n "$N" --seed "$SEED"
    NEW_DS=$(ls -td "$ROOT/synths/SynthV3/datasets/S03D"*/ 2>/dev/null | head -1)
    NEW_DS="${NEW_DS%/}"
    if [ -z "$NEW_DS" ]; then
        echo "ERROR: Dataset olusturulamadi."
        exit 1
    fi
    if [ "$NEW_DS" != "$DS_DIR" ]; then
        mv "$NEW_DS" "$DS_DIR"
    fi
    echo "Dataset hazir: $DS_NAME"
fi

# ── 2. Feature extraction (PNG YOK; .npy cache = python-loadable backup) ───────
echo ""
echo "--- [2] Feature extraction (.npy cache, PNG yok) ---"
cd "$NN_DIR"
$PYTHON feature_extraction_v2.py \
    --config "$SYNTH_CONFIG" \
    --num-workers "$NUM_WORKERS"

# ── 3. Sirali egitim: once DOOR (classification) ──────────────────────────────
echo ""
echo "--- [3] Sirali egitim ---"
cd "$NN_DIR"

if [ -f "$EXP_DIR/logs/door/train_log.json" ]; then
    echo ">>> door: tamamlanmis (atlandi)."
else
    echo ">>> door (classification) egitiliyor..."
    $PYTHON sequenced_train_door.py --config "$SYNTH_CONFIG"
    git_save "Exp015: door (classification) trained"
fi

# ── 4. Sirali egitim: 12 frame TEK TEK ────────────────────────────────────────
for FRAME in "${FRAMES[@]}"; do
    if [ -f "$EXP_DIR/logs/$FRAME/train_log.json" ]; then
        echo ">>> $FRAME: tamamlanmis (atlandi)."
        continue
    fi
    echo ">>> $FRAME egitiliyor..."
    $PYTHON sequenced_train_frames.py --config "$SYNTH_CONFIG" --frame "$FRAME"
    git_save "Exp015: frame '$FRAME' trained"
done

# ── 5. Validation / evaluation (door + tum frameler) ──────────────────────────
echo ""
echo "--- [5] Evaluation ---"
$PYTHON sequenced_evaluate.py --config "$SYNTH_CONFIG"
git_save "Exp015: evaluation results"

echo ""
echo "  Dataset     : $DS_DIR/"
echo "  Checkpoints : $EXP_DIR/checkpoints/"
echo "  Logs        : $EXP_DIR/logs/"
echo "  Eval        : $EXP_DIR/logs/eval_results.json"
echo "=============================================="
echo "  Experiment15 tamamlandi: $(date)"
echo "=============================================="
