#!/bin/bash
# experiment17_bash.sh — Exp017: full sequential training (door + 12 frames) + ONNX export
# RunPod/remote. 30k sample, S03D17 dataset.
#
# Exp015'ten fark (girdi yapisi):
#   main_render = full_frame_hint
#   mel_full  = tam karisim (row["filename"])       — hedef ses, her zaman mevcut
#   mel_prev  = bu frame'e kadarki render (dropout_p=0.3 ile perdeli)  — plugin inference akisi
#   prev_params = onceki frame parametreleri (teacher forcing)
#
# auto_export=true: her frame egitiminden sonra ONNX + meta.json otomatik uretilir.
#
# SIRALI EGITIM (kritik): once door (classification), sonra 12 frame TEK TEK.
# RESUME: ayni scripti tekrar calistirinca kaldigi yerden devam eder
#   - dataset varsa  -> generation atlanir
#   - feature cache  -> mevcut .npy'ler atlanir
#   - bir asamanin train_log.json'i varsa -> o asama atlanir
#
# Calistir (proje kok dizininden):
#   bash bashs/experiment17_bash.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python3 || true)"
EXP_DIR="$ROOT/synths/SynthV3/Experiments/Exp017"
NN_DIR="$ROOT/NeuralNetworks"
DATAGEN_CONFIG="$ROOT/DataGenerator/configs/datagen_base.yaml"
SYNTH_CONFIG="$EXP_DIR/config.yaml"

N=30000
SEED=17
DS_NAME="S03D17"
DS_DIR="$ROOT/synths/SynthV3/datasets/$DS_NAME"

FRAMES=(osc_a osc_b noise filter env1 env2 lfo1 lfo2 lfo3 lfo4 reverb delay)

NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
NUM_WORKERS=$(( NPROC < 16 ? NPROC : 16 ))

echo "=============================================="
echo "  Experiment17 — full sequential + ONNX export"
echo "  Root     : $ROOT"
echo "  Python   : $PYTHON"
echo "  Dataset  : $DS_NAME (n=$N, seed=$SEED)"
echo "  Workers  : $NUM_WORKERS"
echo "=============================================="

# ── git yardimcisi: hata olsa bile egitimi durdurmaz ──────────────────────────
git_save() {
    local msg="$1"
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

# ── 1. Data Generation ────────────────────────────────────────────────────────
echo ""
echo "--- [1] Data generation (S03D17, 30k) ---"
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

# ── 2. Feature extraction ─────────────────────────────────────────────────────
echo ""
echo "--- [2] Feature extraction (.npy cache) ---"
cd "$NN_DIR"
$PYTHON feature_extraction_v2.py \
    --config "$SYNTH_CONFIG" \
    --num-workers "$NUM_WORKERS"

# ── 3. Door (classification) ──────────────────────────────────────────────────
echo ""
echo "--- [3] Sirali egitim ---"
cd "$NN_DIR"

if [ -f "$EXP_DIR/logs/door/train_log.json" ]; then
    echo ">>> door: tamamlanmis (atlandi)."
else
    echo ">>> door (classification) egitiliyor..."
    $PYTHON sequenced_train_door.py --config "$SYNTH_CONFIG"
fi

# ── 4. 12 frame TEK TEK ───────────────────────────────────────────────────────
for FRAME in "${FRAMES[@]}"; do
    if [ -f "$EXP_DIR/logs/$FRAME/train_log.json" ]; then
        echo ">>> $FRAME: tamamlanmis (atlandi)."
        continue
    fi
    echo ">>> $FRAME egitiliyor..."
    $PYTHON sequenced_train_frames.py --config "$SYNTH_CONFIG" --frame "$FRAME"
done

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
echo ""
echo "--- [5] Evaluation ---"
$PYTHON sequenced_evaluate.py --config "$SYNTH_CONFIG"

# ── 6. ONNX export (auto_export=true zaten her frame sonrasi calisir,
#       bu adim eksik kalanlar icin fallback) ────────────────────────────────
echo ""
echo "--- [6] ONNX export (eksik varsa tamamla) ---"
$PYTHON sequenced_export_onnx.py --config "$SYNTH_CONFIG" --frame all

# ── Tek seferlik commit + push ────────────────────────────────────────────────
git_save "Exp017: full sequenced run (full_frame_hint) + ONNX export"

echo ""
echo "  Dataset     : $DS_DIR/"
echo "  Checkpoints : $EXP_DIR/checkpoints/"
echo "  Logs        : $EXP_DIR/logs/"
echo "  Eval        : $EXP_DIR/logs/eval_results.json"
echo "  ONNX        : $EXP_DIR/exports/"
echo "=============================================="
echo "  Experiment17 tamamlandi: $(date)"
echo "=============================================="
