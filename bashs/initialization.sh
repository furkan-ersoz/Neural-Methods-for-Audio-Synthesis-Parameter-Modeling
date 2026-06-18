#!/bin/bash
set -e
echo "=== SS Project Initialization ==="

# System deps (JUCE + audio)
apt-get update -qq && apt-get install -y \
  cmake build-essential pkg-config \
  libfreetype-dev libfontconfig1-dev \
  libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libgl1-mesa-dev libglu1-mesa-dev mesa-common-dev \
  libasound2-dev libpulse-dev libjack-jackd2-dev \
  libgtk-3-dev libwebkit2gtk-4.0-dev \
  libsndfile1-dev ffmpeg

# JUCE submodule
git submodule update --init --recursive

# freetype header symlinks
[ -f /usr/include/ft2build.h ]   || ln -s /usr/include/freetype2/ft2build.h /usr/include/ft2build.h
[ -d /usr/include/freetype ]     || ln -s /usr/include/freetype2/freetype   /usr/include/freetype

# Python deps
pip install -r NeuralNetworks/requirements.txt

# PyTorch Blackwell (sm_120) uyumluluk kontrolu
# RTX 4500 / 5000 serisi gibi yeni GPU'larda nightly gerekir
GPU_CAP=$(python3 -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "0")
if [ "$GPU_CAP" -ge 12 ] 2>/dev/null; then
  echo "[init] Blackwell GPU tespit edildi (sm_${GPU_CAP}x) — PyTorch nightly kuruluyor..."
  pip install --upgrade --pre torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
fi

# JUCE build
cmake -B cmake-build-release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target DataGeneratorCLI -j$(nproc)

# CLI binary path doğrula
CLI_BIN="cmake-build-release/DataGenerator/DataGeneratorCLI_artefacts/Release/DataGeneratorCLI"
if [ ! -f "$CLI_BIN" ]; then
  echo "ERROR: DataGeneratorCLI binary not found at $CLI_BIN"
  exit 1
fi
echo "CLI binary: $CLI_BIN"
echo "=== Initialization complete ==="
