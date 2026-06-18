cmake_minimum_required(VERSION 3.22)
project(SmartSynth VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Shared ─────────────────────────────────────────────────────────────────
add_subdirectory(Shared)

# ─── JUCE ───────────────────────────────────────────────────────────────────
add_subdirectory(libs/JUCE)

# ─── Template ───────────────────────────────────────────────────────────────
add_subdirectory(Templates/SynthTemplate/AudioEngine)
add_subdirectory(Templates/SynthTemplate/Plugin)

# ─── DataGenerator ──────────────────────────────────────────────────────────
add_subdirectory(DataGenerator)
add_subdirectory(synths/SynthV1)
add_subdirectory(synths/SynthV2)
add_subdirectory(synths/SynthV3)

# ─── Synth Versiyonları (SynthVersionAdder.py buraya satır ekler) ───────────