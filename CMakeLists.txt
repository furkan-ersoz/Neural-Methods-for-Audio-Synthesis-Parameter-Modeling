cmake_minimum_required(VERSION 3.22)
project(SmartSynth VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── JUCE ───────────────────────────────────────────────────────────────────
add_subdirectory(libs/JUCE)

# ─── Template ───────────────────────────────────────────────────────────────
add_subdirectory(Templates/SynthTemplate/AudioEngine)
add_subdirectory(Templates/SynthTemplate/Plugin)

# ─── DataGenerator ──────────────────────────────────────────────────────────
add_subdirectory(DataGenerator)
add_subdirectory(synths/SynthV1)

# ─── Synth Versiyonları (SynthVersionAdder.py buraya satır ekler) ───────────
# add_subdirectory(synths/SynthV1/AudioEngine)
# add_subdirectory(synths/SynthV1/Plugin)