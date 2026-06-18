# Core: SynthEngine only — no ONNX, no JUCE DSP — used by DataGenerator (OBJECT = no linker stripping)
add_library(SynthEngineV1Core OBJECT
        SynthEngineV1.cpp
        SynthEngineV1.h
        SynthParametersV1.h
)

target_include_directories(SynthEngineV1Core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
)

target_compile_features(SynthEngineV1Core PUBLIC cxx_std_17)

# Full: Core + InferenceEngine + MelSpectrogram + ONNX — used by Plugin
add_library(SynthEngineV1 STATIC
        SynthEngineV1.cpp
        SynthEngineV1.h
        SynthParametersV1.h
        MelSpectrogram.cpp
        MelSpectrogram.h
        InferenceEngine.cpp
        InferenceEngine.h
)

target_include_directories(SynthEngineV1 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_compile_features(SynthEngineV1 PUBLIC cxx_std_17)

target_link_libraries(SynthEngineV1 PUBLIC
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_dsp
)
