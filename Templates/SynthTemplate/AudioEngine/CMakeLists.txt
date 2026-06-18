# Core: SynthEngine only — no ONNX, no JUCE DSP — used by DataGenerator (OBJECT = no linker stripping)
add_library(SynthEngineTemplateCore OBJECT
        SynthEngineTemplate.cpp
        SynthEngineTemplate.h
        SynthParametersTemplate.h
)

target_include_directories(SynthEngineTemplateCore PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
)

target_compile_features(SynthEngineTemplateCore PUBLIC cxx_std_17)

# Full: Core + InferenceEngine + MelSpectrogram + ONNX — used by Plugin
add_library(SynthEngineTemplate STATIC
        SynthEngineTemplate.cpp
        SynthEngineTemplate.h
        SynthParametersTemplate.h
        MelSpectrogram.cpp
        MelSpectrogram.h
        InferenceEngine.cpp
        InferenceEngine.h
)

target_include_directories(SynthEngineTemplate PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_compile_features(SynthEngineTemplate PUBLIC cxx_std_17)

target_link_libraries(SynthEngineTemplate PUBLIC
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_dsp
)
