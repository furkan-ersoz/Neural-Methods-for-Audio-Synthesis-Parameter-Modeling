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
)

target_compile_features(SynthEngineTemplate PUBLIC cxx_std_17)

# JUCE modulleri — InferenceEngine ve MelSpectrogram icin
target_link_libraries(SynthEngineTemplate PUBLIC
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_dsp
)

# ONNX Runtime
target_include_directories(SynthEngineTemplate PUBLIC
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)