# ONNX Runtime path — sisteme gore ayarla
# macOS: brew install onnxruntime veya manuel indirme
# set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime")  # brew
set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime" CACHE PATH "ONNX Runtime root")

juce_add_plugin(SynthV1Plugin
        PLUGIN_NAME             "SynthV1"
        PLUGIN_MANUFACTURER_CODE Smsy
        PLUGIN_CODE             SV01
        FORMATS                 VST3 Standalone
        PRODUCT_NAME            "SynthV1"
        IS_SYNTH                TRUE
        NEEDS_MIDI_INPUT        TRUE
)

juce_generate_juce_header(SynthV1Plugin)

target_sources(SynthV1Plugin PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
)

target_include_directories(SynthV1Plugin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_link_libraries(SynthV1Plugin PRIVATE
        SynthEngineV1
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_audio_formats
        juce::juce_dsp
        juce::juce_gui_basics
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# ONNX Runtime linkle
if(APPLE)
    target_link_libraries(SynthV1Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.dylib
    )
    # CoreML framework
    target_link_libraries(SynthV1Plugin PRIVATE
            "-framework CoreML"
            "-framework Foundation"
    )
elseif(WIN32)
    target_link_libraries(SynthV1Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib
    )
else()
    target_link_libraries(SynthV1Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )
endif()

# AudioEngine icin ONNX header'lari da lazim
target_include_directories(SynthEngineV1 PUBLIC
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)