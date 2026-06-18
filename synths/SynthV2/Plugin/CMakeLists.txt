# ONNX Runtime path — sisteme gore ayarla
# macOS: brew install onnxruntime veya manuel indirme
# set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime")  # brew
set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime" CACHE PATH "ONNX Runtime root")

juce_add_plugin(SynthV2Plugin
        PLUGIN_NAME             "SynthV2"
        PLUGIN_MANUFACTURER_CODE Smsy
        PLUGIN_CODE             SV02
        FORMATS                 VST3 Standalone
        PRODUCT_NAME            "SynthV2"
        IS_SYNTH                TRUE
        NEEDS_MIDI_INPUT        TRUE
)

juce_generate_juce_header(SynthV2Plugin)

target_sources(SynthV2Plugin PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
)

target_include_directories(SynthV2Plugin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_link_libraries(SynthV2Plugin PRIVATE
        SynthEngineV2
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
    target_link_libraries(SynthV2Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.dylib
    )
    # CoreML framework
    target_link_libraries(SynthV2Plugin PRIVATE
            "-framework CoreML"
            "-framework Foundation"
    )
elseif(WIN32)
    target_link_libraries(SynthV2Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib
    )
else()
    target_link_libraries(SynthV2Plugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )
endif()

# AudioEngine icin ONNX header'lari da lazim
target_include_directories(SynthEngineV2 PUBLIC
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)