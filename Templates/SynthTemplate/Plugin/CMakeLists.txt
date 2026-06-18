# ONNX Runtime path — sisteme gore ayarla
# macOS: brew install onnxruntime veya manuel indirme
# set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime")  # brew
set(ONNX_RUNTIME_DIR "/opt/homebrew/opt/onnxruntime" CACHE PATH "ONNX Runtime root")

juce_add_plugin(SynthTemplatePlugin
        PLUGIN_NAME             "SynthTemplate"
        PLUGIN_MANUFACTURER_CODE Smsy
        PLUGIN_CODE             Stp1
        FORMATS                 VST3 Standalone
        PRODUCT_NAME            "SynthTemplate"
        IS_SYNTH                TRUE
        NEEDS_MIDI_INPUT        TRUE
)

juce_generate_juce_header(SynthTemplatePlugin)

target_sources(SynthTemplatePlugin PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
)

target_include_directories(SynthTemplatePlugin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_link_libraries(SynthTemplatePlugin PRIVATE
        SynthEngineTemplate
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
    target_link_libraries(SynthTemplatePlugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.dylib
    )
    # CoreML framework
    target_link_libraries(SynthTemplatePlugin PRIVATE
            "-framework CoreML"
            "-framework Foundation"
    )
elseif(WIN32)
    target_link_libraries(SynthTemplatePlugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib
    )
else()
    target_link_libraries(SynthTemplatePlugin PRIVATE
            ${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so
    )
endif()

# AudioEngine icin ONNX header'lari da lazim
target_include_directories(SynthEngineTemplate PUBLIC
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)