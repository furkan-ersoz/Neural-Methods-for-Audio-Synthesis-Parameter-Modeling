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

target_link_libraries(SynthV1Plugin PRIVATE
        SynthEngineV1          # AudioEngine static lib
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_gui_basics
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)