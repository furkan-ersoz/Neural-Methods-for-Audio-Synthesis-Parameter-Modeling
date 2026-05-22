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

target_link_libraries(SynthTemplatePlugin PRIVATE
        SynthEngineTemplate          # AudioEngine static lib
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_gui_basics
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)