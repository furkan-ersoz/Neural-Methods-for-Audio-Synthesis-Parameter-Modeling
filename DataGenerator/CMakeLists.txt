juce_add_gui_app(DataGenerator
        PRODUCT_NAME "DataGenerator"
)

juce_generate_juce_header(DataGenerator)

target_sources(DataGenerator PRIVATE
        Source/Main.cpp
        Source/MainComponent.cpp
        Source/DatasetWriter.cpp
)

target_include_directories(DataGenerator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_link_libraries(DataGenerator PRIVATE
        SynthEngineV1                    # SynthV1 AudioEngine static lib
        juce::juce_gui_basics
        juce::juce_audio_formats         # WAV yazma
        juce::juce_core
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)