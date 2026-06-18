juce_add_gui_app(DataGenerator
        PRODUCT_NAME "DataGenerator"
)

juce_generate_juce_header(DataGenerator)

target_sources(DataGenerator PRIVATE
        Source/Main.cpp
        Source/MainComponent.cpp
        Source/DatasetWriter.cpp
        Source/ParamConfigResolver.cpp
)

target_include_directories(DataGenerator PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_SOURCE_DIR}/Shared
)

target_link_libraries(DataGenerator PRIVATE
        SynthEngineV1Core                # SynthV1 — registered at static-init time
        SynthEngineV2Core                # SynthV2 — registered at static-init time
        SynthEngineV3Core                # SynthV3 — registered at static-init time
        # SynthVersionAdder.py appends each new SynthEngineVNCore target above this line
        juce::juce_gui_basics
        juce::juce_audio_formats
        juce::juce_core
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)

# ─── Headless CLI target ───────────────────────────────────────────────────────
juce_add_console_app(DataGeneratorCLI
        PRODUCT_NAME "DataGeneratorCLI"
)

target_compile_definitions(DataGeneratorCLI PRIVATE
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
)

target_sources(DataGeneratorCLI PRIVATE
        Source/DataGeneratorCLI.cpp
        Source/DatasetWriter.cpp
        Source/ParamConfigResolver.cpp
)

target_include_directories(DataGeneratorCLI PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_SOURCE_DIR}/Shared
)

target_link_libraries(DataGeneratorCLI PRIVATE
        SynthEngineV1Core
        SynthEngineV2Core
        SynthEngineV3Core
        # SynthVersionAdder.py appends each new SynthEngineVNCore target above this line
        juce::juce_audio_formats
        juce::juce_core
        PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
