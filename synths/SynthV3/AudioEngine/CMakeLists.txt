# Core: SynthEngine only — no ONNX, no JUCE DSP — used by DataGenerator (OBJECT = no linker stripping)
add_library(SynthEngineV3Core OBJECT
        SynthEngineV3.cpp
        SynthEngineV3.h
        SynthParametersV3.h
)

target_include_directories(SynthEngineV3Core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
)

target_compile_features(SynthEngineV3Core PUBLIC cxx_std_17)

# Do NOT link JUCE modules here — linking them causes JUCE to compile each module's
# .mm implementation into this OBJECT library's objects.  DataGenerator then links
# the same modules directly → 4000+ duplicate symbols at link time.
# Instead, borrow only the include dirs and compile definitions (header-only access),
# exactly as SynthEngineV1Core and SynthEngineV2Core do — they carry no JUCE links.
# The consuming target (DataGenerator / Plugin) provides the actual module compilation.
target_include_directories(SynthEngineV3Core PRIVATE
        $<TARGET_PROPERTY:juce::juce_audio_basics,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:juce::juce_dsp,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(SynthEngineV3Core PRIVATE
        $<TARGET_PROPERTY:juce::juce_audio_basics,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:juce::juce_dsp,INTERFACE_COMPILE_DEFINITIONS>
)

# Full: Core + InferenceEngine + MelSpectrogram + ONNX — used by Plugin
add_library(SynthEngineV3 STATIC
        SynthEngineV3.cpp
        SynthEngineV3.h
        SynthParametersV3.h
        MelSpectrogram.cpp
        MelSpectrogram.h
        InferenceEngine.cpp
        InferenceEngine.h
)

target_include_directories(SynthEngineV3 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/Shared
        ${ONNX_RUNTIME_DIR}/include/onnxruntime
)

target_compile_features(SynthEngineV3 PUBLIC cxx_std_17)

target_link_libraries(SynthEngineV3 PUBLIC
        juce::juce_core
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_dsp
)

# Render test — remove after validating all 72 params render correctly
juce_add_console_app(SynthV3RenderTest PRODUCT_NAME "SynthV3RenderTest")
target_sources(SynthV3RenderTest PRIVATE RenderTest.cpp)
target_link_libraries(SynthV3RenderTest PRIVATE SynthEngineV3)

# ParamCodec verification — remove after P0 discrete-codec validation is done
juce_add_console_app(SynthV3ParamCodecTest PRODUCT_NAME "SynthV3ParamCodecTest")
target_sources(SynthV3ParamCodecTest PRIVATE ParamCodecTest.cpp)
target_link_libraries(SynthV3ParamCodecTest PRIVATE SynthEngineV3)
target_link_libraries(SynthV3ParamCodecTest PRIVATE juce::juce_audio_formats)
