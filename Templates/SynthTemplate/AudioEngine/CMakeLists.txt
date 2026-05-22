add_library(SynthEngineTemplate STATIC
        SynthEngineTemplate.cpp
        SynthEngineTemplate.h
        SynthParametersTemplate.h
)
target_include_directories(SynthEngineTemplate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(SynthEngineTemplate PUBLIC cxx_std_17)