add_library(SynthEngineV1 STATIC
        SynthEngineV1.cpp
        SynthEngineV1.h
        SynthParametersV1.h
)
target_include_directories(SynthEngineV1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(SynthEngineV1 PUBLIC cxx_std_17)