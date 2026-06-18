#pragma once
#include "../SamplingProfile.h"

namespace SamplingProfiles {

inline SamplingProfile makeDataGeneration() {
    return SamplingProfile{
        .name        = "DataGeneration",
        .description = "Balanced ML dataset — stratified LFO, safe level ranges",

        .paramRules  = {
            {.paramName = "osc_a_level",      .randomMin = 0.5f,  .randomMax = 1.0f},
            {.paramName = "osc_b_level",      .randomMin = 0.5f,  .randomMax = 1.0f},
            {.paramName = "noise_level",      .randomMin = 0.20f, .randomMax = 0.40f},
            {.paramName = "noise_stereo",     .randomMin = 0.0f,  .randomMax = 0.65f},

            {.paramName = "filter_cutoff",    .randomMin = 0.35f, .randomMax = 1.0f},
            {.paramName = "filter_resonance", .randomMin = 0.0f,  .randomMax = 0.85f},

            {.paramName = "osc_a_harmonic2",  .randomMax = 0.5f},
            {.paramName = "osc_a_harmonic3",  .randomMax = 0.5f},

            {.paramName = "reverb_mix",       .fixedProb = 1.0f},
            {.paramName = "delay_mix",        .fixedProb = 1.0f},
        },

        .frameRules  = {
            {"osc_b",  0.4f},
            {"noise",  0.2f},
            {"filter", 0.2f},
            {"lfo1",   0.0f},
            {"lfo2",   0.0f},
            {"lfo3",   0.0f},
            {"lfo4",   0.0f},
            {"reverb", 0.2f},
            {"delay",  0.2f},
        },

        .lfoStratified = true,
    };
}

} // namespace SamplingProfiles
