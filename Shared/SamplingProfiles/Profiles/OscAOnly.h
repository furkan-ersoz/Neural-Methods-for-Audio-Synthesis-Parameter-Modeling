#pragma once
#include "../SamplingProfile.h"

namespace SamplingProfiles {

// Sanity-check profile: osc_a always active, all other components always bypassed.
// Full render = pure osc_a signal — clean correlation for frame model training.
inline SamplingProfile makeOscAOnly() {
    return SamplingProfile{
        .name        = "OscAOnly",
        .description = "All components bypassed except osc_a — used for sanity/regression tests",

        .paramRules  = {
            {.paramName = "osc_a_level",      .randomMin = 0.5f,  .randomMax = 1.0f},
            {.paramName = "reverb_mix",       .fixedProb = 1.0f},
            {.paramName = "delay_mix",        .fixedProb = 1.0f},
        },

        .frameRules  = {
            {"osc_b",  0.0f},
            {"noise",  0.0f},
            {"filter", 0.0f},
            {"lfo1",   0.0f},
            {"lfo2",   0.0f},
            {"lfo3",   0.0f},
            {"lfo4",   0.0f},
            {"reverb", 0.0f},
            {"delay",  0.0f},
            // env1 bypassed: flat amplitude, no time-domain shaping in osc_a frame
            {"env1",   0.0f},
        },

        .lfoStratified = false,
    };
}

} // namespace SamplingProfiles
