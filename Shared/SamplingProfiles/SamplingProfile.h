#pragma once
#include "../ISynth.h"
#include <string>
#include <string_view>
#include <vector>

namespace SamplingProfiles {

struct ParamRule {
    std::string_view paramName;
    float            randomMin       = -1.0f;
    float            randomMax       = -1.0f;
    std::string_view distribution    = "";
    int              categoricalSteps = 6;
    float            fixedProb       = 0.0f;
    float            fixedValue      = -1.0f;
};

struct FrameRule {
    std::string_view frameName;
    float activeProb = 0.5f;
};

struct SamplingProfile {
    std::string_view         name;
    std::string_view         description;
    std::vector<ParamRule>   paramRules;
    std::vector<FrameRule>   frameRules;
    bool lfoStratified = false;
    // true → active_lfo_count is drawn uniform 0-4 first,
    //        then that many LFOs are activated randomly.
    // false → each LFO is an independent coin flip (default behaviour).
};

// Returns the ParamRule for a given paramName, or nullptr if not in this profile.
inline const ParamRule* findRule(const SamplingProfile& profile, std::string_view paramName) {
    for (const auto& rule : profile.paramRules)
        if (rule.paramName == paramName) return &rule;
    return nullptr;
}

struct ResolvedRule {
    float       randomMin;
    float       randomMax;
    std::string distribution;  // "uniform", "log", "categorical"
    int         categoricalSteps;
    float       fixedProb;
    float       fixedValue;
};

// Given a descriptor and an optional rule, return effective sampling parameters.
// Falls back to descriptor values when rule fields are -1 or "".
inline ResolvedRule resolveRule(const ParamDescriptor& desc, const ParamRule* rule) {
    ResolvedRule r;
    r.randomMin       = (rule && rule->randomMin >= 0.0f) ? rule->randomMin : desc.minVal;
    r.randomMax       = (rule && rule->randomMax >= 0.0f) ? rule->randomMax : desc.maxVal;
    r.categoricalSteps = (rule && rule->categoricalSteps > 0) ? rule->categoricalSteps : 6;
    r.fixedProb       = rule ? rule->fixedProb : 0.0f;
    r.fixedValue      = (rule && rule->fixedValue >= 0.0f) ? rule->fixedValue : desc.defaultVal;

    if (rule && !rule->distribution.empty()) {
        r.distribution = std::string(rule->distribution);
    } else {
        std::string_view cls = desc.paramClass;
        if (cls == "continuous_log")
            r.distribution = "log";
        else if (cls == "categorical" || cls == "morph" || cls == "integer")
            r.distribution = "categorical";
        else
            r.distribution = "uniform";
    }
    return r;
}

} // namespace SamplingProfiles
