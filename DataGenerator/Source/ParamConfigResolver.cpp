#include "ParamConfigResolver.h"

#include <algorithm>
#include <juce_core/juce_core.h>

namespace DataGen {

std::vector<ParamConfig> resolveParamConfigs(
    const SamplingProfiles::SamplingProfile& profile,
    const std::vector<ParamDescriptor>&      descs)
{
    std::vector<ParamConfig> configs;
    configs.reserve(descs.size());

    for (const auto& desc : descs) {
        const auto* rule     = SamplingProfiles::findRule(profile, desc.name);
        const auto  resolved = SamplingProfiles::resolveRule(desc, rule);

        ParamConfig pc;
        pc.rangeMin = resolved.randomMin;
        pc.rangeMax = resolved.randomMax;
        pc.fixedVal = resolved.fixedValue;

        if (resolved.fixedProb >= 1.0f) {
            pc.strategy = Strategy::Fixed;
        } else if (desc.paramClass == "categorical" || desc.paramClass == "integer") {
            pc.strategy   = Strategy::Discrete;
            pc.numClasses = desc.numClasses;
            if (pc.numClasses < 2)
                juce::Logger::writeToLog("WARNING: param '" + juce::String(desc.name.data())
                    + "' is " + juce::String(desc.paramClass.data())
                    + " but has numClasses < 2; sampling will be fixed at rangeMin.");
        } else if (desc.paramClass == "continuous_log") {
            pc.strategy = Strategy::RandomLog;
            pc.rangeMin = std::max(pc.rangeMin, 1e-4f);
        } else {
            pc.strategy = Strategy::RandomUniform;
        }

        configs.push_back(pc);
    }

    return configs;
}

} // namespace DataGen
