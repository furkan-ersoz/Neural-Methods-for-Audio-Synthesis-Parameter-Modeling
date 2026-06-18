#pragma once
#include "Profiles/DataGeneration.h"
#include "Profiles/OscAOnly.h"
#include <vector>

namespace SamplingProfiles {

inline std::vector<SamplingProfile> getAllProfiles() {
    return { makeDataGeneration(), makeOscAOnly() };
}

} // namespace SamplingProfiles
