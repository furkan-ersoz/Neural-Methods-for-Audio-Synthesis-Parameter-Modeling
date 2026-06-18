#pragma once

#include "DatasetWriter.h"
#include "ISynth.h"
#include "SamplingProfiles/SamplingProfile.h"

#include <vector>

namespace DataGen {

// Resolves a SamplingProfile + param descriptors into the per-param sampling
// recipe (ParamConfig). Single source of truth shared by CLI and GUI.
std::vector<ParamConfig> resolveParamConfigs(
    const SamplingProfiles::SamplingProfile& profile,
    const std::vector<ParamDescriptor>&      descs);

} // namespace DataGen
