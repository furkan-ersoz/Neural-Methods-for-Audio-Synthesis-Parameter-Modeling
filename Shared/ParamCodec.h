#pragma once
#include "ISynth.h"
#include <algorithm>
#include <cmath>

namespace ParamCodec {

inline bool isDiscrete(const ParamDescriptor& d) { return d.numClasses >= 2; }

// norm [0,1] -> class index [0, N-1]
inline int normToIndex(const ParamDescriptor& d, float norm) {
    const int N = d.numClasses;
    if (N < 2) return 0;
    return std::clamp(static_cast<int>(std::lround(norm * (N - 1))), 0, N - 1);
}

// class index [0, N-1] -> representative norm [0,1]
inline float indexToNorm(const ParamDescriptor& d, int k) {
    const int N = d.numClasses;
    if (N < 2) return 0.0f;
    return static_cast<float>(std::clamp(k, 0, N - 1)) / static_cast<float>(N - 1);
}

// Signed value for symmetric-bipolar integer params (center = 0)
// e.g. pitch_coarse N=49 -> -24..+24 ; semitone_ofs N=25 -> -12..+12
inline int normToIntValue(const ParamDescriptor& d, float norm) {
    return normToIndex(d, norm) - (d.numClasses - 1) / 2;
}

} // namespace ParamCodec
