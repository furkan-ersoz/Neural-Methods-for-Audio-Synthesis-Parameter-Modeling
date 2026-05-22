#pragma once
#include <array>
#include <string_view>
#include <cmath>

namespace SynthTemplate {

    struct SynthParameters {
        float amplitude    = 0.8f;   // [0,1]
        float attackTime   = 0.05f;  // [0,1] → 1ms–3s
        float decayTime    = 0.2f;
        float sustainLevel = 0.7f;   // [0,1] doğrudan
        float releaseTime  = 0.3f;

        static constexpr int NUM_PARAMS = 5;

        static constexpr std::array<std::string_view, NUM_PARAMS> PARAM_NAMES = {{
            "amplitude", "attack", "decay", "sustain", "release"
        }};
    };

    // Dönüşüm fonksiyonları — Engine ve DatasetWriter ikisi de buradan kullanır
    inline float midiToHz(int midiNote) {
        return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
    }

    inline float toTimeSeconds(float norm) {
        return 0.001f + norm * 2.999f;         // 1ms – 3s
    }

} // namespace SynthTemplate