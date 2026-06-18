#pragma once
#include <array>
#include <cmath>
#include <string_view>
#include <vector>
#include "ISynth.h"

namespace SynthV1 {

    // ─── ADD / REMOVE PREDICTED PARAMETERS HERE ───────────────────────────────
    // Amplitude is excluded — it is internal to the engine and normalised away
    // during data generation. Only add parameters the model should predict.
    static constexpr std::array PARAM_DESCRIPTORS = {
        ParamDescriptor{ "attack",     0.05f, 0.0f, 1.0f },
        ParamDescriptor{ "decay",      0.20f, 0.0f, 1.0f },
        ParamDescriptor{ "sustain",    0.70f, 0.0f, 1.0f },
        ParamDescriptor{ "release",    0.30f, 0.0f, 1.0f },
        ParamDescriptor{ "harmonic2",  0.30f, 0.0f, 1.0f },  // 2nd harmonic mix (2f)
        ParamDescriptor{ "harmonic3",  0.15f, 0.0f, 1.0f },  // 3rd harmonic mix (3f)
    };
    // ──────────────────────────────────────────────────────────────────────────

    static constexpr int NUM_PARAMS = static_cast<int>(PARAM_DESCRIPTORS.size());

    struct SynthParameters {
        // Internal — not predicted. Engine always plays at full amplitude;
        // DataGenerator normalises to target RMS.
        float amplitude    = 0.8f;

        // Predicted parameters — order must match PARAM_DESCRIPTORS above.
        float attackTime   = PARAM_DESCRIPTORS[0].defaultVal;  // [0,1] → 1ms–3s
        float decayTime    = PARAM_DESCRIPTORS[1].defaultVal;
        float sustainLevel = PARAM_DESCRIPTORS[2].defaultVal;  // [0,1] direct
        float releaseTime  = PARAM_DESCRIPTORS[3].defaultVal;
        float harmonic2    = PARAM_DESCRIPTORS[4].defaultVal;  // 2nd harmonic mix
        float harmonic3    = PARAM_DESCRIPTORS[5].defaultVal;  // 3rd harmonic mix

        // ─── UPDATE THESE TWO WHEN ADDING A PARAMETER ─────────────────────────
        std::vector<float> toVector() const {
            return { attackTime, decayTime, sustainLevel, releaseTime,
                     harmonic2, harmonic3 };
        }

        void fromVector(const std::vector<float>& v) {
            if (static_cast<int>(v.size()) < NUM_PARAMS) return;
            attackTime   = v[0];
            decayTime    = v[1];
            sustainLevel = v[2];
            releaseTime  = v[3];
            harmonic2    = v[4];
            harmonic3    = v[5];
        }
        // ──────────────────────────────────────────────────────────────────────
    };

    inline float midiToHz(int midiNote) {
        return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
    }

    inline float toTimeSeconds(float norm) {
        return 0.001f + norm * 2.999f;  // 1ms – 3s
    }

} // namespace SynthV1
