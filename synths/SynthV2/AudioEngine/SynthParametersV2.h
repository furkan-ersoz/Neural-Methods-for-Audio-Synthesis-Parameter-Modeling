#pragma once
#include <array>
#include <cmath>
#include <string_view>
#include <vector>
#include "ISynth.h"

namespace SynthV2 {

    // ─── ADD / REMOVE PREDICTED PARAMETERS HERE ───────────────────────────────
    // All stored as normalised [0,1].  Engine converts to physical units.
    // Amplitude is excluded — internal to the engine, normalised during data gen.
    static constexpr std::array PARAM_DESCRIPTORS = {
        ParamDescriptor{ "osc_waveform",      0.500f,  0.0f, 1.0f },  // 0=sine 0.5=saw 1=square
        ParamDescriptor{ "osc_detune",        0.000f,  0.0f, 1.0f },  // [0,50] cents, linear
        ParamDescriptor{ "pulse_width",       0.500f,  0.0f, 1.0f },  // [0.1,0.9], linear
        ParamDescriptor{ "fm_ratio",          0.067f,  0.0f, 1.0f },  // [0.5,8], linear → default 1.0
        ParamDescriptor{ "fm_depth",          0.000f,  0.0f, 1.0f },  // 0 = FM inactive
        ParamDescriptor{ "filter_cutoff",     0.820f,  0.0f, 1.0f },  // [200,18k] Hz, log → ~8kHz
        ParamDescriptor{ "filter_resonance",  0.103f,  0.0f, 1.0f },  // [0.1,4] Q, linear → ~0.5
        ParamDescriptor{ "filter_env_amount", 0.000f,  0.0f, 1.0f },
        ParamDescriptor{ "attack",            0.489f,  0.0f, 1.0f },  // [1ms,3s] log → ~50ms
        ParamDescriptor{ "decay",             0.662f,  0.0f, 1.0f },  // [1ms,3s] log → ~200ms
        ParamDescriptor{ "sustain",           0.700f,  0.0f, 1.0f },  // linear, direct
        ParamDescriptor{ "release",           0.750f,  0.0f, 1.0f },  // [1ms,2s] log → ~300ms
        ParamDescriptor{ "noise_amount",      0.000f,  0.0f, 1.0f },
        ParamDescriptor{ "lfo_depth",         0.000f,  0.0f, 1.0f },  // modulates filter_cutoff, 3Hz
    };
    // ──────────────────────────────────────────────────────────────────────────

    static constexpr int NUM_PARAMS = static_cast<int>(PARAM_DESCRIPTORS.size());

    struct SynthParameters {
        // Internal — not predicted.
        float amplitude = 0.8f;

        // Predicted parameters — order must match PARAM_DESCRIPTORS above.
        float oscWaveform  = PARAM_DESCRIPTORS[0].defaultVal;
        float oscDetune    = PARAM_DESCRIPTORS[1].defaultVal;
        float pulseWidth   = PARAM_DESCRIPTORS[2].defaultVal;
        float fmRatio      = PARAM_DESCRIPTORS[3].defaultVal;
        float fmDepth      = PARAM_DESCRIPTORS[4].defaultVal;
        float filterCutoff = PARAM_DESCRIPTORS[5].defaultVal;
        float filterRes    = PARAM_DESCRIPTORS[6].defaultVal;
        float filterEnvAmt = PARAM_DESCRIPTORS[7].defaultVal;
        float attackTime   = PARAM_DESCRIPTORS[8].defaultVal;
        float decayTime    = PARAM_DESCRIPTORS[9].defaultVal;
        float sustainLevel = PARAM_DESCRIPTORS[10].defaultVal;
        float releaseTime  = PARAM_DESCRIPTORS[11].defaultVal;
        float noiseAmount  = PARAM_DESCRIPTORS[12].defaultVal;
        float lfoDepth     = PARAM_DESCRIPTORS[13].defaultVal;

        // ─── UPDATE THESE TWO WHEN ADDING A PARAMETER ─────────────────────────
        std::vector<float> toVector() const {
            return { oscWaveform, oscDetune, pulseWidth, fmRatio, fmDepth,
                     filterCutoff, filterRes, filterEnvAmt,
                     attackTime, decayTime, sustainLevel, releaseTime,
                     noiseAmount, lfoDepth };
        }

        void fromVector(const std::vector<float>& v) {
            if (static_cast<int>(v.size()) < NUM_PARAMS) return;
            oscWaveform  = v[0];
            oscDetune    = v[1];
            pulseWidth   = v[2];
            fmRatio      = v[3];
            fmDepth      = v[4];
            filterCutoff = v[5];
            filterRes    = v[6];
            filterEnvAmt = v[7];
            attackTime   = v[8];
            decayTime    = v[9];
            sustainLevel = v[10];
            releaseTime  = v[11];
            noiseAmount  = v[12];
            lfoDepth     = v[13];
        }
        // ──────────────────────────────────────────────────────────────────────
    };

    inline float midiToHz(int midiNote) {
        return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
    }

    // norm [0,1] → seconds, log-mapped:  min * (max/min)^norm
    inline float normToAttackDecaySec(float norm) {
        return 0.001f * std::pow(3000.0f, norm);   // 1ms – 3.0s
    }
    inline float normToReleaseSec(float norm) {
        return 0.001f * std::pow(2000.0f, norm);   // 1ms – 2.0s
    }

    // norm [0,1] → Hz, log-mapped
    inline float normToFilterHz(float norm) {
        return 200.0f * std::pow(90.0f, norm);     // 200Hz – 18000Hz
    }

    inline float normToLinear(float norm, float minV, float maxV) {
        return minV + norm * (maxV - minV);
    }

} // namespace SynthV2
