#pragma once
#include <array>
#include <cmath>
#include <string_view>
#include <vector>
#include "ISynth.h"

namespace SynthV3 {

    static constexpr std::array PARAM_DESCRIPTORS = {
        // ── OSC A (always active) ──────────────────────────────────────────
        ParamDescriptor{"osc_a_waveform",     0.50f, 0.0f, 1.0f, "osc_a", "morph"},
        ParamDescriptor{"osc_a_warp_amt",     0.00f, 0.0f, 1.0f, "osc_a", "continuous"},
        ParamDescriptor{"osc_a_warp_type",    0.00f, 0.0f, 1.0f, "osc_a", "categorical", 4},
        ParamDescriptor{"osc_a_detune",       0.00f, 0.0f, 1.0f, "osc_a", "continuous"},
        ParamDescriptor{"osc_a_blend",        0.50f, 0.0f, 1.0f, "osc_a", "continuous"},
        ParamDescriptor{"osc_a_pitch_coarse", 0.50f, 0.0f, 1.0f, "osc_a", "integer", 49},
        ParamDescriptor{"osc_a_pitch_fine",   0.50f, 0.0f, 1.0f, "osc_a", "bipolar"},
        ParamDescriptor{"osc_a_level",        0.80f, 0.0f, 1.0f, "osc_a", "continuous"},
        ParamDescriptor{"osc_a_harmonic2",    0.00f, 0.0f, 1.0f, "osc_a", "continuous"},
        ParamDescriptor{"osc_a_harmonic3",    0.00f, 0.0f, 1.0f, "osc_a", "continuous"},

        // ── OSC B (bypassable) ─────────────────────────────────────────────
        ParamDescriptor{"osc_b_waveform",     0.50f, 0.0f, 1.0f, "osc_b", "morph"},
        ParamDescriptor{"osc_b_warp_amt",     0.00f, 0.0f, 1.0f, "osc_b", "continuous"},
        ParamDescriptor{"osc_b_warp_type",    0.00f, 0.0f, 1.0f, "osc_b", "categorical", 4},
        ParamDescriptor{"osc_b_detune",       0.00f, 0.0f, 1.0f, "osc_b", "continuous"},
        ParamDescriptor{"osc_b_blend",        0.50f, 0.0f, 1.0f, "osc_b", "continuous"},
        ParamDescriptor{"osc_b_pitch_coarse", 0.50f, 0.0f, 1.0f, "osc_b", "integer", 49},
        ParamDescriptor{"osc_b_pitch_fine",   0.50f, 0.0f, 1.0f, "osc_b", "bipolar"},
        ParamDescriptor{"osc_b_level",        0.80f, 0.0f, 1.0f, "osc_b", "continuous"},
        ParamDescriptor{"osc_b_phase",        0.00f, 0.0f, 1.0f, "osc_b", "continuous"},
        ParamDescriptor{"osc_b_semitone_ofs", 0.50f, 0.0f, 1.0f, "osc_b", "integer", 25},

        // ── Noise (bypassable) ─────────────────────────────────────────────
        ParamDescriptor{"noise_color",        0.00f, 0.0f, 1.0f, "noise", "categorical", 3},
        ParamDescriptor{"noise_level",        0.50f, 0.0f, 1.0f, "noise", "continuous"}, // BREAKING DEFAULT (was 0.00)
        ParamDescriptor{"noise_stereo",       0.00f, 0.0f, 1.0f, "noise", "continuous"},

        // ── Filter (bypassable) ────────────────────────────────────────────
        ParamDescriptor{"filter_cutoff",      0.82f, 0.0f, 1.0f, "filter", "continuous_log"},
        ParamDescriptor{"filter_resonance",   0.10f, 0.0f, 1.0f, "filter", "continuous"},
        ParamDescriptor{"filter_type",        0.00f, 0.0f, 1.0f, "filter", "categorical", 6},
        ParamDescriptor{"filter_drive",       0.00f, 0.0f, 1.0f, "filter", "continuous"},
        ParamDescriptor{"filter_keytrack",    0.00f, 0.0f, 1.0f, "filter", "continuous"},
        ParamDescriptor{"filter_env_amt",     0.00f, 0.0f, 1.0f, "filter", "bipolar"},

        // ── ENV1 — OSC A amplitude (never bypassed) ────────────────────────
        ParamDescriptor{"env1_attack",        0.05f, 0.0f, 1.0f, "env1", "continuous_log"},
        ParamDescriptor{"env1_hold",          0.00f, 0.0f, 1.0f, "env1", "continuous_log"},
        ParamDescriptor{"env1_decay",         0.20f, 0.0f, 1.0f, "env1", "continuous_log"},
        ParamDescriptor{"env1_sustain",       0.70f, 0.0f, 1.0f, "env1", "continuous"},
        ParamDescriptor{"env1_release",       0.30f, 0.0f, 1.0f, "env1", "continuous_log"},
        ParamDescriptor{"env1_atk_curve",     0.50f, 0.0f, 1.0f, "env1", "bipolar"},
        ParamDescriptor{"env1_dec_curve",     0.50f, 0.0f, 1.0f, "env1", "bipolar"},

        // ── ENV2 — OSC B amplitude (bypassed with OSC B) ───────────────────
        ParamDescriptor{"env2_attack",        0.05f, 0.0f, 1.0f, "env2", "continuous_log"},
        ParamDescriptor{"env2_hold",          0.00f, 0.0f, 1.0f, "env2", "continuous_log"},
        ParamDescriptor{"env2_decay",         0.20f, 0.0f, 1.0f, "env2", "continuous_log"},
        ParamDescriptor{"env2_sustain",       0.70f, 0.0f, 1.0f, "env2", "continuous"},
        ParamDescriptor{"env2_release",       0.30f, 0.0f, 1.0f, "env2", "continuous_log"},
        ParamDescriptor{"env2_atk_curve",     0.50f, 0.0f, 1.0f, "env2", "bipolar"},
        ParamDescriptor{"env2_dec_curve",     0.50f, 0.0f, 1.0f, "env2", "bipolar"},

        // ── LFO 1–4 (each independently bypassable) ───────────────────────
        ParamDescriptor{"lfo1_rate",          0.15f, 0.0f, 1.0f, "lfo1", "continuous_log"},
        ParamDescriptor{"lfo1_depth",         0.00f, 0.0f, 1.0f, "lfo1", "continuous"},
        ParamDescriptor{"lfo1_shape",         0.00f, 0.0f, 1.0f, "lfo1", "categorical", 4},
        ParamDescriptor{"lfo1_target",        0.00f, 0.0f, 1.0f, "lfo1", "categorical", 4},

        ParamDescriptor{"lfo2_rate",          0.15f, 0.0f, 1.0f, "lfo2", "continuous_log"},
        ParamDescriptor{"lfo2_depth",         0.00f, 0.0f, 1.0f, "lfo2", "continuous"},
        ParamDescriptor{"lfo2_shape",         0.00f, 0.0f, 1.0f, "lfo2", "categorical", 4},
        ParamDescriptor{"lfo2_target",        0.00f, 0.0f, 1.0f, "lfo2", "categorical", 4},

        ParamDescriptor{"lfo3_rate",          0.15f, 0.0f, 1.0f, "lfo3", "continuous_log"},
        ParamDescriptor{"lfo3_depth",         0.00f, 0.0f, 1.0f, "lfo3", "continuous"},
        ParamDescriptor{"lfo3_shape",         0.00f, 0.0f, 1.0f, "lfo3", "categorical", 4},
        ParamDescriptor{"lfo3_target",        0.00f, 0.0f, 1.0f, "lfo3", "categorical", 4},

        ParamDescriptor{"lfo4_rate",          0.15f, 0.0f, 1.0f, "lfo4", "continuous_log"},
        ParamDescriptor{"lfo4_depth",         0.00f, 0.0f, 1.0f, "lfo4", "continuous"},
        ParamDescriptor{"lfo4_shape",         0.00f, 0.0f, 1.0f, "lfo4", "categorical", 4},
        ParamDescriptor{"lfo4_target",        0.00f, 0.0f, 1.0f, "lfo4", "categorical", 4},

        // ── Reverb (bypassable) ────────────────────────────────────────────
        ParamDescriptor{"reverb_size",        0.50f, 0.0f, 1.0f, "reverb", "continuous"},
        ParamDescriptor{"reverb_decay",       0.50f, 0.0f, 1.0f, "reverb", "continuous_log"},
        ParamDescriptor{"reverb_damping",     0.50f, 0.0f, 1.0f, "reverb", "continuous"},
        ParamDescriptor{"reverb_predelay",    0.00f, 0.0f, 1.0f, "reverb", "continuous_log"},
        ParamDescriptor{"reverb_mix",         0.40f, 0.0f, 1.0f, "reverb", "continuous"}, // BREAKING DEFAULT (was 0.00)
        ParamDescriptor{"reverb_width",       0.50f, 0.0f, 1.0f, "reverb", "continuous"},

        // ── Delay (bypassable) ─────────────────────────────────────────────
        ParamDescriptor{"delay_time_l",       0.25f, 0.0f, 1.0f, "delay", "continuous_log"},
        ParamDescriptor{"delay_time_r",       0.25f, 0.0f, 1.0f, "delay", "continuous_log"},
        ParamDescriptor{"delay_feedback",     0.30f, 0.0f, 1.0f, "delay", "continuous"},
        ParamDescriptor{"delay_filter_cutoff",0.80f, 0.0f, 1.0f, "delay", "continuous_log"},
        ParamDescriptor{"delay_mix",          0.40f, 0.0f, 1.0f, "delay", "continuous"}, // BREAKING DEFAULT (was 0.00)
    };

    static constexpr int NUM_PARAMS = static_cast<int>(PARAM_DESCRIPTORS.size()); // 70

    struct BypassFlags {
        bool osc_b  = false;
        bool noise  = false;
        bool filter = false;
        bool lfo1   = true;
        bool lfo2   = true;
        bool lfo3   = true;
        bool lfo4   = true;
        bool reverb = true;
        bool delay  = true;
        // env1: when true, amplitude envelope is flat (env=1.0). Follows osc_a — profile-controlled.
        // env2: always mirrors osc_b. Set by DatasetWriter, not independently sampled.
        bool env1   = false;
        bool env2   = false;

        std::vector<float> toVector() const {
            return {
                osc_b  ? 1.f : 0.f,
                noise  ? 1.f : 0.f,
                filter ? 1.f : 0.f,
                lfo1   ? 1.f : 0.f,
                lfo2   ? 1.f : 0.f,
                lfo3   ? 1.f : 0.f,
                lfo4   ? 1.f : 0.f,
                reverb ? 1.f : 0.f,
                delay  ? 1.f : 0.f,
                env1   ? 1.f : 0.f,
                env2   ? 1.f : 0.f,
            };
        }
    };

    struct SynthParameters {
        float amplitude = 0.8f; // internal, not predicted

        // Master — engine-internal, plugin-controlled, not predicted by model
        float masterVolume    = 0.80f;
        float masterPan       = 0.50f;
        float masterTune      = 0.50f;

        // OSC A
        float oscAWaveform    = PARAM_DESCRIPTORS[0].defaultVal;
        float oscAWarpAmt     = PARAM_DESCRIPTORS[1].defaultVal;
        float oscAWarpType    = PARAM_DESCRIPTORS[2].defaultVal;
        float oscADetune      = PARAM_DESCRIPTORS[3].defaultVal;
        float oscABlend       = PARAM_DESCRIPTORS[4].defaultVal;
        float oscAPitchCoarse = PARAM_DESCRIPTORS[5].defaultVal;
        float oscAPitchFine   = PARAM_DESCRIPTORS[6].defaultVal;
        float oscALevel       = PARAM_DESCRIPTORS[7].defaultVal;
        float oscAHarmonic2   = PARAM_DESCRIPTORS[8].defaultVal;
        float oscAHarmonic3   = PARAM_DESCRIPTORS[9].defaultVal;

        // OSC B
        float oscBWaveform    = PARAM_DESCRIPTORS[10].defaultVal;
        float oscBWarpAmt     = PARAM_DESCRIPTORS[11].defaultVal;
        float oscBWarpType    = PARAM_DESCRIPTORS[12].defaultVal;
        float oscBDetune      = PARAM_DESCRIPTORS[13].defaultVal;
        float oscBBlend       = PARAM_DESCRIPTORS[14].defaultVal;
        float oscBPitchCoarse = PARAM_DESCRIPTORS[15].defaultVal;
        float oscBPitchFine   = PARAM_DESCRIPTORS[16].defaultVal;
        float oscBLevel       = PARAM_DESCRIPTORS[17].defaultVal;
        float oscBPhase       = PARAM_DESCRIPTORS[18].defaultVal;
        float oscBSemitoneOfs = PARAM_DESCRIPTORS[19].defaultVal;

        // Noise
        float noiseColor      = PARAM_DESCRIPTORS[20].defaultVal;
        float noiseLevel      = PARAM_DESCRIPTORS[21].defaultVal;
        float noiseStereo     = PARAM_DESCRIPTORS[22].defaultVal;

        // Filter
        float filterCutoff    = PARAM_DESCRIPTORS[23].defaultVal;
        float filterResonance = PARAM_DESCRIPTORS[24].defaultVal;
        float filterType      = PARAM_DESCRIPTORS[25].defaultVal;
        float filterDrive     = PARAM_DESCRIPTORS[26].defaultVal;
        float filterKeytrack  = PARAM_DESCRIPTORS[27].defaultVal;
        float filterEnvAmt    = PARAM_DESCRIPTORS[28].defaultVal;

        // ENV1
        float env1Attack      = PARAM_DESCRIPTORS[29].defaultVal;
        float env1Hold        = PARAM_DESCRIPTORS[30].defaultVal;
        float env1Decay       = PARAM_DESCRIPTORS[31].defaultVal;
        float env1Sustain     = PARAM_DESCRIPTORS[32].defaultVal;
        float env1Release     = PARAM_DESCRIPTORS[33].defaultVal;
        float env1AtkCurve    = PARAM_DESCRIPTORS[34].defaultVal;
        float env1DecCurve    = PARAM_DESCRIPTORS[35].defaultVal;

        // ENV2
        float env2Attack      = PARAM_DESCRIPTORS[36].defaultVal;
        float env2Hold        = PARAM_DESCRIPTORS[37].defaultVal;
        float env2Decay       = PARAM_DESCRIPTORS[38].defaultVal;
        float env2Sustain     = PARAM_DESCRIPTORS[39].defaultVal;
        float env2Release     = PARAM_DESCRIPTORS[40].defaultVal;
        float env2AtkCurve    = PARAM_DESCRIPTORS[41].defaultVal;
        float env2DecCurve    = PARAM_DESCRIPTORS[42].defaultVal;

        // LFO 1
        float lfo1Rate        = PARAM_DESCRIPTORS[43].defaultVal;
        float lfo1Depth       = PARAM_DESCRIPTORS[44].defaultVal;
        float lfo1Shape       = PARAM_DESCRIPTORS[45].defaultVal;
        float lfo1Target      = PARAM_DESCRIPTORS[46].defaultVal;

        // LFO 2
        float lfo2Rate        = PARAM_DESCRIPTORS[47].defaultVal;
        float lfo2Depth       = PARAM_DESCRIPTORS[48].defaultVal;
        float lfo2Shape       = PARAM_DESCRIPTORS[49].defaultVal;
        float lfo2Target      = PARAM_DESCRIPTORS[50].defaultVal;

        // LFO 3
        float lfo3Rate        = PARAM_DESCRIPTORS[51].defaultVal;
        float lfo3Depth       = PARAM_DESCRIPTORS[52].defaultVal;
        float lfo3Shape       = PARAM_DESCRIPTORS[53].defaultVal;
        float lfo3Target      = PARAM_DESCRIPTORS[54].defaultVal;

        // LFO 4
        float lfo4Rate        = PARAM_DESCRIPTORS[55].defaultVal;
        float lfo4Depth       = PARAM_DESCRIPTORS[56].defaultVal;
        float lfo4Shape       = PARAM_DESCRIPTORS[57].defaultVal;
        float lfo4Target      = PARAM_DESCRIPTORS[58].defaultVal;

        // Reverb
        float reverbSize      = PARAM_DESCRIPTORS[59].defaultVal;
        float reverbDecay     = PARAM_DESCRIPTORS[60].defaultVal;
        float reverbDamping   = PARAM_DESCRIPTORS[61].defaultVal;
        float reverbPredelay  = PARAM_DESCRIPTORS[62].defaultVal;
        float reverbMix       = PARAM_DESCRIPTORS[63].defaultVal;
        float reverbWidth     = PARAM_DESCRIPTORS[64].defaultVal;

        // Delay
        float delayTimeL      = PARAM_DESCRIPTORS[65].defaultVal;
        float delayTimeR      = PARAM_DESCRIPTORS[66].defaultVal;
        float delayFeedback   = PARAM_DESCRIPTORS[67].defaultVal;
        float delayFilterCutoff = PARAM_DESCRIPTORS[68].defaultVal;
        float delayMix        = PARAM_DESCRIPTORS[69].defaultVal;

        std::vector<float> toVector() const {
            return {
                oscAWaveform, oscAWarpAmt, oscAWarpType, oscADetune, oscABlend,
                oscAPitchCoarse, oscAPitchFine, oscALevel,
                oscAHarmonic2, oscAHarmonic3,
                oscBWaveform, oscBWarpAmt, oscBWarpType, oscBDetune, oscBBlend,
                oscBPitchCoarse, oscBPitchFine, oscBLevel, oscBPhase, oscBSemitoneOfs,
                noiseColor, noiseLevel, noiseStereo,
                filterCutoff, filterResonance, filterType, filterDrive, filterKeytrack, filterEnvAmt,
                env1Attack, env1Hold, env1Decay, env1Sustain, env1Release, env1AtkCurve, env1DecCurve,
                env2Attack, env2Hold, env2Decay, env2Sustain, env2Release, env2AtkCurve, env2DecCurve,
                lfo1Rate, lfo1Depth, lfo1Shape, lfo1Target,
                lfo2Rate, lfo2Depth, lfo2Shape, lfo2Target,
                lfo3Rate, lfo3Depth, lfo3Shape, lfo3Target,
                lfo4Rate, lfo4Depth, lfo4Shape, lfo4Target,
                reverbSize, reverbDecay, reverbDamping, reverbPredelay, reverbMix, reverbWidth,
                delayTimeL, delayTimeR, delayFeedback, delayFilterCutoff, delayMix,
            };
        }

        static_assert(PARAM_DESCRIPTORS[0].name[0] == 'o', "osc_a_waveform must be index 0");
        static_assert(PARAM_DESCRIPTORS[0].name[4] == 'a', "osc_a_waveform must be index 0");

        void fromVector(const std::vector<float>& v) {
            if (static_cast<int>(v.size()) < NUM_PARAMS) return;
            oscAWaveform    = v[0];
            oscAWarpAmt     = v[1];
            oscAWarpType    = v[2];
            oscADetune      = v[3];
            oscABlend       = v[4];
            oscAPitchCoarse = v[5];
            oscAPitchFine   = v[6];
            oscALevel       = v[7];
            oscAHarmonic2   = v[8];
            oscAHarmonic3   = v[9];
            oscBWaveform    = v[10];
            oscBWarpAmt     = v[11];
            oscBWarpType    = v[12];
            oscBDetune      = v[13];
            oscBBlend       = v[14];
            oscBPitchCoarse = v[15];
            oscBPitchFine   = v[16];
            oscBLevel       = v[17];
            oscBPhase       = v[18];
            oscBSemitoneOfs = v[19];
            noiseColor      = v[20];
            noiseLevel      = v[21];
            noiseStereo     = v[22];
            filterCutoff    = v[23];
            filterResonance = v[24];
            filterType      = v[25];
            filterDrive     = v[26];
            filterKeytrack  = v[27];
            filterEnvAmt    = v[28];
            env1Attack      = v[29];
            env1Hold        = v[30];
            env1Decay       = v[31];
            env1Sustain     = v[32];
            env1Release     = v[33];
            env1AtkCurve    = v[34];
            env1DecCurve    = v[35];
            env2Attack      = v[36];
            env2Hold        = v[37];
            env2Decay       = v[38];
            env2Sustain     = v[39];
            env2Release     = v[40];
            env2AtkCurve    = v[41];
            env2DecCurve    = v[42];
            lfo1Rate        = v[43];
            lfo1Depth       = v[44];
            lfo1Shape       = v[45];
            lfo1Target      = v[46];
            lfo2Rate        = v[47];
            lfo2Depth       = v[48];
            lfo2Shape       = v[49];
            lfo2Target      = v[50];
            lfo3Rate        = v[51];
            lfo3Depth       = v[52];
            lfo3Shape       = v[53];
            lfo3Target      = v[54];
            lfo4Rate        = v[55];
            lfo4Depth       = v[56];
            lfo4Shape       = v[57];
            lfo4Target      = v[58];
            reverbSize      = v[59];
            reverbDecay     = v[60];
            reverbDamping   = v[61];
            reverbPredelay  = v[62];
            reverbMix       = v[63];
            reverbWidth     = v[64];
            delayTimeL      = v[65];
            delayTimeR      = v[66];
            delayFeedback   = v[67];
            delayFilterCutoff = v[68];
            delayMix        = v[69];
        }
    };

    inline float midiToHz(int midiNote) {
        return 440.0f * std::pow(2.0f, (midiNote - 69) / 12.0f);
    }

    inline float normToLinear(float norm, float minV, float maxV) {
        return minV + norm * (maxV - minV);
    }

} // namespace SynthV3
