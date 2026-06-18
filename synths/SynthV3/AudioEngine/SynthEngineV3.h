#pragma once
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include "SynthParametersV3.h"

namespace SynthV3 {

    class SynthEngine : public ISynth {
    public:
        // ── ISynth interface ─────────────────────────────────────────────────
        std::vector<ParamDescriptor> getParams() const override;
        void setParamValues(const std::vector<float>& values) override;
        std::vector<float> getParamValues() const override;
        float computeDuration(float maxSec) const override;
        int   computeNoteOnSamples(double sampleRate, float sustainHoldSec) const override;
        void  prepare(double sampleRate, int samplesPerBlock) override;
        void  process(float* buffer, int numSamples, bool noteOn, float freqHz) override;
        void  processStereo(float* left, float* right, int numSamples, bool noteOn, float freqHz) override;
        void  reset() override;
        void  setBypassFlags(const std::vector<bool>& flags) override;
        std::vector<std::string> getCategoricalLabels(const std::string& paramName) const override;

        // ── Direct parameter access (used by Plugin APVTS) ──────────────────
        void setParameters(const SynthParameters& p);
        const SynthParameters& getParameters() const { return params_; }
        void setBypass(const BypassFlags& b) { bypass_ = b; }
        const BypassFlags& getBypass() const { return bypass_; }
        bool  isEnvelopeDone() const {
            return env1Stage_ == Stage::Idle &&
                   (bypass_.osc_b || env2Stage_ == Stage::Idle);
        }
        bool  isIdle() const {
            const bool envDone = env1Stage_ == Stage::Idle &&
                                 (bypass_.osc_b || env2Stage_ == Stage::Idle);
            if (!envDone) return false;
            if (!bypass_.reverb && reverbEnergy_ > 0.0001f) return false;
            if (!bypass_.delay  && delayEnergy_  > 0.0001f) return false;
            return true;
        }
        bool  isFadingOut()  const { return env1Stage_ == Stage::FadeOut || env2Stage_ == Stage::FadeOut; }
        float getEnvelope()  const { return env1Value_; }
        float getPanMod()    const { return panMod_; }

        void startFadeOut();

    private:
        static constexpr float kFadeOutSec = 0.005f;

        double sampleRate_ = 44100.0;
        SynthParameters params_;

        // OSC A phases
        double phase_       = 0.0;
        double phaseDetune_ = 0.0;
        double fmPhase_     = 0.0;   // warp sync accumulator
        float oscAH2Phase_  = 0.0f;
        float oscAH3Phase_  = 0.0f;

        double lfo1Phase_ = 0.0;
        double lfo2Phase_ = 0.0;
        double lfo3Phase_ = 0.0;
        double lfo4Phase_ = 0.0;

        float lfo1Smooth_ = 0.0f;
        float lfo2Smooth_ = 0.0f;
        float lfo3Smooth_ = 0.0f;
        float lfo4Smooth_ = 0.0f;

        // OSC B phases
        double oscBPhase_       = 0.0;
        double oscBPhaseDetune_ = 0.0;
        double oscBSyncPhase_   = 0.0;

        // First biquad stage
        float fx1_=0, fx2_=0, fy1_=0, fy2_=0;
        float fb0_=1, fb1_=0, fb2_=0, fa1_=0, fa2_=0;
        // Second biquad stage (LP24 and HP24 only)
        float fx1b_=0, fx2b_=0, fy1b_=0, fy2b_=0;
        float fb0b_=1, fb1b_=0, fb2b_=0, fa1b_=0, fa2b_=0;

        float currentFreqHz_ = 440.0f;

        enum class Stage { Idle, Attack, Hold, Decay, Sustain, Release, FadeOut };

        // ENV1 — OSC A amplitude + filter modulation
        Stage env1Stage_    = Stage::Idle;
        float env1Value_    = 0.0f;
        float env1HoldTime_ = 0.0f;

        // ENV2 — OSC B amplitude (inactive when OSC B bypassed)
        Stage env2Stage_    = Stage::Idle;
        float env2Value_    = 0.0f;
        float env2HoldTime_ = 0.0f;

        // Noise state
        juce::Random noiseRng_;
        float pinkB0_ = 0, pinkB1_ = 0, pinkB2_ = 0, pinkB3_ = 0;
        float pinkB4_ = 0, pinkB5_ = 0, pinkB6_ = 0;
        float brownAccum_ = 0.0f;

        BypassFlags bypass_;

        // ── Reverb ───────────────────────────────────────────────────────────
        juce::Reverb reverb_;
        juce::Reverb::Parameters reverbParams_;

        static constexpr int kMaxPredelaySamples = 22050;
        float predelayBuf_[kMaxPredelaySamples] = {};
        int   predelayWrite_ = 0;

        // ── Delay ────────────────────────────────────────────────────────────
        static constexpr int kMaxDelaySamples = 88200;
        float delayBufL_[kMaxDelaySamples] = {};
        float delayBufR_[kMaxDelaySamples] = {};
        int   delayWriteL_ = 0;
        int   delayWriteR_ = 0;
        float delayFilterState_  = 0.0f;
        float delayFilterStateR_ = 0.0f;

        float reverbEnergy_ = 0.0f;
        float delayEnergy_  = 0.0f;

        float smoothedCutoffHz_ = 1000.0f;

        // ── Filter coeff cache (Problem 3) ───────────────────────────────────
        float lastCutoffHz_   = -1.0f;
        float lastQ_          = -1.0f;
        int   lastFilterType_ = -1;
        float makeupGain_     = 1.0f;

        // ── Note tracking (Problem 5) ────────────────────────────────────────
        bool  prevNoteOn_ = false;
        int   samplesAfterNoteOn_ = 0;
        float panMod_ = 0.5f;

        void  computeFilterCoeffs(float cutoffHz, float Q, int type, bool secondStage);
        float tickLfo(double& phase, float rateSec, float shape, double sampleRate,
                      const ParamDescriptor& shapeDesc);
        void  applyLfo(float lfoOut, float depth, float targetNorm,
                       float& filterCutoff, float& oscPitchMod,
                       float& oscLevelMod, float& warpAmtMod);
        float tickEnv(Stage& stage, float& value, float& holdTime,
                      bool noteOn,
                      float normAttack, float normHold, float normDecay,
                      float normSustain, float normRelease,
                      float atkCurve, float decCurve);
        float computeWave(double ph, float pw) const;
        float renderNoise(float color);
        float renderOscVoice(double& phase, double& phaseDetune, double& syncPhase,
                             float freqHz, float waveform, float warpAmt, int warpType,
                             float detuneCents, float blend,
                             float pitchCoarseSemitones, float pitchFineCents);
    };

} // namespace SynthV3
