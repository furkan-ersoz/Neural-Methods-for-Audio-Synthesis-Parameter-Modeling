#pragma once
#include "SynthParametersV2.h"

namespace SynthV2 {

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
        void  reset() override;

        // ── Direct parameter access (used by Plugin APVTS) ──────────────────
        void setParameters(const SynthParameters& p);
        const SynthParameters& getParameters() const { return params_; }
        bool  isIdle()       const { return stage_ == Stage::Idle; }
        bool  isFadingOut()  const { return stage_ == Stage::FadeOut; }
        float getEnvelope()  const { return envelope_; }

        void startFadeOut();

    private:
        static constexpr float kFadeOutSec = 0.005f;

        double sampleRate_ = 44100.0;
        SynthParameters params_;

        // Oscillator & modulation phases
        double phase_       = 0.0;   // main oscillator
        double phaseDetune_ = 0.0;   // detuned second oscillator
        double fmPhase_     = 0.0;   // FM modulator
        double lfoPhase_    = 0.0;   // LFO (3 Hz, modulates filter cutoff)

        // Biquad lowpass filter state
        float filterX1_ = 0.0f, filterX2_ = 0.0f;
        float filterY1_ = 0.0f, filterY2_ = 0.0f;
        // Cached biquad coefficients (recomputed per sample)
        float fb0_ = 1.0f, fb1_ = 0.0f, fb2_ = 0.0f;
        float fa1_ = 0.0f, fa2_ = 0.0f;

        enum class Stage { Idle, Attack, Decay, Sustain, Release, FadeOut };
        Stage stage_    = Stage::Idle;
        float envelope_ = 0.0f;

        float tickADSR(bool noteOn);
        float computeWave(double ph, float pw) const;
    };

} // namespace SynthV2
