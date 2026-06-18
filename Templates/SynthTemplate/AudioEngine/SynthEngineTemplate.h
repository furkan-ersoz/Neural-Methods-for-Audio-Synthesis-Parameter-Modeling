#pragma once
#include "SynthParametersTemplate.h"

namespace SynthTemplate {

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

        // Begin a short fade-out — used before reassigning a stolen voice.
        void startFadeOut();

    private:
        static constexpr float kFadeOutSec = 0.005f;  // 5 ms

        double sampleRate_ = 44100.0;
        SynthParameters params_;

        double phase_ = 0.0;

        enum class Stage { Idle, Attack, Decay, Sustain, Release, FadeOut };
        Stage stage_    = Stage::Idle;
        float envelope_ = 0.0f;

        float tickADSR(bool noteOn);
    };

} // namespace SynthTemplate
