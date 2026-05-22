#pragma once
#include "SynthParametersTemplate.h"

namespace SynthTemplate {

    class SynthEngine {
    public:
        void prepare(double sampleRate, int samplesPerBlock);
        void process(float* buffer, int numSamples, bool noteOn, float frequencyHz);
        void setParameters(const SynthParameters& p);
        const SynthParameters& getParameters() const { return params_; }
        void reset();

    private:
        double sampleRate_ = 44100.0;
        SynthParameters params_;

        // Oscillator
        double phase_ = 0.0;

        // ADSR
        enum class Stage { Idle, Attack, Decay, Sustain, Release };
        Stage  stage_     = Stage::Idle;
        float  envelope_  = 0.0f;

        float tickADSR(bool noteOn);
    };

} // namespace SynthTemplate