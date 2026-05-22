#include "SynthEngineV1.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SynthV1 {

void SynthEngine::prepare(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;
    reset();
}

void SynthEngine::reset() {
    phase_    = 0.0;
    stage_    = Stage::Idle;
    envelope_ = 0.0f;
}

void SynthEngine::setParameters(const SynthParameters& p) {
    params_ = p;
}

float SynthEngine::tickADSR(bool noteOn) {
    // note-on edge: Idle'dan Attack'a geç
    if (noteOn && stage_ == Stage::Idle)
        stage_ = Stage::Attack;

    // note-off edge: her aşamadan Release'e geç
    if (!noteOn && stage_ != Stage::Idle && stage_ != Stage::Release)
        stage_ = Stage::Release;

    auto rateFor = [&](float normTime) -> float {
        return 1.0f / (toTimeSeconds(normTime) * (float)sampleRate_);
    };

    switch (stage_) {
        case Stage::Attack:
            envelope_ += rateFor(params_.attackTime);
            if (envelope_ >= 1.0f) { envelope_ = 1.0f; stage_ = Stage::Decay; }
            break;
        case Stage::Decay:
            envelope_ -= rateFor(params_.decayTime);
            if (envelope_ <= params_.sustainLevel) {
                envelope_ = params_.sustainLevel;
                stage_ = Stage::Sustain;
            }
            break;
        case Stage::Sustain:
            envelope_ = params_.sustainLevel;
            break;
        case Stage::Release:
            envelope_ -= rateFor(params_.releaseTime);
            if (envelope_ <= 0.0f) { envelope_ = 0.0f; stage_ = Stage::Idle; }
            break;
        default:
            break;
    }
    return envelope_;
}

void SynthEngine::process(float* buffer, int numSamples, bool noteOn, float frequencyHz) {
    const float amp  = params_.amplitude;
    const double phaseInc = (2.0 * M_PI * (double)frequencyHz) / sampleRate_;

    for (int i = 0; i < numSamples; ++i) {
        const float env = tickADSR(noteOn);
        buffer[i] = amp * env * (float)std::sin(phase_);
        phase_ += phaseInc;
        if (phase_ >= 2.0 * M_PI) phase_ -= 2.0 * M_PI;
    }
}

} // namespace SynthV1