#include "SynthEngineV1.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SynthV1 {

// ── ISynth ────────────────────────────────────────────────────────────────────

std::vector<ParamDescriptor> SynthEngine::getParams() const {
    return { PARAM_DESCRIPTORS.begin(), PARAM_DESCRIPTORS.end() };
}

void SynthEngine::setParamValues(const std::vector<float>& values) {
    params_.fromVector(values);
}

std::vector<float> SynthEngine::getParamValues() const {
    return params_.toVector();
}

float SynthEngine::computeDuration(float maxSec) const {
    const float total = toTimeSeconds(params_.attackTime)
                      + toTimeSeconds(params_.decayTime)
                      + 0.5f   // sustain hold — must match DatasetWriter::kSustainHold
                      + toTimeSeconds(params_.releaseTime);
    return std::min(total, maxSec);
}

int SynthEngine::computeNoteOnSamples(double sampleRate, float sustainHoldSec) const {
    const float onSec = toTimeSeconds(params_.attackTime)
                      + toTimeSeconds(params_.decayTime)
                      + sustainHoldSec;
    return static_cast<int>(onSec * static_cast<float>(sampleRate));
}

void SynthEngine::prepare(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;
    reset();
}

void SynthEngine::process(float* buffer, int numSamples, bool noteOn, float frequencyHz) {
    const float   amp      = params_.amplitude;
    const double  phaseInc = (2.0 * M_PI * static_cast<double>(frequencyHz)) / sampleRate_;

    const float h2 = params_.harmonic2;
    const float h3 = params_.harmonic3;
    const float norm = 1.0f / (1.0f + h2 + h3);  // keep peak level consistent

    for (int i = 0; i < numSamples; ++i) {
        const float env  = tickADSR(noteOn);
        const float wave = static_cast<float>(std::sin(phase_))
                         + h2 * static_cast<float>(std::sin(2.0 * phase_))
                         + h3 * static_cast<float>(std::sin(3.0 * phase_));
        buffer[i] = amp * env * wave * norm;
        phase_ += phaseInc;
        if (phase_ >= 2.0 * M_PI) phase_ -= 2.0 * M_PI;
    }
}

void SynthEngine::reset() {
    phase_    = (static_cast<double>(std::rand()) / RAND_MAX) * 2.0 * M_PI;
    stage_    = Stage::Idle;
    envelope_ = 0.0f;
}

void SynthEngine::startFadeOut() {
    stage_ = Stage::FadeOut;
    // phase_ and envelope_ are kept — the voice fades out from wherever it is.
}

// ── Direct parameter access ───────────────────────────────────────────────────

void SynthEngine::setParameters(const SynthParameters& p) {
    params_ = p;
}

// ── ADSR ──────────────────────────────────────────────────────────────────────

float SynthEngine::tickADSR(bool noteOn) {
    // FadeOut overrides all normal ADSR logic — drain envelope then go Idle.
    if (stage_ == Stage::FadeOut) {
        envelope_ -= 1.0f / (kFadeOutSec * static_cast<float>(sampleRate_));
        if (envelope_ <= 0.0f) { envelope_ = 0.0f; stage_ = Stage::Idle; }
        return envelope_;
    }

    if (noteOn && stage_ == Stage::Idle)
        stage_ = Stage::Attack;

    if (!noteOn && stage_ != Stage::Idle && stage_ != Stage::Release)
        stage_ = Stage::Release;

    auto rateFor = [&](float normTime) -> float {
        return 1.0f / (toTimeSeconds(normTime) * static_cast<float>(sampleRate_));
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
                stage_    = Stage::Sustain;
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

} // namespace SynthV1

// ── SynthRegistry registration ────────────────────────────────────────────────
namespace {
    const bool _registered = []() {
        SynthRegistry::instance().registerSynth("SynthV1", []() {
            return std::make_unique<SynthV1::SynthEngine>();
        });
        return true;
    }();
}
