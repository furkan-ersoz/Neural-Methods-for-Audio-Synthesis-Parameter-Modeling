#include "SynthEngineV2.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SynthV2 {

static constexpr double kTwoPi  = 2.0 * M_PI;
static constexpr float  kLfoHz  = 3.0f;

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
    const float total = normToAttackDecaySec(params_.attackTime)
                      + normToAttackDecaySec(params_.decayTime)
                      + 0.5f   // sustain hold — must match DatasetWriter::kSustainHold
                      + normToReleaseSec(params_.releaseTime);
    return std::min(total, maxSec);
}

int SynthEngine::computeNoteOnSamples(double sampleRate, float sustainHoldSec) const {
    const float onSec = normToAttackDecaySec(params_.attackTime)
                      + normToAttackDecaySec(params_.decayTime)
                      + sustainHoldSec;
    return static_cast<int>(onSec * static_cast<float>(sampleRate));
}

void SynthEngine::prepare(double sampleRate, int /*samplesPerBlock*/) {
    sampleRate_ = sampleRate;
    reset();
}

// ── Waveform generator ────────────────────────────────────────────────────────
// Continuous morph: [0,0.5] sine→saw, [0.5,1] saw→square.
// ph may be outside [0,2π) due to FM offset — fmod normalises it.

float SynthEngine::computeWave(double ph, float pw) const {
    double p = std::fmod(ph, kTwoPi);
    if (p < 0.0) p += kTwoPi;

    const float pf   = static_cast<float>(p);
    const float sine = std::sin(pf);
    const float saw  = pf * static_cast<float>(1.0 / M_PI) - 1.0f;
    const float sq   = (pf < pw * static_cast<float>(kTwoPi)) ? 1.0f : -1.0f;

    const float wf = params_.oscWaveform;
    if (wf <= 0.5f) {
        const float t = wf * 2.0f;
        return (1.0f - t) * sine + t * saw;
    }
    const float t = (wf - 0.5f) * 2.0f;
    return (1.0f - t) * saw + t * sq;
}

// ── Main DSP loop ─────────────────────────────────────────────────────────────

void SynthEngine::process(float* buffer, int numSamples, bool noteOn, float freqHz) {
    // Pre-compute per-block constants
    const float  detuneCents  = normToLinear(params_.oscDetune, 0.0f, 50.0f);
    const double phaseInc1    = kTwoPi * freqHz / sampleRate_;
    const double phaseInc2    = kTwoPi * freqHz
                                * std::pow(2.0, detuneCents / 1200.0) / sampleRate_;
    const float  fmRatio      = normToLinear(params_.fmRatio, 0.5f, 8.0f);
    const double fmPhaseInc   = kTwoPi * freqHz * fmRatio / sampleRate_;
    const double lfoInc       = kTwoPi * kLfoHz / sampleRate_;
    const float  pw           = normToLinear(params_.pulseWidth, 0.1f, 0.9f);
    const float  fmModIndex   = params_.fmDepth * 5.0f;   // max ±5 rad phase deviation
    const float  noiseAmt     = params_.noiseAmount;
    const float  amp          = params_.amplitude;
    const float  maxCutoff    = static_cast<float>(sampleRate_) * 0.45f;

    for (int i = 0; i < numSamples; ++i) {
        const float env = tickADSR(noteOn);

        // ── Filter cutoff: base + env mod + LFO (±1 octave at depth=1) ──────
        float cutoffNorm = std::clamp(params_.filterCutoff + params_.filterEnvAmt * env,
                                      0.0f, 1.0f);
        float cutoffHz   = normToFilterHz(cutoffNorm)
                         * std::pow(2.0f, params_.lfoDepth
                                          * std::sin(static_cast<float>(lfoPhase_)));
        cutoffHz = std::clamp(cutoffHz, 20.0f, maxCutoff);

        // ── Biquad lowpass coefficients (Audio EQ Cookbook) ──────────────────
        const float q    = normToLinear(params_.filterRes, 0.1f, 4.0f);
        const float w0   = static_cast<float>(kTwoPi) * cutoffHz
                         / static_cast<float>(sampleRate_);
        const float sinw = std::sin(w0);
        const float cosw = std::cos(w0);
        const float alph = sinw / (2.0f * q);
        const float a0   = 1.0f + alph;
        fb0_ = (1.0f - cosw) * 0.5f / a0;
        fb1_ = (1.0f - cosw) / a0;
        fb2_ = fb0_;
        fa1_ = -2.0f * cosw / a0;
        fa2_ = (1.0f - alph) / a0;

        // ── Oscillator ────────────────────────────────────────────────────────
        const float fmMod = fmModIndex * std::sin(static_cast<float>(fmPhase_));
        const float wave1 = computeWave(phase_       + fmMod, pw);
        const float wave2 = computeWave(phaseDetune_ + fmMod, pw);

        // ── Noise mix ─────────────────────────────────────────────────────────
        const float noise = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)
                          * 2.0f - 1.0f;
        const float osc = (1.0f - noiseAmt) * (wave1 + wave2) * 0.5f + noiseAmt * noise;

        // ── Biquad filter ─────────────────────────────────────────────────────
        const float y = fb0_ * osc + fb1_ * filterX1_ + fb2_ * filterX2_
                      - fa1_ * filterY1_ - fa2_ * filterY2_;
        filterX2_ = filterX1_; filterX1_ = osc;
        filterY2_ = filterY1_; filterY1_ = y;

        buffer[i] = amp * env * y;

        // ── Advance phases ────────────────────────────────────────────────────
        phase_       += phaseInc1;  if (phase_       >= kTwoPi) phase_       -= kTwoPi;
        phaseDetune_ += phaseInc2;  if (phaseDetune_ >= kTwoPi) phaseDetune_ -= kTwoPi;
        fmPhase_     += fmPhaseInc; if (fmPhase_     >= kTwoPi) fmPhase_     -= kTwoPi;
        lfoPhase_    += lfoInc;     if (lfoPhase_    >= kTwoPi) lfoPhase_    -= kTwoPi;
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void SynthEngine::reset() {
    const double startPhase = static_cast<double>(std::rand()) / RAND_MAX * kTwoPi;
    phase_       = startPhase;
    phaseDetune_ = startPhase;
    fmPhase_     = 0.0;
    lfoPhase_    = 0.0;
    filterX1_ = filterX2_ = 0.0f;
    filterY1_ = filterY2_ = 0.0f;
    stage_    = Stage::Idle;
    envelope_ = 0.0f;
}

void SynthEngine::startFadeOut() {
    stage_ = Stage::FadeOut;
}

void SynthEngine::setParameters(const SynthParameters& p) {
    params_ = p;
}

// ── ADSR ──────────────────────────────────────────────────────────────────────

float SynthEngine::tickADSR(bool noteOn) {
    if (stage_ == Stage::FadeOut) {
        envelope_ -= 1.0f / (kFadeOutSec * static_cast<float>(sampleRate_));
        if (envelope_ <= 0.0f) { envelope_ = 0.0f; stage_ = Stage::Idle; }
        return envelope_;
    }

    if (noteOn && stage_ == Stage::Idle)
        stage_ = Stage::Attack;

    if (!noteOn && stage_ != Stage::Idle && stage_ != Stage::Release)
        stage_ = Stage::Release;

    switch (stage_) {
        case Stage::Attack: {
            const float rate = 1.0f / (normToAttackDecaySec(params_.attackTime)
                                       * static_cast<float>(sampleRate_));
            envelope_ += rate;
            if (envelope_ >= 1.0f) { envelope_ = 1.0f; stage_ = Stage::Decay; }
            break;
        }
        case Stage::Decay: {
            const float rate = 1.0f / (normToAttackDecaySec(params_.decayTime)
                                       * static_cast<float>(sampleRate_));
            envelope_ -= rate;
            if (envelope_ <= params_.sustainLevel) {
                envelope_ = params_.sustainLevel;
                stage_    = Stage::Sustain;
            }
            break;
        }
        case Stage::Sustain:
            envelope_ = params_.sustainLevel;
            break;
        case Stage::Release: {
            const float rate = 1.0f / (normToReleaseSec(params_.releaseTime)
                                       * static_cast<float>(sampleRate_));
            envelope_ -= rate;
            if (envelope_ <= 0.0f) { envelope_ = 0.0f; stage_ = Stage::Idle; }
            break;
        }
        default:
            break;
    }
    return envelope_;
}

} // namespace SynthV2

// ── SynthRegistry registration ────────────────────────────────────────────────
namespace {
    const bool _registered = []() {
        SynthRegistry::instance().registerSynth("SynthV2", []() {
            return std::make_unique<SynthV2::SynthEngine>();
        });
        return true;
    }();
}
