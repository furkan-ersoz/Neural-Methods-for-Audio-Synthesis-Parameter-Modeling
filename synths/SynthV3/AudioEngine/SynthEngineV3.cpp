#include "SynthEngineV3.h"
#include "ParamCodec.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
    template<typename T> T jlimit(T lo, T hi, T v) { return std::clamp(v, lo, hi); }
    template<typename T> T jmin(T a, T b) { return std::min(a, b); }

    constexpr const ParamDescriptor& descOf(std::string_view name) {
        for (const auto& d : SynthV3::PARAM_DESCRIPTORS)
            if (d.name == name) return d;
        throw std::logic_error("descOf: unknown param name");
    }
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SynthV3 {

static constexpr double kTwoPi = 2.0 * M_PI;

// ── Helpers ───────────────────────────────────────────────────────────────────

static float normToAttackSec(float n)  { return 0.002f  * std::pow(1500.0f, n); } // 2ms–3s
static float normToHoldSec(float n)    { return (n < 0.001f) ? 0.0f : 0.001f * std::pow(500.0f, n); } // 0–500ms
static float normToDecaySec(float n)   { return 0.0005f * std::pow(6000.0f, n); }
static float normToReleaseSec(float n) { return 0.005f  * std::pow(600.0f,  n); } // 5ms–3s

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
    const float total = normToAttackSec(params_.env1Attack)
                      + normToHoldSec(params_.env1Hold)
                      + normToDecaySec(params_.env1Decay)
                      + 0.5f
                      + normToReleaseSec(params_.env1Release);
    return std::min(total, maxSec);
}

int SynthEngine::computeNoteOnSamples(double sampleRate, float sustainHoldSec) const {
    const float onSec = normToAttackSec(params_.env1Attack)
                      + normToHoldSec(params_.env1Hold)
                      + normToDecaySec(params_.env1Decay)
                      + sustainHoldSec;
    return static_cast<int>(onSec * static_cast<float>(sampleRate));
}

void SynthEngine::prepare(double sampleRate, int samplesPerBlock) {
    for (const auto& d : PARAM_DESCRIPTORS) {
        if (!ParamCodec::isDiscrete(d)) continue;
        const auto labels = getCategoricalLabels(std::string(d.name));
        if (!labels.empty() && static_cast<int>(labels.size()) != d.numClasses)
            throw std::logic_error("getCategoricalLabels size mismatch for " + std::string(d.name));
    }

    sampleRate_ = sampleRate;
    reverb_.setSampleRate(sampleRate);
    // Buffer latency: 64→1.5ms, 256→5.8ms, 512→11.6ms, 1024→23.2ms (at 44100Hz)
    DBG("SynthEngineV3 prepare: sampleRate=" << sampleRate
        << " samplesPerBlock=" << samplesPerBlock
        << " latencyMs=" << (samplesPerBlock * 1000.0 / sampleRate));

    reset();
    smoothedCutoffHz_ = -1.0f;
}

// ── OSC A waveform: continuous morph sine(0) → saw(0.5) → square(1.0) ────────

static float oscAWaveform(double ph, float waveform) {
    double p = std::fmod(ph, kTwoPi);
    if (p < 0.0) p += kTwoPi;
    const float pf   = static_cast<float>(p);
    const float sine = std::sin(pf);
    const float saw  = pf * static_cast<float>(1.0 / M_PI) - 1.0f;
    const float sq   = (pf < static_cast<float>(M_PI)) ? 1.0f : -1.0f;

    if (waveform <= 0.5f) {
        const float t = waveform * 2.0f;
        return (1.0f - t) * sine + t * saw;
    }
    const float t = (waveform - 0.5f) * 2.0f;
    return (1.0f - t) * saw + t * sq;
}

// ── OSC A warp: phase distortion before waveform lookup ──────────────────────
// Type 0 (sync) is handled per-sample in renderOscVoice; this covers type 1.
// Types 2 (fold) and 3 (ring) are post-waveform in renderOscVoice.
// At amt=0, all types return the input phase unchanged (identity).

static double applyWarpPhase(double ph, float amt, int type) {
    double p = std::fmod(ph, kTwoPi);
    if (p < 0.0) p += kTwoPi;
    const double t = p * (1.0 / kTwoPi);  // normalised [0,1)

    switch (type) {
        case 1: {  // mirror: fold phase back at midpoint
            const double u = t < 0.5 ? t * 2.0 : (1.0 - t) * 2.0;
            return ((1.0 - amt) * t + (double)amt * u) * kTwoPi;
        }
        case 2:  // fold — post-waveform in renderOscVoice
        case 3:  // ring — post-waveform in renderOscVoice
        default:
            return p;
    }
}

// ── Noise generator (white / pink / brown morph) ─────────────────────────────

float SynthEngine::renderNoise(float color) {
    const float white = noiseRng_.nextFloat() * 2.0f - 1.0f;

    pinkB0_ = 0.99886f * pinkB0_ + white * 0.0555179f;
    pinkB1_ = 0.99332f * pinkB1_ + white * 0.0750759f;
    pinkB2_ = 0.96900f * pinkB2_ + white * 0.1538520f;
    pinkB3_ = 0.86650f * pinkB3_ + white * 0.3104856f;
    pinkB4_ = 0.55000f * pinkB4_ + white * 0.5329522f;
    pinkB5_ = -0.7616f * pinkB5_ - white * 0.0168980f;
    const float pink = (pinkB0_ + pinkB1_ + pinkB2_ + pinkB3_ + pinkB4_
                        + pinkB5_ + pinkB6_ + white * 0.5362f) * 0.11f;
    pinkB6_ = white * 0.115926f;

    brownAccum_ += white * 0.05f; // FIXED: brown-gain (was 0.02f — too slow)
    brownAccum_  = jlimit(-1.0f, 1.0f, brownAccum_);

    float noiseOut;
    if (color <= 0.5f) {
        const float t = color * 2.0f;
        noiseOut = white * (1.0f - t) + pink * t;
    } else {
        const float t = (color - 0.5f) * 2.0f;
        noiseOut = pink * (1.0f - t) + brownAccum_ * t;
    }
    noiseOut = std::tanh(noiseOut * 2.0f); // FIXED: brown-gain — saturation for warmth
    return juce::jlimit(-1.0f, 1.0f, noiseOut * 2.5f);
}

// ── OSC voice renderer (shared by OSC A and OSC B) ───────────────────────────

float SynthEngine::renderOscVoice(
    double& phase, double& phaseDetune, double& syncPhase,
    float freqHz, float waveform, float warpAmt, int warpType,
    float detuneCents, float blend,
    float pitchCoarseSemitones, float pitchFineCents)
{
    const double pitchedHz = freqHz
                           * std::pow(2.0, pitchCoarseSemitones / 12.0)
                           * std::pow(2.0, pitchFineCents / 1200.0);
    const double phaseInc1 = kTwoPi * pitchedHz / sampleRate_;
    const double phaseInc2 = kTwoPi * pitchedHz
                           * std::pow(2.0, detuneCents / 1200.0) / sampleRate_;
    const double syncInc   = phaseInc1 * (1.0 + warpAmt * 2.0);

    double wp1, wp2;
    if (warpType == 0) {  // sync
        syncPhase += syncInc;
        if (syncPhase >= kTwoPi) {
            syncPhase -= kTwoPi;
            phase = 0.0;
        }
        wp1 = phase;
        wp2 = phaseDetune;
    } else {
        wp1 = applyWarpPhase(phase,       warpAmt, warpType);
        wp2 = applyWarpPhase(phaseDetune, warpAmt, warpType);
    }

    float w1 = oscAWaveform(wp1, waveform);
    float w2 = oscAWaveform(wp2, waveform);

    if (warpType == 2) {  // fold
        w1 = std::sin(static_cast<float>(M_PI) * w1 * (1.0f + warpAmt * 3.0f));
        w2 = std::sin(static_cast<float>(M_PI) * w2 * (1.0f + warpAmt * 3.0f));
    }

    if (warpType == 3) {  // ring
        const float ring = std::sin(static_cast<float>(phase * 0.5));
        w1 *= (1.0f - warpAmt) + warpAmt * ring;
        w2 *= (1.0f - warpAmt) + warpAmt * ring;
    }

    phase       += phaseInc1; if (phase       >= kTwoPi) phase       -= kTwoPi;
    phaseDetune += phaseInc2; if (phaseDetune >= kTwoPi) phaseDetune -= kTwoPi;

    return (1.0f - blend * 0.5f) * w1 + blend * 0.5f * w2;
}

// ── Biquad coefficient computation ───────────────────────────────────────────

void SynthEngine::computeFilterCoeffs(float cutoffHz, float Q, int type, bool secondStage) {
    const float omega = static_cast<float>(kTwoPi) * cutoffHz / static_cast<float>(sampleRate_);
    const float sinw  = std::sin(omega);
    const float cosw  = std::cos(omega);

    float b0, b1, b2, a0, a1, a2;

    switch (type) {
        case 0: // LP12
        case 1: // LP24 (same coeffs, applied twice)
        {
            const float alpha = sinw / (2.0f * Q);
            b0 = (1.0f - cosw) * 0.5f;
            b1 = 1.0f - cosw;
            b2 = (1.0f - cosw) * 0.5f;
            a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
            break;
        }
        case 2: // BP
        {
            const float alpha = sinw * std::sinh(std::log(2.0f) / 2.0f * Q * omega / sinw);
            b0 = sinw * 0.5f; b1 = 0.0f; b2 = -sinw * 0.5f;
            a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
            break;
        }
        case 3: // HP12
        case 4: // HP24
        {
            const float alpha = sinw / (2.0f * Q);
            b0 = (1.0f + cosw) * 0.5f;
            b1 = -(1.0f + cosw);
            b2 = (1.0f + cosw) * 0.5f;
            a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
            break;
        }
        case 5: // Notch
        {
            const float alpha = sinw / (2.0f * Q);
            b0 = 1.0f; b1 = -2.0f * cosw; b2 = 1.0f;
            a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
            break;
        }
        default:
            throw std::logic_error("computeFilterCoeffs: invalid filter type");
    }

    if (!secondStage) {
        fb0_ = b0 / a0; fb1_ = b1 / a0; fb2_ = b2 / a0;
        fa1_ = a1 / a0; fa2_ = a2 / a0;
    } else {
        fb0b_ = b0 / a0; fb1b_ = b1 / a0; fb2b_ = b2 / a0;
        fa1b_ = a1 / a0; fa2b_ = a2 / a0;
    }
}

// ── LFO helpers ───────────────────────────────────────────────────────────────

float SynthEngine::tickLfo(double& phase, float rateNorm, float shape, double sampleRate,
                            const ParamDescriptor& shapeDesc) {
    const float rateHz = 0.05f * std::pow(400.0f, rateNorm); // 0.05–20Hz
    phase += rateHz / sampleRate;
    if (phase >= 1.0) phase -= 1.0;

    const float t = static_cast<float>(phase);
    const float sine     = std::sin(static_cast<float>(kTwoPi) * t);
    const float triRaw   = 1.0f - std::abs(4.0f * std::fmod(t, 1.0f) - 2.0f);
    const float triangle = std::clamp(triRaw, -1.0f, 1.0f);
    const float saw      = 2.0f * t - 1.0f;
    const float square   = t < 0.5f ? 1.0f : -1.0f;

    if (ParamCodec::isDiscrete(shapeDesc)) {
        switch (ParamCodec::normToIndex(shapeDesc, shape)) {
            case 0:  return sine;
            case 1:  return triangle;
            case 2:  return saw;
            default: return square;
        }
    }

    if (shape <= 0.333f) {
        const float blend = shape / 0.333f;
        return sine * (1.0f - blend) + triangle * blend;
    } else if (shape <= 0.667f) {
        const float blend = (shape - 0.333f) / 0.333f;
        return triangle * (1.0f - blend) + saw * blend;
    } else {
        const float blend = (shape - 0.667f) / 0.333f;
        return saw * (1.0f - blend) + square * blend;
    }
}

void SynthEngine::applyLfo(float lfoOut, float depth, float targetNorm,
                            float& filterCutoff, float& oscPitchMod,
                            float& oscLevelMod, float& warpAmtMod)
{
    const float cappedDepth = depth * 0.7f; // crop max depth to 0.7
    switch (ParamCodec::normToIndex(descOf("lfo1_target"), targetNorm)) {
        case 0:  // pitch
            oscPitchMod += lfoOut * cappedDepth * 2.0f;
            break;
        case 1:  // filter_cutoff
            filterCutoff = std::clamp(filterCutoff + lfoOut * cappedDepth, 0.0f, 1.0f);
            break;
        case 2:  // level
            oscLevelMod = std::clamp(oscLevelMod * (1.0f + lfoOut * cappedDepth), 0.0f, 2.0f);
            break;
        case 3:  // warp
            warpAmtMod = std::clamp(warpAmtMod + lfoOut * cappedDepth, 0.0f, 1.0f);
            break;
    }
}

// ── Main DSP loop ─────────────────────────────────────────────────────────────

void SynthEngine::process(float* buffer, int numSamples, bool noteOn, float freqHz) {
    std::vector<float> scratchR(static_cast<size_t>(numSamples));
    processStereo(buffer, scratchR.data(), numSamples, noteOn, freqHz);
    for (int i = 0; i < numSamples; ++i)
        buffer[i] = (buffer[i] + scratchR[static_cast<size_t>(i)]) * 0.5f;
}

void SynthEngine::processStereo(float* left, float* right, int numSamples, bool noteOn, float freqHz) {
    freqHz *= std::pow(2.0f, (params_.masterTune - 0.5f) * 2.0f / 12.0f);
    currentFreqHz_ = freqHz;

    // Problem 5: reset all state on note-on rising edge for deterministic sound
    if (noteOn && !prevNoteOn_) {
        reset();
        oscBPhase_ = params_.oscBPhase * kTwoPi;
        samplesAfterNoteOn_ = 0;
        smoothedCutoffHz_ = -1.0f;
    }
    prevNoteOn_ = noteOn;

    // OSC A params
    const float oscACoarse = static_cast<float>(
        ParamCodec::normToIntValue(descOf("osc_a_pitch_coarse"), params_.oscAPitchCoarse));
    const float oscAFine   = (params_.oscAPitchFine - 0.5f) * 200.0f;
    const float oscADetune = params_.oscADetune * 30.0f; // crop 50→30 cents
    const int   oscAWarpType = ParamCodec::normToIndex(descOf("osc_a_warp_type"), params_.oscAWarpType);

    // OSC B params — semitone offset cropped to ±12 (was ±24)
    const float oscBCoarse = static_cast<float>(
        ParamCodec::normToIntValue(descOf("osc_b_pitch_coarse"), params_.oscBPitchCoarse)
      + ParamCodec::normToIntValue(descOf("osc_b_semitone_ofs"), params_.oscBSemitoneOfs));
    const float oscBFine   = (params_.oscBPitchFine - 0.5f) * 200.0f;
    const float oscBDetune = params_.oscBDetune * 50.0f;
    const int   oscBWarpType = ParamCodec::normToIndex(descOf("osc_b_warp_type"), params_.oscBWarpType);
    const float oscBActive = bypass_.osc_b ? 0.0f : 1.0f;

    const float lfoCoeff    = 1.0f - std::exp(-1.0f / (static_cast<float>(sampleRate_) * 0.004f));
    const float cutoffSlew  = 1.0f - std::exp(-1.0f / (static_cast<float>(sampleRate_) * 0.003f));

    for (int i = 0; i < numSamples; ++i) {
        float filterCutoffMod = params_.filterCutoff;
        float oscPitchMod     = 0.0f;
        float oscLevelMod     = 1.0f;
        float warpAmtMod      = params_.oscAWarpAmt;
        float panMod          = 0.5f;

        if (!bypass_.lfo1) {
            float raw = tickLfo(lfo1Phase_, params_.lfo1Rate, params_.lfo1Shape, sampleRate_, descOf("lfo1_shape"));
            lfo1Smooth_ += lfoCoeff * (raw - lfo1Smooth_);
            applyLfo(lfo1Smooth_, params_.lfo1Depth, params_.lfo1Target,
                     filterCutoffMod, oscPitchMod, oscLevelMod, warpAmtMod);
        }
        if (!bypass_.lfo2) {
            float raw = tickLfo(lfo2Phase_, params_.lfo2Rate, params_.lfo2Shape, sampleRate_, descOf("lfo2_shape"));
            lfo2Smooth_ += lfoCoeff * (raw - lfo2Smooth_);
            applyLfo(lfo2Smooth_, params_.lfo2Depth, params_.lfo2Target,
                     filterCutoffMod, oscPitchMod, oscLevelMod, warpAmtMod);
        }
        if (!bypass_.lfo3) {
            float raw = tickLfo(lfo3Phase_, params_.lfo3Rate, params_.lfo3Shape, sampleRate_, descOf("lfo3_shape"));
            lfo3Smooth_ += lfoCoeff * (raw - lfo3Smooth_);
            applyLfo(lfo3Smooth_, params_.lfo3Depth, params_.lfo3Target,
                     filterCutoffMod, oscPitchMod, oscLevelMod, warpAmtMod);
        }
        if (!bypass_.lfo4) {
            float raw = tickLfo(lfo4Phase_, params_.lfo4Rate, params_.lfo4Shape, sampleRate_, descOf("lfo4_shape"));
            lfo4Smooth_ += lfoCoeff * (raw - lfo4Smooth_);
            applyLfo(lfo4Smooth_, params_.lfo4Depth, params_.lfo4Target,
                     filterCutoffMod, oscPitchMod, oscLevelMod, warpAmtMod);
        }

        const float pitchMultiplier = std::pow(2.0f, oscPitchMod / 12.0f);
        const float lfoFreqHz = freqHz * pitchMultiplier;

        // env1: bypassed → flat amplitude (1.0), still ticks to keep stage in sync
        const float env1raw = tickEnv(env1Stage_, env1Value_, env1HoldTime_,
                                      noteOn,
                                      params_.env1Attack, params_.env1Hold,
                                      params_.env1Decay,  params_.env1Sustain,
                                      params_.env1Release,
                                      params_.env1AtkCurve, params_.env1DecCurve);
        const float env1 = bypass_.env1 ? 1.0f : env1raw;

        // env2: follows osc_b — also bypassed if env2 flag is set
        float env2 = 0.0f;
        if (!bypass_.osc_b)
            env2 = bypass_.env2 ? 1.0f
                                : tickEnv(env2Stage_, env2Value_, env2HoldTime_,
                                          noteOn,
                                          params_.env2Attack, params_.env2Hold,
                                          params_.env2Decay,  params_.env2Sustain,
                                          params_.env2Release,
                                          params_.env2AtkCurve, params_.env2DecCurve);

        // ── OSC A ─────────────────────────────────────────────────────────────
        const float oscA = renderOscVoice(phase_, phaseDetune_, fmPhase_,
                                          lfoFreqHz, params_.oscAWaveform,
                                          warpAmtMod, oscAWarpType,
                                          oscADetune, params_.oscABlend,
                                          oscACoarse, oscAFine);

        const double oscAFreqHz = lfoFreqHz
            * std::pow(2.0, (double)oscACoarse / 12.0)
            * std::pow(2.0, (double)oscAFine / 1200.0);
        const double phaseInc1 = kTwoPi * oscAFreqHz / sampleRate_;
        const float h2 = std::sin(2.0 * (phase_ - phaseInc1)) * params_.oscAHarmonic2;
        const float h3 = std::sin(3.0 * (phase_ - phaseInc1)) * params_.oscAHarmonic3;
        const float oscA_final = (oscA + h2 + h3)
            / (1.0f + params_.oscAHarmonic2 + params_.oscAHarmonic3 + 0.001f);

        // ── OSC B ─────────────────────────────────────────────────────────────
        const float oscB = renderOscVoice(oscBPhase_, oscBPhaseDetune_, oscBSyncPhase_,
                                          lfoFreqHz, params_.oscBWaveform,
                                          params_.oscBWarpAmt, oscBWarpType,
                                          oscBDetune, params_.oscBBlend,
                                          oscBCoarse, oscBFine)
                         * oscBActive;

        const float oscOutput = oscA_final + oscB;

        float noiseOut = 0.0f;
        if (!bypass_.noise) {
            const auto& noiseColorDesc = descOf("noise_color");
            const float snappedColor = ParamCodec::indexToNorm(
                noiseColorDesc, ParamCodec::normToIndex(noiseColorDesc, params_.noiseColor));
            noiseOut = renderNoise(snappedColor) * params_.noiseLevel;
        }

        const float mixedInput = oscOutput + noiseOut;

        // ── Filter ────────────────────────────────────────────────────────────
        float filteredOutput;
        if (bypass_.filter) {
            filteredOutput = mixedInput;
        } else {
            float envMod = env1 * ((params_.filterEnvAmt - 0.5f) * 2.0f);
            float effectiveCutoff = jlimit(0.0f, 1.0f, filterCutoffMod + envMod);
            // Crop low end: 80Hz–20kHz (was 20Hz–20kHz)
            float filterCutoffHz = 80.0f * std::pow(250.0f, effectiveCutoff);

            float keytrackMult = 1.0f;
            const float cutoffBeforeKeytrack = filterCutoffHz;
            if (params_.filterKeytrack > 0.0f)
                keytrackMult = 1.0f + params_.filterKeytrack
                              * (currentFreqHz_ / 261.63f - 1.0f);
            filterCutoffHz = jlimit(20.0f, 20000.0f, filterCutoffHz * keytrackMult);

            if (smoothedCutoffHz_ < 0.0f)
                smoothedCutoffHz_ = filterCutoffHz;
            else
                smoothedCutoffHz_ += cutoffSlew * (filterCutoffHz - smoothedCutoffHz_);
            filterCutoffHz = smoothedCutoffHz_;
            // Q crop: max 3.0 (was 4.0) — avoids harsh self-oscillation
            float Q = 0.1f + params_.filterResonance * 2.9f;
            int filterType = ParamCodec::normToIndex(descOf("filter_type"), params_.filterType);

            // Drive crop: max 0.7 (was 1.0) — saturation without signal destruction
            float driveGain = 1.0f + params_.filterDrive * 0.7f * 3.0f;
            float driven = std::tanh(mixedInput * driveGain) / driveGain;

            bool needsUpdate =
                std::abs(filterCutoffHz - lastCutoffHz_) > 0.5f ||
                std::abs(Q             - lastQ_)         > 0.001f ||
                filterType != lastFilterType_;
            if (needsUpdate) {
                computeFilterCoeffs(filterCutoffHz, Q, filterType, false);
                if (filterType == 1 || filterType == 4)
                    computeFilterCoeffs(filterCutoffHz, Q, filterType, true);
                lastCutoffHz_    = filterCutoffHz;
                lastQ_           = Q;
                lastFilterType_  = filterType;

                switch (filterType) {
                    case 0: case 1: {
                        const float hdc = (fb0_ + fb1_ + fb2_) / (1.0f + fa1_ + fa2_);
                        const float lpGain = std::abs(hdc) + 1e-6f;
                        makeupGain_ = (1.0f / lpGain) * (1.0f + (1.0f - params_.filterCutoff) * 0.6f);
                        break;
                    }
                    case 3: case 4: {
                        const float hny = (fb0_ - fb1_ + fb2_) / (1.0f - fa1_ + fa2_);
                        const float hpGain = std::abs(hny) + 1e-6f;
                        makeupGain_ = std::min((1.0f / hpGain) * (1.0f + params_.filterCutoff * 0.4f), 2.0f);
                        break;
                    }
                    case 2:
                        makeupGain_ = 1.4f + Q * 0.15f;
                        break;
                    default:
                        makeupGain_ = 1.0f;
                        break;
                }
            }

            float y1 = fb0_ * driven + fb1_ * fx1_ + fb2_ * fx2_
                     - fa1_ * fy1_ - fa2_ * fy2_;
            fx2_ = fx1_; fx1_ = driven;
            fy2_ = fy1_; fy1_ = y1;
            filteredOutput = y1;

            if (filterType == 1 || filterType == 4) {
                float y2 = fb0b_ * y1 + fb1b_ * fx1b_ + fb2b_ * fx2b_
                         - fa1b_ * fy1b_ - fa2b_ * fy2b_;
                fx2b_ = fx1b_; fx1b_ = y1;
                fy2b_ = fy1b_; fy1b_ = y2;
                filteredOutput = y2;
            }

            filteredOutput *= makeupGain_;
            filteredOutput = std::clamp(filteredOutput, -1.2f, 1.2f);
        }

        if (noteOn) ++samplesAfterNoteOn_;

        const float outA = filteredOutput * env1 * params_.oscALevel;
        float outB = 0.0f;
        if (!bypass_.osc_b)
            outB = filteredOutput * env2 * params_.oscBLevel;

        float envOut = params_.amplitude * (outA + outB) * oscLevelMod;

        // ── Reverb ────────────────────────────────────────────────────────────
        float reverbOutL, reverbOutR;
        if (bypass_.reverb) {
            reverbOutL = reverbOutR = envOut;
        } else {
            const float normSize    = params_.reverbSize;
            const float normDecay   = params_.reverbDecay;
            const float normPre     = params_.reverbPredelay;

            // FIXED: reverb-tail — roomSize and decay decoupled
            reverbParams_.roomSize  = jmin(0.99f, normSize * 0.6f);
            reverbParams_.damping   = params_.reverbDamping;
            reverbParams_.width     = params_.reverbWidth;
            const float reverbWet   = params_.reverbMix * 0.85f;
            reverbParams_.wetLevel  = reverbWet;
            reverbParams_.dryLevel  = 1.0f - reverbWet * (0.3f + normDecay * 0.7f);

            const float predelayMs = (normPre < 0.001f) ? 0.0f
                                   : normPre * normPre * 200.0f;
            const int predelaySamples = jlimit(0, kMaxPredelaySamples - 1,
                static_cast<int>(predelayMs * 0.001f * static_cast<float>(sampleRate_)));

            predelayBuf_[predelayWrite_] = envOut;
            const int readIdx = (predelayWrite_ - predelaySamples
                                 + kMaxPredelaySamples) % kMaxPredelaySamples;
            float predelayed = predelayBuf_[readIdx];
            // FIXED: reverb-predelay-explosion — fade in over 220 samples to avoid initial burst
            if (samplesAfterNoteOn_ < 220)
                predelayed *= samplesAfterNoteOn_ / 220.0f;
            predelayWrite_ = (predelayWrite_ + 1) % kMaxPredelaySamples;

            reverb_.setParameters(reverbParams_);
            float wetL = predelayed, wetR = predelayed;
            reverb_.processStereo(&wetL, &wetR, 1);
            reverbOutL = envOut * reverbParams_.dryLevel + wetL * reverbParams_.wetLevel;
            reverbOutR = envOut * reverbParams_.dryLevel + wetR * reverbParams_.wetLevel;
            reverbEnergy_ = reverbEnergy_ * 0.9999f + std::abs(reverbOutL - envOut) * 0.0001f;
        }

        // ── Delay ─────────────────────────────────────────────────────────────
        // FIXED: delay-rewrite — true stereo L/R lines, wet uses filtered feedback
        float delayOutL, delayOutR;
        if (bypass_.delay) {
            delayOutL = reverbOutL;
            delayOutR = reverbOutR;
        } else {
            const float feedback   = params_.delayFeedback * 0.95f;
            const float normCutoff = params_.delayFilterCutoff;
            const float delayMix   = params_.delayMix * 0.75f;

            const float cutoffHz = 200.0f * std::pow(90.0f, normCutoff);
            const float omega    = 2.0f * static_cast<float>(M_PI) * cutoffHz
                                 / static_cast<float>(sampleRate_);
            const float c = 1.0f - std::cos(omega);
            const float a = -c + std::sqrt(c * c + 2.0f * c);

            // L line
            const float delaySecL = 0.01f * std::pow(300.0f, params_.delayTimeL);
            const int delaySamplesL = jlimit(1, kMaxDelaySamples - 1,
                static_cast<int>(delaySecL * static_cast<float>(sampleRate_)));
            const int readIdxL = (delayWriteL_ - delaySamplesL + kMaxDelaySamples) % kMaxDelaySamples;
            delayFilterState_  += a * (delayBufL_[readIdxL] - delayFilterState_);
            const float filteredL = delayFilterState_;
            delayBufL_[delayWriteL_] = reverbOutL + filteredL * feedback;
            delayWriteL_ = (delayWriteL_ + 1) % kMaxDelaySamples;
            delayOutL = reverbOutL * (1.0f - delayMix) + filteredL * delayMix;

            // R line
            const float delaySecR = 0.01f * std::pow(300.0f, params_.delayTimeR);
            const int delaySamplesR = jlimit(1, kMaxDelaySamples - 1,
                static_cast<int>(delaySecR * static_cast<float>(sampleRate_)));
            const int readIdxR = (delayWriteR_ - delaySamplesR + kMaxDelaySamples) % kMaxDelaySamples;
            delayFilterStateR_ += a * (delayBufR_[readIdxR] - delayFilterStateR_);
            const float filteredR = delayFilterStateR_;
            delayBufR_[delayWriteR_] = reverbOutR + filteredR * feedback;
            delayWriteR_ = (delayWriteR_ + 1) % kMaxDelaySamples;
            delayOutR = reverbOutR * (1.0f - delayMix) + filteredR * delayMix;

            delayEnergy_ = delayEnergy_ * 0.9999f
                + (std::abs(delayOutL - reverbOutL) + std::abs(delayOutR - reverbOutR)) * 0.00005f;
        }

        float side = 0.0f;
        if (!bypass_.noise) {
            const float sideWhite = noiseRng_.nextFloat() * 2.0f - 1.0f;
            side = sideWhite * 2.5f * params_.noiseLevel * params_.noiseStereo;
        }

        panMod_ = panMod;
        left[i]  = delayOutL + side;
        right[i] = delayOutR - side;
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void SynthEngine::reset() {
    // Oscillator phases — deterministic zero for consistent sound on every note-on
    phase_ = phaseDetune_ = fmPhase_ = 0.0;
    oscAH2Phase_ = oscAH3Phase_ = 0.0f;
    oscBPhase_       = params_.oscBPhase * kTwoPi;
    oscBPhaseDetune_ = 0.0;
    oscBSyncPhase_   = 0.0;

    // LFO phases
    lfo1Phase_ = lfo2Phase_ = lfo3Phase_ = lfo4Phase_ = 0.0;
    lfo1Smooth_ = lfo2Smooth_ = lfo3Smooth_ = lfo4Smooth_ = 0.0f;

    // Filter state
    fx1_=fx2_=fy1_=fy2_=0.0f;
    fx1b_=fx2b_=fy1b_=fy2b_=0.0f;
    lastCutoffHz_ = lastQ_ = -1.0f;
    lastFilterType_ = -1;
    smoothedCutoffHz_ = std::clamp(80.0f * std::pow(250.0f, params_.filterCutoff), 20.0f, 20000.0f);
    makeupGain_ = 1.0f;

    // Envelope state
    env1Stage_ = env2Stage_ = Stage::Idle;
    env1Value_ = env2Value_ = 0.0f;
    env1HoldTime_ = env2HoldTime_ = 0.0f;

    // Noise state (do NOT reseed noiseRng_)
    pinkB0_=pinkB1_=pinkB2_=pinkB3_=pinkB4_=pinkB5_=pinkB6_=0.0f;
    brownAccum_ = 0.0f;

    // Reverb state
    reverb_.reset();
    std::fill(std::begin(predelayBuf_), std::end(predelayBuf_), 0.0f);
    predelayWrite_ = 0;

    // Delay state
    std::fill(std::begin(delayBufL_), std::end(delayBufL_), 0.0f);
    std::fill(std::begin(delayBufR_), std::end(delayBufR_), 0.0f);
    delayWriteL_       = 0;
    delayWriteR_       = 0;
    delayFilterState_  = 0.0f;
    delayFilterStateR_ = 0.0f;
    reverbEnergy_      = 0.0f;
    delayEnergy_       = 0.0f;

    samplesAfterNoteOn_ = 0;
}

void SynthEngine::setBypassFlags(const std::vector<bool>& f) {
    if (f.size() < 9) return;
    bypass_.osc_b  = f[0];
    bypass_.noise  = f[1];
    bypass_.filter = f[2];
    bypass_.lfo1   = f[3];
    bypass_.lfo2   = f[4];
    bypass_.lfo3   = f[5];
    bypass_.lfo4   = f[6];
    bypass_.reverb = f[7];
    bypass_.delay  = f[8];
    // env flags — optional, only present in newer datasets (11-element vectors)
    bypass_.env1   = f.size() > 9  ? f[9]  : false;
    bypass_.env2   = f.size() > 10 ? f[10] : false;
}

std::vector<std::string> SynthEngine::getCategoricalLabels(const std::string& paramName) const {
    if (paramName == "osc_a_warp_type" || paramName == "osc_b_warp_type")
        return {"sync", "mirror", "fold", "ring"};
    if (paramName == "filter_type")
        return {"LP12", "LP24", "BP", "HP12", "HP24", "Notch"};
    if (paramName == "lfo1_target" || paramName == "lfo2_target" ||
        paramName == "lfo3_target" || paramName == "lfo4_target")
        return {"pitch", "filter_cutoff", "level", "warp"};
    if (paramName == "lfo1_shape" || paramName == "lfo2_shape" ||
        paramName == "lfo3_shape" || paramName == "lfo4_shape")
        return {"sine", "triangle", "saw", "square"};
    if (paramName == "noise_color")
        return {"white", "pink", "brown"};
    return {};
}

void SynthEngine::startFadeOut() {
    env1Stage_ = Stage::FadeOut;
    env2Stage_ = Stage::FadeOut;
}

void SynthEngine::setParameters(const SynthParameters& p) {
    params_ = p;
}

// ── Envelope tick ─────────────────────────────────────────────────────────────
// holdTime_ doubles as a linear progress accumulator [0,1] for Attack/Decay,
// and as elapsed-time accumulator (seconds) for the Hold stage.

float SynthEngine::tickEnv(Stage& stage, float& value, float& holdTime,
                            bool noteOn,
                            float normAttack, float normHold, float normDecay,
                            float normSustain, float normRelease,
                            float atkCurve, float decCurve)
{
    const float sr  = static_cast<float>(sampleRate_);
    const float inv = 1.0f / sr;

    if (stage == Stage::FadeOut) {
        value -= inv / kFadeOutSec;
        if (value <= 0.0f) { value = 0.0f; stage = Stage::Idle; }
        return value;
    }

    if (noteOn && stage == Stage::Idle) {
        stage = Stage::Attack;
        holdTime = 0.0f;
    }

    if (!noteOn && stage != Stage::Idle && stage != Stage::Release && stage != Stage::FadeOut) {
        stage = Stage::Release;
        holdTime = 0.0f;
    }

    const float atkSec = normToAttackSec(normAttack);
    const float decSec = normToDecaySec(normDecay);
    const float relSec = normToReleaseSec(normRelease);
    const float hldSec = normToHoldSec(normHold);

    // curve_physical ∈ [-1,+1]; shaped_t = t ^ 2^(-cp*2)
    const float atkCp  = (atkCurve - 0.5f) * 2.0f;
    const float decCp  = (decCurve - 0.5f) * 2.0f;
    const float atkExp = std::pow(2.0f, -atkCp * 2.0f);
    const float decExp = std::pow(2.0f, -decCp * 2.0f);

    switch (stage) {
        case Stage::Attack: {
            holdTime += inv / atkSec;          // linear t [0,1]
            if (holdTime >= 1.0f) {
                holdTime = 0.0f;
                value    = 1.0f;
                stage    = Stage::Hold;
            } else {
                value = std::pow(holdTime, atkExp);
            }
            break;
        }
        case Stage::Hold: {
            value     = 1.0f;
            holdTime += inv;                   // elapsed seconds
            if (holdTime >= hldSec) {
                holdTime = 0.0f;
                stage    = Stage::Decay;
            }
            break;
        }
        case Stage::Decay: {
            holdTime += inv / decSec;          // linear t [0,1]
            if (holdTime >= 1.0f) {
                holdTime = 0.0f;
                value    = normSustain;
                stage    = Stage::Sustain;
            } else {
                const float shaped = std::pow(holdTime, decExp);
                value = 1.0f - shaped * (1.0f - normSustain);
            }
            break;
        }
        case Stage::Sustain:
            value = normSustain;
            break;
        case Stage::Release: {
            holdTime += inv;
            value -= value * inv / relSec;
            if (value <= 1e-3f || holdTime > relSec * 3.0f) { value = 0.0f; stage = Stage::Idle; }
            break;
        }
        default:
            break;
    }
    return value;
}

} // namespace SynthV3

// ── SynthRegistry registration ────────────────────────────────────────────────
namespace {
    const bool _registered = []() {
        SynthRegistry::instance().registerSynth("SynthV3", []() -> std::unique_ptr<ISynth> {
            return std::make_unique<SynthV3::SynthEngine>();
        });
        return true;
    }();
}
