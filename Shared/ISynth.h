#pragma once
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ─── Parameter descriptor ────────────────────────────────────────────────────
// The following parameters are engine-internal, plugin-controlled, and NOT
// predicted by the model:
//   amplitude      — internal per-engine level; normalised by DataGenerator RMS
//   master_volume  — post-mix gain; user-adjustable plugin knob
//   master_pan     — stereo panning; user-adjustable plugin knob
//   master_tune    — global pitch offset; user-adjustable plugin knob
//
// Valid frame values:
//   "osc_a"  "osc_b"  "noise"
//   "filter" "env1"   "env2"
//   "lfo1"   "lfo2"   "lfo3"  "lfo4"
//   "reverb" "delay"
//
// Valid paramClass values:
//   "continuous"      0–1 knob, linear response
//   "continuous_log"  0–1 knob, log response
//   "morph"           0–1 knob with named stops
//   "categorical"     discrete named options
//   "bipolar"         -1–+1 knob, center = zero
//   "integer"         stepped integer knob
//   "bool"            toggle (bypass flags only, not in param vector)
struct ParamDescriptor {
    std::string_view name;
    float            defaultVal;
    float            minVal;
    float            maxVal;
    std::string_view frame      = "";
    std::string_view paramClass = "continuous";
    int              numClasses = 0;   // categorical/integer class count; 0 = continuous
};

// ─── Abstract synth interface ─────────────────────────────────────────────────
class ISynth {
public:
    virtual ~ISynth() = default;

    // Returns the ordered list of predictable parameters (no amplitude).
    virtual std::vector<ParamDescriptor> getParams() const = 0;

    // Set all normalised [0,1] parameter values. Vector size must match getParams().
    virtual void setParamValues(const std::vector<float>& values) = 0;

    // Read back the current normalised [0,1] values in the same order as getParams().
    virtual std::vector<float> getParamValues() const = 0;

    // Total audio duration (attack+decay+sustainHold+release), clamped to maxSec.
    virtual float computeDuration(float maxSec) const = 0;

    // Sample index at which note-off fires (attack+decay+sustainHoldSec).
    virtual int computeNoteOnSamples(double sampleRate, float sustainHoldSec) const = 0;

    // Prepare the engine for a new render session.
    virtual void prepare(double sampleRate, int samplesPerBlock) = 0;

    // Render numSamples into buffer. noteOn controls ADSR state per sample.
    virtual void process(float* buffer, int numSamples, bool noteOn, float freqHz) = 0;

    // Render numSamples into left/right buffers. Default: mono process, duplicated to both channels.
    virtual void processStereo(float* left, float* right, int numSamples,
                               bool noteOn, float freqHz) {
        process(left, numSamples, noteOn, freqHz);
        std::copy(left, left + numSamples, right);
    }

    // Reset ADSR/oscillator state without changing parameters.
    virtual void reset() = 0;

    // Set per-frame bypass flags. Default no-op — older synths are unaffected.
    // Flag order (11 flags): osc_b[0], noise[1], filter[2], lfo1-4[3-6], reverb[7], delay[8],
    //   env1[9] (flat amp when true), env2[10] (mirrors osc_b, set by DatasetWriter).
    // Vectors with fewer than 11 elements are handled gracefully — missing flags default to false.
    virtual void setBypassFlags(const std::vector<bool>& /*flags*/) {}

    // Returns the ordered string labels for a categorical parameter (e.g. "lfo1_target").
    // Returns an empty vector for unknown or non-categorical params.
    // Override in each synth to expose its own categorical label sets.
    virtual std::vector<std::string> getCategoricalLabels(
        const std::string& /*paramName*/) const { return {}; }
};

// ─── Synth registry ───────────────────────────────────────────────────────────
// Each synth AudioEngine registers itself at static-init time via
//   SynthRegistry::instance().registerSynth("SynthVN", []{ return std::make_unique<SynthVN::SynthEngine>(); });
// DataGenerator queries getNames() at startup and calls create() on selection.

class SynthRegistry {
public:
    using Factory = std::function<std::unique_ptr<ISynth>()>;

    static SynthRegistry& instance() {
        static SynthRegistry inst;
        return inst;
    }

    void registerSynth(const std::string& name, Factory f) {
        entries_.push_back({ name, std::move(f) });
    }

    std::vector<std::string> getNames() const {
        std::vector<std::string> names;
        names.reserve(entries_.size());
        for (const auto& e : entries_)
            names.push_back(e.first);
        return names;
    }

    std::unique_ptr<ISynth> create(const std::string& name) const {
        for (const auto& e : entries_)
            if (e.first == name) return e.second();
        throw std::runtime_error("SynthRegistry: unknown synth '" + name + "'");
    }

private:
    SynthRegistry() = default;
    std::vector<std::pair<std::string, Factory>> entries_;
};
