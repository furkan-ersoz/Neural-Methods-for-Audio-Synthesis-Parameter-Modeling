// Standalone codec/alignment verification for SynthV3::SynthEngine.
// Build target: SynthV3ParamCodecTest (added in CMakeLists.txt).
//
// Checks:
//  1) getCategoricalLabels(name).size() == numClasses for every categorical param
//     (exercised via engine.prepare(), which throws on mismatch).
//  2) Pure codec round-trip: normToIndex(d, indexToNorm(d, k)) == k for all k in [0,N-1].
//  3) Renders short WAV files for manual spot-check of motor alignment:
//     noise_color {white,pink,brown}, lfo1_shape {sine,tri,saw,square} via level-mod,
//     and warp_type k=0 (sync) on both OSC A and OSC B.

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_dsp/juce_dsp.h>
#include "SynthEngineV3.h"
#include "SynthParametersV3.h"
#include "ParamCodec.h"

static constexpr double kSr = 44100.0;

static int indexOfParam(const std::string& name) {
    for (int i = 0; i < SynthV3::NUM_PARAMS; ++i)
        if (SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].name == name)
            return i;
    return -1;
}

static const ParamDescriptor& descOf(const std::string& name) {
    for (const auto& d : SynthV3::PARAM_DESCRIPTORS)
        if (d.name == name) return d;
    throw std::logic_error("unknown param: " + name);
}

static bool writeWav(const std::vector<float>& buf, double sr, const juce::File& file) {
    juce::WavAudioFormat wavFormat;
    auto stream = std::unique_ptr<juce::FileOutputStream>(file.createOutputStream());
    if (!stream) return false;
    JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE("-Wdeprecated-declarations")
    auto* writer = wavFormat.createWriterFor(stream.get(), sr, 1u, 16, {}, 0);
    JUCE_END_IGNORE_WARNINGS_GCC_LIKE
    if (!writer) return false;
    stream.release();
    std::unique_ptr<juce::AudioFormatWriter> writerOwner(writer);
    juce::AudioBuffer<float> ab(1, static_cast<int>(buf.size()));
    ab.copyFrom(0, 0, buf.data(), static_cast<int>(buf.size()));
    return writerOwner->writeFromAudioSampleBuffer(ab, 0, ab.getNumSamples());
}

static std::vector<float> renderPatch(std::vector<float> vals, std::vector<bool> bypass,
                                       int numSamples, int noteOffAt) {
    SynthV3::SynthEngine engine;
    engine.prepare(kSr, 512);
    engine.setBypassFlags(bypass);
    engine.setParamValues(vals);
    engine.reset();
    std::vector<float> buf(static_cast<size_t>(numSamples), 0.0f);
    for (int i = 0; i < numSamples; ++i)
        engine.process(buf.data() + i, 1, i < noteOffAt, 440.0f);
    return buf;
}

int main() {
    bool allPass = true;
    const auto outDir = juce::File::getSpecialLocation(juce::File::tempDirectory)
                             .getChildFile("SynthV3ParamCodecTest");
    outDir.createDirectory();
    printf("Output dir: %s\n\n", outDir.getFullPathName().toRawUTF8());

    // ── 1) Startup label/numClasses validation ──────────────────────────────
    printf("=== 1) Label/numClasses validation ===\n");
    try {
        SynthV3::SynthEngine engine;
        engine.prepare(kSr, 512);
        for (const auto& d : SynthV3::PARAM_DESCRIPTORS) {
            if (!ParamCodec::isDiscrete(d)) continue;
            auto labels = engine.getCategoricalLabels(std::string(d.name));
            if (labels.empty()) continue;
            const bool ok = static_cast<int>(labels.size()) == d.numClasses;
            printf("  %-20s numClasses=%-2d labels=%-2zu  %s\n",
                   std::string(d.name).c_str(), d.numClasses, labels.size(),
                   ok ? "OK" : "MISMATCH");
            allPass &= ok;
        }
        printf("  prepare() validation: PASS (no throw)\n");
    } catch (const std::exception& e) {
        printf("  prepare() validation: FAIL (%s)\n", e.what());
        allPass = false;
    }

    // ── 2) Pure codec round-trip ─────────────────────────────────────────────
    printf("\n=== 2) Codec round-trip (normToIndex(indexToNorm(k)) == k) ===\n");
    for (const auto& d : SynthV3::PARAM_DESCRIPTORS) {
        if (!ParamCodec::isDiscrete(d)) continue;
        bool ok = true;
        for (int k = 0; k < d.numClasses; ++k) {
            const float norm = ParamCodec::indexToNorm(d, k);
            const int back = ParamCodec::normToIndex(d, norm);
            if (back != k) ok = false;
        }
        printf("  %-20s numClasses=%-2d  %s\n", std::string(d.name).c_str(), d.numClasses,
               ok ? "OK" : "FAIL");
        allPass &= ok;
    }

    // ── 3) Motor-alignment spot-check renders ────────────────────────────────
    printf("\n=== 3) Motor alignment WAV renders (manual spot-check) ===\n");

    // Common bypass: osc_b off, noise off, filter off, all LFOs off, reverb/delay off
    std::vector<bool> bypassBase = { true, true, true, true, true, true, true, true, true };

    // -- noise_color: white/pink/brown --
    {
        const auto& d = descOf("noise_color");
        const char* names[] = { "white", "pink", "brown" };
        for (int k = 0; k < d.numClasses; ++k) {
            std::vector<float> vals(SynthV3::NUM_PARAMS, 0.0f);
            for (int i = 0; i < SynthV3::NUM_PARAMS; ++i)
                vals[static_cast<size_t>(i)] = SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].defaultVal;
            vals[static_cast<size_t>(indexOfParam("osc_a_level"))] = 0.0f; // silence osc A
            vals[static_cast<size_t>(indexOfParam("noise_level"))] = 0.8f;
            vals[static_cast<size_t>(indexOfParam("noise_color"))] = ParamCodec::indexToNorm(d, k);

            auto bypass = bypassBase;
            bypass[1] = false; // noise on
            auto buf = renderPatch(vals, bypass, static_cast<int>(kSr), static_cast<int>(kSr));
            auto file = outDir.getChildFile(juce::String("noise_") + names[k] + ".wav");
            writeWav(buf, kSr, file);
            printf("  noise_color k=%d (%s) -> %s\n", k, names[k], file.getFullPathName().toRawUTF8());
        }
    }

    // -- lfo1_shape via level modulation: sine/triangle/saw/square --
    {
        const auto& shapeDesc  = descOf("lfo1_shape");
        const auto& targetDesc = descOf("lfo1_target");
        const char* names[] = { "sine", "triangle", "saw", "square" };
        // lfo1_target index 2 == "level" (see getCategoricalLabels order)
        const float levelTargetNorm = ParamCodec::indexToNorm(targetDesc, 2);
        // ~2 Hz LFO: rateHz = 0.05 * 400^rateNorm
        const float lfo1RateNorm = std::log(2.0f / 0.05f) / std::log(400.0f);

        for (int k = 0; k < shapeDesc.numClasses; ++k) {
            std::vector<float> vals(SynthV3::NUM_PARAMS, 0.0f);
            for (int i = 0; i < SynthV3::NUM_PARAMS; ++i)
                vals[static_cast<size_t>(i)] = SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].defaultVal;
            vals[static_cast<size_t>(indexOfParam("osc_a_level"))]  = 0.8f;
            vals[static_cast<size_t>(indexOfParam("osc_a_waveform"))] = 0.0f; // sine carrier
            vals[static_cast<size_t>(indexOfParam("lfo1_rate"))]   = lfo1RateNorm;
            vals[static_cast<size_t>(indexOfParam("lfo1_depth"))]  = 0.9f;
            vals[static_cast<size_t>(indexOfParam("lfo1_shape"))]  = ParamCodec::indexToNorm(shapeDesc, k);
            vals[static_cast<size_t>(indexOfParam("lfo1_target"))] = levelTargetNorm;

            auto bypass = bypassBase;
            bypass[3] = false; // lfo1 on
            auto buf = renderPatch(vals, bypass, static_cast<int>(kSr * 2), static_cast<int>(kSr * 2));
            auto file = outDir.getChildFile(juce::String("lfo1_shape_") + names[k] + ".wav");
            writeWav(buf, kSr, file);
            printf("  lfo1_shape k=%d (%s) -> %s\n", k, names[k], file.getFullPathName().toRawUTF8());
        }
    }

    // -- warp_type k=0 (sync) on OSC A and OSC B --
    {
        const auto& warpDescA = descOf("osc_a_warp_type");
        const auto& warpDescB = descOf("osc_b_warp_type");

        // OSC A sync
        {
            std::vector<float> vals(SynthV3::NUM_PARAMS, 0.0f);
            for (int i = 0; i < SynthV3::NUM_PARAMS; ++i)
                vals[static_cast<size_t>(i)] = SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].defaultVal;
            vals[static_cast<size_t>(indexOfParam("osc_a_level"))]     = 0.8f;
            vals[static_cast<size_t>(indexOfParam("osc_a_waveform"))]  = 0.0f;
            vals[static_cast<size_t>(indexOfParam("osc_a_warp_type"))] = ParamCodec::indexToNorm(warpDescA, 0);
            vals[static_cast<size_t>(indexOfParam("osc_a_warp_amt"))]  = 0.5f;

            auto buf = renderPatch(vals, bypassBase, static_cast<int>(kSr), static_cast<int>(kSr));
            auto file = outDir.getChildFile("warp_sync_oscA.wav");
            writeWav(buf, kSr, file);
            printf("  osc_a_warp_type k=0 (sync) -> %s\n", file.getFullPathName().toRawUTF8());
        }

        // OSC B sync (silence OSC A)
        {
            std::vector<float> vals(SynthV3::NUM_PARAMS, 0.0f);
            for (int i = 0; i < SynthV3::NUM_PARAMS; ++i)
                vals[static_cast<size_t>(i)] = SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].defaultVal;
            vals[static_cast<size_t>(indexOfParam("osc_a_level"))]     = 0.0f;
            vals[static_cast<size_t>(indexOfParam("osc_b_level"))]     = 0.8f;
            vals[static_cast<size_t>(indexOfParam("osc_b_waveform"))]  = 0.0f;
            vals[static_cast<size_t>(indexOfParam("osc_b_warp_type"))] = ParamCodec::indexToNorm(warpDescB, 0);
            vals[static_cast<size_t>(indexOfParam("osc_b_warp_amt"))]  = 0.5f;

            auto bypass = bypassBase;
            bypass[0] = false; // osc_b on
            auto buf = renderPatch(vals, bypass, static_cast<int>(kSr), static_cast<int>(kSr));
            auto file = outDir.getChildFile("warp_sync_oscB.wav");
            writeWav(buf, kSr, file);
            printf("  osc_b_warp_type k=0 (sync) -> %s\n", file.getFullPathName().toRawUTF8());
        }
    }

    printf("\n=== %s ===\n", allPass ? "1-2: ALL PASS" : "1-2: SOME FAILURES");
    printf("Listen/inspect WAVs in %s for item 3.\n", outDir.getFullPathName().toRawUTF8());
    return allPass ? 0 : 1;
}
