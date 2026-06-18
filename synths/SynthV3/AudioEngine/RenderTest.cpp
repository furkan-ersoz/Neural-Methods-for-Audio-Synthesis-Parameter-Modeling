// Standalone render test for SynthV3::SynthEngine.
// Build target: SynthV3RenderTest (added in CMakeLists.txt).
// Run from any working directory.

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include "SynthEngineV3.h"
#include "SynthParametersV3.h"

static constexpr int   kSr44 = 44100;
static constexpr double kSr  = 44100.0;

static float computePeakDb(const std::vector<float>& buf, int start, int end)
{
    float peak = 0.0f;
    for (int i = start; i < end && i < (int)buf.size(); ++i)
        peak = std::max(peak, std::abs(buf[i]));
    if (peak < 1e-10f) return -100.0f;
    return 20.0f * std::log10f(peak);
}

static float computeRmsDb(const std::vector<float>& buf, int start, int end)
{
    double sum = 0.0;
    int n = 0;
    for (int i = start; i < end && i < (int)buf.size(); ++i) {
        sum += (double)buf[i] * (double)buf[i];
        ++n;
    }
    if (n == 0 || sum < 1e-20) return -100.0f;
    return 20.0f * std::log10f((float)std::sqrt(sum / n));
}

// Stratified randomize matching the GUI randomize-button strategy:
// continuous_log -> random log, categorical -> fixed at 0, others -> random uniform.
// Master params fixed at defaults (volume=0.80, pan=0.50, tune=0.50).
static std::vector<float> randomizePatch(juce::Random& rng)
{
    std::vector<float> vals(SynthV3::NUM_PARAMS);
    for (int i = 0; i < SynthV3::NUM_PARAMS; ++i) {
        const auto& d = SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)];
        const std::string pc(d.paramClass);
        const std::string pname(d.name);

        const bool isMix = pname.size() >= 4 && pname.substr(pname.size() - 4) == "_mix";
        if (pname == "master_volume" || pname == "master_pan" || pname == "master_tune" || isMix) {
            vals[i] = d.defaultVal;
        } else if (pc == "categorical") {
            vals[i] = 0.0f;
        } else if (pc == "continuous_log") {
            const float eps = 1e-6f;
            const float logLo = std::log(d.minVal + eps);
            const float logHi = std::log(d.maxVal + eps);
            vals[i] = std::exp(logLo + rng.nextFloat() * (logHi - logLo));
        } else {
            vals[i] = d.minVal + rng.nextFloat() * (d.maxVal - d.minVal);
        }
    }
    return vals;
}

static void printParamVector(const std::vector<float>& vals)
{
    for (int i = 0; i < (int)vals.size(); ++i)
        printf("  [%2d] %-28s = %.6f\n", i,
               SynthV3::PARAM_DESCRIPTORS[static_cast<size_t>(i)].name.data(),
               vals[i]);
}

int main()
{
    printf("=== SynthV3 RegTest ===\n\n");

    // Header
    printf("%-4s  %-10s  %-10s  %-9s  %-10s  %s\n",
           "seed", "peak_dBFS", "rms_dBFS", "delay_ok", "reverb_ok", "result");
    printf("%-4s  %-10s  %-10s  %-9s  %-10s  %s\n",
           "----", "----------", "--------", "---------", "----------", "------");

    bool allPass = true;
    const int kTotalSamples = kSr44 * 3;   // 3 seconds
    const int kNoteOffAt    = kSr44;        // note on for 1 second, tail for 2

    for (int seed = 0; seed < 20; ++seed) {
        juce::Random rng(static_cast<int64_t>(seed));
        std::vector<float> vals = randomizePatch(rng);

        // Determine which effects are active based on randomized bypass params
        // (in DataGenerator bypass is rolled separately; here we enable delay+reverb
        // randomly to stress-test them — enable both for all seeds)
        const bool delayActive  = true;
        const bool reverbActive = true;

        // Set bypass: osc_b bypassed, noise bypassed, filter active, all lfos bypassed,
        // reverb active, delay active
        std::vector<bool> bypass = { true, true, false, true, true, true, true, false, false };

        SynthV3::SynthEngine engine;
        engine.prepare(kSr, 512);
        engine.setBypassFlags(bypass);
        engine.setParamValues(vals);

        std::vector<float> buf(kTotalSamples, 0.0f);
        engine.reset();
        for (int i = 0; i < kTotalSamples; ++i)
            engine.process(buf.data() + i, 1, i < kNoteOffAt, 440.0f);

        const float peakDb  = computePeakDb(buf, 0, kTotalSamples);
        const float rmsDb   = computeRmsDb (buf, 0, kSr44);         // first second
        const float rmsEcho = computeRmsDb (buf, kSr44, kSr44 * 2); // s[1..2]
        const float rmsTail = computeRmsDb (buf, kSr44 * 2, kSr44 * 3); // s[2..3]

        const bool noClip  = peakDb  < 0.0f;
        const bool notSilent = rmsDb > -40.0f;

        // delay: echo window RMS must be > first-second RMS * 0.05
        const float rmsFirst = std::pow(10.0f, rmsDb / 20.0f);
        const float rmsEchoLin = std::pow(10.0f, rmsEcho / 20.0f);
        const bool delayOk = !delayActive || (rmsEchoLin > rmsFirst * 0.05f);

        // reverb: tail at 2-3s must be > -55 dB
        const bool reverbOk = !reverbActive || (rmsTail > -55.0f);

        const bool pass = noClip && notSilent && delayOk && reverbOk;
        allPass &= pass;

        printf("%4d  %10.1f  %10.1f  %-9s  %-10s  %s\n",
               seed, peakDb, rmsDb,
               delayOk  ? "ok" : "FAIL",
               reverbOk ? "ok" : "FAIL",
               pass ? "PASS" : "FAIL");

        if (!pass) {
            if (!noClip)    printf("  -> CLIP: peak=%.1f dBFS\n", peakDb);
            if (!notSilent) printf("  -> SILENT: rms=%.1f dB\n", rmsDb);
            if (!delayOk)   printf("  -> NO ECHO: echo_rms=%.1f dB, first_rms=%.1f dB\n",
                                   rmsEcho, rmsDb);
            if (!reverbOk)  printf("  -> NO TAIL: tail_rms=%.1f dB\n", rmsTail);
            printf("  Param vector:\n");
            printParamVector(vals);
        }
    }

    printf("\n=== %s ===\n", allPass ? "ALL PASS" : "SOME FAILURES");
    return allPass ? 0 : 1;
}
