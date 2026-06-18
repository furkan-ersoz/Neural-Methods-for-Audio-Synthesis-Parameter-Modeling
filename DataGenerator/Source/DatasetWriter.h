#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include "ISynth.h"
#include "SamplingProfiles/SamplingProfile.h"

namespace DataGen {

enum class Strategy { RandomUniform, RandomLog, Linear, Fixed, Range, Discrete };

enum class RenderStatus { Ok, Silent, Loud };

struct ParamConfig {
    Strategy strategy   = Strategy::RandomUniform;
    float    rangeMin   = 0.0f;
    float    rangeMax   = 1.0f;
    float    fixedVal   = 0.5f;
    int      numClasses = 0;   // Discrete only (>=2)
};


enum class RenderMode {
    AllFrames,      // save all active frame WAVs + full WAV
    FinalOnly,      // save only full WAV (legacy behaviour)
    SelectedFrames  // save only frames listed in frameSelection
};

struct FrameSelection {
    bool osc_a  = true;
    bool osc_b  = true;
    bool noise  = true;
    bool filter = true;
    bool env1   = true;
    bool env2   = true;
    bool lfo1   = true;
    bool lfo2   = true;
    bool lfo3   = true;
    bool lfo4   = true;
    bool reverb = true;
    bool delay  = true;
};

// filenames[0]=f01 … filenames[11]=f12; empty string if not saved
struct FrameRenderResult {
    std::array<juce::String, 12> filenames;
};

struct WriterSettings {
    juce::File   outputDir;
    int          numSamples         = 500;
    uint32_t     baseSeed           = 42;
    float        maxDuration        = 10.0f;
    float        targetRmsDb        = -16.0f;
    float        silenceThresholdDb = -24.0f;
    float        loudThresholdDb    = -6.0f;
    bool         normalizeRms       = false; // legacy: per-sample RMS normalization

    std::function<std::unique_ptr<ISynth>()> synthFactory;
    std::vector<ParamConfig>                    paramConfigs;
    std::vector<SamplingProfiles::FrameRule>    frameRules;

    std::string profileName;
    std::string synthName;
    bool        lfoStratified  = false;

    RenderMode     renderMode     = RenderMode::FinalOnly;
    FrameSelection frameSelection;

    // Parallel generation: 1 = sequential (legacy path), >1 = worker pool.
    int   numWorkers      = 1;
    // Task pool size = numSamples * oversampleFactor, to absorb silence-skips.
    float oversampleFactor = 2.5f;
};

struct WriterCallbacks {
    std::function<void(int current, int total)> onProgress;
    std::function<void(const juce::String&)>    onLog;
    std::function<void(bool success)>           onFinished;
};

class DatasetWriter : public juce::Thread {
public:
    DatasetWriter();
    ~DatasetWriter() override { stopAndWait(); }

    void configure(const WriterSettings& settings, const WriterCallbacks& callbacks);
    void run() override;
    void stopAndWait();
    float estimatedSizeMB() const;

private:
    // One pre-generated sample's parameters/bypass flags (deterministic, RNG-derived).
    struct SampleTask {
        int                linearIndex = 0;
        std::vector<float> paramValues;
        std::vector<bool>  bypassFlags;
    };

    struct ParallelRunResult {
        int written  = 0;
        int attempts = 0;
    };

    SampleTask generateTask(juce::Random& rng,
                            const std::vector<ParamDescriptor>& paramDescs,
                            int linearIndex, int total) const;

    ParallelRunResult runParallel(const juce::File& samplesDir,
                                   const juce::File& samplesFullDir,
                                   const std::vector<ParamDescriptor>& paramDescs,
                                   juce::Random& rng);

    RenderStatus renderSample(ISynth& engine,
                       juce::AudioBuffer<float>& buffer,
                       double sampleRate);

    RenderStatus renderAndWrite(ISynth& engine,
                         const juce::File& dir,
                         int index,
                         const juce::String& suffix);

    // Renders full + all active/selected frames RAW into buffers, then applies a
    // SINGLE per-sample gain to every buffer (full + frames). No file I/O here.
    // Returns non-Ok if the FULL render is silent/too loud (caller skips, no index consumed).
    RenderStatus renderSampleGroup(ISynth& engine,
                           const SampleTask& task,
                           juce::AudioBuffer<float>& fullBufOut,
                           std::array<juce::AudioBuffer<float>, 12>& frameBufsOut,
                           std::array<bool, 12>& frameActiveOut);

    // Writes the already-gained buffers to disk and emits the CSV row.
    // CSV write is guarded by csvMutex_ internally (safe for both paths).
    void writeSampleGroup(const juce::File& samplesDir,
                          const juce::File& samplesFullDir,
                          int idx,
                          const juce::AudioBuffer<float>& fullBuf,
                          const std::array<juce::AudioBuffer<float>, 12>& frameBufs,
                          const std::array<bool, 12>& frameActive,
                          const SampleTask& task,
                          const std::vector<ParamDescriptor>& paramDescs);

    float sampleParam(const ParamConfig& cfg, juce::Random& rng,
                      int linearIndex, int total) const;

    bool  writeWav(const juce::AudioBuffer<float>& buffer,
                   double sampleRate, const juce::File& file);

    bool  openCsv(const juce::File& csvFile,
                  const std::vector<ParamDescriptor>& paramDescriptors);

    // FinalOnly mode
    void  writeCsvRow(const juce::String& filenameFull,
                      const std::vector<float>& values,
                      const std::vector<bool>& bypassFlags,
                      const juce::String& filenameOsc,
                      const juce::String& filenameFiltered);

    // AllFrames / SelectedFrames mode
    void  writeCsvRowFrames(const juce::String& filenameFull,
                            const std::vector<float>& values,
                            const std::vector<bool>& bypassFlags,
                            const std::vector<ParamDescriptor>& paramDescs,
                            const FrameRenderResult& frameResult);

    void  closeCsv();

    static float        computeRmsDb(const juce::AudioBuffer<float>& buffer);
    static int          findParamIndex(const std::vector<ParamDescriptor>& descs,
                                       std::string_view name);

    void writeLog(const juce::String& msg);

    WriterSettings   settings_;
    WriterCallbacks  callbacks_;

    std::unique_ptr<juce::FileOutputStream> csvStream_;
    std::mutex                               csvMutex_;
    std::mutex                               logMutex_;
    std::atomic<bool>                        shouldStop_ { false };
    std::array<int, 4>                      lfoTargetIndices_ = {-1, -1, -1, -1};

    static constexpr double kSampleRate  = 44100.0;
    static constexpr float  kSustainHold = 0.5f;

    std::array<std::vector<std::string>, 4> lfoTargetLabels_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DatasetWriter)
};

} // namespace DataGen
