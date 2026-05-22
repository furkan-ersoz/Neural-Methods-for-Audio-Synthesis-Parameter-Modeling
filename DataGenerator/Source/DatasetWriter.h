#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include "SynthEngineV1.h"
#include "SynthParametersV1.h"

namespace DataGen {

enum class Strategy { RandomUniform, RandomLog, Linear, Fixed, Range };

struct ParamConfig {
    Strategy strategy = Strategy::RandomUniform;
    float    rangeMin = 0.0f;
    float    rangeMax = 1.0f;
    float    fixedVal = 0.5f;
};

struct WriterSettings {
    juce::File   outputDir;
    int          numSamples          = 500;
    uint32_t     baseSeed            = 42;
    float        maxDuration         = 10.0f;
    float        targetRmsDb         = -18.0f;
    float        silenceThresholdDb  = -40.0f;

    // attack(0), decay(1), sustain(2), release(3)
    std::array<ParamConfig, 4> paramConfigs;
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
    bool  renderSample(const SynthV1::SynthParameters& params,
                       juce::AudioBuffer<float>& buffer,
                       double sampleRate);

    float sampleParam(const ParamConfig& cfg, juce::Random& rng,
                      int linearIndex, int total) const;

    bool  writeWav(const juce::AudioBuffer<float>& buffer,
                   double sampleRate, const juce::File& file);

    bool  openCsv(const juce::File& csvFile);
    void  writeCsvRow(const juce::String& filename,
                      const SynthV1::SynthParameters& params);
    void  closeCsv();

    static float computeRmsDb(const juce::AudioBuffer<float>& buffer);
    static void  applyNormalization(juce::AudioBuffer<float>& buffer, float targetDb);

    float computeDuration(const SynthV1::SynthParameters& params) const;

    // log() ismi math.h ile çakışıyor — writeLog kullanıyoruz
    void writeLog(const juce::String& msg);

    WriterSettings   settings_;
    WriterCallbacks  callbacks_;

    std::unique_ptr<juce::FileOutputStream> csvStream_;

    static constexpr double kSampleRate  = 44100.0;
    static constexpr float  kSustainHold = 0.5f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DatasetWriter)
};

} // namespace DataGen