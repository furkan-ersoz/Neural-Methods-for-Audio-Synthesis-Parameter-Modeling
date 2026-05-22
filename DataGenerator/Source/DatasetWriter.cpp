#include "DatasetWriter.h"
#include <cmath>
#include <algorithm>

namespace DataGen {

// ─────────────────────────────────────────────────────────────────────────────
DatasetWriter::DatasetWriter()
    : juce::Thread("DatasetWriter")
{
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::configure(const WriterSettings& s, const WriterCallbacks& cb)
{
    settings_  = s;
    callbacks_ = cb;
}

// ─────────────────────────────────────────────────────────────────────────────
float DatasetWriter::estimatedSizeMB() const
{
    // 16-bit mono WAV: sampleRate × maxDuration × 2 byte
    const float bytesPerSample = (float)kSampleRate * settings_.maxDuration * 2.0f;
    const float totalBytes     = bytesPerSample * (float)settings_.numSamples;
    return totalBytes / (1024.0f * 1024.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::run()
{
    // ── Session klasör adı: S##D## ───────────────────────────────────────
    // Synth numarasini outputDir'in üst dizininden parse et: SynthV1 → 01
    int synthNum = 1;
    {
        const auto synthDirName = settings_.outputDir.getParentDirectory().getFileName();
        const auto vPart = synthDirName.fromFirstOccurrenceOf("SynthV", false, false);
        if (vPart.isNotEmpty())
            synthNum = vPart.getIntValue();
    }

    // Mevcut S##D## klasörlerine bakarak sıradaki dataset numarasini bul
    int datasetNum = 1;
    {
        juce::Array<juce::File> existing;
        settings_.outputDir.findChildFiles(existing, juce::File::findDirectories, false, "S??D??");
        for (const auto& f : existing) {
            const auto dPart = f.getFileName().fromFirstOccurrenceOf("D", false, false);
            if (dPart.isNotEmpty())
                datasetNum = juce::jmax(datasetNum, dPart.getIntValue() + 1);
        }
    }

    const auto sessionName = juce::String::formatted("S%02dD%02d", synthNum, datasetNum);
    const auto sessionDir  = settings_.outputDir.getChildFile(sessionName);

    if (! sessionDir.createDirectory()) {
        writeLog("ERROR: Could not create folder: " + sessionDir.getFullPathName());
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    const auto samplesDir = sessionDir.getChildFile("samples");
    samplesDir.createDirectory();

    // ── CSV aç ────────────────────────────────────────────────────────────
    const auto csvFile = sessionDir.getChildFile("dataset.csv");
    if (! openCsv(csvFile)) {
        writeLog("ERROR: Could not open CSV file.");
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    writeLog("Started -> " + sessionDir.getFullPathName());

    // ── Ana döngü ─────────────────────────────────────────────────────────
    int written     = 0;
    int attempts    = 0;
    int linearIndex = 0;

    // RNG once seeded outside loop — same baseSeed always produces same sequence.
    juce::Random rng(static_cast<int64_t>(settings_.baseSeed));

    while (written < settings_.numSamples && ! threadShouldExit())
    {
        ++attempts;

        // ── Parametre örnekle ─────────────────────────────────────────────
        SynthV1::SynthParameters params;
        params.amplitude    = 1.0f;  // sabit — normalize edilecek
        params.attackTime   = sampleParam(settings_.paramConfigs[0], rng, linearIndex, settings_.numSamples);
        params.decayTime    = sampleParam(settings_.paramConfigs[1], rng, linearIndex, settings_.numSamples);
        params.sustainLevel = sampleParam(settings_.paramConfigs[2], rng, linearIndex, settings_.numSamples);
        params.releaseTime  = sampleParam(settings_.paramConfigs[3], rng, linearIndex, settings_.numSamples);

        // ── Buffer boyutlandır ────────────────────────────────────────────
        const float duration  = computeDuration(params);
        const int   numFrames = static_cast<int>(duration * (float)kSampleRate);
        juce::AudioBuffer<float> buffer(1, numFrames);
        buffer.clear();

        // ── Render ───────────────────────────────────────────────────────
        if (! renderSample(params, buffer, kSampleRate)) {
            writeLog("Silent sample skipped (attempt " + juce::String(attempts) + ")");
            continue;
        }

        // ── WAV dosya adı ─────────────────────────────────────────────────
        const auto filename = juce::String::formatted("sample_%05d.wav", written + 1);
        const auto wavFile  = samplesDir.getChildFile(filename);

        // ── WAV yaz ──────────────────────────────────────────────────────
        if (! writeWav(buffer, kSampleRate, wavFile)) {
            writeLog("ERROR: Could not write WAV -> " + filename);
            continue;
        }

        // ── CSV satırı ────────────────────────────────────────────────────
        writeCsvRow(filename, params);

        ++written;
        ++linearIndex;

        if (callbacks_.onProgress)
            callbacks_.onProgress(written, settings_.numSamples);
    }

    closeCsv();

    if (threadShouldExit() && written < settings_.numSamples) {
        writeLog("Stopped. " + juce::String(written) + " samples written.");
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    writeLog("Done. " + juce::String(written) + "/" + juce::String(attempts)
             + "  samples (skipped: " + juce::String(attempts - written) + ")");
    if (callbacks_.onFinished) callbacks_.onFinished(true);
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::stopAndWait()
{
    signalThreadShouldExit();
    stopThread(3000);
}

// ─────────────────────────────────────────────────────────────────────────────
bool DatasetWriter::renderSample(const SynthV1::SynthParameters& params,
                                 juce::AudioBuffer<float>& buffer,
                                 double sampleRate)
{
    SynthV1::SynthEngine engine;
    engine.prepare(sampleRate, buffer.getNumSamples());
    engine.setParameters(params);

    const float attackSec = SynthV1::toTimeSeconds(params.attackTime);
    const float decaySec  = SynthV1::toTimeSeconds(params.decayTime);
    const int   noteOffAt = static_cast<int>((attackSec + decaySec + kSustainHold) * (float)sampleRate);
    const int   total     = buffer.getNumSamples();

    auto* data = buffer.getWritePointer(0);

    for (int i = 0; i < total; ++i) {
        const bool noteOn = (i < noteOffAt);
        float s = 0.0f;
        engine.process(&s, 1, noteOn, SynthV1::midiToHz(60));
        data[i] = s;
    }

    // ── Sessizlik kontrolü ────────────────────────────────────────────────
    const float rmsDb = computeRmsDb(buffer);
    if (rmsDb < settings_.silenceThresholdDb)
        return false;

    // ── Normalize ─────────────────────────────────────────────────────────
    applyNormalization(buffer, settings_.targetRmsDb);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
float DatasetWriter::sampleParam(const ParamConfig& cfg,
                                 juce::Random& rng,
                                 int linearIndex,
                                 int total) const
{
    const float lo = cfg.rangeMin;
    const float hi = cfg.rangeMax;

    switch (cfg.strategy)
    {
        case Strategy::Fixed:
            return cfg.fixedVal;

        case Strategy::RandomUniform:
            return lo + rng.nextFloat() * (hi - lo);

        case Strategy::RandomLog:
        {
            const float eps   = 1e-6f;
            const float logLo = std::log(lo + eps);
            const float logHi = std::log(hi + eps);
            return std::exp(logLo + rng.nextFloat() * (logHi - logLo));
        }

        case Strategy::Linear:
        {
            if (total <= 1) return lo;
            const float t = static_cast<float>(linearIndex) / static_cast<float>(total - 1);
            return lo + t * (hi - lo);
        }

        case Strategy::Range:
            return lo + rng.nextFloat() * (hi - lo);

        default:
            return rng.nextFloat();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool DatasetWriter::writeWav(const juce::AudioBuffer<float>& buffer,
                             double sampleRate,
                             const juce::File& file)
{
    juce::WavAudioFormat wavFormat;
    auto stream = std::unique_ptr<juce::FileOutputStream>(file.createOutputStream());
    if (! stream) return false;

    // Deprecated uyarısını sustur — bu JUCE versiyonunda tek çalışan overload bu
    JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE("-Wdeprecated-declarations")
    auto* writer = wavFormat.createWriterFor(stream.get(), sampleRate, 1u, 16, {}, 0);
    JUCE_END_IGNORE_WARNINGS_GCC_LIKE

    if (! writer) return false;
    stream.release();  // writer stream'i sahipleniyor
    std::unique_ptr<juce::AudioFormatWriter> writerOwner(writer);
    return writerOwner->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
}

// ─────────────────────────────────────────────────────────────────────────────
bool DatasetWriter::openCsv(const juce::File& csvFile)
{
    csvStream_ = std::make_unique<juce::FileOutputStream>(csvFile);
    if (csvStream_->failedToOpen()) {
        csvStream_.reset();
        return false;
    }
    csvStream_->writeText("filename;attack;decay;sustain;release\n", false, false, nullptr);
    return true;
}

void DatasetWriter::writeCsvRow(const juce::String& filename,
                                const SynthV1::SynthParameters& params)
{
    if (! csvStream_) return;

    auto fmt = [](float v) { return juce::String::formatted("%.6f", v); };

    const juce::String row = filename
        + ";" + fmt(params.attackTime)
        + ";" + fmt(params.decayTime)
        + ";" + fmt(params.sustainLevel)
        + ";" + fmt(params.releaseTime)
        + "\n";

    csvStream_->writeText(row, false, false, nullptr);
    csvStream_->flush();
}

void DatasetWriter::closeCsv()
{
    if (csvStream_) {
        csvStream_->flush();
        csvStream_.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
float DatasetWriter::computeRmsDb(const juce::AudioBuffer<float>& buffer)
{
    const int n = buffer.getNumSamples();
    if (n == 0) return -100.0f;

    const auto* data = buffer.getReadPointer(0);
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<double>(data[i]) * static_cast<double>(data[i]);

    const float rms = static_cast<float>(std::sqrt(sum / n));
    if (rms < 1e-10f) return -100.0f;
    return 20.0f * std::log10(rms);
}

void DatasetWriter::applyNormalization(juce::AudioBuffer<float>& buffer, float targetDb)
{
    const float currentDb = computeRmsDb(buffer);
    if (currentDb < -99.0f) return;

    const float gainDb = targetDb - currentDb;
    const float gain   = std::pow(10.0f, gainDb / 20.0f);
    buffer.applyGain(gain);
}

// ─────────────────────────────────────────────────────────────────────────────
float DatasetWriter::computeDuration(const SynthV1::SynthParameters& params) const
{
    const float attack  = SynthV1::toTimeSeconds(params.attackTime);
    const float decay   = SynthV1::toTimeSeconds(params.decayTime);
    const float release = SynthV1::toTimeSeconds(params.releaseTime);
    const float total   = attack + decay + kSustainHold + release;
    return std::min(total, settings_.maxDuration);
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::writeLog(const juce::String& msg)
{
    if (callbacks_.onLog)
        callbacks_.onLog(msg);
}

} // namespace DataGen