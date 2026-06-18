#include "DatasetWriter.h"
#include "ParamCodec.h"
#include <algorithm>
#include <cmath>

namespace DataGen {

namespace {
    const char* skipReason(RenderStatus status) {
        return status == RenderStatus::Loud ? "Loud" : "Silent";
    }
}

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
    const float bytesPerSample = static_cast<float>(kSampleRate) * settings_.maxDuration * 2.0f;
    const float totalBytes     = bytesPerSample * static_cast<float>(settings_.numSamples);
    return totalBytes / (1024.0f * 1024.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::run()
{
    shouldStop_.store(false);

    if (!settings_.synthFactory) {
        writeLog("ERROR: No synth factory configured.");
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    // ── Resolve synth number from output path (SynthVN/datasets/) ────────────
    int synthNum = 1;
    {
        const auto synthDirName = settings_.outputDir.getParentDirectory().getFileName();
        const auto vPart = synthDirName.fromFirstOccurrenceOf("SynthV", false, false);
        if (vPart.isNotEmpty())
            synthNum = vPart.getIntValue();
    }

    // ── Auto-increment dataset number ─────────────────────────────────────────
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

    if (!sessionDir.createDirectory()) {
        writeLog("ERROR: Could not create folder: " + sessionDir.getFullPathName());
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    const auto samplesDir = sessionDir.getChildFile("samples");
    samplesDir.createDirectory();
    const auto samplesFullDir = sessionDir.getChildFile("samples_full");
    samplesFullDir.createDirectory();

    // ── Create engine once; reuse across samples ───────────────────────────────
    auto engine = settings_.synthFactory();
    engine->prepare(kSampleRate, 512);
    const auto paramDescs = engine->getParams();

    // Cache lfo_target string labels from the synth (generalises across synth versions)
    const std::array<std::string, 4> lfoTargetParamNames = {
        "lfo1_target", "lfo2_target", "lfo3_target", "lfo4_target"
    };
    for (int i = 0; i < 4; ++i)
        lfoTargetLabels_[static_cast<size_t>(i)] =
            engine->getCategoricalLabels(lfoTargetParamNames[static_cast<size_t>(i)]);

    // ── Open CSV ──────────────────────────────────────────────────────────────
    if (!openCsv(sessionDir.getChildFile("dataset.csv"), paramDescs)) {
        writeLog("ERROR: Could not open CSV file.");
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    writeLog("Started -> " + sessionDir.getFullPathName());

    // ── Main loop ─────────────────────────────────────────────────────────────
    int written     = 0;
    int attempts    = 0;
    int linearIndex = 0;

    juce::Random rng(static_cast<int64_t>(settings_.baseSeed));

    if (settings_.numWorkers > 1)
    {
        const auto result = runParallel(samplesDir, samplesFullDir, paramDescs, rng);
        written  = result.written;
        attempts = result.attempts;
    }
    else
    while (written < settings_.numSamples && !threadShouldExit())
    {
        ++attempts;

        const auto task = generateTask(rng, paramDescs, linearIndex, settings_.numSamples);
        const auto& paramValues  = task.paramValues;
        const auto& bypassFlags  = task.bypassFlags;

        if (settings_.renderMode == RenderMode::FinalOnly)
        {
            const int idx = written + 1;

            // ── Legacy 3-render path ──────────────────────────────────────────
            engine->setBypassFlags(bypassFlags);
            engine->setParamValues(paramValues);
            engine->reset();
            const auto fullStatus = renderAndWrite(*engine, samplesDir, idx, "_full");
            if (fullStatus != RenderStatus::Ok) {
                writeLog(juce::String(skipReason(fullStatus)) + " sample skipped (attempt " + juce::String(attempts) + ")");
                continue;
            }

            std::vector<bool> filteredFlags = bypassFlags;
            filteredFlags[3] = filteredFlags[4] = filteredFlags[5] = filteredFlags[6] = true;
            filteredFlags[7] = filteredFlags[8] = true;
            engine->setBypassFlags(filteredFlags);
            engine->setParamValues(paramValues);
            engine->reset();
            renderAndWrite(*engine, samplesDir, idx, "_filtered");

            std::vector<bool> oscFlags = filteredFlags;
            oscFlags[2] = true;
            engine->setBypassFlags(oscFlags);
            engine->setParamValues(paramValues);
            engine->reset();
            renderAndWrite(*engine, samplesDir, idx, "_osc");

            engine->setBypassFlags(bypassFlags);

            const auto nameFull     = juce::String::formatted("sample_%05d_full.wav",     idx);
            const auto nameOsc      = juce::String::formatted("sample_%05d_osc.wav",      idx);
            const auto nameFiltered = juce::String::formatted("sample_%05d_filtered.wav", idx);
            writeCsvRow(nameFull, paramValues, bypassFlags, nameOsc, nameFiltered);
        }
        else // AllFrames / SelectedFrames
        {
            juce::AudioBuffer<float> fullBuf;
            std::array<juce::AudioBuffer<float>, 12> frameBufs;
            std::array<bool, 12> frameActive {};

            const auto groupStatus = renderSampleGroup(*engine, task, fullBuf, frameBufs, frameActive);
            if (groupStatus != RenderStatus::Ok) {
                writeLog(juce::String(skipReason(groupStatus)) + " sample skipped (attempt " + juce::String(attempts) + ")");
                continue;
            }

            const int idx = written + 1;
            writeSampleGroup(samplesDir, samplesFullDir, idx, fullBuf, frameBufs, frameActive, task, paramDescs);
        }

        ++written;
        ++linearIndex;

        if (callbacks_.onProgress)
            callbacks_.onProgress(written, settings_.numSamples);
    }

    closeCsv();

    if ((threadShouldExit() || shouldStop_.load()) && written < settings_.numSamples) {
        writeLog("Stopped. " + juce::String(written) + " samples written.");
        if (callbacks_.onFinished) callbacks_.onFinished(false);
        return;
    }

    if (written < settings_.numSamples)
        writeLog("WARNING: task pool exhausted - only " + juce::String(written)
                 + "/" + juce::String(settings_.numSamples) + " samples written.");

    writeLog("Done. " + juce::String(written) + "/" + juce::String(attempts)
             + " samples (skipped: " + juce::String(attempts - written) + ")");
    if (callbacks_.onFinished) callbacks_.onFinished(true);
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::stopAndWait()
{
    shouldStop_.store(true);
    signalThreadShouldExit();
    stopThread(3000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministically draws one sample's parameters/bypass flags from rng.
// Same RNG-consumption order as the original sequential loop.
DatasetWriter::SampleTask DatasetWriter::generateTask(juce::Random& rng,
                                                       const std::vector<ParamDescriptor>& paramDescs,
                                                       int linearIndex, int total) const
{
    const int numParams = static_cast<int>(paramDescs.size());

    SampleTask task;
    task.linearIndex = linearIndex;
    task.paramValues.resize(static_cast<size_t>(numParams));
    for (int p = 0; p < numParams; ++p)
        task.paramValues[static_cast<size_t>(p)] =
            sampleParam(settings_.paramConfigs[static_cast<size_t>(p)], rng, linearIndex, total);

    auto bypassProbFor = [&](const char* frameName) -> float {
        for (const auto& fr : settings_.frameRules)
            if (fr.frameName == frameName)
                return 1.0f - fr.activeProb;
        return 0.0f;
    };

    auto& bypassFlags = task.bypassFlags;
    bypassFlags.resize(11);
    bypassFlags[0] = rng.nextFloat() < bypassProbFor("osc_b");
    bypassFlags[1] = rng.nextFloat() < bypassProbFor("noise");
    bypassFlags[2] = rng.nextFloat() < bypassProbFor("filter");
    if (settings_.lfoStratified)
    {
        const int activeCount = rng.nextInt(5);

        std::array<int, 4> lfoSlots = {3, 4, 5, 6};
        for (int s = 3; s > 0; --s)
        {
            const int j = rng.nextInt(s + 1);
            std::swap(lfoSlots[static_cast<size_t>(s)],
                      lfoSlots[static_cast<size_t>(j)]);
        }

        bypassFlags[3] = bypassFlags[4] = bypassFlags[5] = bypassFlags[6] = true;

        for (int a = 0; a < activeCount; ++a)
            bypassFlags[static_cast<size_t>(lfoSlots[static_cast<size_t>(a)])] = false;
    }
    else
    {
        bypassFlags[3] = rng.nextFloat() < bypassProbFor("lfo1");
        bypassFlags[4] = rng.nextFloat() < bypassProbFor("lfo2");
        bypassFlags[5] = rng.nextFloat() < bypassProbFor("lfo3");
        bypassFlags[6] = rng.nextFloat() < bypassProbFor("lfo4");
    }
    bypassFlags[7] = rng.nextFloat() < bypassProbFor("reverb");
    bypassFlags[8] = rng.nextFloat() < bypassProbFor("delay");
    // env1: profile-controlled. env2 always mirrors osc_b — no independent sampling.
    bypassFlags[9]  = rng.nextFloat() < bypassProbFor("env1");
    bypassFlags[10] = bypassFlags[0]; // env2 follows osc_b

    return task;
}

// ─────────────────────────────────────────────────────────────────────────────
// Renders the task pool with a worker-thread pool. Each worker owns its own
// ISynth instance (created via synthFactory) and AudioBuffers. Output sample
// indices are assigned only once a render is confirmed non-silent, so the
// resulting dataset is gap-free even though tasks are processed out of order.
DatasetWriter::ParallelRunResult DatasetWriter::runParallel(const juce::File& samplesDir,
                                                             const juce::File& samplesFullDir,
                                                             const std::vector<ParamDescriptor>& paramDescs,
                                                             juce::Random& rng)
{
    const int total = settings_.numSamples;
    const int poolSize = std::max(total,
        static_cast<int>(std::ceil(static_cast<float>(total) * std::max(1.0f, settings_.oversampleFactor))));

    const bool hasLinear = std::any_of(settings_.paramConfigs.begin(), settings_.paramConfigs.end(),
        [](const ParamConfig& c) { return c.strategy == Strategy::Linear; });
    if (hasLinear)
        writeLog("WARNING: Strategy::Linear with numWorkers > 1 sweeps across the "
                 "oversampled task pool, not the requested sample count.");

    std::vector<SampleTask> tasks;
    tasks.reserve(static_cast<size_t>(poolSize));
    for (int i = 0; i < poolSize; ++i)
        tasks.push_back(generateTask(rng, paramDescs, i, poolSize));

    std::atomic<int> cursor    { 0 };
    std::atomic<int> nextIndex { 0 };
    std::atomic<int> completed { 0 };
    std::atomic<int> attempts  { 0 };

    auto workerFn = [&]()
    {
        auto engine = settings_.synthFactory();
        engine->prepare(kSampleRate, 512);

        while (!shouldStop_.load(std::memory_order_relaxed) && !threadShouldExit())
        {
            if (completed.load(std::memory_order_relaxed) >= total)
                break;

            const int taskIdx = cursor.fetch_add(1, std::memory_order_relaxed);
            if (taskIdx >= poolSize)
                break;

            const auto& task = tasks[static_cast<size_t>(taskIdx)];
            const int   attemptNum = attempts.fetch_add(1, std::memory_order_relaxed) + 1;

            if (settings_.renderMode == RenderMode::FinalOnly)
            {
                engine->setBypassFlags(task.bypassFlags);
                engine->setParamValues(task.paramValues);
                engine->reset();

                const float duration = engine->computeDuration(settings_.maxDuration);
                const int   nf       = static_cast<int>(duration * static_cast<float>(kSampleRate));
                juce::AudioBuffer<float> fullBuf(2, nf);
                fullBuf.clear();

                const auto sampleStatus = renderSample(*engine, fullBuf, kSampleRate);
                if (sampleStatus != RenderStatus::Ok) {
                    writeLog(juce::String(skipReason(sampleStatus)) + " sample skipped (attempt " + juce::String(attemptNum) + ")");
                    continue;
                }

                const int prevIndex = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (prevIndex >= total) {
                    nextIndex.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                const int idx = prevIndex + 1;

                const auto nameFull = juce::String::formatted("sample_%05d_full.wav", idx);
                if (!writeWav(fullBuf, kSampleRate, samplesDir.getChildFile(nameFull))) {
                    writeLog("ERROR: Could not write WAV -> " + nameFull);
                    continue;
                }

                std::vector<bool> filteredFlags = task.bypassFlags;
                filteredFlags[3] = filteredFlags[4] = filteredFlags[5] = filteredFlags[6] = true;
                filteredFlags[7] = filteredFlags[8] = true;
                engine->setBypassFlags(filteredFlags);
                engine->setParamValues(task.paramValues);
                engine->reset();
                renderAndWrite(*engine, samplesDir, idx, "_filtered");

                std::vector<bool> oscFlags = filteredFlags;
                oscFlags[2] = true;
                engine->setBypassFlags(oscFlags);
                engine->setParamValues(task.paramValues);
                engine->reset();
                renderAndWrite(*engine, samplesDir, idx, "_osc");

                const auto nameOsc      = juce::String::formatted("sample_%05d_osc.wav",      idx);
                const auto nameFiltered = juce::String::formatted("sample_%05d_filtered.wav", idx);

                std::lock_guard<std::mutex> lock(csvMutex_);
                writeCsvRow(nameFull, task.paramValues, task.bypassFlags, nameOsc, nameFiltered);
            }
            else // AllFrames / SelectedFrames
            {
                juce::AudioBuffer<float> fullBuf;
                std::array<juce::AudioBuffer<float>, 12> frameBufs;
                std::array<bool, 12> frameActive {};

                const auto groupStatus = renderSampleGroup(*engine, task, fullBuf, frameBufs, frameActive);
                if (groupStatus != RenderStatus::Ok) {
                    writeLog(juce::String(skipReason(groupStatus)) + " sample skipped (attempt " + juce::String(attemptNum) + ")");
                    continue;
                }

                const int prevIndex = nextIndex.fetch_add(1, std::memory_order_relaxed);
                if (prevIndex >= total) {
                    nextIndex.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                const int idx = prevIndex + 1;

                writeSampleGroup(samplesDir, samplesFullDir, idx, fullBuf, frameBufs, frameActive, task, paramDescs);
            }

            const int done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (callbacks_.onProgress)
                callbacks_.onProgress(done, total);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(settings_.numWorkers));
    for (int w = 0; w < settings_.numWorkers; ++w)
        workers.emplace_back(workerFn);
    for (auto& t : workers)
        t.join();

    return { completed.load(), attempts.load() };
}

// ─────────────────────────────────────────────────────────────────────────────
RenderStatus DatasetWriter::renderSample(ISynth& engine,
                                  juce::AudioBuffer<float>& buffer,
                                  double sampleRate)
{
    const int   noteOffAt = engine.computeNoteOnSamples(sampleRate, kSustainHold);
    const int   total     = buffer.getNumSamples();
    auto*       dataL     = buffer.getWritePointer(0);
    auto*       dataR     = buffer.getWritePointer(1);
    const float freqHz    = 440.0f * std::pow(2.0f, (60 - 69) / 12.0f); // MIDI 60

    constexpr int kBlockSize = 512;
    int pos = 0;
    while (pos < total) {
        const int segEnd  = std::min(pos + kBlockSize, total);
        const int segLen  = segEnd - pos;
        const bool noteOn = (pos < noteOffAt);
        engine.processStereo(dataL + pos, dataR + pos, segLen, noteOn, freqHz);
        pos = segEnd;
    }

    const int kFadeInSamples = static_cast<int>(sampleRate * 0.020);
    for (int i = 0; i < std::min(kFadeInSamples, buffer.getNumSamples()); ++i) {
        const float g = static_cast<float>(i) / static_cast<float>(kFadeInSamples);
        dataL[i] *= g;
        dataR[i] *= g;
    }

    const float rmsDb = computeRmsDb(buffer);
    if (rmsDb < settings_.silenceThresholdDb)
        return RenderStatus::Silent;
    if (rmsDb > settings_.loudThresholdDb)
        return RenderStatus::Loud;

    return RenderStatus::Ok;
}

// ─────────────────────────────────────────────────────────────────────────────
RenderStatus DatasetWriter::renderAndWrite(ISynth& engine,
                                    const juce::File& dir,
                                    int index,
                                    const juce::String& suffix)
{
    const float duration  = engine.computeDuration(settings_.maxDuration);
    const int   numFrames = static_cast<int>(duration * static_cast<float>(kSampleRate));
    juce::AudioBuffer<float> buffer(2, numFrames);
    buffer.clear();

    const auto status = renderSample(engine, buffer, kSampleRate);
    if (status != RenderStatus::Ok)
        return status;

    const auto filename = juce::String::formatted("sample_%05d", index) + suffix + ".wav";
    if (!writeWav(buffer, kSampleRate, dir.getChildFile(filename))) {
        writeLog("ERROR: Could not write WAV -> " + filename);
        return RenderStatus::Silent;
    }
    return RenderStatus::Ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Renders full + all active/selected frames RAW, then applies a single
// per-sample gain across full + all active frame buffers.
RenderStatus DatasetWriter::renderSampleGroup(ISynth& engine,
                                       const SampleTask& task,
                                       juce::AudioBuffer<float>& fullBufOut,
                                       std::array<juce::AudioBuffer<float>, 12>& frameBufsOut,
                                       std::array<bool, 12>& frameActiveOut)
{
    engine.setBypassFlags(task.bypassFlags);
    engine.setParamValues(task.paramValues);
    engine.reset();

    const float duration  = engine.computeDuration(settings_.maxDuration);
    const int   numFrames = static_cast<int>(duration * static_cast<float>(kSampleRate));

    fullBufOut.setSize(2, numFrames);
    fullBufOut.clear();
    const auto status = renderSample(engine, fullBufOut, kSampleRate);
    if (status != RenderStatus::Ok)
        return status;

    const auto& bypassFlags = task.bypassFlags;
    const bool B0 = bypassFlags[0]; // osc_b
    const bool B1 = bypassFlags[1]; // noise
    const bool B2 = bypassFlags[2]; // filter
    const bool B3 = bypassFlags[3]; // lfo1
    const bool B4 = bypassFlags[4]; // lfo2
    const bool B5 = bypassFlags[5]; // lfo3
    const bool B6 = bypassFlags[6]; // lfo4
    const bool B7 = bypassFlags[7]; // reverb
    const bool B8 = bypassFlags[8]; // delay

    // 12 frame configs: {bypass vector (9 flags), active condition}
    struct FrameCfg {
        std::vector<bool> flags;
        bool              active;
    };

    const std::array<FrameCfg, 12> frames = {{
        // f01: osc_a — always
        {{true,true,true,true,true,true,true,true,true},   true},
        // f02: osc_b
        {{B0,  true,true,true,true,true,true,true,true},   !B0},
        // f03: noise
        {{B0,  B1,  true,true,true,true,true,true,true},   !B1},
        // f04: filter
        {{B0,  B1,  B2,  true,true,true,true,true,true},   !B2},
        // f05: env1 — active unless explicitly bypassed via profile (bypassFlags[9])
        {{B0,  B1,  B2,  true,true,true,true,true,true},   !bypassFlags[9]},
        // f06: env2 — active with osc_b; bypassFlags[10] mirrors osc_b so condition is same
        {{B0,  B1,  B2,  true,true,true,true,true,true},   !B0},
        // f07: lfo1
        {{B0,  B1,  B2,  B3,  true,true,true,true,true},   !B3},
        // f08: lfo2
        {{B0,  B1,  B2,  B3,  B4,  true,true,true,true},   !B4},
        // f09: lfo3
        {{B0,  B1,  B2,  B3,  B4,  B5,  true,true,true},   !B5},
        // f10: lfo4
        {{B0,  B1,  B2,  B3,  B4,  B5,  B6,  true,true},   !B6},
        // f11: reverb
        {{B0,  B1,  B2,  B3,  B4,  B5,  B6,  B7,  true},   !B7},
        // f12: delay (= full bypass state)
        {bypassFlags,                                        !B8},
    }};

    // Per-frame selection bitmask (only used for SelectedFrames mode)
    const std::array<bool, 12> selected = {
        settings_.frameSelection.osc_a,  settings_.frameSelection.osc_b,
        settings_.frameSelection.noise,  settings_.frameSelection.filter,
        settings_.frameSelection.env1,   settings_.frameSelection.env2,
        settings_.frameSelection.lfo1,   settings_.frameSelection.lfo2,
        settings_.frameSelection.lfo3,   settings_.frameSelection.lfo4,
        settings_.frameSelection.reverb, settings_.frameSelection.delay,
    };

    for (int i = 0; i < 12; ++i) {
        const auto& fc = frames[static_cast<size_t>(i)];
        frameActiveOut[static_cast<size_t>(i)] = false;
        if (!fc.active) continue;
        if (settings_.renderMode == RenderMode::SelectedFrames && !selected[static_cast<size_t>(i)])
            continue;

        auto& buf = frameBufsOut[static_cast<size_t>(i)];
        buf.setSize(2, numFrames);
        buf.clear();
        engine.setBypassFlags(fc.flags);
        engine.reset();
        renderSample(engine, buf, kSampleRate); // silence return ignored for frame renders

        frameActiveOut[static_cast<size_t>(i)] = true;
    }

    // ── Single per-sample gain across full + all active frames ───────────────
    float globalPeak = fullBufOut.getMagnitude(0, fullBufOut.getNumSamples());
    for (int i = 0; i < 12; ++i)
        if (frameActiveOut[static_cast<size_t>(i)])
            globalPeak = std::max(globalPeak,
                frameBufsOut[static_cast<size_t>(i)].getMagnitude(0, frameBufsOut[static_cast<size_t>(i)].getNumSamples()));

    float gain = 1.0f;
    if (settings_.normalizeRms)
    {
        const float fullRmsDb = computeRmsDb(fullBufOut);
        if (fullRmsDb > -99.0f)
            gain = std::pow(10.0f, (settings_.targetRmsDb - fullRmsDb) / 20.0f);
    }
    else
    {
        gain = (globalPeak > 1.0f) ? (0.999f / globalPeak) : 1.0f;
    }

    if (gain != 1.0f)
    {
        fullBufOut.applyGain(gain);
        for (int i = 0; i < 12; ++i)
            if (frameActiveOut[static_cast<size_t>(i)])
                frameBufsOut[static_cast<size_t>(i)].applyGain(gain);
    }

    return RenderStatus::Ok;
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::writeSampleGroup(const juce::File& samplesDir,
                                      const juce::File& samplesFullDir,
                                      int idx,
                                      const juce::AudioBuffer<float>& fullBuf,
                                      const std::array<juce::AudioBuffer<float>, 12>& frameBufs,
                                      const std::array<bool, 12>& frameActive,
                                      const SampleTask& task,
                                      const std::vector<ParamDescriptor>& paramDescs)
{
    const auto fullBasename = juce::String::formatted("s%05d.wav", idx);
    const auto fullCsvPath  = "samples_full/" + fullBasename;
    if (!writeWav(fullBuf, kSampleRate, samplesFullDir.getChildFile(fullBasename))) {
        writeLog("ERROR: Could not write WAV -> " + fullCsvPath);
        return;
    }

    FrameRenderResult frameResult;
    for (int i = 0; i < 12; ++i) {
        if (!frameActive[static_cast<size_t>(i)]) continue;

        const auto basename = juce::String::formatted("s%05df%02d.wav", idx, i + 1);
        if (writeWav(frameBufs[static_cast<size_t>(i)], kSampleRate, samplesDir.getChildFile(basename)))
            frameResult.filenames[static_cast<size_t>(i)] = "samples/" + basename;
        else
            writeLog("ERROR: Could not write frame WAV -> " + basename);
    }

    std::lock_guard<std::mutex> lock(csvMutex_);
    writeCsvRowFrames(fullCsvPath, task.paramValues, task.bypassFlags, paramDescs, frameResult);
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

        case Strategy::Discrete:
        {
            if (cfg.numClasses < 2) return lo;
            const int k = rng.nextInt(cfg.numClasses);
            return static_cast<float>(k) / static_cast<float>(cfg.numClasses - 1);
        }

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
    if (!stream) return false;

    JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE("-Wdeprecated-declarations")
    auto* writer = wavFormat.createWriterFor(stream.get(), sampleRate, 2u, 16, {}, 0);
    JUCE_END_IGNORE_WARNINGS_GCC_LIKE

    if (!writer) return false;
    stream.release();
    std::unique_ptr<juce::AudioFormatWriter> writerOwner(writer);
    return writerOwner->writeFromAudioSampleBuffer(buffer, 0, buffer.getNumSamples());
}

// ─────────────────────────────────────────────────────────────────────────────
bool DatasetWriter::openCsv(const juce::File& csvFile,
                              const std::vector<ParamDescriptor>& paramDescs)
{
    csvStream_ = std::make_unique<juce::FileOutputStream>(csvFile);
    if (csvStream_->failedToOpen()) {
        csvStream_.reset();
        return false;
    }

    const auto now = juce::Time::getCurrentTime();
    csvStream_->writeText("# profile: " + juce::String(settings_.profileName) + "\n", false, false, nullptr);
    csvStream_->writeText("# synth: "   + juce::String(settings_.synthName)   + "\n", false, false, nullptr);
    csvStream_->writeText("# date: "    + now.formatted("%Y-%m-%d")            + "\n", false, false, nullptr);

    if (settings_.renderMode == RenderMode::FinalOnly)
    {
        juce::String header = "filename_full";
        for (const auto& d : paramDescs)
            header += ";" + juce::String(d.name.data());
        header += ";bypass_osc_b;bypass_noise;bypass_filter"
                  ";bypass_lfo1;bypass_lfo2;bypass_lfo3;bypass_lfo4"
                  ";bypass_reverb;bypass_delay;bypass_env1"
                  ";filename_osc;filename_filtered\n";
        csvStream_->writeText(header, false, false, nullptr);
    }
    else // AllFrames / SelectedFrames
    {
        // Cache lfo_target param indices for string encoding in writeCsvRowFrames
        lfoTargetIndices_.fill(-1);
        for (int i = 0; i < static_cast<int>(paramDescs.size()); ++i) {
            const std::string_view name = paramDescs[static_cast<size_t>(i)].name;
            if      (name == "lfo1_target") lfoTargetIndices_[0] = i;
            else if (name == "lfo2_target") lfoTargetIndices_[1] = i;
            else if (name == "lfo3_target") lfoTargetIndices_[2] = i;
            else if (name == "lfo4_target") lfoTargetIndices_[3] = i;
        }

        juce::String header = "filename";
        for (const auto& d : paramDescs)
            header += ";" + juce::String(d.name.data());
        header += ";bypass_osc_b;bypass_noise;bypass_filter;bypass_reverb;bypass_delay;bypass_env1"
                  ";active_lfo_count"
                  ";lfo1_target_label;lfo2_target_label;lfo3_target_label;lfo4_target_label";
        for (int i = 1; i <= 12; ++i)
            header += juce::String::formatted(";filename_frame%02d", i);
        header += "\n";
        csvStream_->writeText(header, false, false, nullptr);
    }

    return true;
}

// FinalOnly mode CSV row — signature and behaviour unchanged
void DatasetWriter::writeCsvRow(const juce::String& filenameFull,
                                 const std::vector<float>& values,
                                 const std::vector<bool>& bypassFlags,
                                 const juce::String& filenameOsc,
                                 const juce::String& filenameFiltered)
{
    if (!csvStream_) return;

    juce::String row = filenameFull;
    for (float v : values)
        row += ";" + juce::String::formatted("%.6f", v);
    for (bool b : bypassFlags)
        row += ";" + juce::String(b ? 1 : 0);
    row += ";" + filenameOsc + ";" + filenameFiltered + "\n";

    csvStream_->writeText(row, false, false, nullptr);
    csvStream_->flush();
}

// AllFrames / SelectedFrames mode CSV row
void DatasetWriter::writeCsvRowFrames(const juce::String& filenameFull,
                                       const std::vector<float>& values,
                                       const std::vector<bool>& bypassFlags,
                                       const std::vector<ParamDescriptor>& paramDescs,
                                       const FrameRenderResult& frameResult)
{
    if (!csvStream_) return;

    const bool B0 = bypassFlags[0]; // osc_b
    const bool B1 = bypassFlags[1]; // noise
    const bool B2 = bypassFlags[2]; // filter
    const bool B3 = bypassFlags[3]; // lfo1
    const bool B4 = bypassFlags[4]; // lfo2
    const bool B5 = bypassFlags[5]; // lfo3
    const bool B6 = bypassFlags[6]; // lfo4
    const bool B7 = bypassFlags[7]; // reverb
    const bool B8 = bypassFlags[8]; // delay

    const int activeLfoCount = (!B3 ? 1 : 0) + (!B4 ? 1 : 0) + (!B5 ? 1 : 0) + (!B6 ? 1 : 0);

    juce::String row = filenameFull;
    for (float v : values)
        row += ";" + juce::String::formatted("%.6f", v);

    row += ";" + juce::String(B0 ? 1 : 0); // bypass_osc_b
    row += ";" + juce::String(B1 ? 1 : 0); // bypass_noise
    row += ";" + juce::String(B2 ? 1 : 0); // bypass_filter
    row += ";" + juce::String(B7 ? 1 : 0); // bypass_reverb
    row += ";" + juce::String(B8 ? 1 : 0); // bypass_delay
    row += ";" + juce::String(bypassFlags.size() > 9 && bypassFlags[9] ? 1 : 0); // bypass_env1
    row += ";" + juce::String(activeLfoCount);

    for (int i = 0; i < 4; ++i) {
        if (bypassFlags[3 + static_cast<size_t>(i)]) {
            row += ";";
            continue;
        }
        const int idx = lfoTargetIndices_[static_cast<size_t>(i)];
        if (idx >= 0 && idx < static_cast<int>(values.size())) {
            const auto& labels = lfoTargetLabels_[static_cast<size_t>(i)];
            if (!labels.empty()) {
                const int classIdx = ParamCodec::normToIndex(
                    paramDescs[static_cast<size_t>(idx)], values[static_cast<size_t>(idx)]);
                const int clamped = std::clamp(classIdx, 0, static_cast<int>(labels.size()) - 1);
                row += ";" + juce::String(labels[static_cast<size_t>(clamped)]);
            } else {
                row += ";";
            }
        } else {
            row += ";";
        }
    }

    for (const auto& fn : frameResult.filenames)
        row += ";" + fn;

    row += "\n";
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

    const auto* dataL = buffer.getReadPointer(0);
    const auto* dataR = buffer.getReadPointer(1);
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<double>(dataL[i]) * static_cast<double>(dataL[i])
             + static_cast<double>(dataR[i]) * static_cast<double>(dataR[i]);

    const float rms = static_cast<float>(std::sqrt(sum / (2.0 * n)));
    if (rms < 1e-10f) return -100.0f;
    return 20.0f * std::log10(rms);
}

int DatasetWriter::findParamIndex(const std::vector<ParamDescriptor>& descs,
                                   std::string_view name)
{
    for (int i = 0; i < static_cast<int>(descs.size()); ++i)
        if (descs[static_cast<size_t>(i)].name == name) return i;
    return -1;
}

// ─────────────────────────────────────────────────────────────────────────────
void DatasetWriter::writeLog(const juce::String& msg)
{
    std::lock_guard<std::mutex> lock(logMutex_);
    if (callbacks_.onLog)
        callbacks_.onLog(msg);
}

} // namespace DataGen
