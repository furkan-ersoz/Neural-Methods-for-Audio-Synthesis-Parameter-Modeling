#include "InferenceEngine.h"
#include "SynthParametersV3.h"
#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <map>

namespace SynthV3 {

namespace {

// Sequential pipeline frame order — door once, then 12 frame predictors
struct FrameSpec {
    const char* name;
    const char* file;
    int         prevParamsDim; // fallback if meta'da yoksa
    bool        hasPrevMel;
};

constexpr FrameSpec kFrameSpecs[] = {
    {"door",   "model002_door.onnx",   0,  false},
    {"osc_a",  "model003_osc_a.onnx",  0,  true},
    {"osc_b",  "model004_osc_b.onnx",  9,  true},
    {"noise",  "model005_noise.onnx",  20, true},
    {"filter", "model006_filter.onnx", 23, true},
    {"env1",   "model007_env1.onnx",   29, true},
    {"env2",   "model007_env2.onnx",   36, true},
    {"lfo1",   "model008_lfo1.onnx",   43, true},
    {"lfo2",   "model008_lfo2.onnx",   47, true},
    {"lfo3",   "model008_lfo3.onnx",   51, true},
    {"lfo4",   "model008_lfo4.onnx",   55, true},
    {"reverb", "model009_reverb.onnx", 59, true},
    {"delay",  "model010_delay.onnx",  65, true},
};

juce::File metaFileFor(const juce::File& exportsDir, const juce::String& onnxFile)
{
    return exportsDir.getChildFile(onnxFile.replace(".onnx", "_meta.json"));
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
InferenceEngine::InferenceEngine()
    : juce::Thread("InferenceEngine")
{
    // ONNX Runtime env — tek instance, ORT_LOGGING_LEVEL_WARNING
    ortEnv_        = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SynthV3");
    sessionOptions_= std::make_unique<Ort::SessionOptions>();

    // CoreML provider — Apple Silicon GPU / Neural Engine
    // Desteklenmiyorsa otomatik CPU'ya duser
#if JUCE_MAC
    try {
        (void)OrtSessionOptionsAppendExecutionProvider_CoreML(*sessionOptions_, 0);
    } catch (...) {
        // CoreML yoksa sessizce devam et
    }
#endif

    sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Mel config — feature_extraction.py ile birebir
    melCfg_.sampleRate = 44100;
    melCfg_.nMels      = 128;
    melCfg_.nFft       = 2048;
    melCfg_.hopLength  = 512;
    melCfg_.fMin       = 0.0f;
    melCfg_.fMax       = 22050.0f;
    melCfg_.normalize  = true;

    melSpec_ = std::make_unique<MelSpectrogram>(melCfg_);

    formatManager_.registerBasicFormats();

    for (const auto& desc : PARAM_DESCRIPTORS)
        paramNames_.emplace_back(desc.name);
}

// ─────────────────────────────────────────────────────────────────────────────
InferenceEngine::~InferenceEngine()
{
    signalThreadShouldExit();
    stopThread(3000);
}

// ─────────────────────────────────────────────────────────────────────────────
bool InferenceEngine::loadModel(const juce::File& exportsDir)
{
    if (! exportsDir.isDirectory()) {
        juce::Logger::writeToLog("InferenceEngine: exports directory not found: "
                                 + exportsDir.getFullPathName());
        return false;
    }

    frames_.clear();

    for (const auto& spec : kFrameSpecs) {
        FrameModel fm;
        fm.name          = spec.name;
        fm.prevParamsDim = spec.prevParamsDim;
        fm.hasPrevMel    = spec.hasPrevMel;

        const auto onnxFile = exportsDir.getChildFile(spec.file);
        const auto metaFile = metaFileFor(exportsDir, spec.file);

        // Meta JSON — continuous/categorical param isimleri ve prev_params_dim
        if (metaFile.existsAsFile()) {
            const auto meta = juce::JSON::parse(metaFile);

            fm.prevParamsDim = (int)meta.getProperty("prev_params_dim", (int)spec.prevParamsDim);

            if (auto* contArr = meta.getProperty("continuous_params", juce::var()).getArray()) {
                for (const auto& v : *contArr)
                    fm.continuousParams.emplace_back(v.toString().toStdString());
            }

            if (auto* catObj = meta.getProperty("categorical_params", juce::var()).getDynamicObject()) {
                for (const auto& prop : catObj->getProperties()) {
                    int numClasses = 0;
                    if (auto* classes = prop.value.getArray())
                        numClasses = classes->size();
                    fm.categoricalParams.emplace_back(prop.name.toString().toStdString(), numClasses);
                }
            }
        }

        // ONNX session
        if (onnxFile.existsAsFile()) {
            try {
#if JUCE_WINDOWS
                const auto pathStr = onnxFile.getFullPathName().toWideCharPointer();
                fm.session = std::make_unique<Ort::Session>(*ortEnv_, pathStr, *sessionOptions_);
#else
                const auto pathStr = onnxFile.getFullPathName().toStdString();
                fm.session = std::make_unique<Ort::Session>(*ortEnv_, pathStr.c_str(), *sessionOptions_);
#endif
                juce::Logger::writeToLog("InferenceEngine: model loaded: " + onnxFile.getFileName());
            }
            catch (const Ort::Exception& e) {
                juce::Logger::writeToLog(juce::String("InferenceEngine: ORT error loading ")
                                         + onnxFile.getFileName() + ": " + e.what());
            }
        } else {
            juce::Logger::writeToLog("InferenceEngine: frame model not found, skipping: "
                                     + onnxFile.getFullPathName());
        }

        if (juce::String(spec.name) == "door" && ! fm.session) {
            juce::Logger::writeToLog("InferenceEngine: door model is required but missing — abort load");
            frames_.clear();
            return false;
        }

        frames_.push_back(std::move(fm));
    }

    modelLoaded_ = frames_.empty() ? false : (frames_.front().session != nullptr);
    return modelLoaded_;
}

// ─────────────────────────────────────────────────────────────────────────────
void InferenceEngine::runInference(const juce::File& audioFile,
                                   const ResultCallback& callback)
{
    {
        juce::ScopedLock sl(queueLock_);
        pendingFile_     = audioFile;
        pendingCallback_ = callback;
        hasPending_      = true;
    }

    if (! isThreadRunning())
        startThread();
    else
        notify();  // thread uyandır
}

// ─────────────────────────────────────────────────────────────────────────────
void InferenceEngine::run()
{
    while (! threadShouldExit())
    {
        juce::File     file;
        ResultCallback callback;
        bool           hasWork = false;

        {
            juce::ScopedLock sl(queueLock_);
            if (hasPending_) {
                file         = pendingFile_;
                callback     = pendingCallback_;
                hasPending_  = false;
                hasWork      = true;
            }
        }

        if (! hasWork) {
            wait(500);
            continue;
        }

        // ── WAV yukle ─────────────────────────────────────────────────────
        std::vector<float> samples;
        double sampleRate = 44100.0;

        if (! loadAudio(file, samples, sampleRate)) {
            juce::MessageManager::callAsync([callback] {
                callback({}, "Could not load audio file.");
            });
            continue;
        }

        // ── Resample gerekiyorsa ──────────────────────────────────────────
        if ((int)sampleRate != melCfg_.sampleRate) {
            // Basit linear resample — pitch-correct degil ama prototype icin yeterli
            // Ileride juce::dsp::Oversampling veya libsamplerate eklenebilir
            const double ratio = melCfg_.sampleRate / sampleRate;
            const int    newN  = (int)(samples.size() * ratio);
            std::vector<float> resampled((size_t)newN);
            for (int i = 0; i < newN; ++i) {
                const double srcPos = i / ratio;
                const int    lo     = (int)srcPos;
                const int    hi     = std::min(lo + 1, (int)samples.size() - 1);
                const float  frac   = (float)(srcPos - lo);
                resampled[(size_t)i] = samples[(size_t)lo] * (1.0f - frac)
                                     + samples[(size_t)hi] * frac;
            }
            samples = std::move(resampled);
        }

        // ── Mel-spectrogram ───────────────────────────────────────────────
        const int nFrames = melSpec_->numFrames((int)samples.size());
        const auto melData = computeMelTensor(samples);

        // ── Sequential pipeline inference ──────────────────────────────────
        std::vector<bool> bypassFlags;
        int lfoCount = 0;
        const auto params = runSequentialPipeline(melData, melCfg_.nMels, nFrames,
                                                    bypassFlags, lfoCount);

        if (params.empty()) {
            juce::MessageManager::callAsync([callback] {
                callback({}, "ONNX inference failed.");
            });
            continue;
        }

        // ── Callback — GUI thread ─────────────────────────────────────────
        juce::MessageManager::callAsync([callback, params] {
            callback(params, {});
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
bool InferenceEngine::loadAudio(const juce::File& file,
                                std::vector<float>& samples,
                                double& sampleRate)
{
    std::unique_ptr<juce::AudioFormatReader> reader(
        formatManager_.createReaderFor(file));

    if (! reader) return false;

    sampleRate = reader->sampleRate;
    const int numSamples = (int)reader->lengthInSamples;

    juce::AudioBuffer<float> buffer(1, numSamples);
    reader->read(&buffer, 0, numSamples, 0, true, false);

    // Stereo ise mono'ya indir
    if (reader->numChannels > 1) {
        juce::AudioBuffer<float> stereo(2, numSamples);
        reader->read(&stereo, 0, numSamples, 0, true, true);
        buffer.clear();
        buffer.addFrom(0, 0, stereo, 0, 0, numSamples, 0.5f);
        buffer.addFrom(0, 0, stereo, 1, 0, numSamples, 0.5f);
    }

    samples.resize((size_t)numSamples);
    std::copy(buffer.getReadPointer(0),
              buffer.getReadPointer(0) + numSamples,
              samples.begin());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> InferenceEngine::computeMelTensor(const std::vector<float>& samples)
{
    // MelSpectrogram::compute() ciktisi zaten flat (1 * nMels * nFrames), row-major
    return melSpec_->compute(samples.data(), (int)samples.size());
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> InferenceEngine::runSequentialPipeline(const std::vector<float>& melFlat,
                                                            int /*nMels*/, int /*nFrames*/,
                                                            std::vector<bool>& bypassFlagsOut,
                                                            int& lfoCountOut)
{
    bypassFlagsOut.assign(5, false);
    lfoCountOut = 0;

    if (! modelLoaded_ || frames_.empty() || ! frames_.front().session)
        return {};

    const auto& nMels   = melCfg_.nMels;
    const int   nFrames = (int)melFlat.size() / nMels;

    // mel_prev — cheat sheet inference'ta devre disi, sifir tensor
    const std::vector<float> melPrev(melFlat.size(), 0.0f);

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── Door classifier ──────────────────────────────────────────────────
    try {
        std::array<int64_t, 3> doorShape = {1, (int64_t)nMels, (int64_t)nFrames};
        auto doorInput = Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(melFlat.data()), melFlat.size(),
            doorShape.data(), doorShape.size());

        const char* doorInputNames[]  = {"mel_full"};
        const char* doorOutputNames[] = {"bypass_flags", "lfo_count_logits"};

        auto doorOutputs = frames_.front().session->Run(
            Ort::RunOptions{nullptr},
            doorInputNames, &doorInput, 1,
            doorOutputNames, 2);

        const float* bypassLogits = doorOutputs[0].GetTensorData<float>();
        for (int i = 0; i < 5; ++i)
            bypassFlagsOut[(size_t)i] = (1.0f / (1.0f + std::exp(-bypassLogits[i]))) > 0.5f;

        const float* lfoLogits  = doorOutputs[1].GetTensorData<float>();
        const auto   lfoShape   = doorOutputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const int    numClasses = (int)lfoShape[1];

        int best = 0;
        for (int c = 1; c < numClasses; ++c)
            if (lfoLogits[c] > lfoLogits[best]) best = c;
        lfoCountOut = best;
    }
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog(juce::String("InferenceEngine ORT error (door): ") + e.what());
        return {};
    }

    // ── Frame predictors — door, osc_a, osc_b, ..., delay (sequential) ─────
    std::map<std::string, float> paramMap;
    std::vector<float> prevParams;

    for (size_t i = 1; i < frames_.size(); ++i) {
        const auto& fm = frames_[i];

        bool active = true;
        if      (fm.name == "osc_b")  active = ! bypassFlagsOut[0];
        else if (fm.name == "noise")  active = ! bypassFlagsOut[1];
        else if (fm.name == "filter") active = ! bypassFlagsOut[2];
        else if (fm.name == "reverb") active = ! bypassFlagsOut[3];
        else if (fm.name == "delay")  active = ! bypassFlagsOut[4];
        else if (fm.name == "lfo1")   active = lfoCountOut >= 1;
        else if (fm.name == "lfo2")   active = lfoCountOut >= 2;
        else if (fm.name == "lfo3")   active = lfoCountOut >= 3;
        else if (fm.name == "lfo4")   active = lfoCountOut >= 4;

        std::map<std::string, float> frameResult;
        if (active && fm.session) {
            frameResult = runFrameModel(fm, melFlat, melPrev, prevParams);
            for (const auto& kv : frameResult)
                paramMap[kv.first] = kv.second;
        }

        // prevParams must grow by exactly the total param count for this frame
        // (continuous + categorical, in the order they appear in frame config params)
        std::vector<std::string> allFrameParams = fm.continuousParams;
        for (const auto& cat : fm.categoricalParams)
            allFrameParams.push_back(cat.first);
        for (const auto& pname : allFrameParams) {
            const auto it = frameResult.find(pname);
            prevParams.push_back(it != frameResult.end() ? it->second : 0.0f);
        }
    }

    // ── Final assembly — PARAM_DESCRIPTORS sirasinda 75 deger ───────────────
    std::vector<float> result;
    result.reserve(PARAM_DESCRIPTORS.size());
    for (const auto& desc : PARAM_DESCRIPTORS) {
        const std::string name(desc.name);
        const auto it = paramMap.find(name);
        result.push_back(it != paramMap.end() ? it->second : desc.defaultVal);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
std::map<std::string, float> InferenceEngine::runFrameModel(const FrameModel& fm,
                                                   const std::vector<float>& melFull4d,
                                                   const std::vector<float>& melPrev4d,
                                                   const std::vector<float>& prevParams)
{
    if (! fm.session) return {};

    try {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const int64_t nMels   = melCfg_.nMels;
        const int64_t nFrames = (int64_t)melFull4d.size() / nMels;
        std::array<int64_t, 4> melShape = {1, 1, nMels, nFrames};
        std::array<int64_t, 2> prevShape = {1, (int64_t)fm.prevParamsDim};

        std::vector<Ort::Value>  inputs;
        std::vector<const char*> inputNames;

        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(melFull4d.data()), melFull4d.size(),
            melShape.data(), melShape.size()));
        inputNames.push_back("mel_full");

        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(melPrev4d.data()), melPrev4d.size(),
            melShape.data(), melShape.size()));
        inputNames.push_back("mel_prev");

        if (fm.prevParamsDim > 0) {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memInfo, const_cast<float*>(prevParams.data()), prevParams.size(),
                prevShape.data(), prevShape.size()));
            inputNames.push_back("prev_params");
        }

        // Cikti isimleri: continuous_out + cat_<param> (kategorik sirasinda)
        std::vector<std::string> catOutputNameStrs;
        for (const auto& cat : fm.categoricalParams)
            catOutputNameStrs.push_back("cat_" + cat.first);

        std::vector<const char*> outputNames;
        outputNames.push_back("continuous_out");
        for (const auto& s : catOutputNameStrs)
            outputNames.push_back(s.c_str());

        auto outputs = fm.session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),  inputs.data(),  inputs.size(),
            outputNames.data(), outputNames.size());

        const float* contData  = outputs[0].GetTensorData<float>();
        const auto   contShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const int    numCont   = (int)contShape[1];

        std::map<std::string, float> result;

        for (size_t p = 0; p < fm.continuousParams.size() && (int)p < numCont; ++p)
            result[fm.continuousParams[p]] = contData[p];

        // Kategorik ciktilar — argmax, [0,1]'e normalize
        for (size_t i = 0; i < fm.categoricalParams.size(); ++i) {
            const auto& paramName  = fm.categoricalParams[i].first;
            const int   numClasses = fm.categoricalParams[i].second;
            if (numClasses <= 0) continue;

            const float* logits = outputs[1 + i].GetTensorData<float>();
            int best = 0;
            for (int c = 1; c < numClasses; ++c)
                if (logits[c] > logits[best]) best = c;

            result[paramName] = numClasses > 1 ? (float)best / (float)(numClasses - 1) : 0.0f;
        }

        return result;
    }
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog(juce::String("InferenceEngine ORT error (")
                                 + fm.name + "): " + e.what());
        return {};
    }
}

} // namespace SynthV3
