#include "InferenceEngine.h"
#include "SynthParametersTemplate.h"
#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>
#include <cmath>
#include <algorithm>

namespace SynthTemplate {

// ─────────────────────────────────────────────────────────────────────────────
InferenceEngine::InferenceEngine()
    : juce::Thread("InferenceEngine")
{
    // ONNX Runtime env — tek instance, ORT_LOGGING_LEVEL_WARNING
    ortEnv_        = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SynthTemplate");
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
        paramNames_.emplace_back(desc.name.data());
}

// ─────────────────────────────────────────────────────────────────────────────
InferenceEngine::~InferenceEngine()
{
    signalThreadShouldExit();
    stopThread(3000);
}

// ─────────────────────────────────────────────────────────────────────────────
bool InferenceEngine::loadModel(const juce::File& onnxFile)
{
    if (! onnxFile.existsAsFile()) {
        juce::Logger::writeToLog("InferenceEngine: model not found: "
                                 + onnxFile.getFullPathName());
        return false;
    }

    try {
#if JUCE_WINDOWS
        // Windows: wstring path gerekiyor
        const auto pathStr = onnxFile.getFullPathName().toWideCharPointer();
        session_ = std::make_unique<Ort::Session>(*ortEnv_, pathStr, *sessionOptions_);
#else
        const auto pathStr = onnxFile.getFullPathName().toStdString();
        session_ = std::make_unique<Ort::Session>(*ortEnv_, pathStr.c_str(),
                                                   *sessionOptions_);
#endif
        modelLoaded_ = true;
        juce::Logger::writeToLog("InferenceEngine: model loaded: "
                                 + onnxFile.getFileName());
        return true;
    }
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog(juce::String("InferenceEngine: ORT error: ")
                                 + e.what());
        return false;
    }
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
        const auto melData = melSpec_->compute(samples.data(), (int)samples.size());

        // ── ONNX inference ────────────────────────────────────────────────
        const auto params = runOnnx(melData, melCfg_.nMels, nFrames);

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
std::vector<float> InferenceEngine::runOnnx(const std::vector<float>& melData,
                                             int nMels, int nFrames)
{
    if (! modelLoaded_ || ! session_) return {};

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // Input: (1, nMels, nFrames)
        std::array<int64_t, 3> inputShape = {1, (int64_t)nMels, (int64_t)nFrames};

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(melData.data()),
            melData.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Input / output isimleri
        const char* inputNames[]  = {"spectrogram"};
        const char* outputNames[] = {"params"};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inputTensor, 1,
            outputNames, 1
        );

        // Output: (1, num_params)
        const float* outData = outputs[0].GetTensorData<float>();
        const auto   outShape= outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const int    numOut  = (int)outShape[1];

        return std::vector<float>(outData, outData + numOut);
    }
    catch (const Ort::Exception& e) {
        juce::Logger::writeToLog(juce::String("InferenceEngine ORT error: ")
                                 + e.what());
        return {};
    }
}

} // namespace SynthTemplate