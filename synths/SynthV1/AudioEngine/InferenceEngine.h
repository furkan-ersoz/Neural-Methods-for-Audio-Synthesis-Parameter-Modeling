#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <functional>
#include <vector>
#include <string>

#include "MelSpectrogram.h"

// ONNX Runtime forward declarations
namespace Ort { class Session; class Env; class SessionOptions; }

namespace SynthV1 {

/**
 * InferenceEngine
 *
 * Arka plan thread'inde calisir — audio thread'e dokunmaz.
 * 1. WAV yukle
 * 2. Mel-spectrogram hesapla
 * 3. ONNX Runtime ile inference
 * 4. Parametreleri callback ile ilet
 */
class InferenceEngine : public juce::Thread {
public:
    // Inference tamamlaninca cagrilir — GUI thread'inde dispatch edilir
    using ResultCallback = std::function<void(const std::vector<float>& params,
                                              const juce::String& error)>;

    InferenceEngine();
    ~InferenceEngine() override;

    // Model yukle — plugin baslarken bir kez cagrilir
    bool loadModel(const juce::File& onnxFile);
    bool isModelLoaded() const { return modelLoaded_; }

    // Inference baslat — audio dosyasini arka planda isle
    // onceki islem bitmemisse bekler
    void runInference(const juce::File& audioFile,
                      const ResultCallback& callback);

    // juce::Thread
    void run() override;

    // Parametre isimleri (ONNX meta'dan veya config'den)
    const std::vector<std::string>& getParamNames() const { return paramNames_; }

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env>            ortEnv_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::Session>        session_;
    bool                                 modelLoaded_ = false;

    // Mel config — feature_extraction.py ile eslesmeli
    MelSpectrogram::Config melCfg_;
    std::unique_ptr<MelSpectrogram> melSpec_;

    // Audio format manager
    juce::AudioFormatManager formatManager_;

    // Islem kuyruğu
    juce::CriticalSection  queueLock_;
    juce::File             pendingFile_;
    ResultCallback         pendingCallback_;
    bool                   hasPending_ = false;

    // Parametre isimleri
    std::vector<std::string> paramNames_;

    // Yardimcilar
    bool loadAudio(const juce::File& file,
                   std::vector<float>& samples,
                   double& sampleRate);

    std::vector<float> runOnnx(const std::vector<float>& melData,
                                int nMels, int nFrames);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InferenceEngine)
};

} // namespace SynthV1
