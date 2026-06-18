#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "MelSpectrogram.h"

// ONNX Runtime forward declarations
namespace Ort { class Session; class Env; class SessionOptions; }

namespace SynthV3 {

/**
 * InferenceEngine
 *
 * Arka plan thread'inde calisir — audio thread'e dokunmaz.
 * 1. WAV yukle
 * 2. Mel-spectrogram hesapla
 * 3. Sequential frame-model pipeline ile inference (door -> osc_a -> ... -> delay)
 * 4. Parametreleri callback ile ilet
 */
class InferenceEngine : public juce::Thread {
public:
    // Inference tamamlaninca cagrilir — GUI thread'inde dispatch edilir
    using ResultCallback = std::function<void(const std::vector<float>& params,
                                              const juce::String& error)>;

    InferenceEngine();
    ~InferenceEngine() override;

    // Model yukle — plugin baslarken bir kez cagrilir.
    // exportsDir: Exp003/exports klasoru (her frame icin model00X_*.onnx + _meta.json icerir)
    bool loadModel(const juce::File& exportsDir);
    bool isModelLoaded() const { return modelLoaded_; }

    // Inference baslat — audio dosyasini arka planda isle
    // onceki islem bitmemisse bekler
    void runInference(const juce::File& audioFile,
                      const ResultCallback& callback);

    // juce::Thread
    void run() override;

    // Parametre isimleri (PARAM_DESCRIPTORS sirasi)
    const std::vector<std::string>& getParamNames() const { return paramNames_; }

private:
    // Tek bir sequential pipeline frame'i (door, osc_a, osc_b, ...)
    struct FrameModel {
        std::string name;                                    // "door", "osc_a", ...
        std::unique_ptr<Ort::Session> session;
        int prevParamsDim = 0;
        std::vector<std::string> continuousParams;           // param adlari, model siralamasinda
        std::vector<std::pair<std::string, int>> categoricalParams; // {param_name, num_classes}
        bool hasPrevMel = true;                               // false sadece door icin
    };

    // ONNX Runtime
    std::unique_ptr<Ort::Env>            ortEnv_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    bool                                 modelLoaded_ = false;

    // Sequential pipeline frame'leri (door, osc_a, osc_b, ..., delay)
    std::vector<FrameModel> frames_;

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

    // MelSpectrogram::compute() ciktisini flat (1*nMels*nFrames) vektore donusturur
    std::vector<float> computeMelTensor(const std::vector<float>& samples);

    // Tum pipeline'i sirayla calistirir, 75-elemanli params vektoru dondurur
    std::vector<float> runSequentialPipeline(const std::vector<float>& melFlat,
                                              int nMels, int nFrames,
                                              std::vector<bool>& bypassFlagsOut,
                                              int& lfoCountOut);

    // Tek bir frame modelini calistirir — continuous + (argmax) categorical ciktiyi
    // param adi -> deger map'i olarak dondurur
    std::map<std::string, float> runFrameModel(const FrameModel& fm,
                                     const std::vector<float>& melFull4d,
                                     const std::vector<float>& melPrev4d,
                                     const std::vector<float>& prevParams);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(InferenceEngine)
};

} // namespace SynthV3
