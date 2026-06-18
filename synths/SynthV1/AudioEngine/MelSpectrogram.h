#pragma once

#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <cmath>
#include <numeric>

namespace SynthV1 {

/**
 * MelSpectrogram
 *
 * librosa.feature.melspectrogram() + power_to_db() + z-score
 * ile uyumlu C++ implementasyonu.
 *
 * Config, Python tarafindaki feature_extraction.py ile eslesmeli:
 *   sample_rate: 44100
 *   n_mels:      128
 *   n_fft:       2048
 *   hop_length:  512
 *
 * Cikti: row-major (n_mels, T) float vektor
 *   index(mel, t) = mel * numFrames + t
 */
class MelSpectrogram {
public:
    struct Config {
        int   sampleRate = 44100;
        int   nMels      = 128;
        int   nFft       = 2048;
        int   hopLength  = 512;
        float fMin       = 0.0f;
        float fMax       = 22050.0f;
        bool  normalize  = true;
    };

    explicit MelSpectrogram(const Config& cfg);

    // Ana fonksiyon — mono float audio → (nMels * T) vektoru
    std::vector<float> compute(const float* samples, int numSamples) const;

    int getNumMels()   const { return cfg_.nMels; }
    int numFrames(int numSamples) const {
        return 1 + (numSamples / cfg_.hopLength);
    }

private:
    Config cfg_;

    // Mel filterbank: (nMels, nFft/2+1)
    std::vector<std::vector<float>> filterbank_;

    // Hann penceresi
    std::vector<float> window_;

    void buildFilterbank();
    void buildWindow();

    // Hz <-> Mel donusum (librosa slinear scale)
    static float hzToMel(float hz);
    static float melToHz(float mel);

    // Tek frame FFT magnitude karesi
    std::vector<float> fftPowerFrame(const float* frame, int fftSize) const;

    // power_to_db — librosa ile ayni
    static std::vector<float> powerToDb(const std::vector<float>& power);

    // Z-score per mel bin
    static void zScoreNormalize(std::vector<float>& data, int nMels, int nFrames);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MelSpectrogram)
};

} // namespace SynthV1
