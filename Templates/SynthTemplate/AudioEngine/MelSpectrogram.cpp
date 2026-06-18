#include "MelSpectrogram.h"
#include <algorithm>
#include <numeric>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SynthTemplate {

// ─────────────────────────────────────────────────────────────────────────────
MelSpectrogram::MelSpectrogram(const Config& cfg) : cfg_(cfg)
{
    buildWindow();
    buildFilterbank();
}

// ─────────────────────────────────────────────────────────────────────────────
float MelSpectrogram::hzToMel(float hz)
{
    // librosa mel_to_hz / hz_to_mel — slinear (O'Shaughnessy 1987)
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float MelSpectrogram::melToHz(float mel)
{
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
void MelSpectrogram::buildWindow()
{
    window_.resize((size_t)cfg_.nFft);
    for (int i = 0; i < cfg_.nFft; ++i)
        window_[(size_t)i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i
                                                       / (float)(cfg_.nFft - 1)));
}

// ─────────────────────────────────────────────────────────────────────────────
void MelSpectrogram::buildFilterbank()
{
    const int nBins = cfg_.nFft / 2 + 1;

    const float melMin = hzToMel(cfg_.fMin);
    const float melMax = hzToMel(cfg_.fMax);

    // cfg_.nMels + 2 nokta: baslangic, bitis, ve aradaki merkez noktalari
    std::vector<float> melPoints((size_t)(cfg_.nMels + 2));
    for (int i = 0; i < cfg_.nMels + 2; ++i)
        melPoints[(size_t)i] = melMin + (melMax - melMin)
                                * (float)i / (float)(cfg_.nMels + 1);

    // Mel → Hz → FFT bin
    std::vector<float> hzPoints((size_t)(cfg_.nMels + 2));
    for (int i = 0; i < cfg_.nMels + 2; ++i)
        hzPoints[(size_t)i] = melToHz(melPoints[(size_t)i]);

    std::vector<int> binPoints((size_t)(cfg_.nMels + 2));
    for (int i = 0; i < cfg_.nMels + 2; ++i)
        binPoints[(size_t)i] = (int)std::floor(
            (float)(cfg_.nFft + 1) * hzPoints[(size_t)i] / (float)cfg_.sampleRate);

    // Filterbank olustur: (nMels, nBins)
    filterbank_.assign((size_t)cfg_.nMels, std::vector<float>((size_t)nBins, 0.0f));

    for (int m = 0; m < cfg_.nMels; ++m) {
        const int   left   = binPoints[(size_t)m];
        const int   center = binPoints[(size_t)(m + 1)];
        const int   right  = binPoints[(size_t)(m + 2)];

        for (int k = left; k < center; ++k)
            if (k >= 0 && k < nBins)
                filterbank_[(size_t)m][(size_t)k] =
                    (float)(k - left) / (float)(center - left);

        for (int k = center; k < right; ++k)
            if (k >= 0 && k < nBins)
                filterbank_[(size_t)m][(size_t)k] =
                    (float)(right - k) / (float)(right - center);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> MelSpectrogram::fftPowerFrame(const float* frame,
                                                  int fftSize) const
{
    // Windowed frame
    std::vector<float> windowed((size_t)fftSize, 0.0f);
    for (int i = 0; i < fftSize; ++i)
        windowed[(size_t)i] = frame[i] * window_[(size_t)i];

    // JUCE FFT — 2^order >= fftSize
    const int order = (int)std::ceil(std::log2((double)fftSize));
    juce::dsp::FFT fft(order);

    // FFT interleaved complex buffer: re, im, re, im, ...
    std::vector<float> buf((size_t)(fft.getSize() * 2), 0.0f);
    for (int i = 0; i < fftSize; ++i)
        buf[(size_t)(i * 2)] = windowed[(size_t)i];

    fft.performRealOnlyForwardTransform(buf.data(), true);

    // Magnitude squared (power spectrum): nFft/2+1 bins
    const int nBins = fftSize / 2 + 1;
    std::vector<float> power((size_t)nBins);
    for (int k = 0; k < nBins; ++k) {
        const float re = buf[(size_t)(k * 2)];
        const float im = buf[(size_t)(k * 2 + 1)];
        power[(size_t)k] = re * re + im * im;
    }
    return power;
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> MelSpectrogram::powerToDb(const std::vector<float>& power)
{
    // librosa.power_to_db(S, ref=np.max)
    // out = 10 * log10(S / ref), clipped to top_db=80
    const float refVal = *std::max_element(power.begin(), power.end());
    const float ref    = std::max(refVal, 1e-10f);

    std::vector<float> db(power.size());
    for (size_t i = 0; i < power.size(); ++i)
        db[i] = 10.0f * std::log10(std::max(power[i], 1e-10f) / ref);

    // Clip: max - 80 dB
    const float minDb = *std::max_element(db.begin(), db.end()) - 80.0f;
    for (auto& v : db)
        v = std::max(v, minDb);

    return db;
}

// ─────────────────────────────────────────────────────────────────────────────
void MelSpectrogram::zScoreNormalize(std::vector<float>& data,
                                     int nMels, int nFrames)
{
    for (int m = 0; m < nMels; ++m) {
        float* row = data.data() + m * nFrames;

        // Mean
        float mean = 0.0f;
        for (int t = 0; t < nFrames; ++t) mean += row[t];
        mean /= (float)nFrames;

        // Std
        float var = 0.0f;
        for (int t = 0; t < nFrames; ++t) {
            float d = row[t] - mean;
            var += d * d;
        }
        const float std = std::sqrt(var / (float)nFrames);
        const float safeStd = std < 1e-8f ? 1.0f : std;

        for (int t = 0; t < nFrames; ++t)
            row[t] = (row[t] - mean) / safeStd;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
std::vector<float> MelSpectrogram::compute(const float* samples,
                                            int numSamples) const
{
    const int nBins   = cfg_.nFft / 2 + 1;
    const int nFrames = 1 + (numSamples / cfg_.hopLength);

    // Her frame icin mel energileri: (nMels, nFrames) row-major
    std::vector<float> melSpec((size_t)(cfg_.nMels * nFrames), 0.0f);

    // Frame buffer — zero-padded
    std::vector<float> frameBuf((size_t)cfg_.nFft, 0.0f);

    for (int t = 0; t < nFrames; ++t) {
        // Frame basi
        const int start = t * cfg_.hopLength - cfg_.nFft / 2;  // center-aligned

        // Zero-pad doldur
        std::fill(frameBuf.begin(), frameBuf.end(), 0.0f);
        for (int i = 0; i < cfg_.nFft; ++i) {
            const int srcIdx = start + i;
            if (srcIdx >= 0 && srcIdx < numSamples)
                frameBuf[(size_t)i] = samples[srcIdx];
        }

        // FFT power spectrum
        const auto power = fftPowerFrame(frameBuf.data(), cfg_.nFft);

        // Mel filterbank uygula
        for (int m = 0; m < cfg_.nMels; ++m) {
            float energy = 0.0f;
            for (int k = 0; k < nBins; ++k)
                energy += filterbank_[(size_t)m][(size_t)k] * power[(size_t)k];
            melSpec[(size_t)(m * nFrames + t)] = energy;
        }
    }

    // power_to_db — frame bazinda degil, tum matris uzerinden
    const float refVal = *std::max_element(melSpec.begin(), melSpec.end());
    const float ref    = std::max(refVal, 1e-10f);
    for (auto& v : melSpec)
        v = 10.0f * std::log10(std::max(v, 1e-10f) / ref);

    // Clip -80 dB
    const float maxDb = *std::max_element(melSpec.begin(), melSpec.end());
    const float minDb = maxDb - 80.0f;
    for (auto& v : melSpec)
        v = std::max(v, minDb);

    // Z-score normalize
    if (cfg_.normalize)
        zScoreNormalize(melSpec, cfg_.nMels, nFrames);

    return melSpec;  // (nMels * nFrames) row-major
}

} // namespace SynthTemplate
