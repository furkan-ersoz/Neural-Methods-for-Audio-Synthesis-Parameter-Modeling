#include "PluginProcessor.h"
#include "PluginEditor.h"

using namespace juce;

// APVTS parametre tanımları — ID'ler CSV header ile birebir eşleşmeli
AudioProcessorValueTreeState::ParameterLayout
SynthV1Processor::createParameterLayout() {
    std::vector<std::unique_ptr<RangedAudioParameter>> params;

    // Tüm parametreler [0,1] normalize — gerçek değer dönüşümü Engine'de
    auto add = [&](const char* id, float def) {
        params.push_back(std::make_unique<AudioParameterFloat>(
            ParameterID{id, 1}, id,
            NormalisableRange<float>(0.0f, 1.0f), def));
    };

    add("amplitude",    0.8f);
    add("attack",       0.05f);
    add("decay",        0.2f);
    add("sustain",      0.7f);
    add("release",      0.3f);

    return { params.begin(), params.end() };
}

SynthV1Processor::SynthV1Processor()
    : AudioProcessor(BusesProperties()
          .withOutput("Output", AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "STATE", createParameterLayout())
{}

void SynthV1Processor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    engine_.prepare(sampleRate, samplesPerBlock);
}

void SynthV1Processor::syncEngineFromAPVTS() {
    SynthV1::SynthParameters p;
    p.amplitude    = *apvts_.getRawParameterValue("amplitude");
    p.attackTime   = *apvts_.getRawParameterValue("attack");
    p.decayTime    = *apvts_.getRawParameterValue("decay");
    p.sustainLevel = *apvts_.getRawParameterValue("sustain");
    p.releaseTime  = *apvts_.getRawParameterValue("release");
    engine_.setParameters(p);
}

void SynthV1Processor::processBlock(AudioBuffer<float>& buffer,
                                          MidiBuffer& midiMessages) {
    ScopedNoDenormals noDenormals;
    buffer.clear();

    // MIDI event'leri işle
    for (const auto meta : midiMessages) {
        const auto msg = meta.getMessage();
        if (msg.isNoteOn())  { noteOn_ = true;  midiNote_ = msg.getNoteNumber(); }
        if (msg.isNoteOff()) { noteOn_ = false; }
    }

    syncEngineFromAPVTS();

    const float frequencyHz = SynthV1::midiToHz(midiNote_);

    // Mono üret, stereo'ya kopyala
    auto* left  = buffer.getWritePointer(0);
    auto* right = buffer.getWritePointer(1);
    const int numSamples = buffer.getNumSamples();

    engine_.process(left, numSamples, noteOn_, frequencyHz);

    // MIDI note → frekans override (APVTS'deki normalize değeri geçersiz kılar)
    // Not: Şu an APVTS'deki frequency parametresi kullanılıyor.
    // İleride midiNote_ → Hz → normalize dönüşümü eklenecek.

    FloatVectorOperations::copy(right, left, numSamples);
}

AudioProcessorEditor* SynthV1Processor::createEditor() {
    return new SynthV1Editor(*this);
}

// JUCE plugin factory
AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new SynthV1Processor();
}