#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "../../AudioEngine/SynthEngineTemplate.h"

class SynthTemplateProcessor : public juce::AudioProcessor {
public:
    SynthTemplateProcessor();
    ~SynthTemplateProcessor() override = default;

    // --- JUCE zorunlu override'lar ---
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "SynthTemplate"; }
    bool   acceptsMidi()  const override { return true; }
    bool   producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int  getNumPrograms()    override { return 1; }
    int  getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    // DataGenerator ve Editor buradan parametrelere erişir
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts_; }

private:
    SynthTemplate::SynthEngine engine_;
    juce::AudioProcessorValueTreeState apvts_;

    // MIDI note tracking
    bool noteOn_  = false;
    int  midiNote_ = 60;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void syncEngineFromAPVTS();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthTemplateProcessor)
};