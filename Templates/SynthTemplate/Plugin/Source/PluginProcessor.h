#pragma once
#include <array>
#include <juce_audio_processors/juce_audio_processors.h>
#include "../../AudioEngine/SynthEngineTemplate.h"
#include "InferenceEngine.h"

class SynthTemplateProcessor : public juce::AudioProcessor {
public:
    SynthTemplateProcessor();
    ~SynthTemplateProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "SynthTemplate"; }
    bool   acceptsMidi()  const override { return true; }
    bool   producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }

    int  getNumPrograms()    override { return 1; }
    int  getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock&) override {}
    void setStateInformation(const void*, int) override {}

    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts_; }
    SynthTemplate::InferenceEngine&     getInferenceEngine() { return inferenceEngine_; }

    void applyInferenceResult(const std::vector<float>& params);

private:
    static constexpr int kNumVoices = 8;

    struct Voice {
        SynthTemplate::SynthEngine engine;
        int  midiNote        = 60;
        bool noteOn          = false;
        bool active          = false;
        int  ageCounter      = 0;
        bool hasPendingNote  = false;
        int  pendingMidiNote = 60;
    };

    std::array<Voice, kNumVoices> voices_;
    int blockCounter_ = 0;

    SynthTemplate::InferenceEngine    inferenceEngine_;
    juce::AudioProcessorValueTreeState apvts_;

    int  findFreeVoice() const;
    SynthTemplate::SynthParameters paramsFromAPVTS() const;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    juce::File findModelFile() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthTemplateProcessor)
};
