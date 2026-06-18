#pragma once
#include <array>
#include <juce_audio_processors/juce_audio_processors.h>
#include "../../AudioEngine/SynthEngineV3.h"
#include "InferenceEngine.h"

class SynthV3Processor : public juce::AudioProcessor {
public:
    SynthV3Processor();
    ~SynthV3Processor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "SynthV3"; }
    bool   acceptsMidi()  const override { return true; }
    bool   producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }

    int  getNumPrograms()    override { return 1; }
    int  getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts_; }
    SynthV3::InferenceEngine&           getInferenceEngine() { return inferenceEngine_; }

    // Returns the bypass AudioParameterBool for a given frame name, or nullptr.
    juce::AudioParameterBool* getBypassParam(const juce::String& frame) const;

    void applyInferenceResult(const std::vector<float>& params);

private:
    static constexpr int kNumVoices = 8;

    struct Voice {
        SynthV3::SynthEngine engine;
        int  midiNote    = 60;
        bool noteOn      = false;
        bool active      = false;
        int  ageCounter  = 0;
        bool isGhost     = false;

        static constexpr int kPendingQueueSize = 4;
        int pendingQueue[kPendingQueueSize] = {};
        int pendingHead  = 0;
        int pendingTail  = 0;
        int pendingCount = 0;

        bool pendingEmpty() const { return pendingCount == 0; }
        bool pendingFull()  const { return pendingCount == kPendingQueueSize; }
        void pendingPush(int note) {
            if (!pendingFull()) {
                pendingQueue[pendingTail % kPendingQueueSize] = note;
                ++pendingTail;
                ++pendingCount;
            }
        }
        int pendingPop() {
            int n = pendingQueue[pendingHead % kPendingQueueSize];
            ++pendingHead;
            --pendingCount;
            return n;
        }
    };

    std::array<Voice, kNumVoices> voices_;
    int blockCounter_ = 0;

    SynthV3::InferenceEngine    inferenceEngine_;
    juce::AudioProcessorValueTreeState apvts_;

    // Bypass params — stored as "active" (1=active/not-bypassed, 0=bypassed).
    // Separate from APVTS so they don't pollute the model parameter namespace.
    juce::AudioParameterBool* bypassOscB_   = nullptr;
    juce::AudioParameterBool* bypassNoise_  = nullptr;
    juce::AudioParameterBool* bypassFilter_ = nullptr;
    juce::AudioParameterBool* bypassLfo1_   = nullptr;
    juce::AudioParameterBool* bypassLfo2_   = nullptr;
    juce::AudioParameterBool* bypassLfo3_   = nullptr;
    juce::AudioParameterBool* bypassLfo4_   = nullptr;
    juce::AudioParameterBool* bypassReverb_ = nullptr;
    juce::AudioParameterBool* bypassDelay_  = nullptr;

    int  findFreeVoice() const;
    SynthV3::SynthParameters paramsFromAPVTS() const;
    SynthV3::BypassFlags     bypassFromParams() const;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    juce::File findModelFile() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthV3Processor)
};
