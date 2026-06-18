#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"
#include "FrameSection.h"
#include "SamplingProfiles/AllProfiles.h"

class SynthV3Editor : public juce::AudioProcessorEditor,
                      public juce::FileDragAndDropTarget,
                      public juce::AudioProcessorParameter::Listener
{
public:
    explicit SynthV3Editor(SynthV3Processor&);
    ~SynthV3Editor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;

    // AudioProcessorParameter::Listener — OSC B bypass → ENV2 dimming
    void parameterValueChanged(int, float newValue) override;
    void parameterGestureChanged(int, bool) override {}

private:
    SynthV3Processor& processor_;

    // ── Frame sections ────────────────────────────────────────────────────────
    std::unique_ptr<FrameSection> master_;
    std::unique_ptr<FrameSection> oscA_, oscB_;
    std::unique_ptr<FrameSection> noise_, filter_;
    std::unique_ptr<FrameSection> env1_, env2_;
    std::unique_ptr<FrameSection> lfo1_, lfo2_, lfo3_, lfo4_;
    std::unique_ptr<FrameSection> reverb_, delay_;

    // ── Audio input panel buttons ─────────────────────────────────────────────
    juce::TextButton randomiseButton_   { "Randomise" };
    juce::TextButton predictButton_     { "Predict" };
    juce::TextButton savePresetButton_  { "Save Preset" };
    juce::TextButton loadPresetButton_  { "Load Preset" };
    juce::ComboBox   profileCombo_;

    std::vector<SamplingProfiles::SamplingProfile> profiles_;
    int activeProfileIndex_ = 0;

    // ── OSC B bypass → ENV2 dimming ───────────────────────────────────────────
    juce::AudioParameterBool* bypassOscB_ = nullptr;

    void randomisePatch();
    void applyEnv2Dim(bool oscBActive);
    void savePreset();
    void loadPreset();

    // ── Drop zone ─────────────────────────────────────────────────────────────
    struct DropZone : public juce::Component {
        bool isDragOver = false;
        juce::String statusText = "Drop audio here";
        void paint(juce::Graphics&) override;
    };
    DropZone dropZone_;
    juce::Label statusLabel_;

    std::unique_ptr<juce::FileChooser> fileChooser_;

    static std::vector<const ParamDescriptor*> paramsForFrame(const char* frame);
    void processFile(const juce::File& file);
    void setStatus(const juce::String& msg, bool isError = false);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthV3Editor)
};
