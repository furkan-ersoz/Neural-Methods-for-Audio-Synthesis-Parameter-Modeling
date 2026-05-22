#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include "DatasetWriter.h"

class MainComponent : public juce::Component,
                      public juce::Timer
{
public:
    MainComponent();
    ~MainComponent() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    juce::Label    synthLabel_  { {}, "Synth:" };
    juce::ComboBox synthCombo_;

    juce::Label      outputLabel_     { {}, "Output:" };
    juce::TextEditor outputPathBox_;
    juce::TextButton outputBrowseBtn_ { "Browse..." };

    juce::Label      nLabel_      { {}, "Samples:" };
    juce::TextEditor nBox_;
    juce::Label      seedLabel_   { {}, "Seed:" };
    juce::TextEditor seedBox_;
    juce::Label      maxDurLabel_ { {}, "Max dur. (s):" };
    juce::TextEditor maxDurBox_;

    static constexpr int kNumParams = 4;
    static constexpr const char* kParamNames[kNumParams] = {
        "attack", "decay", "sustain", "release"
    };

    struct ParamRow {
        juce::Label      nameLabel;
        juce::ComboBox   strategyCombo;
        juce::Label      minLabel   { {}, "Min:" };
        juce::TextEditor minBox;
        juce::Label      maxLabel   { {}, "Max:" };
        juce::TextEditor maxBox;
        juce::Label      fixedLabel { {}, "Fixed:" };
        juce::TextEditor fixedBox;
    };
    std::array<ParamRow, kNumParams> paramRows_;

    juce::Label      diskEstLabel_;
    juce::TextButton startBtn_ { "Start" };
    juce::TextButton stopBtn_  { "Stop"  };

    double            progressVal_ = 0.0;
    juce::ProgressBar progressBar_ { progressVal_ };
    juce::Label       progressLabel_;

    juce::TextEditor logBox_;

    DataGen::DatasetWriter              writer_;
    std::unique_ptr<juce::FileChooser>  fileChooser_;

    juce::CriticalSection logMutex_;
    juce::StringArray     pendingLogs_;
    std::atomic<int>      progressCurrent_ { 0 };
    std::atomic<int>      progressTotal_   { 0 };
    std::atomic<bool>     jobFinished_     { false };
    std::atomic<bool>     jobSuccess_      { false };

    void startGeneration();
    void stopGeneration();
    void updateDiskEstimate();
    void appendLog(const juce::String& msg);

    DataGen::WriterSettings buildSettings() const;
    DataGen::ParamConfig    buildParamConfig(int idx) const;
    void populateSynthCombo();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};