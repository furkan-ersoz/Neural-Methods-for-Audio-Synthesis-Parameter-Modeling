#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DatasetWriter.h"
#include "ISynth.h"

// ── Scrollable panel that holds all parameter rows ───────────────────────────
class ParamPanel : public juce::Component
{
public:
    // Set by MainComponent after each layout pass so we can repaint header tints
    std::vector<juce::Rectangle<int>> headerRects;

    void paint(juce::Graphics& g) override
    {
        g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
        g.setColour(juce::Colours::white.withAlpha(0.07f));
        for (auto& r : headerRects)
            g.fillRect(r);
    }
};

// ── Scrollable panel that holds frame output toggles ─────────────────────────
class FrameOutputsPanel : public juce::Component
{
public:
    void paint(juce::Graphics& g) override
    {
        g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
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
    juce::Label    synthLabel_   { {}, "Synth:" };
    juce::ComboBox synthCombo_;
    juce::Label    profileLabel_ { {}, "Sampling Profile:" };
    juce::ComboBox profileCombo_;

    juce::Label      outputLabel_;
    juce::TextEditor outputPathBox_;
    juce::TextButton outputBrowseBtn_ { "Browse..." };

    juce::Label      nLabel_      { {}, "Samples:" };
    juce::TextEditor nBox_;
    juce::Label      seedLabel_   { {}, "Seed:" };
    juce::TextEditor seedBox_;
    juce::Label      maxDurLabel_ { {}, "Max dur. (s):" };
    juce::TextEditor maxDurBox_;
    juce::Label      workersLabel_ { {}, "Workers:" };
    juce::TextEditor workersBox_;

    struct ParamRow {
        std::string      frameName;
        int              numClasses = 0;   // categorical/integer class count (0 = continuous)
        juce::Label      nameLabel;
        juce::ComboBox   strategyCombo;
        juce::Label      minLabel   { {}, "Min:" };
        juce::TextEditor minBox;
        juce::Label      maxLabel   { {}, "Max:" };
        juce::TextEditor maxBox;
        juce::Label      fixedLabel { {}, "Fixed:" };
        juce::TextEditor fixedBox;
    };

    juce::OwnedArray<ParamRow> paramRows_;

    // ── Scrollable param area ─────────────────────────────────────────────────
    ParamPanel        paramPanel_;
    juce::Viewport    paramViewport_;

    struct SectionHeaderEntry {
        std::string      frameName;
        juce::Label      label;
        bool             hasBypass = false;
        int              layoutY   = 0;
    };

    juce::OwnedArray<SectionHeaderEntry>                 sectionHeaders_;
    juce::OwnedArray<juce::ToggleButton>                    bypassToggleOwner_;
    juce::OwnedArray<juce::TextEditor>                      bypassProbEditorOwner_;
    std::unordered_map<std::string, juce::ToggleButton*>    bypassToggles_;
    std::unordered_map<std::string, juce::TextEditor*>      bypassProbLabels_;

    struct LayoutItem {
        enum class Type { Header, Param };
        Type type;
        int  index;
    };
    std::vector<LayoutItem> layoutItems_;

    // ── Frame Outputs panel ───────────────────────────────────────────────────
    FrameOutputsPanel            frameOutPanel_;
    juce::Viewport               frameOutViewport_;
    juce::Label                  frameOutTitleLabel_  { {}, "Frame Outputs" };
    juce::Label                  fullRenderLabel_     { {}, "Full render \xe2\x80\x94 always saved" };
    juce::TextButton             selectAllBtn_        { "Select All" };
    juce::TextButton             deselectAllBtn_      { "Deselect All" };
    juce::OwnedArray<juce::ToggleButton> frameToggleOwner_;
    std::vector<std::string>     frameInternalNames_;

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
    void populateSynthCombo();
    void populateProfileCombo();
    void applyProfileToRows();
    void rebuildParamRows();
    int  paramAreaHeight() const;

    DataGen::WriterSettings  buildSettings() const;
    DataGen::ParamConfig     buildParamConfig(int idx) const;
    DataGen::FrameSelection  buildFrameSelection() const;

    juce::String defaultOutputForSynth(const juce::String& synthName) const;
    void         applyBypassAlpha(const std::string& frameName, bool bypassed);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
