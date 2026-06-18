#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

class SynthTemplateEditor : public juce::AudioProcessorEditor,
                             public juce::FileDragAndDropTarget
{
public:
    explicit SynthTemplateEditor(SynthTemplateProcessor&);
    ~SynthTemplateEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

    // FileDragAndDropTarget
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;

private:
    SynthTemplateProcessor& processor_;

    // ── Drop Zone ─────────────────────────────────────────────────────────
    struct DropZone : public juce::Component {
        bool isDragOver = false;
        juce::String statusText = "Drop audio file or click to browse";
        std::function<void()> onClicked;

        void paint(juce::Graphics&) override;
        void mouseUp(const juce::MouseEvent&) override {
            if (onClicked) onClicked();
        }
    };
    DropZone dropZone_;

    // ── Status label ──────────────────────────────────────────────────────
    juce::Label statusLabel_;

    // ── Sliders — auto-built from amplitude + PARAM_DESCRIPTORS ──────────
    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    std::vector<std::unique_ptr<juce::Slider>>    sliders_;
    std::vector<std::unique_ptr<Attachment>>      attachments_;
    juce::StringArray                             paramLabels_;

    // ── File chooser ──────────────────────────────────────────────────────
    std::unique_ptr<juce::FileChooser> fileChooser_;

    void processFile(const juce::File& file);
    void setStatus(const juce::String& msg, bool isError = false);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthTemplateEditor)
};
