#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "../../AudioEngine/SynthParametersV3.h"

class FrameSection : public juce::Component,
                     public juce::AudioProcessorParameter::Listener
{
public:
    FrameSection(const juce::String& frameName,
                 const std::vector<const ParamDescriptor*>& params,
                 juce::AudioProcessorValueTreeState& apvts,
                 juce::AudioParameterBool* bypassParam = nullptr);
    ~FrameSection() override;

    void paint(juce::Graphics&) override;
    void paintOverChildren(juce::Graphics&) override;
    void resized() override;

    int getPreferredHeight(int availWidth) const;

    // AudioProcessorParameter::Listener — bypass param changes
    void parameterValueChanged(int, float newValue) override;
    void parameterGestureChanged(int, bool) override {}

    void setSubtitle(const juce::String& s) { subtitle_ = s; repaint(); }

    friend class SynthV3Editor;

private:
    // ── LED toggle button ─────────────────────────────────────────────────────
    struct LedButton : public juce::ToggleButton {
        void paintButton(juce::Graphics&, bool highlighted, bool) override;
    };

    // ── Bidirectional ComboBox ↔ float[0,1] APVTS param sync ─────────────────
    struct CatParamSync : public juce::AudioProcessorValueTreeState::Listener {
        juce::AudioProcessorValueTreeState& apvts;
        juce::ComboBox& combo;
        juce::String paramId;
        int numChoices;

        CatParamSync(juce::AudioProcessorValueTreeState&, juce::ComboBox&,
                     const juce::String& id, int numChoices);
        ~CatParamSync() override;
        void parameterChanged(const juce::String&, float newValue) override;
    };

    // ── Per-parameter widget bundle ───────────────────────────────────────────
    struct ParamWidget {
        const ParamDescriptor* desc = nullptr;
        std::unique_ptr<juce::Slider>   slider;
        std::unique_ptr<juce::ComboBox> combo;
        std::unique_ptr<juce::Label>    label;
        std::unique_ptr<juce::Label>    valueLabel;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> sliderAttach;
        std::unique_ptr<CatParamSync> catSync;
    };

    juce::String  frameName_;
    juce::String  subtitle_;
    juce::Colour  headerColour_;
    juce::AudioProcessorValueTreeState& apvts_;
    juce::AudioParameterBool* bypassParam_ = nullptr;
    bool isBypassed_ = false;

    LedButton bypassButton_;
    std::vector<ParamWidget> widgets_;

    static constexpr int kHeaderH  = 22;
    static constexpr int kCellW    = 72;
    static constexpr int kCellH    = 76;
    static constexpr int kPad      = 6;
    static constexpr int kSliderSz = 46;

    static juce::Colour      colourForFrame(const juce::String& name);
    static juce::String      labelForParam(const juce::String& id);
    static juce::String      formatParamValue(float norm, const juce::String& cls, const ParamDescriptor* desc);
    static float             parseValueToNorm(const juce::String& text, const juce::String& cls, const ParamDescriptor* desc);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(FrameSection)
};
