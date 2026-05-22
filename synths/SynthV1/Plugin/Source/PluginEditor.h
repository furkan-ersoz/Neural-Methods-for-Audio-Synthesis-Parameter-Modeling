#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

class SynthV1Editor : public juce::AudioProcessorEditor {
public:
    explicit SynthV1Editor(SynthV1Processor&);
    ~SynthV1Editor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    SynthV1Processor& processor_;

    // Her parametre için bir slider + attachment
    juce::Slider ampSlider_, atkSlider_,
                 decSlider_, susSlider_, relSlider_;

    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    Attachment ampAtt_, atkAtt_, decAtt_, susAtt_, relAtt_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthV1Editor)
};