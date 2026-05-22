#include "PluginEditor.h"

SynthV1Editor::SynthV1Editor(SynthV1Processor& p)
    : AudioProcessorEditor(&p), processor_(p),
      ampAtt_ (p.getAPVTS(), "amplitude", ampSlider_),
      atkAtt_ (p.getAPVTS(), "attack",    atkSlider_),
      decAtt_ (p.getAPVTS(), "decay",     decSlider_),
      susAtt_ (p.getAPVTS(), "sustain",   susSlider_),
      relAtt_ (p.getAPVTS(), "release",   relSlider_)
{
    for (auto* s : { &ampSlider_, &atkSlider_,
                     &decSlider_,  &susSlider_, &relSlider_ }) {
        s->setSliderStyle(juce::Slider::RotaryVerticalDrag);
        s->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s);
                     }
    setSize(480, 200);
}

void SynthV1Editor::paint(juce::Graphics& g) {
    g.fillAll(juce::Colours::darkgrey);
    g.setColour(juce::Colours::white);
    g.setFont(14.0f);

    const juce::StringArray labels {
        "Amp", "Attack", "Decay", "Sustain", "Release"
    };
    const int w = getWidth() / 5;
    for (int i = 0; i < 5; ++i)
        g.drawText(labels[i], i * w, 0, w, 20, juce::Justification::centred);
}

void SynthV1Editor::resized() {
    const int w = getWidth() / 5;
    const int h = getHeight() - 20;
    juce::Slider* sliders[] = {
        &ampSlider_, &atkSlider_,
        &decSlider_,  &susSlider_, &relSlider_
    };
    for (int i = 0; i < 5; ++i)
        sliders[i]->setBounds(i * w, 20, w, h);
}