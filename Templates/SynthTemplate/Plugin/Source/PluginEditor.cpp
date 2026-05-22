#include "PluginEditor.h"

using namespace juce;

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::DropZone::paint(Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(2.0f);
    const auto bg     = isDragOver ? Colour(0xff2a5caa) : Colour(0xff1e1e2e);
    const auto border = isDragOver ? Colour(0xff5b9cf6) : Colour(0xff44475a);

    g.setColour(bg);
    g.fillRoundedRectangle(bounds, 8.0f);
    g.setColour(border);
    g.drawRoundedRectangle(bounds, 8.0f, isDragOver ? 2.0f : 1.0f);

    // Icon — kopya alarak non-const sorunu coz
    auto top = bounds;
    auto iconBounds = top.removeFromTop(bounds.getHeight() * 0.55f);
    g.setColour(border.brighter(0.4f));
    g.setFont(FontOptions(28.0f));
    g.drawText(isDragOver ? "+" : "~", iconBounds, Justification::centredBottom);

    g.setColour(Colours::lightgrey);
    g.setFont(FontOptions(12.0f));
    g.drawText(statusText, top, Justification::centredTop);
}

// ─────────────────────────────────────────────────────────────────────────────
SynthTemplateEditor::SynthTemplateEditor(SynthTemplateProcessor& p)
    : AudioProcessorEditor(&p), processor_(p),
      ampAtt_(p.getAPVTS(), "amplitude", ampSlider_),
      atkAtt_(p.getAPVTS(), "attack",    atkSlider_),
      decAtt_(p.getAPVTS(), "decay",     decSlider_),
      susAtt_(p.getAPVTS(), "sustain",   susSlider_),
      relAtt_(p.getAPVTS(), "release",   relSlider_)
{
    for (auto* s : {&ampSlider_, &atkSlider_, &decSlider_, &susSlider_, &relSlider_}) {
        s->setSliderStyle(Slider::RotaryVerticalDrag);
        s->setTextBoxStyle(Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s);
    }

    dropZone_.setInterceptsMouseClicks(true, false);
    dropZone_.onClicked = [this] {
        fileChooser_ = std::make_unique<FileChooser>(
            "Select audio file",
            File::getSpecialLocation(File::userHomeDirectory),
            "*.wav;*.aif;*.aiff;*.mp3;*.flac"
        );
        fileChooser_->launchAsync(
            FileBrowserComponent::openMode | FileBrowserComponent::canSelectFiles,
            [this](const FileChooser& fc) {
                if (fc.getResult() != File{})
                    processFile(fc.getResult());
            });
    };
    addAndMakeVisible(dropZone_);

    statusLabel_.setJustificationType(Justification::centred);
    statusLabel_.setFont(FontOptions(11.0f));
    statusLabel_.setColour(Label::textColourId, Colours::lightgrey);

    if (!processor_.getInferenceEngine().isModelLoaded())
        setStatus("Model not loaded", true);
    else
        setStatus("Ready");

    addAndMakeVisible(statusLabel_);
    setSize(520, 320);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::paint(Graphics& g)
{
    g.fillAll(Colour(0xff13131f));

    g.setColour(Colours::lightgrey.withAlpha(0.7f));
    g.setFont(FontOptions(11.0f));
    const StringArray labels {"Amp", "Attack", "Decay", "Sustain", "Release"};
    const int w = getWidth() / 5;
    for (int i = 0; i < 5; ++i)
        g.drawText(labels[i], i * w, 8, w, 16, Justification::centred);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::resized()
{
    const int pad     = 8;
    const int sliderH = 140;
    const int w       = getWidth() / 5;
    const int dropH   = getHeight() - sliderH - 50;

    Slider* sliders[] = {&ampSlider_, &atkSlider_, &decSlider_, &susSlider_, &relSlider_};
    for (int i = 0; i < 5; ++i)
        sliders[i]->setBounds(i * w, 24, w, sliderH);

    const int dzY = 24 + sliderH + pad;
    dropZone_.setBounds(pad, dzY, getWidth() - 2 * pad, dropH);
    statusLabel_.setBounds(pad, getHeight() - 24, getWidth() - 2 * pad, 20);
}

// ─────────────────────────────────────────────────────────────────────────────
bool SynthTemplateEditor::isInterestedInFileDrag(const StringArray& files)
{
    for (auto& f : files) {
        auto ext = File(f).getFileExtension().toLowerCase();
        if (ext == ".wav" || ext == ".aif" || ext == ".aiff"
            || ext == ".mp3" || ext == ".flac")
            return true;
    }
    return false;
}

void SynthTemplateEditor::filesDropped(const StringArray& files, int, int)
{
    dropZone_.isDragOver = false;
    dropZone_.repaint();
    if (!files.isEmpty())
        processFile(File(files[0]));
}

void SynthTemplateEditor::fileDragEnter(const StringArray&, int, int)
{
    dropZone_.isDragOver = true;
    dropZone_.repaint();
}

void SynthTemplateEditor::fileDragExit(const StringArray&)
{
    dropZone_.isDragOver = false;
    dropZone_.repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::processFile(const File& file)
{
    if (!processor_.getInferenceEngine().isModelLoaded()) {
        setStatus("Model not loaded.", true);
        return;
    }

    setStatus("Processing: " + file.getFileName());
    dropZone_.statusText = file.getFileName();
    dropZone_.repaint();

    processor_.getInferenceEngine().runInference(
        file,
        [this](const std::vector<float>& params, const String& error) {
            if (!error.isEmpty()) {
                setStatus("Error: " + error, true);
                return;
            }
            processor_.applyInferenceResult(params);

            String msg = "Done:";
            const StringArray names {"atk", "dec", "sus", "rel"};
            for (int i = 0; i < (int)params.size() && i < names.size(); ++i)
                msg += " " + names[i] + "=" + String(params[(size_t)i], 3);
            setStatus(msg);
        }
    );
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::setStatus(const String& msg, bool isError)
{
    statusLabel_.setColour(Label::textColourId,
                           isError ? Colour(0xffff6b6b) : Colours::lightgrey);
    statusLabel_.setText(msg, dontSendNotification);
}