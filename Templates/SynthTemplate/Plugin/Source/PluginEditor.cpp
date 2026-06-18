#include "PluginEditor.h"

using namespace juce;

static constexpr int kSliderH  = 140;
static constexpr int kLabelH   = 24;
static constexpr int kDropH    = 100;
static constexpr int kStatusH  = 24;
static constexpr int kPad      = 8;

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
    : AudioProcessorEditor(&p), processor_(p)
{
    // ── Build sliders from amplitude + PARAM_DESCRIPTORS ─────────────────────
    // Amplitude first (internal param, always present)
    {
        paramLabels_.add("Amp");
        auto s = std::make_unique<Slider>();
        s->setSliderStyle(Slider::RotaryVerticalDrag);
        s->setTextBoxStyle(Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s.get());
        attachments_.push_back(std::make_unique<Attachment>(p.getAPVTS(), "amplitude", *s));
        sliders_.push_back(std::move(s));
    }

    // All predicted params — driven by PARAM_DESCRIPTORS, no hardcoding needed
    for (const auto& desc : SynthTemplate::PARAM_DESCRIPTORS) {
        paramLabels_.add(desc.name.data());
        auto s = std::make_unique<Slider>();
        s->setSliderStyle(Slider::RotaryVerticalDrag);
        s->setTextBoxStyle(Slider::TextBoxBelow, false, 60, 20);
        addAndMakeVisible(s.get());
        attachments_.push_back(std::make_unique<Attachment>(p.getAPVTS(), desc.name.data(), *s));
        sliders_.push_back(std::move(s));
    }

    // ── Drop zone ─────────────────────────────────────────────────────────────
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

    const int numSliders = static_cast<int>(sliders_.size());
    const int minWidth   = numSliders * 70;
    const int totalH     = kLabelH + kSliderH + kPad + kDropH + kPad + kStatusH + kPad;
    setSize(std::max(minWidth, 400), totalH);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::paint(Graphics& g)
{
    g.fillAll(Colour(0xff13131f));

    const int n = static_cast<int>(sliders_.size());
    if (n == 0) return;

    const int w = getWidth() / n;
    g.setColour(Colours::lightgrey.withAlpha(0.7f));
    g.setFont(FontOptions(11.0f));
    for (int i = 0; i < n; ++i)
        g.drawText(paramLabels_[i], i * w, kPad, w, kLabelH - kPad, Justification::centred);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateEditor::resized()
{
    const int n = static_cast<int>(sliders_.size());
    if (n == 0) return;

    const int w    = getWidth() / n;
    const int topY = kLabelH;

    for (int i = 0; i < n; ++i)
        sliders_[static_cast<size_t>(i)]->setBounds(i * w, topY, w, kSliderH);

    const int dzY = topY + kSliderH + kPad;
    dropZone_.setBounds(kPad, dzY, getWidth() - 2 * kPad, kDropH);
    statusLabel_.setBounds(kPad, dzY + kDropH + kPad, getWidth() - 2 * kPad, kStatusH);
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
            for (int i = 0; i < static_cast<int>(params.size())
                            && i < static_cast<int>(SynthTemplate::PARAM_DESCRIPTORS.size()); ++i)
                msg += " " + String(SynthTemplate::PARAM_DESCRIPTORS[static_cast<size_t>(i)].name.data())
                     + "=" + String(params[static_cast<size_t>(i)], 3);
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
