#include "FrameSection.h"
#include "ParamCodec.h"
using namespace juce;

// ── LedButton ─────────────────────────────────────────────────────────────────
void FrameSection::LedButton::paintButton(Graphics& g, bool highlighted, bool)
{
    auto b = getLocalBounds().toFloat().reduced(2.0f);
    g.setColour(getToggleState() ? Colour(0xff44dd88) : Colour(0xff2a2a3a));
    g.fillEllipse(b);
    if (highlighted) {
        g.setColour(Colours::white.withAlpha(0.15f));
        g.fillEllipse(b);
    }
    g.setColour(Colours::white.withAlpha(0.25f));
    g.drawEllipse(b, 0.7f);
}

// ── CatParamSync ──────────────────────────────────────────────────────────────
FrameSection::CatParamSync::CatParamSync(AudioProcessorValueTreeState& a,
                                          ComboBox& c,
                                          const String& id, int n)
    : apvts(a), combo(c), paramId(id), numChoices(n)
{
    apvts.addParameterListener(id, this);
    int idx = 0;
    if (auto* cp = dynamic_cast<AudioParameterChoice*>(apvts.getParameter(id)))
        idx = jlimit(0, n - 1, cp->getIndex());
    combo.setSelectedItemIndex(idx, dontSendNotification);

    combo.onChange = [this]() {
        const int   i    = combo.getSelectedItemIndex();
        const float norm = (numChoices > 1) ? (float)i / (float)(numChoices - 1) : 0.0f;
        apvts.getParameter(paramId)->setValueNotifyingHost(norm);
    };
}

FrameSection::CatParamSync::~CatParamSync()
{
    combo.onChange = nullptr;
    apvts.removeParameterListener(paramId, this);
}

void FrameSection::CatParamSync::parameterChanged(const String& pid, float)
{
    int idx = 0;
    if (auto* cp = dynamic_cast<AudioParameterChoice*>(apvts.getParameter(pid)))
        idx = jlimit(0, numChoices - 1, cp->getIndex());
    MessageManager::callAsync([this, idx]() {
        combo.setSelectedItemIndex(idx, dontSendNotification);
    });
}

// ── Value formatting helpers ───────────────────────────────────────────────────
String FrameSection::formatParamValue(float norm, const String& cls, const ParamDescriptor* desc)
{
    if (cls == "morph")   return String(norm, 1);
    if (cls == "bipolar") {
        float v = (norm - 0.5f) * 2.0f;
        return (v >= 0.0f ? "+" : "") + String(v, 1);
    }
    if (cls == "integer" && desc) {
        const int v = ParamCodec::normToIntValue(*desc, norm);
        return (v > 0 ? "+" : "") + String(v);
    }
    return String(norm, 2);
}

float FrameSection::parseValueToNorm(const String& text, const String& cls, const ParamDescriptor* desc)
{
    if (cls == "bipolar") return jlimit(0.f, 1.f, text.getFloatValue() * 0.5f + 0.5f);
    if (cls == "integer" && desc) {
        const int center = (desc->numClasses - 1) / 2;
        const int idx    = jlimit(0, desc->numClasses - 1, text.getIntValue() + center);
        return ParamCodec::indexToNorm(*desc, idx);
    }
    return jlimit(0.f, 1.f, text.getFloatValue());
}

// ── Static helpers ─────────────────────────────────────────────────────────────
Colour FrameSection::colourForFrame(const String& name)
{
    if (name == "OSC A" || name == "OSC B") return Colour(0x667F77DD);
    if (name == "Noise")                    return Colour(0x66888780);
    if (name == "Filter")                   return Colour(0x661D9E75);
    if (name == "ENV 1" || name == "ENV 2") return Colour(0x66D85A30);
    if (name.startsWith("LFO"))             return Colour(0x66BA7517);
    return Colour(0x66378ADD); // Reverb / Delay
}

String FrameSection::labelForParam(const String& id)
{
    static const char* kPrefixes[] = {
        "osc_a_", "osc_b_", "noise_", "filter_", "env1_", "env2_",
        "lfo1_", "lfo2_", "lfo3_", "lfo4_", "reverb_", "delay_"
    };
    String s = id;
    for (auto* p : kPrefixes)
        if (s.startsWith(p)) { s = s.fromFirstOccurrenceOf(p, false, false); break; }
    s = s.replace("_", " ");
    return s.isEmpty() ? id : (s.substring(0, 1).toUpperCase() + s.substring(1));
}

// ── Constructor ───────────────────────────────────────────────────────────────
FrameSection::FrameSection(const String& frameName,
                            const std::vector<const ParamDescriptor*>& params,
                            AudioProcessorValueTreeState& apvts,
                            AudioParameterBool* bypassParam)
    : frameName_(frameName),
      headerColour_(colourForFrame(frameName)),
      apvts_(apvts),
      bypassParam_(bypassParam)
{
    const Colour sliderFill    = headerColour_.withAlpha(1.0f).brighter(0.3f);
    const Colour sliderOutline = Colour(0xff2a2a3e);

    for (auto* desc : params) {
        ParamWidget pw;
        pw.desc = desc;
        const String id(desc->name.data());
        const String cls(desc->paramClass.data());

        // Label
        auto lbl = std::make_unique<Label>("", labelForParam(id));
        lbl->setFont(FontOptions(10.0f));
        lbl->setJustificationType(Justification::centred);
        lbl->setColour(Label::textColourId, Colours::lightgrey.withAlpha(0.75f));
        addAndMakeVisible(lbl.get());
        pw.label = std::move(lbl);

        if (cls == "categorical") {
            StringArray choices;
            if (auto* cp = dynamic_cast<AudioParameterChoice*>(apvts_.getParameter(id)))
                choices = cp->choices;
            jassert(choices.size() == desc->numClasses);
            const int n = choices.size();
            auto cb = std::make_unique<ComboBox>(id);
            cb->addItemList(choices, 1);
            cb->setJustificationType(Justification::centred);
            cb->setColour(ComboBox::backgroundColourId, Colour(0xff1e1e2e));
            cb->setColour(ComboBox::textColourId, Colours::lightgrey);
            cb->setColour(ComboBox::outlineColourId, headerColour_.withAlpha(0.5f));
            addAndMakeVisible(cb.get());
            pw.catSync = std::make_unique<CatParamSync>(apvts_, *cb, id, n);
            pw.combo   = std::move(cb);
        } else {
            auto sl = std::make_unique<Slider>();
            sl->setSliderStyle(Slider::RotaryVerticalDrag);
            sl->setTextBoxStyle(Slider::NoTextBox, false, 0, 0);
            sl->setColour(Slider::rotarySliderFillColourId,    sliderFill);
            sl->setColour(Slider::rotarySliderOutlineColourId, sliderOutline);
            sl->setColour(Slider::thumbColourId, Colours::white.withAlpha(0.85f));

            addAndMakeVisible(sl.get());

            // Attachment before range overrides
            pw.sliderAttach = std::make_unique<
                AudioProcessorValueTreeState::SliderAttachment>(apvts_, id, *sl);

            if (cls == "bipolar")
                sl->setDoubleClickReturnValue(true, 0.5);

            if (cls == "integer")
                sl->setRange(0.0, 1.0, 1.0 / (double)(desc->numClasses - 1));

            // Value label — shows scaled value, double-click to enter directly
            auto vl = std::make_unique<Label>("", formatParamValue((float)sl->getValue(), cls, desc));
            vl->setFont(FontOptions(10.0f));
            vl->setJustificationType(Justification::centred);
            vl->setColour(Label::textColourId, Colours::white.withAlpha(0.85f));
            vl->setColour(Label::backgroundWhenEditingColourId, Colour(0xff1e1e2e));
            vl->setColour(Label::textWhenEditingColourId, Colours::white);
            vl->setEditable(false, true, true);
            addAndMakeVisible(vl.get());

            auto* slRaw = sl.get();
            auto* vlRaw = vl.get();
            slRaw->onValueChange = [slRaw, vlRaw, cls, desc]() {
                vlRaw->setText(formatParamValue((float)slRaw->getValue(), cls, desc),
                               dontSendNotification);
            };
            vlRaw->onTextChange = [this, vlRaw, id, cls, desc]() {
                const float norm = parseValueToNorm(vlRaw->getText(), cls, desc);
                apvts_.getParameter(id)->setValueNotifyingHost(norm);
            };

            pw.valueLabel = std::move(vl);
            pw.slider     = std::move(sl);
        }
        widgets_.push_back(std::move(pw));
    }

    if (bypassParam_) {
        isBypassed_ = !bypassParam_->get();
        bypassButton_.setToggleState(bypassParam_->get(), dontSendNotification);
        bypassButton_.onStateChange = [this]() {
            if (!bypassParam_) return;
            const bool active = bypassButton_.getToggleState();
            bypassParam_->setValueNotifyingHost(active ? 1.0f : 0.0f);
            isBypassed_ = !active;
            repaint();
        };
        bypassParam_->addListener(this);
        addAndMakeVisible(bypassButton_);
    }
}

FrameSection::~FrameSection()
{
    if (bypassParam_)
        bypassParam_->removeListener(this);
}

// ── AudioProcessorParameter::Listener ────────────────────────────────────────
void FrameSection::parameterValueChanged(int, float newValue)
{
    const bool active = newValue > 0.5f;
    MessageManager::callAsync([safe = Component::SafePointer<FrameSection>(this), active]() {
        if (auto* c = safe.getComponent()) {
            c->isBypassed_ = !active;
            c->bypassButton_.setToggleState(active, dontSendNotification);
            c->repaint();
        }
    });
}

// ── paint ──────────────────────────────────────────────────────────────────────
void FrameSection::paint(Graphics& g)
{
    const auto b = getLocalBounds().toFloat();

    // Background
    g.setColour(Colour(0xff181825));
    g.fillRoundedRectangle(b, 8.0f);

    // Header bar — rounded on top only
    const auto hdr = b.withHeight((float)kHeaderH);
    {
        Path p;
        p.addRoundedRectangle(hdr.getX(), hdr.getY(), hdr.getWidth(), hdr.getHeight(),
                               8.0f, 8.0f, true, true, false, false);
        g.setColour(headerColour_);
        g.fillPath(p);
    }

    // Title
    g.setColour(Colours::white.withAlpha(0.92f));
    g.setFont(FontOptions(13.0f).withStyle("Bold"));
    auto titleR = hdr.toNearestInt();
    titleR.removeFromLeft(8);
    if (bypassParam_) titleR.removeFromRight(kHeaderH + 2);
    g.drawText(frameName_, titleR, Justification::centredLeft);

    // Optional subtitle (right-aligned in header)
    if (subtitle_.isNotEmpty()) {
        g.setColour(Colours::white.withAlpha(0.5f));
        g.setFont(FontOptions(9.0f));
        auto subR = hdr.toNearestInt();
        subR.removeFromRight(bypassParam_ ? kHeaderH + 2 : 4);
        g.drawText(subtitle_, subR, Justification::centredRight);
    }

    // Border
    g.setColour(headerColour_.withAlpha(0.55f));
    g.drawRoundedRectangle(b.reduced(0.5f), 8.0f, 0.5f);
}

void FrameSection::paintOverChildren(Graphics& g)
{
    if (!isBypassed_) return;
    g.setColour(Colour(0xaa0d0d18));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 8.0f);
}

// ── resized ────────────────────────────────────────────────────────────────────
void FrameSection::resized()
{
    auto b = getLocalBounds();

    // Reserve header
    auto hdr = b.removeFromTop(kHeaderH);
    if (bypassParam_)
        bypassButton_.setBounds(hdr.removeFromRight(kHeaderH).reduced(3));

    b.reduce(kPad, kPad);
    const int contentW = b.getWidth();
    const int numCols  = jmax(1, contentW / kCellW);

    int col = 0, row = 0;
    for (auto& pw : widgets_) {
        const int cx = b.getX() + col * kCellW;
        const int cy = b.getY() + row * kCellH;

        if (pw.slider) {
            const int off = (kCellW - kSliderSz) / 2;
            pw.slider->setBounds(cx + off, cy, kSliderSz, kSliderSz);
            pw.valueLabel->setBounds(cx, cy + kSliderSz, kCellW, 13);
            pw.label->setBounds(cx, cy + kSliderSz + 13, kCellW, kCellH - kSliderSz - 13);
        } else if (pw.combo) {
            pw.label->setBounds(cx, cy + 2, kCellW, 14);
            pw.combo->setBounds(cx + 2, cy + 18, kCellW - 4, 22);
        }

        if (++col >= numCols) { col = 0; ++row; }
    }
}

// ── getPreferredHeight ─────────────────────────────────────────────────────────
int FrameSection::getPreferredHeight(int availWidth) const
{
    const int contentW = availWidth - 2 * kPad;
    const int numCols  = jmax(1, contentW / kCellW);
    const int numRows  = (static_cast<int>(widgets_.size()) + numCols - 1) / numCols;
    return kHeaderH + numRows * kCellH + 2 * kPad;
}
