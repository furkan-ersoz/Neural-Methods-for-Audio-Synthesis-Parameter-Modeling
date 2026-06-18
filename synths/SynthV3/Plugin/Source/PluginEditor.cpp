#include "PluginEditor.h"
using namespace juce;

// ── DropZone ──────────────────────────────────────────────────────────────────
void SynthV3Editor::DropZone::paint(Graphics& g)
{
    auto b = getLocalBounds().toFloat().reduced(1.0f);
    g.setColour(isDragOver ? Colour(0xff1e2a40) : Colour(0xff121220));
    g.fillRoundedRectangle(b, 6.0f);
    g.setColour(isDragOver ? Colour(0xff5b9cf6) : Colour(0xff2e2e48));
    g.drawRoundedRectangle(b, 6.0f, isDragOver ? 1.5f : 0.7f);
    g.setColour(Colours::grey.withAlpha(0.65f));
    g.setFont(FontOptions(11.0f));
    g.drawText(isDragOver ? "Drop here" : statusText, getLocalBounds(), Justification::centred);
}

// ── Helper ────────────────────────────────────────────────────────────────────
std::vector<const ParamDescriptor*>
SynthV3Editor::paramsForFrame(const char* frame)
{
    std::vector<const ParamDescriptor*> result;
    for (const auto& d : SynthV3::PARAM_DESCRIPTORS)
        if (d.frame == frame) result.push_back(&d);
    return result;
}

// ── Constructor ───────────────────────────────────────────────────────────────
SynthV3Editor::SynthV3Editor(SynthV3Processor& p)
    : AudioProcessorEditor(&p), processor_(p)
{
    auto& apvts = p.getAPVTS();

    auto make = [&](const char* frame, const String& name) {
        return std::make_unique<FrameSection>(
            name, paramsForFrame(frame), apvts, p.getBypassParam(frame));
    };

    master_ = make("master", "Master");
    oscA_   = make("osc_a",  "OSC A");
    oscB_   = make("osc_b",  "OSC B");
    noise_  = make("noise",  "Noise");
    filter_ = make("filter", "Filter");
    env1_   = make("env1",   "ENV 1");
    env2_   = make("env2",   "ENV 2");
    lfo1_   = make("lfo1",   "LFO 1");
    lfo2_   = make("lfo2",   "LFO 2");
    lfo3_   = make("lfo3",   "LFO 3");
    lfo4_   = make("lfo4",   "LFO 4");
    reverb_ = make("reverb", "Reverb");
    delay_  = make("delay",  "Delay");

    for (auto* sec : { master_.get(), oscA_.get(), oscB_.get(), noise_.get(), filter_.get(),
                       env1_.get(), env2_.get(), lfo1_.get(), lfo2_.get(),
                       lfo3_.get(), lfo4_.get(), reverb_.get(), delay_.get() })
        addAndMakeVisible(sec);

    dropZone_.statusText = "Drop audio here";
    addAndMakeVisible(dropZone_);

    statusLabel_.setJustificationType(Justification::centred);
    statusLabel_.setFont(FontOptions(11.0f));
    setStatus(p.getInferenceEngine().isModelLoaded() ? "Ready" : "Model not loaded",
              !p.getInferenceEngine().isModelLoaded());
    addAndMakeVisible(statusLabel_);

    auto styleBtn = [](juce::TextButton& b) {
        b.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff2a2a3e));
        b.setColour(juce::TextButton::textColourOffId, juce::Colours::lightgrey);
    };

    randomiseButton_.onClick  = [this] { randomisePatch(); };
    predictButton_.onClick    = [this] {
        fileChooser_ = std::make_unique<juce::FileChooser>(
            "Select audio file",
            juce::File::getSpecialLocation(juce::File::userHomeDirectory),
            "*.wav;*.aif;*.aiff;*.mp3;*.flac");
        fileChooser_->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [this](const juce::FileChooser& fc) {
                if (fc.getResult() != juce::File{}) processFile(fc.getResult());
            });
    };
    savePresetButton_.onClick = [this] { savePreset(); };
    loadPresetButton_.onClick = [this] { loadPreset(); };

    styleBtn(randomiseButton_);
    styleBtn(predictButton_);
    styleBtn(savePresetButton_);
    styleBtn(loadPresetButton_);

    addAndMakeVisible(randomiseButton_);
    addAndMakeVisible(predictButton_);
    addAndMakeVisible(savePresetButton_);
    addAndMakeVisible(loadPresetButton_);

    profiles_ = SamplingProfiles::getAllProfiles();
    for (size_t i = 0; i < profiles_.size(); ++i)
        profileCombo_.addItem(juce::String(profiles_[i].name.data()), static_cast<int>(i) + 1);
    profileCombo_.setSelectedId(1, juce::dontSendNotification);
    profileCombo_.onChange = [this] { activeProfileIndex_ = profileCombo_.getSelectedId() - 1; };
    addAndMakeVisible(profileCombo_);

    // ENV2 auto-visibility (follows OSC B)
    bypassOscB_ = p.getBypassParam("osc_b");
    env2_->setSubtitle("(follows OSC B)");
    if (bypassOscB_) {
        bypassOscB_->addListener(this);
        applyEnv2Dim(bypassOscB_->get());
    }

    setResizable(true, true);
    setResizeLimits(900, 640, 3000, 2000);
    setSize(900, 740);
}

SynthV3Editor::~SynthV3Editor()
{
    if (bypassOscB_)
        bypassOscB_->removeListener(this);
}

// ── paint ──────────────────────────────────────────────────────────────────────
void SynthV3Editor::paint(Graphics& g)
{
    g.fillAll(Colour(0xff0d0d18));
}

// ── resized ────────────────────────────────────────────────────────────────────
void SynthV3Editor::resized()
{
    const int W  = getWidth();
    const int kP = 6;
    const int kDH = 42;
    const int kSH = 18;

    // Compute required row height as max of preferred heights in that row
    auto rowH = [&](std::initializer_list<std::pair<FrameSection*, float>> secs) {
        int h = 0;
        for (auto [s, f] : secs)
            h = jmax(h, s->getPreferredHeight(jmax(1, (int)(W * f) - kP)));
        return h;
    };

    const int r0 = jmax(master_->getPreferredHeight(jmax(1, W / 2 - kP)), kDH);
    const int r1 = rowH({ {oscA_.get(), 0.5f},    {oscB_.get(), 0.5f} });
    const int r2 = rowH({ {noise_.get(), 1.0f/3}, {filter_.get(), 2.0f/3} });
    const int r3 = rowH({ {env1_.get(), 0.5f},    {env2_.get(), 0.5f} });
    const int r4 = rowH({ {lfo1_.get(), 0.25f},   {lfo2_.get(), 0.25f},
                           {lfo3_.get(), 0.25f},   {lfo4_.get(), 0.25f} });
    const int r5 = rowH({ {reverb_.get(), 0.5f},  {delay_.get(), 0.5f} });

    int y = kP;

    auto layoutRow = [&](int rh, std::initializer_list<std::pair<FrameSection*, float>> secs) {
        int x = kP;
        for (auto [s, f] : secs) {
            const int w = jmax(1, (int)(W * f) - kP);
            s->setBounds(x, y, w, rh);
            x += w + kP;
        }
        y += rh + kP;
    };

    // Top row: Master (left half) + Audio input panel (right half)
    {
        const int masterW = W / 2 - kP;
        const int panelX  = kP + masterW + kP;
        const int panelW  = W - masterW - 3 * kP;

        master_->setBounds(kP, y, masterW, r0);

        // 4 buttons in a 2x2 grid at top of panel, drop zone fills the rest
        const int btnH      = 24;
        const int comboW    = 120;
        const int randW     = (panelW - kP) / 2 - comboW - 4;
        const int btnW      = (panelW - kP) / 2;
        const int btnY      = y;
        randomiseButton_ .setBounds(panelX,                        btnY, randW, btnH);
        profileCombo_    .setBounds(panelX + randW + 4,            btnY, comboW, btnH);
        predictButton_   .setBounds(panelX + btnW + kP,            btnY, btnW, btnH);
        savePresetButton_.setBounds(panelX,           btnY + btnH + kP, btnW, btnH);
        loadPresetButton_.setBounds(panelX + btnW + kP, btnY + btnH + kP, btnW, btnH);

        const int dropY = btnY + 2 * (btnH + kP);
        const int dropH = jmax(kDH, r0 - 2 * (btnH + kP));
        dropZone_.setBounds(panelX, dropY, panelW, dropH);

        y += r0 + kP;
    }

    layoutRow(r1, { {oscA_.get(), 0.5f},    {oscB_.get(), 0.5f} });
    layoutRow(r2, { {noise_.get(), 1.0f/3}, {filter_.get(), 2.0f/3} });
    layoutRow(r3, { {env1_.get(), 0.5f},    {env2_.get(), 0.5f} });
    layoutRow(r4, { {lfo1_.get(), 0.25f},   {lfo2_.get(), 0.25f},
                    {lfo3_.get(), 0.25f},    {lfo4_.get(), 0.25f} });
    layoutRow(r5, { {reverb_.get(), 0.5f},  {delay_.get(), 0.5f} });

    statusLabel_.setBounds(kP, y, W - 2 * kP, kSH);
}

// ── FileDragAndDropTarget ─────────────────────────────────────────────────────
bool SynthV3Editor::isInterestedInFileDrag(const StringArray& files)
{
    for (auto& f : files) {
        auto ext = File(f).getFileExtension().toLowerCase();
        if (ext == ".wav" || ext == ".aif" || ext == ".aiff"
            || ext == ".mp3" || ext == ".flac")
            return true;
    }
    return false;
}

void SynthV3Editor::filesDropped(const StringArray& files, int, int)
{
    dropZone_.isDragOver = false;
    dropZone_.repaint();
    if (!files.isEmpty()) processFile(File(files[0]));
}

void SynthV3Editor::fileDragEnter(const StringArray&, int, int)
{
    dropZone_.isDragOver = true;
    dropZone_.repaint();
}

void SynthV3Editor::fileDragExit(const StringArray&)
{
    dropZone_.isDragOver = false;
    dropZone_.repaint();
}

// ── processFile / setStatus ───────────────────────────────────────────────────
void SynthV3Editor::processFile(const File& file)
{
    if (!processor_.getInferenceEngine().isModelLoaded()) {
        setStatus("Model not loaded.", true);
        return;
    }
    setStatus("Processing: " + file.getFileName());
    dropZone_.statusText = file.getFileName();
    dropZone_.repaint();

    processor_.getInferenceEngine().runInference(file,
        [this](const std::vector<float>& params, const String& error) {
            if (!error.isEmpty()) { setStatus("Error: " + error, true); return; }
            processor_.applyInferenceResult(params);
            setStatus("Done — " + String(params.size()) + " params applied");
        });
}

void SynthV3Editor::setStatus(const String& msg, bool isError)
{
    statusLabel_.setColour(Label::textColourId,
                           isError ? Colour(0xffff6b6b) : Colours::grey);
    statusLabel_.setText(msg, dontSendNotification);
}

// ── Randomise ─────────────────────────────────────────────────────────────────
void SynthV3Editor::randomisePatch()
{
    auto& apvts = processor_.getAPVTS();
    juce::Random rng;

    // ── 1. Build randomised values in memory (don't touch APVTS yet) ──────────
    std::vector<std::pair<juce::String, float>> pending;
    pending.reserve(SynthV3::NUM_PARAMS);

    const auto& profile = profiles_[static_cast<size_t>(activeProfileIndex_)];

    for (const auto& desc : SynthV3::PARAM_DESCRIPTORS) {
        if (std::string_view(desc.name) == "master_volume") continue;

        const auto* rule     = SamplingProfiles::findRule(profile, desc.name);
        const auto  resolved = SamplingProfiles::resolveRule(desc, rule);

        if (resolved.fixedProb >= 1.0f) {
            float val = (resolved.fixedValue >= 0.0f) ? resolved.fixedValue : desc.defaultVal;
            pending.push_back({ juce::String(desc.name.data()), val });
            continue;
        }
        if (resolved.fixedProb > 0.0f && rng.nextFloat() < resolved.fixedProb) {
            float val = (resolved.fixedValue >= 0.0f) ? resolved.fixedValue : desc.defaultVal;
            pending.push_back({ juce::String(desc.name.data()), val });
            continue;
        }

        float v;
        if (resolved.distribution == "categorical") {
            v = (float)rng.nextInt(resolved.categoricalSteps)
                / (float)(resolved.categoricalSteps - 1);
            v = resolved.randomMin + v * (resolved.randomMax - resolved.randomMin);
        } else if (resolved.distribution == "log") {
            v = resolved.randomMin + (resolved.randomMax - resolved.randomMin) * (0.1f + rng.nextFloat() * 0.8f);
        } else {
            v = resolved.randomMin + rng.nextFloat() * (resolved.randomMax - resolved.randomMin);
        }
        pending.push_back({ juce::String(desc.name.data()), v });
    }

    // ── 2. Silent level measurement with the new values ───────────────────────
    // Uses a temp engine that writes to a local buffer — no audio output.
    // Accounts for the amplitude * masterVolume * 2.0 gain chain in processBlock.
    const float amplitude = *apvts.getRawParameterValue("amplitude"); // usually 0.8

    std::vector<float> newVals(SynthV3::NUM_PARAMS);
    {
        // Seed newVals from current APVTS (preserves fixed params like pan/tune/mix),
        // then overlay the pending randomised values.
        for (size_t i = 0; i < static_cast<size_t>(SynthV3::NUM_PARAMS); ++i)
            newVals[i] = apvts.getParameter(juce::String(SynthV3::PARAM_DESCRIPTORS[i].name.data()))->getValue();
        for (const auto& [id, v] : pending) {
            for (size_t i = 0; i < static_cast<size_t>(SynthV3::NUM_PARAMS); ++i)
                if (juce::String(SynthV3::PARAM_DESCRIPTORS[i].name.data()) == id)
                    { newVals[i] = v; break; }
        }
    }

    SynthV3::SynthParameters tempParams;
    tempParams.fromVector(newVals);
    tempParams.amplitude = amplitude;

    SynthV3::SynthEngine tempEngine;
    tempEngine.prepare(44100.0, 512);
    tempEngine.setParameters(tempParams);
    // All effects bypassed for a clean short render (no reverb/delay tail noise)
    tempEngine.setBypassFlags({ false, true, false, true, true, true, true, true, true });

    std::vector<float> buf(1024, 0.0f);
    tempEngine.process(buf.data(),       512, true,  440.0f);
    tempEngine.process(buf.data() + 512, 512, false, 440.0f);

    float sumSq = 0.0f;
    for (float s : buf) sumSq += s * s;
    const float monoRms = std::sqrt(sumSq / static_cast<float>(buf.size()));

    // Actual output level = monoRms * amplitude * masterVolume * 2.0
    // Solve for masterVolume to hit targetRms:
    const float targetRms  = 0.15f;
    const float effectiveGain = amplitude * 2.0f;
    const float masterVol = (monoRms > 0.001f)
        ? std::clamp(targetRms / (monoRms * effectiveGain), 0.05f, 1.0f)
        : 0.8f;

    // ── 3. Apply everything to APVTS in one pass — audio thread sees final state
    for (const auto& [id, v] : pending)
        apvts.getParameter(id)->setValueNotifyingHost(v);

    const char* bypassFrames[] = { "osc_b","noise","filter","lfo1","lfo2","lfo3","lfo4","reverb","delay" };
    for (auto* frame : bypassFrames) {
        float activeProb = 0.5f;
        for (const auto& fr : profile.frameRules)
            if (fr.frameName == frame) { activeProb = fr.activeProb; break; }
        if (auto* bp = processor_.getBypassParam(frame))
            bp->setValueNotifyingHost(rng.nextFloat() < activeProb ? 1.0f : 0.0f);
    }

    if (auto* p = apvts.getParameter("master_volume"))
        p->setValueNotifyingHost(masterVol);
}

// ── Preset save / load ────────────────────────────────────────────────────────
void SynthV3Editor::savePreset()
{
    fileChooser_ = std::make_unique<juce::FileChooser>(
        "Save preset",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.synthv3");
    fileChooser_->launchAsync(
        juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& fc) {
            auto f = fc.getResult().withFileExtension("synthv3");
            if (f == juce::File{}) return;
            juce::MemoryBlock data;
            processor_.getStateInformation(data);
            if (!f.replaceWithData(data.getData(), data.getSize()))
                setStatus("Failed to save preset", true);
            else
                setStatus("Saved: " + f.getFileName());
        });
}

void SynthV3Editor::loadPreset()
{
    fileChooser_ = std::make_unique<juce::FileChooser>(
        "Load preset",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.synthv3");
    fileChooser_->launchAsync(
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& fc) {
            auto f = fc.getResult();
            if (f == juce::File{}) return;
            juce::MemoryBlock data;
            if (!f.loadFileAsData(data)) { setStatus("Failed to read file", true); return; }
            processor_.setStateInformation(data.getData(), (int)data.getSize());
            setStatus("Loaded: " + f.getFileName());
        });
}

// ── ENV2 dimming ──────────────────────────────────────────────────────────────
void SynthV3Editor::applyEnv2Dim(bool oscBActive)
{
    env2_->setAlpha(oscBActive ? 1.0f : 0.4f);
    env2_->setEnabled(oscBActive);
}

void SynthV3Editor::parameterValueChanged(int, float newValue)
{
    const bool active = newValue > 0.5f;
    MessageManager::callAsync([safe = Component::SafePointer<SynthV3Editor>(this), active]() {
        if (auto* c = safe.getComponent())
            c->applyEnv2Dim(active);
    });
}
