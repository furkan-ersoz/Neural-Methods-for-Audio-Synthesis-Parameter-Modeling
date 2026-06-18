#include "MainComponent.h"

#include <thread>
#include <unordered_set>
#include "ParamConfigResolver.h"
#include "SamplingProfiles/AllProfiles.h"

using namespace juce;

static constexpr int kRowHeight    = 52;
static constexpr int kHeaderHeight = 22;

static const std::unordered_map<std::string, std::string> kFrameDisplayNames = {
    {"osc_a",  "OSC A"},  {"osc_b",  "OSC B"},  {"noise",  "NOISE"},
    {"filter", "FILTER"}, {"env1",   "ENV 1"},  {"env2",   "ENV 2"},
    {"lfo1",   "LFO 1"},  {"lfo2",   "LFO 2"},  {"lfo3",   "LFO 3"},
    {"lfo4",   "LFO 4"},  {"reverb", "REVERB"}, {"delay",  "DELAY"}
};

static const std::array<std::pair<const char*, const char*>, 12> kFrameOrder = {{
    { "OSC A",  "osc_a"  },
    { "OSC B",  "osc_b"  },
    { "Noise",  "noise"  },
    { "Filter", "filter" },
    { "ENV 1",  "env1"   },
    { "ENV 2",  "env2"   },
    { "LFO 1",  "lfo1"   },
    { "LFO 2",  "lfo2"   },
    { "LFO 3",  "lfo3"   },
    { "LFO 4",  "lfo4"   },
    { "Reverb", "reverb" },
    { "Delay",  "delay"  },
}};

static const std::unordered_set<std::string> kBypassableFrames = {
    "osc_b", "noise", "filter", "lfo1", "lfo2", "lfo3", "lfo4", "reverb", "delay"
};

// ─────────────────────────────────────────────────────────────────────────────
MainComponent::MainComponent()
{
    addAndMakeVisible(synthLabel_);
    addAndMakeVisible(synthCombo_);
    populateSynthCombo();

    addAndMakeVisible(profileLabel_);
    addAndMakeVisible(profileCombo_);
    populateProfileCombo();

    outputLabel_.setText("Output:", dontSendNotification);
    addAndMakeVisible(outputLabel_);
    outputPathBox_.setReadOnly(true);

    {
        const auto path = defaultOutputForSynth(synthCombo_.getText());
        if (path.isEmpty()) {
            outputPathBox_.setText("(not selected)");
            appendLog("WARNING: synths/ folder not found. Please select output manually.");
        } else {
            outputPathBox_.setText(path);
            appendLog("Default output: " + path);
        }
    }

    addAndMakeVisible(outputPathBox_);

    outputBrowseBtn_.onClick = [this] {
        fileChooser_ = std::make_unique<FileChooser>("Select output folder");
        fileChooser_->launchAsync(FileBrowserComponent::openMode |
                                  FileBrowserComponent::canSelectDirectories,
            [this](const FileChooser& fc) {
                if (fc.getResult() != File{})
                    outputPathBox_.setText(fc.getResult().getFullPathName());
                updateDiskEstimate();
            });
    };
    addAndMakeVisible(outputBrowseBtn_);

    nBox_.setText("500");
    seedBox_.setText("42");
    maxDurBox_.setText("10");
    workersBox_.setText(juce::String(juce::jmax(1u, std::thread::hardware_concurrency())));
    for (auto* e : { &nBox_, &seedBox_, &maxDurBox_ })
        e->onTextChange = [this] { updateDiskEstimate(); };
    addAndMakeVisible(nLabel_);       addAndMakeVisible(nBox_);
    addAndMakeVisible(seedLabel_);    addAndMakeVisible(seedBox_);
    addAndMakeVisible(maxDurLabel_);  addAndMakeVisible(maxDurBox_);
    addAndMakeVisible(workersLabel_); addAndMakeVisible(workersBox_);

    diskEstLabel_.setText("Estimated size: --", dontSendNotification);
    addAndMakeVisible(diskEstLabel_);

    startBtn_.onClick = [this] { startGeneration(); };
    stopBtn_.onClick  = [this] { stopGeneration();  };
    stopBtn_.setEnabled(false);
    addAndMakeVisible(startBtn_);
    addAndMakeVisible(stopBtn_);

    addAndMakeVisible(progressBar_);
    addAndMakeVisible(progressLabel_);

    logBox_.setMultiLine(true);
    logBox_.setReadOnly(true);
    logBox_.setScrollbarsShown(true);
    addAndMakeVisible(logBox_);

    paramViewport_.setViewedComponent(&paramPanel_, false);
    paramViewport_.setScrollBarsShown(true, false); // vertical only
    addAndMakeVisible(paramViewport_);

    // ── Frame Outputs panel ───────────────────────────────────────────────────
    frameOutTitleLabel_.setFont(FontOptions(11.0f).withStyle("Bold"));
    frameOutPanel_.addAndMakeVisible(frameOutTitleLabel_);
    frameOutPanel_.addAndMakeVisible(fullRenderLabel_);

    selectAllBtn_.onClick = [this] {
        for (auto* btn : frameToggleOwner_)
            btn->setToggleState(true, dontSendNotification);
    };
    deselectAllBtn_.onClick = [this] {
        for (auto* btn : frameToggleOwner_)
            btn->setToggleState(false, dontSendNotification);
    };
    frameOutPanel_.addAndMakeVisible(selectAllBtn_);
    frameOutPanel_.addAndMakeVisible(deselectAllBtn_);

    for (const auto& [display, internal] : kFrameOrder) {
        auto* btn = frameToggleOwner_.add(new ToggleButton(display));
        btn->setToggleState(true, dontSendNotification);
        frameOutPanel_.addAndMakeVisible(*btn);
        frameInternalNames_.push_back(internal);
    }

    frameOutViewport_.setViewedComponent(&frameOutPanel_, false);
    frameOutViewport_.setScrollBarsShown(true, false);
    addAndMakeVisible(frameOutViewport_);

    updateDiskEstimate();
    startTimer(100);

    setSize(640, 780); // fixed window height — param area scrolls inside
}

MainComponent::~MainComponent()
{
    stopTimer();
    writer_.stopAndWait();
}

int MainComponent::paramAreaHeight() const
{
    return sectionHeaders_.size() * kHeaderHeight + paramRows_.size() * kRowHeight + 8;
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::paint(Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(ResizableWindow::backgroundColourId));
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::resized()
{
    const int pad = 8;
    const int W   = getWidth() - 2 * pad;
    int y         = pad;

    // ── Fixed controls at top ─────────────────────────────────────────────────
    synthLabel_.setBounds(pad, y, 60, 24);
    synthCombo_.setBounds(pad + 64, y, 180, 24);
    profileLabel_.setBounds(pad + 256, y, 110, 24);
    profileCombo_.setBounds(pad + 370, y, 180, 24);
    y += 32;

    outputLabel_.setBounds(pad, y, 60, 24);
    outputPathBox_.setBounds(pad + 64, y, W - 130, 24);
    outputBrowseBtn_.setBounds(pad + 64 + W - 126, y, 60, 24);
    y += 32;

    nLabel_.setBounds(pad, y, 70, 24);
    nBox_.setBounds(pad + 74, y, 60, 24);
    seedLabel_.setBounds(pad + 144, y, 40, 24);
    seedBox_.setBounds(pad + 188, y, 60, 24);
    maxDurLabel_.setBounds(pad + 258, y, 90, 24);
    maxDurBox_.setBounds(pad + 352, y, 50, 24);
    workersLabel_.setBounds(pad + 412, y, 60, 24);
    workersBox_.setBounds(pad + 476, y, 40, 24);
    y += 32;

    y += 8; // gap before param viewport

    // ── Bottom controls (laid out from the bottom up) ─────────────────────────
    const int bottomH = 20 + 8   // diskEstLabel
                      + 28 + 8   // start/stop buttons
                      + 20 + 8   // progress bar
                      + 120 + pad; // log box
    const int viewportH  = getHeight() - y - bottomH;
    const int frameOutW  = 196;
    const int paramW     = W - frameOutW - pad;
    paramViewport_.setBounds(pad, y, paramW, juce::jmax(40, viewportH));
    frameOutViewport_.setBounds(pad + paramW + pad, y, frameOutW - pad, juce::jmax(40, viewportH));
    y += viewportH + 8;

    // ── Layout rows inside frame outputs panel ────────────────────────────────
    {
        const int fw  = frameOutW - pad - frameOutViewport_.getScrollBarThickness();
        const int bh  = 20;
        const int rh  = 24;
        int fy = 4;

        frameOutTitleLabel_.setBounds(2, fy, fw, 16);
        fy += 20;
        fullRenderLabel_.setBounds(2, fy, fw, 14);
        fy += 20;

        const int bw = (fw - 4) / 2;
        selectAllBtn_.setBounds(2, fy, bw, bh);
        deselectAllBtn_.setBounds(4 + bw, fy, bw, bh);
        fy += bh + 4;

        for (auto* btn : frameToggleOwner_) {
            btn->setBounds(2, fy, fw, rh);
            fy += rh;
        }
        fy += 4;
        frameOutPanel_.setSize(fw, fy);
    }

    // ── Layout rows inside the param panel (coords relative to paramPanel_) ───
    const int pW = paramW - paramViewport_.getScrollBarThickness();
    int py = 0;

    paramPanel_.headerRects.clear();

    for (const auto& item : layoutItems_) {
        if (item.type == LayoutItem::Type::Header) {
            auto* h    = sectionHeaders_[item.index];
            h->layoutY = py;
            h->label.setBounds(4, py + 3, 200, 16);

            if (h->hasBypass) {
                auto it = bypassToggles_.find(h->frameName);
                if (it != bypassToggles_.end())
                    it->second->setBounds(pW - 130, py + 3, 80, 16);

                auto it2 = bypassProbLabels_.find(h->frameName);
                if (it2 != bypassProbLabels_.end())
                    it2->second->setBounds(pW - 46, py + 3, 40, 16);
            }

            paramPanel_.headerRects.push_back({ 0, py, pW, kHeaderHeight });
            py += kHeaderHeight;
        } else {
            auto& row      = *paramRows_[item.index];
            const bool isFixed = (row.strategyCombo.getSelectedId() == 4);

            row.nameLabel.setBounds(4, py, 70, 20);
            row.strategyCombo.setBounds(78, py, 140, 20);

            if (isFixed) {
                row.fixedLabel.setBounds(228, py, 44, 20);
                row.fixedBox.setBounds(276, py, 80, 20);
            } else {
                row.minLabel.setBounds(228, py, 30, 20);
                row.minBox.setBounds(262, py, 80, 20);
                row.maxLabel.setBounds(350, py, 30, 20);
                row.maxBox.setBounds(384, py, 80, 20);
            }
            py += kRowHeight;
        }
    }
    py += 8; // bottom padding inside panel

    paramPanel_.setSize(pW, py);
    paramPanel_.repaint();

    // ── Bottom controls ───────────────────────────────────────────────────────
    diskEstLabel_.setBounds(pad, y, W, 20);
    y += 28;

    startBtn_.setBounds(pad, y, (W - pad) / 2, 28);
    stopBtn_.setBounds(pad + (W + pad) / 2, y, (W - pad) / 2, 28);
    y += 36;

    progressBar_.setBounds(pad, y, W - 80, 20);
    progressLabel_.setBounds(pad + W - 76, y, 76, 20);
    y += 28;

    logBox_.setBounds(pad, y, W, getHeight() - y - pad);
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::timerCallback()
{
    {
        ScopedLock sl(logMutex_);
        for (auto& msg : pendingLogs_) {
            logBox_.moveCaretToEnd();
            logBox_.insertTextAtCaret(msg + "\n");
        }
        pendingLogs_.clear();
    }

    const int cur   = progressCurrent_.load();
    const int total = progressTotal_.load();
    if (total > 0) {
        progressVal_ = static_cast<double>(cur) / static_cast<double>(total);
        progressLabel_.setText(String(cur) + "/" + String(total), dontSendNotification);
    }

    if (jobFinished_.exchange(false)) {
        startBtn_.setEnabled(true);
        stopBtn_.setEnabled(false);
        if (jobSuccess_.load()) progressVal_ = 1.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::startGeneration()
{
    if (outputPathBox_.getText().isEmpty() ||
        outputPathBox_.getText() == "(not selected)") {
        appendLog("ERROR: Please select an output folder first.");
        return;
    }

    const auto settings = buildSettings();

    DataGen::WriterCallbacks cb;
    cb.onProgress = [this](int cur, int tot) {
        progressCurrent_.store(cur);
        progressTotal_.store(tot);
    };
    cb.onLog = [this](const String& msg) {
        ScopedLock sl(logMutex_);
        pendingLogs_.add(msg);
    };
    cb.onFinished = [this](bool success) {
        jobSuccess_.store(success);
        jobFinished_.store(true);
    };

    writer_.configure(settings, cb);

    appendLog(String::formatted("Starting - estimated size: %.1f MB",
                                writer_.estimatedSizeMB()));

    progressCurrent_.store(0);
    progressTotal_.store(settings.numSamples);
    progressVal_ = 0.0;

    startBtn_.setEnabled(false);
    stopBtn_.setEnabled(true);

    writer_.startThread();
}

void MainComponent::stopGeneration()
{
    appendLog("Stopping...");
    writer_.stopAndWait();
    startBtn_.setEnabled(true);
    stopBtn_.setEnabled(false);
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::updateDiskEstimate()
{
    const int   n      = nBox_.getText().getIntValue();
    const float maxDur = maxDurBox_.getText().getFloatValue();
    const float mb     = static_cast<float>(n) * maxDur * 44100.0f * 2.0f / (1024.0f * 1024.0f);
    diskEstLabel_.setText(String::formatted("Estimated max size: %.1f MB", mb),
                          dontSendNotification);
}

void MainComponent::appendLog(const String& msg)
{
    ScopedLock sl(logMutex_);
    pendingLogs_.add(msg);
}

// ─────────────────────────────────────────────────────────────────────────────
void MainComponent::populateSynthCombo()
{
    synthCombo_.clear();
    const auto names = SynthRegistry::instance().getNames();
    for (int i = 0; i < static_cast<int>(names.size()); ++i)
        synthCombo_.addItem(names[static_cast<size_t>(i)], i + 1);

    if (!names.empty())
        synthCombo_.setSelectedId(1, dontSendNotification);

    synthCombo_.onChange = [this] { rebuildParamRows(); };
    rebuildParamRows();
}

void MainComponent::populateProfileCombo()
{
    profileCombo_.clear();
    const auto profiles = SamplingProfiles::getAllProfiles();
    for (int i = 0; i < static_cast<int>(profiles.size()); ++i)
        profileCombo_.addItem(String(profiles[static_cast<size_t>(i)].name.data()), i + 1);

    if (!profiles.empty())
        profileCombo_.setSelectedId(1, dontSendNotification);

    profileCombo_.onChange = [this] { applyProfileToRows(); };
}

void MainComponent::applyProfileToRows()
{
    const auto profiles = SamplingProfiles::getAllProfiles();
    const int selIdx = profileCombo_.getSelectedId() - 1;
    if (selIdx < 0 || selIdx >= static_cast<int>(profiles.size())) return;
    const auto& profile = profiles[static_cast<size_t>(selIdx)];

    // Update bypass prob text boxes from profile frameRules
    for (const auto& fr : profile.frameRules) {
        auto it = bypassProbLabels_.find(std::string(fr.frameName));
        if (it != bypassProbLabels_.end()) {
            const int bypassPct = static_cast<int>((1.0f - fr.activeProb) * 100.0f + 0.5f);
            it->second->setText(String(bypassPct) + "%", dontSendNotification);
        }
    }

    const auto synthName = synthCombo_.getText().toStdString();
    if (synthName.empty()) return;

    std::unique_ptr<ISynth> probe;
    try { probe = SynthRegistry::instance().create(synthName); }
    catch (...) { return; }

    const auto paramDescs = probe->getParams();
    const auto configs    = DataGen::resolveParamConfigs(profile, paramDescs);

    for (int i = 0; i < paramRows_.size() && i < static_cast<int>(configs.size()); ++i) {
        const auto& pc  = configs[static_cast<size_t>(i)];
        auto&       row = *paramRows_[i];

        if (pc.strategy == DataGen::Strategy::Fixed) {
            row.strategyCombo.setSelectedId(4, sendNotification);
            row.fixedBox.setText(String(pc.fixedVal, 6));
        } else if (pc.strategy == DataGen::Strategy::RandomLog) {
            row.strategyCombo.setSelectedId(2, sendNotification);
            row.minBox.setText(String(pc.rangeMin, 6));
            row.maxBox.setText(String(pc.rangeMax, 6));
        } else if (pc.strategy == DataGen::Strategy::Discrete) {
            row.strategyCombo.setSelectedId(6, sendNotification);
        } else {
            row.strategyCombo.setSelectedId(1, sendNotification);
            row.minBox.setText(String(pc.rangeMin, 6));
            row.maxBox.setText(String(pc.rangeMax, 6));
        }
    }
}

void MainComponent::applyBypassAlpha(const std::string& frameName, bool bypassed)
{
    const float alpha = bypassed ? 0.4f : 1.0f;
    for (auto* row : paramRows_) {
        if (row->frameName != frameName) continue;
        row->nameLabel.setAlpha(alpha);
        row->strategyCombo.setAlpha(alpha);
        row->minLabel.setAlpha(alpha);
        row->minBox.setAlpha(alpha);
        row->maxLabel.setAlpha(alpha);
        row->maxBox.setAlpha(alpha);
        row->fixedLabel.setAlpha(alpha);
        row->fixedBox.setAlpha(alpha);
    }
}

void MainComponent::rebuildParamRows()
{
    // Remove section headers
    for (auto* h : sectionHeaders_)
        paramPanel_.removeChildComponent(&h->label);
    sectionHeaders_.clear();

    // Remove bypass toggles
    for (auto& [name, btn] : bypassToggles_)
        paramPanel_.removeChildComponent(btn);
    bypassToggles_.clear();
    bypassToggleOwner_.clear();

    // Remove bypass prob editors
    for (auto& [name, ed] : bypassProbLabels_)
        paramPanel_.removeChildComponent(ed);
    bypassProbLabels_.clear();
    bypassProbEditorOwner_.clear();

    layoutItems_.clear();

    // Remove param rows
    for (auto* row : paramRows_) {
        paramPanel_.removeChildComponent(&row->nameLabel);
        paramPanel_.removeChildComponent(&row->strategyCombo);
        paramPanel_.removeChildComponent(&row->minLabel);
        paramPanel_.removeChildComponent(&row->minBox);
        paramPanel_.removeChildComponent(&row->maxLabel);
        paramPanel_.removeChildComponent(&row->maxBox);
        paramPanel_.removeChildComponent(&row->fixedLabel);
        paramPanel_.removeChildComponent(&row->fixedBox);
    }
    paramRows_.clear();

    const auto synthName = synthCombo_.getText().toStdString();
    if (synthName.empty()) return;

    std::unique_ptr<ISynth> probe;
    try { probe = SynthRegistry::instance().create(synthName); }
    catch (...) { appendLog("ERROR: Could not create synth: " + synthCombo_.getText()); return; }

    const auto paramDescs = probe->getParams();
    std::string lastFrame;

    for (int i = 0; i < static_cast<int>(paramDescs.size()); ++i) {
        const auto& desc      = paramDescs[static_cast<size_t>(i)];
        const std::string frame(desc.frame);

        // Insert section header when frame changes
        if (frame != lastFrame) {
            lastFrame = frame;

            auto* h        = sectionHeaders_.add(new SectionHeaderEntry());
            h->frameName   = frame;
            h->hasBypass   = kBypassableFrames.count(frame) > 0;

            auto it        = kFrameDisplayNames.find(frame);
            const String displayName = it != kFrameDisplayNames.end()
                                     ? String(it->second)
                                     : String(frame).toUpperCase();
            h->label.setText(displayName, dontSendNotification);
            h->label.setFont(FontOptions(11.0f).withStyle("Bold"));
            paramPanel_.addAndMakeVisible(h->label);

            if (h->hasBypass) {
                auto* btn = bypassToggleOwner_.add(new ToggleButton("Bypass"));
                bypassToggles_[frame] = btn;
                btn->onStateChange = [this, frame] {
                    applyBypassAlpha(frame, bypassToggles_[frame]->getToggleState());
                };
                paramPanel_.addAndMakeVisible(*btn);

                auto* probEd = bypassProbEditorOwner_.add(new TextEditor());
                bypassProbLabels_[frame] = probEd;
                probEd->setText("0%", dontSendNotification);
                probEd->setFont(FontOptions(10.0f));
                probEd->setJustification(Justification::centred);
                paramPanel_.addAndMakeVisible(*probEd);
            }

            layoutItems_.push_back({ LayoutItem::Type::Header,
                                     sectionHeaders_.size() - 1 });
        }

        // Build param row
        auto* row      = paramRows_.add(new ParamRow());
        row->frameName = frame;

        row->nameLabel.setText(desc.name.data(), dontSendNotification);
        row->nameLabel.setFont(FontOptions(14.0f).withStyle("Bold"));

        row->strategyCombo.addItem("Random Uniform", 1);
        row->strategyCombo.addItem("Random Log",     2);
        row->strategyCombo.addItem("Linear",         3);
        row->strategyCombo.addItem("Fixed",          4);
        row->strategyCombo.addItem("Range",          5);
        row->strategyCombo.addItem("Discrete",       6);

        row->numClasses = desc.numClasses;

        const std::string pc(desc.paramClass);

        if      (pc == "continuous_log")                 row->strategyCombo.setSelectedId(2);
        else if (pc == "categorical" || pc == "integer") row->strategyCombo.setSelectedId(6);
        else                                             row->strategyCombo.setSelectedId(1);

        row->minBox.setText(String(desc.minVal, 6));
        row->maxBox.setText(String(desc.maxVal, 6));
        row->fixedBox.setText(String(desc.defaultVal, 6));

        const int idx = paramRows_.size() - 1;
        row->strategyCombo.onChange = [this, idx] {
            auto& r = *paramRows_[idx];
            const int  selId     = r.strategyCombo.getSelectedId();
            const bool isFixed   = (selId == 4);
            const bool isDiscrete = (selId == 6);
            r.minLabel.setVisible(!isFixed && !isDiscrete);
            r.minBox.setVisible(!isFixed && !isDiscrete);
            r.maxLabel.setVisible(!isFixed && !isDiscrete);
            r.maxBox.setVisible(!isFixed && !isDiscrete);
            r.fixedLabel.setVisible(isFixed);
            r.fixedBox.setVisible(isFixed);
            resized();
        };
        row->strategyCombo.onChange();

        paramPanel_.addAndMakeVisible(row->nameLabel);
        paramPanel_.addAndMakeVisible(row->strategyCombo);
        paramPanel_.addAndMakeVisible(row->minLabel);
        paramPanel_.addAndMakeVisible(row->minBox);
        paramPanel_.addAndMakeVisible(row->maxLabel);
        paramPanel_.addAndMakeVisible(row->maxBox);
        paramPanel_.addAndMakeVisible(row->fixedLabel);
        paramPanel_.addAndMakeVisible(row->fixedBox);

        layoutItems_.push_back({ LayoutItem::Type::Param, idx });
    }

    const auto path = defaultOutputForSynth(synthCombo_.getText());
    if (!path.isEmpty())
        outputPathBox_.setText(path);

    applyProfileToRows();
    resized();
    repaint();
}

// ─────────────────────────────────────────────────────────────────────────────
juce::String MainComponent::defaultOutputForSynth(const juce::String& synthName) const
{
    auto candidate = File::getSpecialLocation(File::currentApplicationFile).getParentDirectory();
    for (int up = 0; up < 8; ++up) {
        const auto parentName = candidate.getFileName();
        const bool isBuildDir = parentName.containsIgnoreCase("cmake-build")
                             || parentName.containsIgnoreCase("build");
        auto synthsDir = candidate.getChildFile("synths");
        if (!isBuildDir && synthsDir.isDirectory()) {
            auto datasets = synthsDir.getChildFile(synthName).getChildFile("datasets");
            datasets.createDirectory();
            return datasets.getFullPathName();
        }
        candidate = candidate.getParentDirectory();
    }
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────
DataGen::WriterSettings MainComponent::buildSettings() const
{
    DataGen::WriterSettings s;
    s.outputDir   = File(outputPathBox_.getText());
    s.numSamples  = nBox_.getText().getIntValue();
    s.baseSeed    = static_cast<uint32_t>(seedBox_.getText().getIntValue());
    s.maxDuration = maxDurBox_.getText().getFloatValue();
    s.numWorkers  = juce::jmax(1, workersBox_.getText().getIntValue());

    const auto synthName = synthCombo_.getText().toStdString();
    s.synthFactory = [synthName]() {
        return SynthRegistry::instance().create(synthName);
    };

    s.paramConfigs.resize(static_cast<size_t>(paramRows_.size()));
    for (int i = 0; i < paramRows_.size(); ++i)
        s.paramConfigs[static_cast<size_t>(i)] = buildParamConfig(i);

    {
        const auto profiles = SamplingProfiles::getAllProfiles();
        const int selIdx = profileCombo_.getSelectedId() - 1;
        if (selIdx >= 0 && selIdx < static_cast<int>(profiles.size()))
            s.frameRules = profiles[static_cast<size_t>(selIdx)].frameRules;
    }

    s.profileName = profileCombo_.getText().toStdString();
    s.synthName   = synthCombo_.getText().toStdString();

    {
        const auto profiles = SamplingProfiles::getAllProfiles();
        const int selIdx = profileCombo_.getSelectedId() - 1;
        if (selIdx >= 0 && selIdx < static_cast<int>(profiles.size()))
            s.lfoStratified = profiles[static_cast<size_t>(selIdx)].lfoStratified;
    }

    s.frameSelection = buildFrameSelection();
    const bool allOn = frameToggleOwner_.size() == 12 &&
                       [&] {
                           for (auto* b : frameToggleOwner_)
                               if (!b->getToggleState()) return false;
                           return true;
                       }();
    const bool allOff = [&] {
        for (auto* b : frameToggleOwner_)
            if (b->getToggleState()) return false;
        return true;
    }();
    if (allOff)
        s.renderMode = DataGen::RenderMode::FinalOnly;
    else if (allOn)
        s.renderMode = DataGen::RenderMode::AllFrames;
    else
        s.renderMode = DataGen::RenderMode::SelectedFrames;

    return s;
}


DataGen::FrameSelection MainComponent::buildFrameSelection() const
{
    DataGen::FrameSelection sel;
    for (int i = 0; i < frameToggleOwner_.size() && i < static_cast<int>(frameInternalNames_.size()); ++i) {
        const bool on = frameToggleOwner_[i]->getToggleState();
        const auto& name = frameInternalNames_[static_cast<size_t>(i)];
        if      (name == "osc_a")  sel.osc_a  = on;
        else if (name == "osc_b")  sel.osc_b  = on;
        else if (name == "noise")  sel.noise  = on;
        else if (name == "filter") sel.filter = on;
        else if (name == "env1")   sel.env1   = on;
        else if (name == "env2")   sel.env2   = on;
        else if (name == "lfo1")   sel.lfo1   = on;
        else if (name == "lfo2")   sel.lfo2   = on;
        else if (name == "lfo3")   sel.lfo3   = on;
        else if (name == "lfo4")   sel.lfo4   = on;
        else if (name == "reverb") sel.reverb = on;
        else if (name == "delay")  sel.delay  = on;
    }
    return sel;
}

DataGen::ParamConfig MainComponent::buildParamConfig(int idx) const
{
    const auto& row = *paramRows_[idx];
    DataGen::ParamConfig cfg;

    switch (row.strategyCombo.getSelectedId()) {
        case 1: cfg.strategy = DataGen::Strategy::RandomUniform; break;
        case 2: cfg.strategy = DataGen::Strategy::RandomLog;     break;
        case 3: cfg.strategy = DataGen::Strategy::Linear;        break;
        case 4: cfg.strategy = DataGen::Strategy::Fixed;         break;
        case 5: cfg.strategy = DataGen::Strategy::Range;         break;
        case 6: cfg.strategy = DataGen::Strategy::Discrete;      break;
        default: break;
    }

    cfg.rangeMin   = row.minBox.getText().getFloatValue();
    cfg.rangeMax   = row.maxBox.getText().getFloatValue();
    cfg.fixedVal   = row.fixedBox.getText().getFloatValue();
    cfg.numClasses = row.numClasses;

    return cfg;
}
