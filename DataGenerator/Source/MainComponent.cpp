#include "MainComponent.h"

using namespace juce;

MainComponent::MainComponent()
{
    setSize(640, 620);

    addAndMakeVisible(synthLabel_);
    addAndMakeVisible(synthCombo_);
    populateSynthCombo();

    addAndMakeVisible(outputLabel_);
    outputPathBox_.setReadOnly(true);

    // Default output: traverse up from executable to find synths/SynthV1/datasets/
    {
        auto candidate = juce::File::getSpecialLocation(
                             juce::File::currentApplicationFile).getParentDirectory();
        bool found = false;
        for (int up = 0; up < 8 && !found; ++up) {
            auto synthsDir = candidate.getChildFile("synths");
            // cmake-build-debug icinde sahte synths/ olusabilir — onu atla
            const auto parentName = candidate.getFileName();
            const bool isBuildDir = parentName.containsIgnoreCase("cmake-build")
                                 || parentName.containsIgnoreCase("build");
            if (synthsDir.isDirectory() && !isBuildDir) {
                auto datasets = synthsDir.getChildFile("SynthV1")
                                         .getChildFile("datasets");
                datasets.createDirectory();
                outputPathBox_.setText(datasets.getFullPathName());
                found = true;
            }
            candidate = candidate.getParentDirectory();
        }
        if (!found) {
            outputPathBox_.setText("(not selected)");
            appendLog("WARNING: synths/ folder not found. Please select output manually.");
        } else {
            appendLog("Default output: " + outputPathBox_.getText());
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
    for (auto* e : { &nBox_, &seedBox_, &maxDurBox_ })
        e->onTextChange = [this] { updateDiskEstimate(); };
    addAndMakeVisible(nLabel_);      addAndMakeVisible(nBox_);
    addAndMakeVisible(seedLabel_);   addAndMakeVisible(seedBox_);
    addAndMakeVisible(maxDurLabel_); addAndMakeVisible(maxDurBox_);

    for (int i = 0; i < kNumParams; ++i) {
        auto& row = paramRows_[(size_t)i];

        row.nameLabel.setText(kParamNames[i], dontSendNotification);
        row.nameLabel.setFont(FontOptions(14.0f).withStyle("Bold"));

        row.strategyCombo.addItem("Random Uniform", 1);
        row.strategyCombo.addItem("Random Log",     2);
        row.strategyCombo.addItem("Linear",         3);
        row.strategyCombo.addItem("Fixed",          4);
        row.strategyCombo.addItem("Range",          5);
        row.strategyCombo.setSelectedId(1);

        row.minBox.setText("0.000000");
        row.maxBox.setText("1.000000");
        row.fixedBox.setText("0.500000");

        row.strategyCombo.onChange = [this, i] {
            auto& r = paramRows_[(size_t)i];
            const bool isFixed = (r.strategyCombo.getSelectedId() == 4);
            r.minLabel.setVisible(! isFixed);
            r.minBox.setVisible(! isFixed);
            r.maxLabel.setVisible(! isFixed);
            r.maxBox.setVisible(! isFixed);
            r.fixedLabel.setVisible(isFixed);
            r.fixedBox.setVisible(isFixed);
            resized();
        };
        row.strategyCombo.onChange();

        addAndMakeVisible(row.nameLabel);
        addAndMakeVisible(row.strategyCombo);
        addAndMakeVisible(row.minLabel);
        addAndMakeVisible(row.minBox);
        addAndMakeVisible(row.maxLabel);
        addAndMakeVisible(row.maxBox);
        addAndMakeVisible(row.fixedLabel);
        addAndMakeVisible(row.fixedBox);
    }

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

    updateDiskEstimate();
    startTimer(100);
}

MainComponent::~MainComponent()
{
    stopTimer();
    writer_.stopAndWait();
}

void MainComponent::paint(Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(ResizableWindow::backgroundColourId));

    g.setColour(Colours::white.withAlpha(0.1f));
    const int paramTop = 132;
    const int paramH   = kNumParams * 52 + 8;
    g.fillRoundedRectangle(8.0f, (float)paramTop,
                           (float)(getWidth() - 16), (float)paramH, 6.0f);

    g.setColour(Colours::lightgrey);
    g.setFont(FontOptions(12.0f));
    g.drawText("Parameters", 16, paramTop + 2, 120, 16, Justification::left);
}

void MainComponent::resized()
{
    const int pad = 8;
    const int W   = getWidth() - 2 * pad;
    int y         = pad;

    synthLabel_.setBounds(pad, y, 60, 24);
    synthCombo_.setBounds(pad + 64, y, 200, 24);
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
    y += 32;

    y += 18;
    for (int i = 0; i < kNumParams; ++i) {
        auto& row      = paramRows_[(size_t)i];
        const int rowY = y + i * 52;
        const bool isFixed = (row.strategyCombo.getSelectedId() == 4);

        row.nameLabel.setBounds(pad + 4, rowY, 70, 20);
        row.strategyCombo.setBounds(pad + 78, rowY, 140, 20);

        if (isFixed) {
            row.fixedLabel.setBounds(pad + 228, rowY, 44, 20);
            row.fixedBox.setBounds(pad + 276, rowY, 80, 20);
        } else {
            row.minLabel.setBounds(pad + 228, rowY, 30, 20);
            row.minBox.setBounds(pad + 262, rowY, 80, 20);
            row.maxLabel.setBounds(pad + 350, rowY, 30, 20);
            row.maxBox.setBounds(pad + 384, rowY, 80, 20);
        }
    }
    y += kNumParams * 52 + 8;

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
        progressVal_ = (double)cur / (double)total;
        progressLabel_.setText(String(cur) + "/" + String(total), dontSendNotification);
    }

    if (jobFinished_.exchange(false)) {
        startBtn_.setEnabled(true);
        stopBtn_.setEnabled(false);
        if (jobSuccess_.load()) progressVal_ = 1.0;
    }
}

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

void MainComponent::updateDiskEstimate()
{
    const int   n      = nBox_.getText().getIntValue();
    const float maxDur = maxDurBox_.getText().getFloatValue();
    const float mb     = (float)n * maxDur * 44100.0f * 2.0f / (1024.0f * 1024.0f);
    diskEstLabel_.setText(String::formatted("Estimated max size: %.1f MB", mb),
                          dontSendNotification);
}

void MainComponent::appendLog(const String& msg)
{
    ScopedLock sl(logMutex_);
    pendingLogs_.add(msg);
}

DataGen::WriterSettings MainComponent::buildSettings() const
{
    DataGen::WriterSettings s;
    s.outputDir   = File(outputPathBox_.getText());
    s.numSamples  = nBox_.getText().getIntValue();
    s.baseSeed    = (uint32_t)seedBox_.getText().getIntValue();
    s.maxDuration = maxDurBox_.getText().getFloatValue();

    for (int i = 0; i < kNumParams; ++i)
        s.paramConfigs[(size_t)i] = buildParamConfig(i);

    return s;
}

DataGen::ParamConfig MainComponent::buildParamConfig(int idx) const
{
    const auto& row = paramRows_[(size_t)idx];
    DataGen::ParamConfig cfg;

    switch (row.strategyCombo.getSelectedId()) {
        case 1: cfg.strategy = DataGen::Strategy::RandomUniform; break;
        case 2: cfg.strategy = DataGen::Strategy::RandomLog;     break;
        case 3: cfg.strategy = DataGen::Strategy::Linear;        break;
        case 4: cfg.strategy = DataGen::Strategy::Fixed;         break;
        case 5: cfg.strategy = DataGen::Strategy::Range;         break;
        default: break;
    }

    cfg.rangeMin = row.minBox.getText().getFloatValue();
    cfg.rangeMax = row.maxBox.getText().getFloatValue();
    cfg.fixedVal = row.fixedBox.getText().getFloatValue();

    return cfg;
}

void MainComponent::populateSynthCombo()
{
    synthCombo_.addItem("SynthV1", 1);
    synthCombo_.setSelectedId(1);
}