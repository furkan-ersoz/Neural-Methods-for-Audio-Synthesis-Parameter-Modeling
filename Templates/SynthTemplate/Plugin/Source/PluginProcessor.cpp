#include "PluginProcessor.h"
#include "PluginEditor.h"

using namespace juce;

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorValueTreeState::ParameterLayout
SynthTemplateProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<RangedAudioParameter>> params;

    auto add = [&](const char* id, float def) {
        params.push_back(std::make_unique<AudioParameterFloat>(
            ParameterID{id, 1}, id,
            NormalisableRange<float>(0.0f, 1.0f), def));
    };

    add("amplitude", 0.8f);
    add("attack",    0.05f);
    add("decay",     0.2f);
    add("sustain",   0.7f);
    add("release",   0.3f);

    return { params.begin(), params.end() };
}

// ─────────────────────────────────────────────────────────────────────────────
SynthTemplateProcessor::SynthTemplateProcessor()
    : AudioProcessor(BusesProperties()
          .withOutput("Output", AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "STATE", createParameterLayout())
{
    // Model dosyasini bul ve yukle
    const auto modelFile = findModelFile();
    if (modelFile.existsAsFile())
        inferenceEngine_.loadModel(modelFile);
    else
        Logger::writeToLog("SynthTemplate: ONNX model not found at: "
                           + modelFile.getFullPathName());
}

// ─────────────────────────────────────────────────────────────────────────────
juce::File SynthTemplateProcessor::findModelFile() const
{
    // Arama sirasi:
    // 1. Plugin bundle icindeki Resources/ klasoru (Release)
    // 2. Executable yanindaki klasor (Debug/standalone)
    // 3. Proje kokunden goreceli yol (gelistirme)

    const auto bundleDir = File::getSpecialLocation(
                               File::currentApplicationFile).getParentDirectory();

    const juce::String modelName = "SynthTemplateModel.onnx";

    // 1. Bundle Resources
    auto candidate = bundleDir.getChildFile("Resources").getChildFile(modelName);
    if (candidate.existsAsFile()) return candidate;

    // 2. Executable yaninda
    candidate = bundleDir.getChildFile(modelName);
    if (candidate.existsAsFile()) return candidate;

    // 3. Proje kokunden: synths/SynthV1/exports/
    auto search = bundleDir;
    for (int up = 0; up < 8; ++up) {
        auto exp = search.getChildFile("synths")
                         .getChildFile("SynthV1")
                         .getChildFile("exports");
        if (exp.isDirectory()) {
            // exports/ icindeki ilk .onnx dosyasini al
            Array<File> onnxFiles;
            exp.findChildFiles(onnxFiles, File::findFiles, false, "*.onnx");
            if (! onnxFiles.isEmpty())
                return onnxFiles[0];
        }
        search = search.getParentDirectory();
    }

    return bundleDir.getChildFile(modelName);  // bulunamazsa bos doner
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engine_.prepare(sampleRate, samplesPerBlock);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateProcessor::syncEngineFromAPVTS()
{
    SynthTemplate::SynthParameters p;
    p.amplitude    = *apvts_.getRawParameterValue("amplitude");
    p.attackTime   = *apvts_.getRawParameterValue("attack");
    p.decayTime    = *apvts_.getRawParameterValue("decay");
    p.sustainLevel = *apvts_.getRawParameterValue("sustain");
    p.releaseTime  = *apvts_.getRawParameterValue("release");
    engine_.setParameters(p);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateProcessor::processBlock(AudioBuffer<float>& buffer,
                                          MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;
    buffer.clear();

    for (const auto meta : midiMessages) {
        const auto msg = meta.getMessage();
        if (msg.isNoteOn())  { noteOn_ = true;  midiNote_ = msg.getNoteNumber(); }
        if (msg.isNoteOff()) { noteOn_ = false; }
    }

    syncEngineFromAPVTS();

    const float frequencyHz = SynthTemplate::midiToHz(midiNote_);
    auto* left  = buffer.getWritePointer(0);
    auto* right = buffer.getWritePointer(1);
    const int numSamples = buffer.getNumSamples();

    engine_.process(left, numSamples, noteOn_, frequencyHz);
    FloatVectorOperations::copy(right, left, numSamples);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthTemplateProcessor::applyInferenceResult(const std::vector<float>& params)
{
    // Parametre sirasi: attack, decay, sustain, release (amplitude degismez)
    // inference engine paramNames_ ile eslesmeli
    const std::vector<const char*> ids = {"attack", "decay", "sustain", "release"};

    for (size_t i = 0; i < ids.size() && i < params.size(); ++i) {
        if (auto* p = apvts_.getParameter(ids[i]))
            p->setValueNotifyingHost(params[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorEditor* SynthTemplateProcessor::createEditor()
{
    return new SynthTemplateEditor(*this);
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SynthTemplateProcessor();
}
