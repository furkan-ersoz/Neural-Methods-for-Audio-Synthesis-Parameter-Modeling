#include "PluginProcessor.h"
#include "PluginEditor.h"

using namespace juce;

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorValueTreeState::ParameterLayout
SynthV1Processor::createParameterLayout()
{
    std::vector<std::unique_ptr<RangedAudioParameter>> params;

    // Amplitude is internal — always present, never predicted.
    params.push_back(std::make_unique<AudioParameterFloat>(
        ParameterID{"amplitude", 1}, "amplitude",
        NormalisableRange<float>(0.0f, 1.0f), 0.8f));

    // All predicted parameters are driven by PARAM_DESCRIPTORS — no edits needed here.
    for (const auto& desc : SynthV1::PARAM_DESCRIPTORS)
        params.push_back(std::make_unique<AudioParameterFloat>(
            ParameterID{desc.name.data(), 1}, desc.name.data(),
            NormalisableRange<float>(desc.minVal, desc.maxVal), desc.defaultVal));

    return { params.begin(), params.end() };
}

// ─────────────────────────────────────────────────────────────────────────────
SynthV1Processor::SynthV1Processor()
    : AudioProcessor(BusesProperties()
          .withOutput("Output", AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "STATE", createParameterLayout())
{
    const auto modelFile = findModelFile();
    if (modelFile.existsAsFile())
        inferenceEngine_.loadModel(modelFile);
    else
        Logger::writeToLog("SynthV1: ONNX model not found at: "
                           + modelFile.getFullPathName());
}

// ─────────────────────────────────────────────────────────────────────────────
juce::File SynthV1Processor::findModelFile() const
{
    const auto bundleDir = File::getSpecialLocation(
                               File::currentApplicationFile).getParentDirectory();

    const juce::String modelName = "SynthV1Model.onnx";

    auto candidate = bundleDir.getChildFile("Resources").getChildFile(modelName);
    if (candidate.existsAsFile()) return candidate;

    candidate = bundleDir.getChildFile(modelName);
    if (candidate.existsAsFile()) return candidate;

    auto search = bundleDir;
    for (int up = 0; up < 8; ++up) {
        auto exp = search.getChildFile("synths")
                         .getChildFile("SynthV1")
                         .getChildFile("exports");
        if (exp.isDirectory()) {
            Array<File> onnxFiles;
            exp.findChildFiles(onnxFiles, File::findFiles, false, "*.onnx");
            if (!onnxFiles.isEmpty()) return onnxFiles[0];
        }
        search = search.getParentDirectory();
    }

    return bundleDir.getChildFile(modelName);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV1Processor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    for (auto& v : voices_)
        v.engine.prepare(sampleRate, samplesPerBlock);
}

// ─────────────────────────────────────────────────────────────────────────────
SynthV1::SynthParameters SynthV1Processor::paramsFromAPVTS() const
{
    SynthV1::SynthParameters p;
    p.amplitude = *apvts_.getRawParameterValue("amplitude");

    // Predicted params: loop over PARAM_DESCRIPTORS, collect values, push into struct via fromVector.
    std::vector<float> vals;
    vals.reserve(SynthV1::NUM_PARAMS);
    for (const auto& desc : SynthV1::PARAM_DESCRIPTORS)
        vals.push_back(*apvts_.getRawParameterValue(desc.name.data()));
    p.fromVector(vals);

    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
int SynthV1Processor::findFreeVoice() const
{
    for (size_t i = 0; i < kNumVoices; ++i)
        if (!voices_[i].active) return static_cast<int>(i);

    // Voice steal: prefer the voice already fading out, otherwise the quietest.
    size_t best = 0;
    float  minEnv = voices_[0].engine.getEnvelope();
    for (size_t i = 0; i < kNumVoices; ++i) {
        if (voices_[i].engine.isFadingOut()) return static_cast<int>(i);
        if (voices_[i].engine.getEnvelope() < minEnv) {
            minEnv = voices_[i].engine.getEnvelope();
            best   = i;
        }
    }
    return static_cast<int>(best);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV1Processor::processBlock(AudioBuffer<float>& buffer,
                                          MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;
    buffer.clear();

    const auto params    = paramsFromAPVTS();
    const int  numSamples = buffer.getNumSamples();
    auto* left  = buffer.getWritePointer(0);
    auto* right = buffer.getWritePointer(1);

    std::vector<float> monoBuf(static_cast<size_t>(numSamples), 0.0f);

    // Process audio in segments separated by MIDI event timestamps so that
    // note-on/off take effect at their exact sample position within the block.
    int cursor = 0;

    const float voiceGain = 1.0f / static_cast<float>(kNumVoices);

    auto renderUntil = [&](int endSample) {
        if (endSample <= cursor) return;
        const int segLen = endSample - cursor;

        for (auto& v : voices_) {
            if (!v.active) continue;

            // Fade finished — start the pending note now at this segment boundary.
            if (v.engine.isIdle() && v.hasPendingNote) {
                v.engine.reset();
                v.engine.setParameters(params);
                v.midiNote       = v.pendingMidiNote;
                v.noteOn         = true;
                v.hasPendingNote = false;
            }

            v.engine.setParameters(params);
            std::fill(monoBuf.begin(), monoBuf.begin() + segLen, 0.0f);
            v.engine.process(monoBuf.data(), segLen, v.noteOn,
                             SynthV1::midiToHz(v.midiNote));

            if (!v.noteOn && v.engine.isIdle() && !v.hasPendingNote)
                v.active = false;

            for (int i = 0; i < segLen; ++i) {
                const auto ui = static_cast<size_t>(i);
                left [cursor + i] += monoBuf[ui] * voiceGain;
                right[cursor + i] += monoBuf[ui] * voiceGain;
            }
        }
        cursor = endSample;
    };

    for (const auto meta : midiMessages) {
        renderUntil(meta.samplePosition);

        const auto msg = meta.getMessage();

        if (msg.isNoteOn()) {
            const auto idx = static_cast<size_t>(findFreeVoice());
            auto& v = voices_[idx];
            v.engine.setParameters(params);

            if (v.active && !v.engine.isIdle()) {
                // Voice is still audible — fade it out, queue the new note.
                v.engine.startFadeOut();
                v.noteOn          = false;
                v.hasPendingNote  = true;
                v.pendingMidiNote = msg.getNoteNumber();
            } else {
                // Voice is free (or already idle) — start immediately.
                v.engine.reset();
                v.midiNote       = msg.getNoteNumber();
                v.noteOn         = true;
                v.hasPendingNote = false;
            }
            v.active     = true;
            v.ageCounter = blockCounter_;
        }
        else if (msg.isNoteOff()) {
            for (auto& v : voices_) {
                if (v.active && v.noteOn && v.midiNote == msg.getNoteNumber()) {
                    v.noteOn = false;
                    break;
                }
            }
        }
    }

    renderUntil(numSamples);

    ++blockCounter_;
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV1Processor::applyInferenceResult(const std::vector<float>& params)
{
    for (size_t i = 0; i < SynthV1::PARAM_DESCRIPTORS.size() && i < params.size(); ++i) {
        if (auto* p = apvts_.getParameter(SynthV1::PARAM_DESCRIPTORS[i].name.data()))
            p->setValueNotifyingHost(params[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorEditor* SynthV1Processor::createEditor()
{
    return new SynthV1Editor(*this);
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SynthV1Processor();
}
