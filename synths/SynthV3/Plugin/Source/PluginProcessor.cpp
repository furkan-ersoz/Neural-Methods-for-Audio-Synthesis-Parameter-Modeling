#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "ParamCodec.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace juce;

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorValueTreeState::ParameterLayout
SynthV3Processor::createParameterLayout()
{
    std::vector<std::unique_ptr<RangedAudioParameter>> params;

    params.push_back(std::make_unique<AudioParameterFloat>(
        ParameterID{"amplitude", 1}, "amplitude",
        NormalisableRange<float>(0.0f, 1.0f), 0.8f));

    // Engine-internal, NOT predicted by the model — explicit plugin controls.
    params.push_back(std::make_unique<AudioParameterFloat>(
        ParameterID{"master_volume", 1}, "master_volume",
        NormalisableRange<float>(0.0f, 1.0f), 0.8f));
    params.push_back(std::make_unique<AudioParameterFloat>(
        ParameterID{"master_pan", 1}, "master_pan",
        NormalisableRange<float>(0.0f, 1.0f), 0.5f));
    params.push_back(std::make_unique<AudioParameterFloat>(
        ParameterID{"master_tune", 1}, "master_tune",
        NormalisableRange<float>(0.0f, 1.0f), 0.5f));

    auto labelSrc = std::make_unique<SynthV3::SynthEngine>();

    for (const auto& desc : SynthV3::PARAM_DESCRIPTORS) {
        const String name(desc.name.data());
        const String cls(desc.paramClass.data());

        if (cls == "categorical") {
            const auto labelsStd = labelSrc->getCategoricalLabels(std::string(desc.name));
            StringArray choices;
            for (const auto& l : labelsStd) choices.add(l);
            jassert(choices.size() == desc.numClasses);
            params.push_back(std::make_unique<AudioParameterChoice>(
                ParameterID{name, 1}, name, choices,
                ParamCodec::normToIndex(desc, desc.defaultVal)));
        } else if (cls == "integer") {
            params.push_back(std::make_unique<AudioParameterInt>(
                ParameterID{name, 1}, name, 0, desc.numClasses - 1,
                ParamCodec::normToIndex(desc, desc.defaultVal)));
        } else {
            params.push_back(std::make_unique<AudioParameterFloat>(
                ParameterID{name, 1}, name,
                NormalisableRange<float>(desc.minVal, desc.maxVal), desc.defaultVal));
        }
    }

    return { params.begin(), params.end() };
}

// ─────────────────────────────────────────────────────────────────────────────
SynthV3Processor::SynthV3Processor()
    : AudioProcessor(BusesProperties()
          .withOutput("Output", AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "STATE", createParameterLayout())
{
    // Bypass params — "active" semantics: 1.0 = frame active, 0.0 = bypassed.
    // Defaults mirror BypassFlags defaults (lfo1-4, reverb, delay default bypassed).
    auto addBP = [&](const char* id, bool defaultActive) -> AudioParameterBool* {
        auto* p = new AudioParameterBool({id, 1}, id, defaultActive);
        addParameter(p);
        return p;
    };
    bypassOscB_   = addBP("bypass_osc_b",   true);
    bypassNoise_  = addBP("bypass_noise",   true);
    bypassFilter_ = addBP("bypass_filter",  true);
    bypassLfo1_   = addBP("bypass_lfo1",    false);
    bypassLfo2_   = addBP("bypass_lfo2",    false);
    bypassLfo3_   = addBP("bypass_lfo3",    false);
    bypassLfo4_   = addBP("bypass_lfo4",    false);
    bypassReverb_ = addBP("bypass_reverb",  false);
    bypassDelay_  = addBP("bypass_delay",   false);

    jassert(static_cast<int>(SynthV3::PARAM_DESCRIPTORS.size()) == SynthV3::NUM_PARAMS);

    const auto modelDir = findModelFile();
    if (modelDir.isDirectory())
        inferenceEngine_.loadModel(modelDir);
    else
        Logger::writeToLog("SynthV3: ONNX exports directory not found: "
                           + modelDir.getFullPathName());
}

// ─────────────────────────────────────────────────────────────────────────────
AudioParameterBool* SynthV3Processor::getBypassParam(const String& frame) const
{
    if (frame == "osc_b")  return bypassOscB_;
    if (frame == "noise")  return bypassNoise_;
    if (frame == "filter") return bypassFilter_;
    if (frame == "lfo1")   return bypassLfo1_;
    if (frame == "lfo2")   return bypassLfo2_;
    if (frame == "lfo3")   return bypassLfo3_;
    if (frame == "lfo4")   return bypassLfo4_;
    if (frame == "reverb") return bypassReverb_;
    if (frame == "delay")  return bypassDelay_;
    return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV3Processor::getStateInformation(MemoryBlock& destData)
{
    auto state = apvts_.copyState();
    auto xml   = state.createXml();
    if (!xml) return;

    // Append bypass states as attributes
    auto getBP = [](AudioParameterBool* p) { return p ? (int)p->get() : 1; };
    xml->setAttribute("bypass_osc_b",   getBP(bypassOscB_));
    xml->setAttribute("bypass_noise",   getBP(bypassNoise_));
    xml->setAttribute("bypass_filter",  getBP(bypassFilter_));
    xml->setAttribute("bypass_lfo1",    getBP(bypassLfo1_));
    xml->setAttribute("bypass_lfo2",    getBP(bypassLfo2_));
    xml->setAttribute("bypass_lfo3",    getBP(bypassLfo3_));
    xml->setAttribute("bypass_lfo4",    getBP(bypassLfo4_));
    xml->setAttribute("bypass_reverb",  getBP(bypassReverb_));
    xml->setAttribute("bypass_delay",   getBP(bypassDelay_));

    copyXmlToBinary(*xml, destData);
}

void SynthV3Processor::setStateInformation(const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary(data, sizeInBytes);
    if (!xml) return;

    apvts_.replaceState(ValueTree::fromXml(*xml));

    auto loadBP = [&](const char* id, AudioParameterBool* p, bool def) {
        if (p && xml->hasAttribute(id))
            p->setValueNotifyingHost(xml->getIntAttribute(id, def ? 1 : 0) ? 1.0f : 0.0f);
    };
    loadBP("bypass_osc_b",   bypassOscB_,   true);
    loadBP("bypass_noise",   bypassNoise_,  true);
    loadBP("bypass_filter",  bypassFilter_, true);
    loadBP("bypass_lfo1",    bypassLfo1_,   false);
    loadBP("bypass_lfo2",    bypassLfo2_,   false);
    loadBP("bypass_lfo3",    bypassLfo3_,   false);
    loadBP("bypass_lfo4",    bypassLfo4_,   false);
    loadBP("bypass_reverb",  bypassReverb_, false);
    loadBP("bypass_delay",   bypassDelay_,  false);
}

// ─────────────────────────────────────────────────────────────────────────────
juce::File SynthV3Processor::findModelFile() const
{
    // Walk up from bundle dir looking for Experiments/*/exports/
    juce::File dir = File::getSpecialLocation(
                         File::currentApplicationFile).getParentDirectory();

    for (int i = 0; i < 10; ++i) {
        auto candidate = dir.getChildFile("synths/SynthV3/Experiments/Exp003/exports");
        if (candidate.isDirectory()
            && candidate.getChildFile("model002_door.onnx").existsAsFile())
            return candidate;
        dir = dir.getParentDirectory();
    }

    return juce::File(); // not found
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV3Processor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    for (auto& v : voices_)
        v.engine.prepare(sampleRate, samplesPerBlock);
}

// ─────────────────────────────────────────────────────────────────────────────
SynthV3::SynthParameters SynthV3Processor::paramsFromAPVTS() const
{
    SynthV3::SynthParameters p;
    p.amplitude    = *apvts_.getRawParameterValue("amplitude");
    p.masterVolume = *apvts_.getRawParameterValue("master_volume");
    p.masterPan    = *apvts_.getRawParameterValue("master_pan");
    p.masterTune   = *apvts_.getRawParameterValue("master_tune");

    std::vector<float> vals;
    vals.reserve(SynthV3::NUM_PARAMS);
    for (const auto& desc : SynthV3::PARAM_DESCRIPTORS)
        vals.push_back(apvts_.getParameter(juce::String(desc.name.data()))->getValue());
    p.fromVector(vals);

    return p;
}

SynthV3::BypassFlags SynthV3Processor::bypassFromParams() const
{
    SynthV3::BypassFlags b;
    // "active" param: 1=active(not bypassed), 0=bypassed → BypassFlags uses true=bypassed
    auto isB = [](AudioParameterBool* p, bool def) {
        return p ? !p->get() : def;
    };
    b.osc_b  = isB(bypassOscB_,   false);
    b.noise  = isB(bypassNoise_,  false);
    b.filter = isB(bypassFilter_, false);
    b.lfo1   = isB(bypassLfo1_,   true);
    b.lfo2   = isB(bypassLfo2_,   true);
    b.lfo3   = isB(bypassLfo3_,   true);
    b.lfo4   = isB(bypassLfo4_,   true);
    b.reverb = isB(bypassReverb_, true);
    b.delay  = isB(bypassDelay_,  true);
    return b;
}

// ─────────────────────────────────────────────────────────────────────────────
int SynthV3Processor::findFreeVoice() const
{
    // Pass 1: inactive voice
    for (size_t i = 0; i < kNumVoices; ++i)
        if (!voices_[i].active) return static_cast<int>(i);

    // Pass 2: oldest fading-out non-ghost voice
    int   fadeIdx = -1;
    int   fadeAge = INT_MAX;
    for (size_t i = 0; i < kNumVoices; ++i) {
        if (voices_[i].isGhost) continue;
        if (voices_[i].engine.isFadingOut() && voices_[i].ageCounter < fadeAge) {
            fadeAge = voices_[i].ageCounter;
            fadeIdx = static_cast<int>(i);
        }
    }
    if (fadeIdx >= 0) return fadeIdx;

    // Pass 3: lowest envelope non-ghost voice
    size_t best      = 0;
    float  minEnv    = std::numeric_limits<float>::max();
    bool   foundNonGhost = false;
    for (size_t i = 0; i < kNumVoices; ++i) {
        if (voices_[i].isGhost) continue;
        foundNonGhost = true;
        if (voices_[i].engine.getEnvelope() < minEnv) {
            minEnv = voices_[i].engine.getEnvelope();
            best   = i;
        }
    }
    // Last-resort: all voices are ghost — pick lowest envelope ignoring ghost flag
    if (!foundNonGhost) {
        for (size_t i = 0; i < kNumVoices; ++i) {
            if (voices_[i].engine.getEnvelope() < minEnv) {
                minEnv = voices_[i].engine.getEnvelope();
                best   = i;
            }
        }
    }
    return static_cast<int>(best);
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV3Processor::processBlock(AudioBuffer<float>& buffer,
                                    MidiBuffer& midiMessages)
{
    ScopedNoDenormals noDenormals;
    buffer.clear();

    const auto params     = paramsFromAPVTS();
    const auto bypass     = bypassFromParams();
    const int  numSamples = buffer.getNumSamples();
    auto* left  = buffer.getWritePointer(0);
    auto* right = buffer.getWritePointer(1);

    for (auto& v : voices_)
        v.engine.setBypass(bypass);

    std::vector<float> voiceL(static_cast<size_t>(numSamples), 0.0f);
    std::vector<float> voiceR(static_cast<size_t>(numSamples), 0.0f);

    int cursor = 0;
    const float voiceGain = 1.0f / static_cast<float>(kNumVoices);

    auto renderUntil = [&](int endSample) {
        if (endSample <= cursor) return;
        const int segLen = endSample - cursor;

        for (auto& v : voices_) {
            if (!v.active) continue;

if (!v.isGhost && v.engine.isEnvelopeDone() && !v.pendingEmpty()) {
                auto freeIdx = static_cast<size_t>(findFreeVoice());
                if (voices_[freeIdx].isGhost) {
                    // All non-ghost voices busy — defer to next block, leave queue intact
                } else {
                    v.isGhost = true;
                    int nextNote  = v.pendingPop();
                    auto& nv      = voices_[freeIdx];
                    nv.engine.reset();
                    nv.engine.setParameters(params);
                    nv.midiNote    = nextNote;
                    nv.noteOn      = true;
                    nv.active      = true;
                    nv.isGhost     = false;
                    nv.pendingHead = nv.pendingTail = nv.pendingCount = 0;
                    nv.ageCounter  = blockCounter_;
                }
            }

            v.engine.setParameters(params);
            std::fill(voiceL.begin(), voiceL.begin() + segLen, 0.0f);
            std::fill(voiceR.begin(), voiceR.begin() + segLen, 0.0f);
            v.engine.processStereo(voiceL.data(), voiceR.data(), segLen, v.noteOn,
                                   SynthV3::midiToHz(v.midiNote));

            if (v.isGhost && v.engine.isIdle())
                v.active = false;

            if (!v.isGhost && !v.noteOn && v.engine.isIdle() && v.pendingEmpty())
                v.active = false;

            for (int i = 0; i < segLen; ++i) {
                const auto ui = static_cast<size_t>(i);
                left [cursor + i] += voiceL[ui] * voiceGain;
                right[cursor + i] += voiceR[ui] * voiceGain;
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
            const bool voiceIsAvailable = !v.active || v.engine.isEnvelopeDone();

            if (!voiceIsAvailable) {
                v.engine.startFadeOut();
                v.noteOn = false;
                v.pendingPush(msg.getNoteNumber());
                v.active = true;
                // ageCounter NOT updated — keep original age for steal priority
            } else {
                v.engine.reset();
                v.midiNote    = msg.getNoteNumber();
                v.noteOn      = true;
                v.isGhost     = false;
                v.pendingHead = 0;
                v.pendingTail = 0;
                v.pendingCount = 0;
                v.active      = true;
                v.ageCounter  = blockCounter_;
            }
        }
        else if (msg.isNoteOff()) {
            const int note = msg.getNoteNumber();
            for (auto& v : voices_) {
                if (v.active && v.noteOn && v.midiNote == note) {
                    v.noteOn = false;
                    break;
                }
            }
        }
    }

    renderUntil(numSamples);

    {
        const float vol  = params.masterVolume;
        const float pan  = params.masterPan;
        const float panL = std::cos((pan - 0.5f) * static_cast<float>(M_PI) * 0.5f + static_cast<float>(M_PI) * 0.25f);
        const float panR = std::sin((pan - 0.5f) * static_cast<float>(M_PI) * 0.5f + static_cast<float>(M_PI) * 0.25f);
        for (int i = 0; i < numSamples; ++i) {
            left[i]  *= vol * panL * 2.0f;
            right[i] *= vol * panR * 2.0f;
        }
    }

    ++blockCounter_;
}

// ─────────────────────────────────────────────────────────────────────────────
void SynthV3Processor::applyInferenceResult(const std::vector<float>& params)
{
    for (size_t i = 0; i < SynthV3::PARAM_DESCRIPTORS.size() && i < params.size(); ++i) {
        if (auto* p = apvts_.getParameter(SynthV3::PARAM_DESCRIPTORS[i].name.data()))
            p->setValueNotifyingHost(params[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
AudioProcessorEditor* SynthV3Processor::createEditor()
{
    return new SynthV3Editor(*this);
}

AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SynthV3Processor();
}
