#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <mutex>
#include <thread>

#include "DatasetWriter.h"
#include "ISynth.h"
#include "ParamConfigResolver.h"
#include "SamplingProfiles/AllProfiles.h"

// ─── Minimal YAML parser ──────────────────────────────────────────────────────
// Handles:  key: value   (top-level and one indented block)
//           # comments
//           blank lines
// No multi-doc, no anchors, no flow syntax.

struct YamlNode {
    std::map<std::string, std::string>              scalars;
    std::map<std::string, std::map<std::string,std::string>> sections;
};

static std::string trimStr(const std::string& s) {
    auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static YamlNode parseYaml(const juce::File& file) {
    YamlNode node;
    auto lines = juce::StringArray::fromLines(file.loadFileAsString());
    std::string currentSection;

    for (auto& juceL : lines) {
        std::string raw = juceL.toStdString();

        // strip comment
        auto commentPos = raw.find('#');
        if (commentPos != std::string::npos) raw = raw.substr(0, commentPos);

        if (trimStr(raw).empty()) continue;

        bool indented = (!raw.empty() && (raw[0] == ' ' || raw[0] == '\t'));
        auto colon = raw.find(':');
        if (colon == std::string::npos) continue;

        std::string key = trimStr(raw.substr(0, colon));
        std::string val = trimStr(raw.substr(colon + 1));

        if (indented && !currentSection.empty()) {
            node.sections[currentSection][key] = val;
        } else {
            if (val.empty()) {
                currentSection = key;
            } else {
                currentSection.clear();
                node.scalars[key] = val;
            }
        }
    }
    return node;
}

// ─── Config helpers ───────────────────────────────────────────────────────────

static bool parseBool(const std::string& s, bool def) {
    if (s == "true"  || s == "yes" || s == "1") return true;
    if (s == "false" || s == "no"  || s == "0") return false;
    return def;
}

static float parseFloat(const std::string& s, float def) {
    try { return std::stof(s); } catch (...) { return def; }
}

static int parseInt(const std::string& s, int def) {
    try { return std::stoi(s); } catch (...) { return def; }
}

static std::string getScalar(const YamlNode& y, const std::string& key,
                              const std::string& def, bool& warned) {
    auto it = y.scalars.find(key);
    if (it == y.scalars.end()) {
        fprintf(stderr, "[warn] config missing '%s', using default '%s'\n",
                key.c_str(), def.c_str());
        warned = true;
        return def;
    }
    return it->second;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // ── Parse CLI args ────────────────────────────────────────────────────────
    juce::File configFile;
    int  cliN    = -1;
    int  cliSeed = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config") && i + 1 < argc) {
            configFile = juce::File(juce::String(argv[++i]));
        } else if ((arg == "--n") && i + 1 < argc) {
            try { cliN = std::stoi(argv[++i]); } catch (...) {}
        } else if ((arg == "--seed") && i + 1 < argc) {
            try { cliSeed = std::stoi(argv[++i]); } catch (...) {}
        }
    }

    if (!configFile.existsAsFile()) {
        fprintf(stderr,
            "Usage: DataGeneratorCLI --config /path/to/datagen_config.yaml"
            " [--n <count>] [--seed <seed>]\n");
        return 1;
    }

    // ── Parse YAML ────────────────────────────────────────────────────────────
    YamlNode cfg = parseYaml(configFile);
    bool warnedDefaults = false;

    std::string synthName   = getScalar(cfg, "synth",      "SynthV3",       warnedDefaults);
    std::string profileName = getScalar(cfg, "profile",   "DataGeneration", warnedDefaults);
    int    numSamples  = parseInt  (getScalar(cfg, "n",            "1000", warnedDefaults), 1000);
    int    seed        = parseInt  (getScalar(cfg, "seed",         "42",   warnedDefaults), 42);
    float  maxDuration = parseFloat(getScalar(cfg, "max_duration", "4.0",  warnedDefaults), 4.0f);
    int    numWorkers  = parseInt  (getScalar(cfg, "num_workers",
                              std::to_string(std::thread::hardware_concurrency()), warnedDefaults),
                              static_cast<int>(std::thread::hardware_concurrency()));
    // CLI overrides
    if (cliN    >= 0) numSamples = cliN;
    if (cliSeed >= 0) seed       = cliSeed;

    // Resolve output dir: walk up from config file to find synths/{synthName}/datasets/
    juce::File resolvedOutputRoot;
    {
        auto candidate = configFile.getParentDirectory();
        for (int up = 0; up < 10; ++up) {
            auto synthsDir = candidate.getChildFile("synths");
            if (synthsDir.isDirectory()) {
                resolvedOutputRoot = synthsDir.getChildFile(juce::String(synthName))
                                              .getChildFile("datasets");
                resolvedOutputRoot.createDirectory();
                break;
            }
            candidate = candidate.getParentDirectory();
        }
        if (resolvedOutputRoot == juce::File{}) {
            fprintf(stderr, "[error] Could not locate synths/%s/datasets/ from config path.\n",
                    synthName.c_str());
            return 1;
        }
    }

    // ── Resolve synth factory ─────────────────────────────────────────────────
    auto& registry = SynthRegistry::instance();
    auto  names    = registry.getNames();
    if (std::find(names.begin(), names.end(), synthName) == names.end()) {
        fprintf(stderr, "[error] Synth '%s' not found in SynthRegistry.\n"
                        "        Available: ", synthName.c_str());
        for (auto& n : names) fprintf(stderr, "%s ", n.c_str());
        fprintf(stderr, "\n");
        return 1;
    }

    // ── Build WriterSettings ──────────────────────────────────────────────────
    // Resolve sampling profile
    auto allProfiles = SamplingProfiles::getAllProfiles();
    SamplingProfiles::SamplingProfile* resolvedProfile = nullptr;
    for (auto& p : allProfiles) {
        if (std::string(p.name) == profileName) {
            resolvedProfile = &p;
            break;
        }
    }
    if (!resolvedProfile) {
        fprintf(stderr, "[error] Profile '%s' not found. Available:", profileName.c_str());
        for (auto& p : allProfiles) fprintf(stderr, " %s", std::string(p.name).c_str());
        fprintf(stderr, "\n");
        return 1;
    }

    DataGen::WriterSettings settings;
    settings.synthName      = synthName;
    settings.profileName    = profileName;
    settings.lfoStratified  = resolvedProfile->lfoStratified;
    settings.frameRules     = resolvedProfile->frameRules;
    settings.numSamples   = numSamples;
    settings.baseSeed     = static_cast<uint32_t>(seed);
    settings.maxDuration  = maxDuration;
    settings.numWorkers   = juce::jmax(1, numWorkers);
    settings.renderMode   = DataGen::RenderMode::AllFrames;
    settings.synthFactory = [&synthName]{ return SynthRegistry::instance().create(synthName); };

    // DatasetWriter manages S{N}D{M} session subdirs — pass the datasets root directly
    settings.outputDir = resolvedOutputRoot;

    // param configs — resolved from sampling profile (shared CLI/GUI resolver)
    {
        auto tmpSynth = settings.synthFactory();
        settings.paramConfigs = DataGen::resolveParamConfigs(*resolvedProfile, tmpSynth->getParams());
    }

    // frame selection
    auto& fs = settings.frameSelection;
    auto framesIt = cfg.sections.find("frames");
    if (framesIt != cfg.sections.end()) {
        auto& fm = framesIt->second;
        auto get = [&](const std::string& k, bool def) {
            auto it = fm.find(k);
            return it != fm.end() ? parseBool(it->second, def) : def;
        };
        fs.osc_a  = get("osc_a",  true);
        fs.osc_b  = get("osc_b",  true);
        fs.noise  = get("noise",  true);
        fs.filter = get("filter", true);
        fs.env1   = get("env1",   true);
        fs.env2   = get("env2",   true);
        fs.lfo1   = get("lfo1",   true);
        fs.lfo2   = get("lfo2",   true);
        fs.lfo3   = get("lfo3",   true);
        fs.lfo4   = get("lfo4",   true);
        fs.reverb = get("reverb", true);
        fs.delay  = get("delay",  true);

        // if any frame is false, use SelectedFrames mode
        bool anyFalse = !fs.osc_a || !fs.osc_b || !fs.noise  || !fs.filter ||
                        !fs.env1  || !fs.env2  || !fs.lfo1   || !fs.lfo2   ||
                        !fs.lfo3  || !fs.lfo4  || !fs.reverb || !fs.delay;
        if (anyFalse)
            settings.renderMode = DataGen::RenderMode::SelectedFrames;
    }

    // ── Progress + callbacks ──────────────────────────────────────────────────
    auto startTime = juce::Time::getCurrentTime();
    int  lastPct   = -1;

    std::mutex stdoutMutex;

    DataGen::WriterCallbacks callbacks;
    callbacks.onProgress = [&](int current, int total) {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        int pct = (total > 0) ? (current * 100 / total) : 0;
        if (pct != lastPct) {
            lastPct = pct;
            printf("\r[%3d%%] %d/%d samples written", pct, current, total);
            fflush(stdout);
        }
    };
    callbacks.onLog = [&](const juce::String& msg) {
        std::lock_guard<std::mutex> lock(stdoutMutex);
        printf("\n[log] %s\n", msg.toRawUTF8());
    };
    callbacks.onFinished = [&](bool success) {
        printf("\n");
        if (success) {
            double elapsed = (juce::Time::getCurrentTime() - startTime).inSeconds();
            printf("[done] %.1fs  output: %s\n",
                   elapsed, settings.outputDir.getFullPathName().toRawUTF8());
        } else {
            fprintf(stderr, "[error] DatasetWriter reported failure.\n");
        }
    };

    // ── Run synchronously ─────────────────────────────────────────────────────
    printf("[info] synth=%s  profile=%s  n=%d  seed=%d  workers=%d\n",
           synthName.c_str(), profileName.c_str(), numSamples, seed, settings.numWorkers);
    printf("[info] output: %s\n", resolvedOutputRoot.getFullPathName().toRawUTF8());

    DataGen::DatasetWriter writer;
    writer.configure(settings, callbacks);
    writer.run();   // synchronous — runs on this thread

    return 0;
}
