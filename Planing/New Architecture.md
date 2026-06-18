# SynthV3 — Sequential Frame Prediction Architecture

**Version:** 4.0 — Phase 1 (Raw Prediction, no Differentiable stage) **Date:** 2026-06-06

---

## 1. Overview

SynthV3 is a 70-parameter synthesizer. Instead of predicting all parameters with a single model, a sequence of small specialized models (frame predictors) follows the signal chain. Each model predicts only its own frame's parameters and receives the previous frames' information as conditioning.

**Core principles:**

- Identical roles share one model (ENV1+ENV2 → model006, LFO1–4 → model007)
- Different roles get separate models (OSC A and OSC B — different signal role and data distribution)
- Each frame has its own incremental "cheat sheet" WAV (cumulative render of all previous frames)
- During training, the cheat sheet is provided with input dropout (70% real, 30% zero tensor)
- During inference, cheat sheet is always zero tensor — no intermediate renders needed
- All structure is config-driven — adding SynthV4/V5 requires zero Python changes

---

## 2. Model List (current)

|File|Purpose|Runs|Output|Note|
|---|---|---|---|---|
|model011.py|Frame Predictor (base v2)|all frames|dynamic|pool_mode + temporal configurable|
|model012.py|Frame Predictor variant|—|dynamic|alternative architecture|

**Current status:** model003–model010 are superseded by model011 (base architecture v2). model011 adds configurable `pool_mode` (freq/stats), optional `temporal` GRU, and dynamic `in_channels`. All frame predictors currently use model011.

All frame predictors share the same base architecture. Output size, conditioning vector size, and input configuration are all read from config at runtime — nothing is hardcoded.

---

## 3. Door Classifier Output

```
bypass_osc_b      → binary Sigmoid
bypass_noise      → binary Sigmoid
bypass_filter     → binary Sigmoid
bypass_reverb     → binary Sigmoid
bypass_delay      → binary Sigmoid
active_lfo_count  → 5-class Softmax  (0, 1, 2, 3, 4)
```

**Loss:** Binary Cross-Entropy (5 flags) + Cross-Entropy (lfo_count)

**LFO logic:** `active_lfo_count = 2` → model008 runs twice at inference. Which parameter each LFO modulates (`lfo_target`) is determined by model008's own output. There is no "off" class in lfo_target — the predictor only runs for active LFOs.

---

## 4. Frame Sequence and Cheat Sheets

Ordering follows the signal flow. Each frame receives the previous step's cumulative render as its cheat sheet input.

| #   | Frame  | Model    | Params    | Active when     | Cheat sheet (previous step) |
| --- | ------ | -------- | --------- | --------------- | --------------------------- |
| 0   | Door   | model002 | 6 outputs | always          | `_full.wav`                 |
| 1   | OSC A  | model003 | 9         | always          | `s001f01.wav`               |
| 2   | OSC B  | model004 | 11        | bypass_osc_b=0  | `s001f02.wav`               |
| 3   | Noise  | model005 | 3         | bypass_noise=0  | `s001f03.wav`               |
| 4   | Filter | model006 | 6         | bypass_filter=0 | `s001f04.wav`               |
| 5   | ENV1   | model007 | 7         | always          | `s001f05.wav`               |
| 6   | ENV2   | model007 | 7         | bypass_osc_b=0  | `s001f06.wav`               |
| 7   | LFO1   | model008 | 4         | lfo_count >= 1  | `s001f07.wav`               |
| 8   | LFO2   | model008 | 4         | lfo_count >= 2  | `s001f08.wav`               |
| 9   | LFO3   | model008 | 4         | lfo_count >= 3  | `s001f09.wav`               |
| 10  | LFO4   | model008 | 4         | lfo_count >= 4  | `s001f10.wav`               |
| 11  | Reverb | model009 | 6         | bypass_reverb=0 | `s001f11.wav`               |
| 12  | Delay  | model010 | 5         | bypass_delay=0  | `s001f12.wav` (= full)      |

**Bypass + WAV naming rule:**

- If a frame is bypassed, its WAV file is NOT saved to disk.
- Frame numbers in filenames are fixed and semantic — f04 always means Filter, regardless of whether f03 (Noise) was saved or not.
- The CSV column for a bypassed frame is left empty.
- Python reads the CSV, detects which frame columns are filled, and loads only those files.

---

## 5. Frame Parameters (full list)

**OSC A — model011 (10):** osc_a_waveform, osc_a_warp_amt, osc_a_warp_type*, osc_a_detune, osc_a_blend, osc_a_pitch_coarse, osc_a_pitch_fine, osc_a_level, osc_a_harmonic2, osc_a_harmonic3

**OSC B — model011 (11):** osc_b_waveform, osc_b_warp_amt, osc_b_warp_type*, osc_b_detune, osc_b_blend, osc_b_pitch_coarse, osc_b_pitch_fine, osc_b_pan, osc_b_level, osc_b_phase, osc_b_semitone_ofs

**Noise — model011 (3):** noise_color*, noise_level, noise_stereo

**Filter — model011 (6):** filter_cutoff, filter_resonance, filter_type*, filter_drive, filter_keytrack, filter_env_amt

**ENV1 / ENV2 — model011 (7 each, shared model):** envN_attack, envN_hold, envN_decay, envN_sustain, envN_release, envN_atk_curve, envN_dec_curve

**LFO 1–4 — model011 (4 each, shared model):** lfoN_rate, lfoN_depth, lfoN_shape*, lfoN_target*

**Reverb — model011 (6):** reverb_size, reverb_decay, reverb_damping, reverb_predelay, reverb_mix, reverb_width

**Delay — model011 (5):** delay_time_l, delay_time_r, delay_feedback, delay_filter_cutoff, delay_mix

`*` = categorical param → Softmax output, Cross-Entropy loss

**Categorical values:**

- `warp_type`: [sync, fm, rm, fold] — 4 classes
- `noise_color`: [white, pink, brown] — 3 classes
- `filter_type`: [lp12, lp24, hp12, hp24, bp] — 5 classes
- `lfo_shape`: [sine, triangle, saw, square] — 4 classes
- `lfo_target`: [pitch, filter_cutoff, level, warp] — 4 classes

---

## 6. Model Architectures

### model011 — Frame Predictor (base architecture v2)

All frame predictors use model011. It supersedes model003–model010 with configurable pooling and optional temporal GRU.

**Config options:**
- `pool_mode: freq` — AdaptiveAvgPool2d keeps frequency axis
- `pool_mode: stats` — mean+std over (freq, time) → 2×C_out dims
- `temporal: none` — no temporal processing
- `temporal: gru` — single-layer GRU over CNN time axis, last hidden state appended

```
Input A: concat(mel_full, mel_prev * mask)  →  (2, 128, T)
         mel_prev is zero tensor if frame has no cheat sheet or dropout fires
Input B: prev_params_vector                 →  (D,)
         D = total number of params from all previous frames (fixed layout,
             bypassed frame params are zeroed, not removed — vector stays same size)

    ↓ [Input A]
CNN Encoder:
  Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
  Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
  Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
  Global Average Pooling  →  128-dim

    ↓ [Input B]
Conditioning Projection:
  FC(64) → ReLU  →  64-dim
  (If D=0 for first frame, skip this path and use zero vector)

    ↓ [Fusion]
concat(128-dim, 64-dim)  →  192-dim
FC(256) → ReLU → Dropout(0.3)
    ↓
FC(N_continuous) → Sigmoid
FC(N_categorical_i) → Softmax  ← one head per categorical param
```

**Dynamic dimensions (all read from config, never hardcoded):**

- `N_continuous`: number of continuous params in this frame
- `N_categorical_i`: number of classes for each categorical param
- `D`: sum of all param counts from preceding frames (bypassed = zeroed, not removed)
- Input channels: 2 if cheat sheet WAV exists in config, 1 if not

**Loss per predictor:** MSE (continuous params) + CrossEntropy (each categorical param) → weighted sum

---

## 7. Each Frame Predictor's Input

```
INPUT = concat(
    mel(full.wav),              # (128, T) — target sound, always present
    mel(prev_frame.wav) * mask, # (128, T) — cheat sheet, with dropout
    prev_params_vector          # (D,)     — all previous frames' params, fixed layout
)
```

**prev_params_vector layout:**

- Always the same length (sum of all preceding frame param counts)
- Bypassed frame slots are filled with zeros, not removed
- This ensures every frame predictor always receives the same input shape
- At training time: filled with ground truth values (teacher forcing)
- At inference time: filled with predictions from previous frames

**Input Dropout (cheat sheet masking):**

- Training: Bernoulli(p=0.7) per sample — 70% real mel, 30% zero tensor
- Inference: always zero tensor — no intermediate renders needed
- Model sees both conditions during training → no inference degradation

**Teacher Forcing:**

- Training: ground truth params from CSV
- Inference: predictions from previous frames passed forward
- V2: scheduled sampling (gradually replace ground truth with own predictions)

**Door Classifier input:**

```
INPUT = mel(full.wav)   # no cheat sheet, no conditioning
```

---

## 8. Dataset Structure

### CSV Columns

```
filename_full                          ← always saved, full render
osc_a_waveform ... delay_mix           ← 70 params, normalized [0,1]
bypass_osc_b, bypass_noise,
bypass_filter, bypass_reverb,
bypass_delay                           ← 5 binary flags
active_lfo_count                       ← integer 0–4
lfo1_target, lfo2_target,
lfo3_target, lfo4_target               ← categorical string
filename_frame01 ... filename_frame12  ← per-frame WAV paths, empty if bypassed
```

### WAV Naming Convention

Format: `s{sample_no}f{frame_no}.wav`

- Sample number: zero-padded to 5 digits
- Frame number: zero-padded to 2 digits (matches frame index in sequence table)
- Full render: `s{sample_no}.wav` (no frame suffix)

Example — OSC B and Noise bypassed:

```
s00001.wav          ← full render, always saved
s00001f01.wav       ← OSC A
# f02 not saved    ← OSC B bypassed
# f03 not saved    ← Noise bypassed
s00001f04.wav       ← Filter
s00001f05.wav       ← ENV1
s00001f06.wav       ← ENV2
s00001f07.wav       ← LFO1
# f08–f10 not saved ← LFO2–4 inactive (lfo_count=1)
s00001f11.wav       ← Reverb
s00001f12.wav       ← Delay (= full render content)
```

CSV columns filename_frame02, filename_frame03, filename_frame08–10 are empty for this sample.

### Directory Structure

```
synths/SynthV3/datasets/S03D01/
    dataset.csv
    samples/
        s00001.wav
        s00001f01.wav
        s00001f04.wav       ← f02, f03 skipped (bypassed)
        ...
        s00002.wav
        s00002f01.wav
        s00002f02.wav       ← OSC B active this time
        ...
```

---

## 9. File Structure

```
SS/
├── DataGenerator/
│   └── Source/
│       ├── DatasetWriter.h/.cpp    ← UPDATE: incremental render, CLI support
│       ├── DataGeneratorCLI.cpp    ← NEW: headless terminal runner for RunPod
│       └── MainComponent.cpp       ← UPDATE: per-frame save selection UI
│
├── NeuralNetworks/
│   ├── feature_extraction.py       ← UNTOUCHED
│   ├── dataset.py                  ← UNTOUCHED (V1/V2 pipeline)
│   ├── train.py                    ← UNTOUCHED (V1/V2 pipeline)
│   ├── evaluate.py                 ← UNTOUCHED
│   ├── export_onnx.py              ← UNTOUCHED
│   │
│   ├── models/
│   │   ├── model011.py             ← Base frame predictor v2 (pool_mode + GRU)
│   │   └── model012.py             ← Alternative architecture
│   │
│   ├── feature_extraction_v2.py    ← Feature extraction (v2, active)
│   ├── sequenced_dataset.py        ← multi-WAV + CSV reader
│   ├── sequenced_train_door.py     ← Phase 1 — door classifier training
│   ├── sequenced_train_frames.py   ← Phase 2 — frame predictor training
│   ├── sequenced_evaluate.py       ← per-frame evaluation
│   └── sequenced_export_onnx.py    ← export ONNX models
│
└── synths/SynthV3/
    ├── Experiments/
    │   ├── Exp009/
    │   └── Exp010/                 ← active experiment (osc_a sanity check)
    │       ├── config.yaml
    │       └── logs/
    └── datasets/S03D01/            ← 20k samples, osc_a only
        ├── dataset.csv
        └── samples/
```

---

## 10. Config Structure (SynthV3/Experiments/Exp010/config.yaml)

```yaml
experiment:
  name: "Exp010"
  tag:  "exp03-sanity"
  synth_root: "../../"
  pipeline: "sequenced"

dataset:
  path: "datasets/S03D01"
  train_ratio: 0.80
  val_ratio:   0.10
  seed:        42

features:
  type:        mel_spectrogram
  sample_rate: 44100
  n_mels:      128
  n_fft:       2048
  hop_length:  512
  normalize:   true
  cache:       true

conditioning:
  dropout_p:       0.30       # cheat sheet masking probability
  teacher_forcing: true       # use ground truth params during training

door:
  model_class: model002
  bypass_columns:
    - bypass_osc_b
    - bypass_noise
    - bypass_filter
    - bypass_reverb
    - bypass_delay
  lfo_count_column:  active_lfo_count
  lfo_count_classes: 5

frames:
  - name: osc_a
    model_file: model011
    frame_index: 1
    always_active: true
    wav_column: filename_frame01
    params:
      - osc_a_waveform
      - osc_a_warp_amt
      - osc_a_warp_type
      - osc_a_detune
      - osc_a_blend
      - osc_a_pitch_coarse
      - osc_a_pitch_fine
      - osc_a_level
      - osc_a_harmonic2
      - osc_a_harmonic3
    categorical_params: {}

  - name: osc_b
    model_file: model011
    frame_index: 2
    bypass_column: bypass_osc_b
    wav_column: filename_frame02
    params:
      - osc_b_waveform
      - osc_b_warp_amt
      - osc_b_warp_type
      - osc_b_detune
      - osc_b_blend
      - osc_b_pitch_coarse
      - osc_b_pitch_fine
      - osc_b_pan
      - osc_b_level
      - osc_b_phase
      - osc_b_semitone_ofs
    categorical_params:
      osc_b_warp_type: [sync, fm, rm, fold]

  - name: noise
    model_file: model011
    frame_index: 3
    bypass_column: bypass_noise
    wav_column: filename_frame03
    params:
      - noise_color
      - noise_level
      - noise_stereo
    categorical_params:
      noise_color: [white, pink, brown]

  - name: filter
    model_file: model011
    frame_index: 4
    bypass_column: bypass_filter
    wav_column: filename_frame04
    params:
      - filter_cutoff
      - filter_resonance
      - filter_type
      - filter_drive
      - filter_keytrack
      - filter_env_amt
    categorical_params:
      filter_type: [lp12, lp24, hp12, hp24, bp]

  - name: env1
    model_file: model011
    frame_index: 5
    always_active: true
    wav_column: filename_frame05
    params:
      - env1_attack
      - env1_hold
      - env1_decay
      - env1_sustain
      - env1_release
      - env1_atk_curve
      - env1_dec_curve

  - name: env2
    model_file: model011
    frame_index: 6
    bypass_column: bypass_osc_b
    wav_column: filename_frame06
    params:
      - env2_attack
      - env2_hold
      - env2_decay
      - env2_sustain
      - env2_release
      - env2_atk_curve
      - env2_dec_curve

  - name: lfo1
    model_file: model011
    frame_index: 7
    active_when: "active_lfo_count >= 1"
    wav_column: filename_frame07
    params:
      - lfo1_rate
      - lfo1_depth
      - lfo1_shape
      - lfo1_target
    categorical_params:
      lfo1_shape:  [sine, triangle, saw, square]
      lfo1_target: [pitch, filter_cutoff, level, warp]

  - name: lfo2
    model_file: model011
    frame_index: 8
    active_when: "active_lfo_count >= 2"
    wav_column: filename_frame08
    params:
      - lfo2_rate
      - lfo2_depth
      - lfo2_shape
      - lfo2_target
    categorical_params:
      lfo2_shape:  [sine, triangle, saw, square]
      lfo2_target: [pitch, filter_cutoff, level, warp]

  - name: lfo3
    model_file: model011
    frame_index: 9
    active_when: "active_lfo_count >= 3"
    wav_column: filename_frame09
    params:
      - lfo3_rate
      - lfo3_depth
      - lfo3_shape
      - lfo3_target
    categorical_params:
      lfo3_shape:  [sine, triangle, saw, square]
      lfo3_target: [pitch, filter_cutoff, level, warp]

  - name: lfo4
    model_file: model011
    frame_index: 10
    active_when: "active_lfo_count >= 4"
    wav_column: filename_frame10
    params:
      - lfo4_rate
      - lfo4_depth
      - lfo4_shape
      - lfo4_target
    categorical_params:
      lfo4_shape:  [sine, triangle, saw, square]
      lfo4_target: [pitch, filter_cutoff, level, warp]

  - name: reverb
    model_file: model011
    frame_index: 11
    bypass_column: bypass_reverb
    wav_column: filename_frame11
    params:
      - reverb_size
      - reverb_decay
      - reverb_damping
      - reverb_predelay
      - reverb_mix
      - reverb_width

  - name: delay
    model_file: model011
    frame_index: 12
    bypass_column: bypass_delay
    wav_column: filename_frame12
    params:
      - delay_time_l
      - delay_time_r
      - delay_feedback
      - delay_filter_cutoff
      - delay_mix

training:
  epochs:        100
  learning_rate: 0.001
  batch_size:    32
  loss:
    continuous:  mse
    categorical: cross_entropy
    cat_weight:  1.0
  optimizer:     adam
  checkpoint:
    save_best: true
    save_last: true
  early_stopping:
    patience: 20

export:
  auto_export:   true
  output_dir:    "exports"
  opset_version: 17
```

---

## 11. DataGenerator Updates

### Current state

DatasetWriter currently produces 3 WAVs per sample: `_osc.wav`, `_filtered.wav`, `_full.wav`

### Required changes

**DatasetWriter.h / .cpp:**

- `renderFrameSequence()` — incremental render per frame, saves only active frames
- Frame WAV filenames follow `s{N}f{F}.wav` convention; full render is `s{N}.wav`
- Bypassed frames: no file written, CSV column left empty
- CSV extended with: `active_lfo_count`, `lfo1–4_target`, `filename_frame01–12`

**DataGeneratorCLI.cpp (NEW):** Headless JUCE ConsoleApplication for RunPod / remote GPU machines. All options are driven by a single YAML config file — no GUI required.

```bash
./DataGeneratorCLI --config /data/datagen_config.yaml
```

If `--config` is not provided, the binary prints usage instructions and exits. Missing fields in the config fall back to their default values.

**datagen_config.yaml format:**

```yaml
synth:       SynthV3
dataset:     S03D01
n:           100000
seed:        42
output_dir:  /data/datasets

# Per-frame WAV save selection.
# full WAV (s{N}.wav) is always saved — not listed here.
# Default for all frames: true
frames:
  osc_a:   true
  osc_b:   true
  noise:   true
  filter:  true
  env1:    true
  env2:    true
  lfo1:    true
  lfo2:    true
  lfo3:    true
  lfo4:    true
  reverb:  true
  delay:   true

# Per-parameter sampling strategies (optional, default: random_uniform)
param_strategies:
  osc_a_waveform:  random_uniform
  filter_cutoff:   random_log
  env1_attack:     random_log
```

Every option available in the MainComponent GUI is also controllable via this config file, making the CLI suitable for fully automated remote dataset generation on RunPod, Vast.ai, or any headless machine.

**MainComponent.cpp — Per-frame save selection UI:**

- The UI reads the synth's frame structure (from ISynth / config) at startup
- Displays a scrollable panel listing all frames in fixed sequence order
- Each frame has an individual checkbox: "Save this frame's WAV"
- Default: all frames checked
- "Select all" / "Deselect all" buttons available
- Full render (`s{N}.wav`) has no checkbox — always saved, cannot be deselected
- Frame order in the panel matches the sequence table above and never changes

---

## 12. Training Flow

```bash
# Phase 1 — Door Classifier
python sequenced_train_door.py \
  --config synths/SynthV3/Experiments/Exp010/config.yaml

# Phase 2 — Frame Predictors (all can run in parallel)
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame osc_a
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame osc_b
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame noise
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame filter
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame env1
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame lfo1
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame reverb
python sequenced_train_frames.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame delay

# Evaluation
python sequenced_evaluate.py --config synths/SynthV3/Experiments/Exp010/config.yaml --frame osc_a

# Export
python sequenced_export_onnx.py --config synths/SynthV3/Experiments/Exp010/config.yaml
```

---

## 13. Modularity Guarantee

This architecture contains nothing specific to SynthV3. When SynthV4 arrives:

1. Run `synths/SynthV3/ExperimentAdder.py` → new experiment folder is created
2. Write `synths/SynthV4/Experiments/Exp001/config.yaml` → define your frames
3. Generate dataset with DataGenerator
4. Run `sequenced_train_door.py` and `sequenced_train_frames.py`

**Zero Python changes required.**

The config is the only interface between the synth design and the ML pipeline. Frame count, param count, bypass logic, categorical classes — all come from config.