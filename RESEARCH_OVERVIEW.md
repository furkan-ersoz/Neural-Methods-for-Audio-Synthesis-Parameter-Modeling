# 1. Problem Definitions

**Learnable Audio Synthesizer:** A machine learning framework that estimates synthesizer parameters from audio to reproduce or approximate a target sound.

**Inverse Audio Synthesis:** The task of recovering synthesizer parameters from an observed audio signal.

**Differentiable Audio Synthesis:** A synthesis framework in which the signal generation process is differentiable and optimized via gradient-based learning.

# 2. Approach Variants

**Inverse Synthesis Approach:** A supervised regression method that directly predicts synthesizer parameters from audio representations.

**Differentiable Synthesis Approach:** gradient-based method where predicted parameters are rendered through a differentiable synthesizer and optimized using audio reconstruction loss.

# 3. Objectives and Use Cases

**Sound Matching:** Automatically estimating synthesizer parameters to recreate a target sound.

**Preset Generation:** Learning parameter distributions to generate musically meaningful synthesizer patches.

**Audio-to-Patch Translation:** Converting arbitrary input audio into editable synthesizer settings.

**Intelligent Sound Design Assistance:** Recommending parameter adjustments to achieve a desired timbre.

**Educational Audio Synthesis Modeling:** Demonstrating the relationship between synthesis parameters and perceptual sound characteristics.
