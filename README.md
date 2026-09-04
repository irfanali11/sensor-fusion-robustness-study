# Sensor Fusion Robustness Study

## Research Question
When a vision+IMU human activity recognition system experiences partial sensor
failure (dropped camera frames, sustained occlusion, noisy IMU readings), does
it fail gracefully or catastrophically — and does the fusion strategy (early
vs. late) change how gracefully it degrades?

## Why this matters
Most human activity recognition research reports accuracy under clean,
ideal sensor conditions. Real-world robotic and wearable systems experience
partial sensor failure routinely (a blocked camera, a glitchy accelerometer).
This project measures — rather than assumes — how much that matters, and
whether a common architectural choice (early vs. late fusion) offers any
real protection against it.

## Dataset
[UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) — 27 actions,
8 subjects, 4 trials, synchronized RGB video + wrist/thigh-worn IMU data.
Not included in this repo (download instructions below) due to size.

## Method
- Vision branch: lightweight CNN over sampled video frames
- IMU branch: 1D-CNN over accelerometer/gyroscope signal
- Two fusion strategies compared: early fusion (concatenated features) vs.
  late fusion (averaged prediction logits)
- Robustness evaluated under three corruption types at increasing severity:
  random frame drop, sustained block occlusion, IMU sensor noise
- A dropout-augmented training variant tested as a mitigation

## Setup
\`\`\`bash
wget http://www.utdallas.edu/~kehtar/UTD-MAD/RGB.zip
wget http://www.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip
unzip RGB.zip -d data/rgb
unzip Inertial.zip -d data/inertial
\`\`\`
Then open `sensor_dropout_experiment.ipynb` in Colab/Jupyter and run cells in order.

## Results
*(fill in once you have them — accuracy-vs-severity charts, early vs. late
fusion comparison, and whether dropout-augmented training closed the gap)*

## Status
Work in progress — baseline training pipeline complete, robustness
experiments in progress.
