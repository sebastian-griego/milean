# Mechanistic steering of a Lean tactic predictor: intro vs apply

## Overview
This repo contains a focused mechanistic interpretability study of the Lean 4 tactic generator `kaiyuy/leandojo-lean4-tacgen-byt5-small`. Using LeanDojo Benchmark 4 proof traces, I extracted a balanced dataset of proof states where the next human tactic begins with `intro` or `apply`. I defined a robust behavioral score using teacher forcing, `score(x) = logP(intro | x) - logP(apply | x)`, avoiding the brittleness of decoded strings. I then computed class-mean directions from train-only pooled encoder states (layers 11 and 12), performed dense alpha sweeps with strong controls, localized the signal to goal vs context spans, and diagnosed an architectural confound (encoder RMSNorm suppressing internal edits) before implementing RMS-scaled in-encoder injection.

## Key results

| Result | Value |
| --- | --- |
| Baseline accuracy (test500, max_length=384) | 0.798 |
| Layer 11 slope near 0 (global, alpha +/-0.25) | 9.36, CI [8.86, 9.83] |
| Layer 12 slope near 0 (global, alpha +/-0.25) | 17.73, CI [16.81, 18.59] |
| Token localized L12 slopes (alpha +/-0.25) | global 17.73, goal 7.67, context 10.18, goal_head 7.14 |
| In-encoder RMS-scaled slope (test200, alpha +/-0.25) | global ~0.83, goal ~0.37, context ~0.45 |

Controls (random, shuffled labels, within-class split) stayed near zero across the dense sweep.

## Figures

Control sweep (layer 12, score shift vs alpha, real plus controls):

![Control sweep, layer 12](data/plots_test500/steer_shift_layer12.png)

Token localized steering (layer 12, score shift vs alpha, global vs goal vs context):

![Token localized, layer 12](data/plots_tokenloc_test500/steer_shift_layer12_final.png)

In-encoder RMS-scaled injection (layer 12, score shift vs alpha):

![In-encoder RMS scaled, layer 12](data/plots_inencoder_rms_test200_goalcontext/steer_shift_layer12_block.png)

## Reproduction (CPU friendly)

Install deps, then run the one-command pipeline:

```bash
bash scripts/quick_reproduce.sh
```

This script downloads Benchmark 4, builds a balanced 5k dataset, fixes a train/test split, computes directions, runs the control sweep and token-localized sweep on test500, and runs a lightweight in-encoder RMS-scaled sweep on test200. It can take a few hours on CPU.

## Artifacts

- Control sweep outputs: `data/steer_sweep_test500.json`, `data/plots_test500/`
- Token localized outputs: `data/steer_tokenloc_test500.json`, `data/plots_tokenloc_test500/`
- In-encoder RMS outputs: `data/steer_inencoder_rms_test200_goalcontext.json`, `data/plots_inencoder_rms_test200_goalcontext/`
- Directions: `data/directions/` (raw), `data/directions_rms/` (RMS-normalized)
