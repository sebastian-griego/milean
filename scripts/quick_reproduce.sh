#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ ! -d "data/leandojo_benchmark_4" ]; then
  python external/ReProver/scripts/download_data.py --data-path data
fi

python scripts/extract_intro_apply.py \
  --input data/leandojo_benchmark_4/random/train.json \
  --output data/intro_apply_5000.jsonl \
  --max-per-class 2500 \
  --balance \
  --exact-per-class \
  --max-state-chars 2000 \
  --seed 42

python scripts/split_intro_apply.py \
  --input data/intro_apply_5000.jsonl \
  --train-output data/intro_apply_train.jsonl \
  --test-output data/intro_apply_test.jsonl \
  --train-size 4000 \
  --test-size 1000 \
  --balance \
  --seed 42

python scripts/compute_directions.py \
  --train data/intro_apply_train.jsonl \
  --output-dir data/directions \
  --layers 11,12 \
  --max-length 384 \
  --batch-size 16 \
  --seed 42

python - <<'PY'
import json
import random
from pathlib import Path

def sample_balanced(src, out_path, per_class, seed):
    by_label = {"intro": [], "apply": []}
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        label = ex.get("label")
        if label in by_label:
            by_label[label].append(ex)
    rng = random.Random(seed)
    for k in by_label:
        rng.shuffle(by_label[k])
    subset = by_label["intro"][:per_class] + by_label["apply"][:per_class]
    rng.shuffle(subset)
    Path(out_path).write_text("\n".join(json.dumps(ex, ensure_ascii=True) for ex in subset) + "\n", encoding="utf-8")

sample_balanced("data/intro_apply_test.jsonl", "data/intro_apply_test_500.jsonl", 250, 42)
sample_balanced("data/intro_apply_test.jsonl", "data/intro_apply_test_200.jsonl", 100, 42)
PY

python scripts/steer_sweep.py \
  --test data/intro_apply_test_500.jsonl \
  --directions data/directions/directions_layer11.pt data/directions/directions_layer12.pt \
  --output data/steer_sweep_test500.json \
  --plot-dir data/plots_test500 \
  --batch-size 32 \
  --max-length 384 \
  --seed 42

python scripts/steer_sweep.py \
  --test data/intro_apply_test_500.jsonl \
  --directions data/directions/directions_layer11.pt data/directions/directions_layer12.pt \
  --output data/steer_tokenloc_test500.json \
  --plot-dir data/plots_tokenloc_test500 \
  --batch-size 32 \
  --max-length 384 \
  --alphas -0.25 -0.1 0 0.1 0.25 \
  --mask-modes global,goal,context,goal_head \
  --goal-head-tokens 64 \
  --injection final \
  --seed 42

python scripts/compute_directions.py \
  --train data/intro_apply_train.jsonl \
  --output-dir data/directions_rms \
  --layers 11,12 \
  --max-length 384 \
  --batch-size 16 \
  --seed 42 \
  --apply-rms-norm

python scripts/steer_sweep.py \
  --test data/intro_apply_test_200.jsonl \
  --directions data/directions_rms/directions_layer11.pt data/directions_rms/directions_layer12.pt \
  --output data/steer_inencoder_rms_test200_goalcontext.json \
  --plot-dir data/plots_inencoder_rms_test200_goalcontext \
  --batch-size 16 \
  --max-length 384 \
  --alphas -0.25 -0.1 0 0.1 0.25 \
  --mask-modes global,goal,context \
  --direction-names u \
  --injection block \
  --block-scale rms \
  --seed 42
