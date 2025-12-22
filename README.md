This repo contains a minimal pipeline to probe and steer a Lean tactic generator (ByT5) on the binary behavior `intro` vs `apply` using teacher-forced scoring, layerwise probes, and a simple activation addition intervention.

Quickstart

1) Create a venv and install deps:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
```

2) Download the ReProver Benchmark 4 data, then extract a labeled JSONL:

```bash
python external/ReProver/scripts/download_data.py --data-path data

python scripts/extract_intro_apply.py \
  --input /path/to/leandojo_benchmark_4 \
  --output data/intro_apply.jsonl \
  --max-total 5000 \
  --balance \
  --max-state-chars 2000
```

3) Compute teacher-forced scores:

```bash
python scripts/score_intro_apply.py \
  --data data/intro_apply.jsonl \
  --output data/scores.jsonl
```

4) Save pooled encoder activations:

```bash
python scripts/save_activations.py \
  --data data/intro_apply.jsonl \
  --output data/activations.pt
```

5) Train layerwise probes and plot accuracy:

```bash
python scripts/train_probes.py \
  --activations data/activations.pt \
  --output data/probe_results.csv \
  --plot data/probe_plot.png
```

6) Run activation addition steering:

```bash
python scripts/steer_activation_addition.py \
  --data data/intro_apply.jsonl \
  --activations data/activations.pt \
  --layer 8 \
  --alphas -5 -2 -1 0 1 2 5 \
  --output data/steer_results.json
```

Notes

- Default model is `kaiyuy/leandojo-lean4-tacgen-byt5-small`. Override with `--model`.
- If your dataset schema differs, `extract_intro_apply.py` accepts `--state-key` and `--tactic-key`.
- ByT5 is byte-level; attention and token indices are byte-based.
