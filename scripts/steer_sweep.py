#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
import matplotlib.pyplot as plt


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test JSONL with state/label")
    parser.add_argument("--directions", nargs="+", required=True, help="Direction .pt files")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--plot-dir", default=None, help="Optional plot output dir")
    parser.add_argument("--model", default="kaiyuy/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--alphas", type=float, nargs="*", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def score_from_encoder(model, encoder_hidden, attention_mask, intro_ids, apply_ids):
    device = encoder_hidden.device
    intro_ids = torch.tensor(intro_ids, device=device, dtype=torch.long)
    apply_ids = torch.tensor(apply_ids, device=device, dtype=torch.long)
    max_len = max(intro_ids.numel(), apply_ids.numel())
    bsz = encoder_hidden.size(0)

    labels = torch.full((2 * bsz, max_len), -100, device=device, dtype=torch.long)
    labels[:bsz, : intro_ids.numel()] = intro_ids
    labels[bsz:, : apply_ids.numel()] = apply_ids
    labels = labels.contiguous()

    enc2 = encoder_hidden.repeat_interleave(2, dim=0)
    attn2 = attention_mask.repeat_interleave(2, dim=0)

    outputs = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=enc2),
        attention_mask=attn2,
        labels=labels,
        return_dict=True,
    )
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * (labels != -100)
    seq_logps = token_logps.sum(dim=-1)

    logp_intro = seq_logps[:bsz]
    logp_apply = seq_logps[bsz:]
    return logp_intro - logp_apply


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.alphas is None or len(args.alphas) == 0:
        alphas = [-2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
    else:
        alphas = args.alphas
    alphas = sorted(set(float(a) for a in alphas))
    if 0.0 not in alphas:
        alphas = sorted(alphas + [0.0])

    alpha_pos = float(args.bootstrap_alpha)
    alpha_neg = -alpha_pos
    if alpha_pos not in alphas:
        alphas = sorted(alphas + [alpha_pos])
    if alpha_neg not in alphas:
        alphas = sorted(alphas + [alpha_neg])

    data = [ex for ex in iter_jsonl(args.test) if ex.get("label") in ("intro", "apply")]
    if not data:
        raise ValueError("no labeled examples")

    labels = [1 if ex.get("label") == "intro" else 0 for ex in data]
    n = len(labels)

    # load directions
    directions_by_layer = {}
    for path in args.directions:
        payload = torch.load(path, map_location="cpu")
        layer = int(payload["layer"])
        directions_by_layer[layer] = payload

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    encoder = model.get_encoder()

    target_intro = tokenizer("intro", add_special_tokens=False).input_ids
    target_apply = tokenizer("apply", add_special_tokens=False).input_ids

    # stats[(layer, direction, alpha)] = dict
    stats = {}
    for layer in directions_by_layer:
        for direction in ["u", "random", "shuffled", "within_class"]:
            for alpha in alphas:
                stats[(layer, direction, alpha)] = {
                    "sum_shift": 0.0,
                    "sumsq_shift": 0.0,
                    "count": 0,
                    "correct": 0,
                    "flip": 0,
                }

    # for bootstrap
    boot_shifts = {}
    for layer in directions_by_layer:
        for direction in ["u", "random", "shuffled", "within_class"]:
            boot_shifts[(layer, direction, alpha_pos)] = []
            boot_shifts[(layer, direction, alpha_neg)] = []

    base_correct = 0
    base_count = 0

    with torch.inference_mode():
        for i in tqdm(range(0, n, args.batch_size)):
            batch = data[i : i + args.batch_size]
            states = [ex["state"] for ex in batch]
            label_batch = torch.tensor(labels[i : i + args.batch_size], device=device)

            inputs = tokenizer(
                states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            enc_out = encoder(**inputs, return_dict=True)
            enc_hidden = enc_out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            score_base = score_from_encoder(model, enc_hidden, inputs["attention_mask"], target_intro, target_apply)
            pred_base = score_base > 0
            base_correct += int((pred_base == label_batch).sum().item())
            base_count += label_batch.numel()

            for layer, payload in directions_by_layer.items():
                for direction_name, u in payload["directions"].items():
                    u = u.to(device)

                    for alpha in alphas:
                        key = (layer, direction_name, alpha)
                        if alpha == 0.0:
                            shift = torch.zeros_like(score_base)
                            pred = pred_base
                        else:
                            delta = alpha * u.view(1, 1, -1) * mask
                            enc_mod = enc_hidden + delta
                            score_alpha = score_from_encoder(
                                model, enc_mod, inputs["attention_mask"], target_intro, target_apply
                            )
                            shift = score_alpha - score_base
                            pred = score_alpha > 0

                        stats[key]["sum_shift"] += float(shift.sum().item())
                        stats[key]["sumsq_shift"] += float((shift * shift).sum().item())
                        stats[key]["count"] += label_batch.numel()
                        stats[key]["correct"] += int((pred == label_batch).sum().item())
                        if alpha == 0.0:
                            stats[key]["flip"] += 0
                        else:
                            stats[key]["flip"] += int((pred != pred_base).sum().item())

                        if alpha in (alpha_pos, alpha_neg):
                            boot_shifts[key].extend(shift.detach().cpu().tolist())

    results = []
    for (layer, direction, alpha), s in stats.items():
        count = s["count"]
        mean_shift = s["sum_shift"] / count if count else 0.0
        var_shift = s["sumsq_shift"] / count - mean_shift * mean_shift if count else 0.0
        std_shift = float(np.sqrt(max(var_shift, 0.0)))
        acc = s["correct"] / count if count else 0.0
        flip_rate = s["flip"] / count if count else 0.0
        results.append(
            {
                "layer": layer,
                "direction": direction,
                "alpha": alpha,
                "mean_shift": mean_shift,
                "std_shift": std_shift,
                "accuracy": acc,
                "flip_rate": flip_rate,
                "n": count,
            }
        )

    # compute slopes and bootstrap CIs
    bootstrap = []
    rng = np.random.default_rng(args.seed)
    for layer in directions_by_layer:
        for direction in ["u", "random", "shuffled", "within_class"]:
            shifts_pos = np.array(boot_shifts[(layer, direction, alpha_pos)], dtype=np.float64)
            shifts_neg = np.array(boot_shifts[(layer, direction, alpha_neg)], dtype=np.float64)
            if shifts_pos.size == 0 or shifts_neg.size == 0:
                continue
            slope = (shifts_pos.mean() - shifts_neg.mean()) / (alpha_pos - alpha_neg)
            reps = []
            for _ in range(args.bootstrap_reps):
                idx = rng.integers(0, shifts_pos.size, size=shifts_pos.size)
                mean_pos = shifts_pos[idx].mean()
                mean_neg = shifts_neg[idx].mean()
                reps.append((mean_pos - mean_neg) / (alpha_pos - alpha_neg))
            reps = np.array(reps)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            bootstrap.append(
                {
                    "layer": layer,
                    "direction": direction,
                    "slope": float(slope),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "alpha_pos": alpha_pos,
                    "alpha_neg": alpha_neg,
                    "reps": args.bootstrap_reps,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "test": str(args.test),
                    "directions": [str(p) for p in args.directions],
                    "alphas": alphas,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "model": args.model,
                    "seed": args.seed,
                    "baseline_accuracy": base_correct / base_count if base_count else 0.0,
                },
                "results": results,
                "bootstrap": bootstrap,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        directions = ["u", "random", "shuffled", "within_class"]
        for layer in sorted(directions_by_layer.keys()):
            layer_rows = [r for r in results if r["layer"] == layer]

            plt.figure(figsize=(6, 4))
            for direction in directions:
                rows = [r for r in layer_rows if r["direction"] == direction]
                rows.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in rows]
                ys = [r["mean_shift"] for r in rows]
                plt.plot(xs, ys, marker="o", label=direction)
            plt.axhline(0.0, color="black", linewidth=0.5)
            plt.xlabel("alpha")
            plt.ylabel("mean score shift")
            plt.title(f"Layer {layer} mean score shift vs alpha")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"steer_shift_layer{layer}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6, 4))
            for direction in directions:
                rows = [r for r in layer_rows if r["direction"] == direction]
                rows.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in rows]
                ys = [r["accuracy"] for r in rows]
                plt.plot(xs, ys, marker="o", label=direction)
            plt.axhline(0.5, color="black", linewidth=0.5)
            plt.xlabel("alpha")
            plt.ylabel("accuracy")
            plt.title(f"Layer {layer} accuracy vs alpha")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"steer_accuracy_layer{layer}.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
