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
    parser.add_argument(
        "--mask-modes",
        default="global",
        help="Comma-separated: global,goal,context,goal_head",
    )
    parser.add_argument("--goal-head-tokens", type=int, default=64)
    parser.add_argument("--injection", choices=["final", "block"], default="final")
    parser.add_argument(
        "--direction-names",
        default="u,random,shuffled,within_class",
        help="Comma-separated direction keys to use",
    )
    parser.add_argument(
        "--block-scale",
        choices=["none", "rms"],
        default="none",
        help="Scale delta inside encoder by per-token RMS of block output",
    )
    parser.add_argument(
        "--block-override",
        type=int,
        default=None,
        help="Override block index for in-encoder injection",
    )
    parser.add_argument("--flips-output", default=None)
    parser.add_argument("--flips-alpha", type=float, default=0.25)
    parser.add_argument("--max-flips", type=int, default=10)
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

    enc2 = torch.cat([encoder_hidden, encoder_hidden], dim=0)
    attn2 = torch.cat([attention_mask, attention_mask], dim=0)

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


def compute_goal_start_tokens(states, tokenizer, attention_mask, max_length):
    raw = tokenizer(states, add_special_tokens=False, truncation=True, max_length=max_length)
    raw_lens = [len(ids) for ids in raw["input_ids"]]
    actual_lens = attention_mask.sum(dim=1).tolist()

    goal_tokens = []
    for text, raw_len, actual_len in zip(states, raw_lens, actual_lens):
        offset = max(0, int(actual_len) - int(raw_len))
        idx = text.rfind("⊢")
        if idx == -1:
            goal_byte = raw_len
        else:
            goal_byte = len(text[:idx].encode("utf-8"))
            if goal_byte > raw_len:
                goal_byte = raw_len
        goal_tok = min(goal_byte + offset, int(actual_len))
        goal_tokens.append(goal_tok)
    return goal_tokens


def build_masks(states, tokenizer, attention_mask, max_length, mask_modes, goal_head_tokens):
    masks = {}
    mask_global = attention_mask.clone()
    masks["global"] = mask_global

    if any(m in ("goal", "context", "goal_head") for m in mask_modes):
        goal_starts = compute_goal_start_tokens(states, tokenizer, attention_mask, max_length)
        actual_lens = attention_mask.sum(dim=1).tolist()
        for mode in mask_modes:
            if mode == "global":
                continue
            mask = torch.zeros_like(attention_mask)
            for i, (goal_tok, actual_len) in enumerate(zip(goal_starts, actual_lens)):
                goal_tok = int(goal_tok)
                actual_len = int(actual_len)
                if actual_len == 0:
                    continue
                if mode == "goal":
                    if goal_tok < actual_len:
                        mask[i, goal_tok:actual_len] = 1
                elif mode == "context":
                    if goal_tok > 0:
                        mask[i, : min(goal_tok, actual_len)] = 1
                elif mode == "goal_head":
                    if goal_tok < actual_len:
                        end = min(goal_tok + goal_head_tokens, actual_len)
                        mask[i, goal_tok:end] = 1
            mask = mask * attention_mask
            masks[mode] = mask

    return masks


def make_hook(u, alpha, mask, block_scale, eps=1e-6):
    mask_f = mask.unsqueeze(-1)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            hs = out[0]
        else:
            hs = out
        if block_scale == "rms":
            rms = hs.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=eps)
            delta = alpha * rms * mask_f * u.view(1, 1, -1)
        else:
            delta = alpha * mask_f * u.view(1, 1, -1)
        hs.add_(delta)
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    return hook


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mask_modes = [m.strip() for m in args.mask_modes.split(",") if m.strip()]
    if not mask_modes:
        mask_modes = ["global"]

    direction_names = [d.strip() for d in args.direction_names.split(",") if d.strip()]
    if not direction_names:
        direction_names = ["u"]

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

    stats = {}
    for layer in directions_by_layer:
        for direction in direction_names:
            for mask in mask_modes:
                for alpha in alphas:
                    stats[(layer, direction, mask, alpha)] = {
                        "sum_shift": 0.0,
                        "sumsq_shift": 0.0,
                        "count": 0,
                        "correct": 0,
                        "flip": 0,
                    }

    boot_shifts = {}
    for layer in directions_by_layer:
        for direction in direction_names:
            for mask in mask_modes:
                boot_shifts[(layer, direction, mask, alpha_pos)] = []
                boot_shifts[(layer, direction, mask, alpha_neg)] = []

    flips = {}
    if args.flips_output:
        for layer in directions_by_layer:
            for mask in mask_modes:
                flips[(layer, mask)] = []

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

            masks = build_masks(states, tokenizer, inputs["attention_mask"], args.max_length, mask_modes, args.goal_head_tokens)
            masks = {k: v.to(device) for k, v in masks.items()}

            enc_out = encoder(**inputs, return_dict=True)
            enc_hidden = enc_out.last_hidden_state

            score_base = score_from_encoder(model, enc_hidden, inputs["attention_mask"], target_intro, target_apply)
            pred_base = score_base > 0
            base_correct += int((pred_base == label_batch).sum().item())
            base_count += label_batch.numel()

            if args.injection == "final":
                for layer, payload in directions_by_layer.items():
                    for direction_name, u in payload["directions"].items():
                        if direction_name not in direction_names:
                            continue
                        u = u.to(device)
                        for mask_name, mask in masks.items():
                            mask_f = mask.unsqueeze(-1)
                            for alpha in alphas:
                                key = (layer, direction_name, mask_name, alpha)
                                if alpha == 0.0:
                                    shift = torch.zeros_like(score_base)
                                    pred = pred_base
                                    score_alpha = score_base
                                else:
                                    delta = alpha * mask_f * u.view(1, 1, -1)
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
                                stats[key]["flip"] += int((pred != pred_base).sum().item()) if alpha != 0.0 else 0

                                if alpha in (alpha_pos, alpha_neg):
                                    boot_shifts[key].extend(shift.detach().cpu().tolist())

                                if args.flips_output and direction_name == "u" and abs(alpha - args.flips_alpha) < 1e-9:
                                    for idx in range(label_batch.numel()):
                                        if pred[idx] != pred_base[idx]:
                                            flips[(layer, mask_name)].append(
                                                {
                                                    "state": states[idx],
                                                    "label": "intro" if label_batch[idx].item() == 1 else "apply",
                                                    "pred_base": "intro" if pred_base[idx].item() else "apply",
                                                    "pred_alpha": "intro" if pred[idx].item() else "apply",
                                                    "score_base": float(score_base[idx].item()),
                                                    "score_alpha": float(score_alpha[idx].item()),
                                                    "alpha": float(alpha),
                                                    "layer": layer,
                                                    "mask": mask_name,
                                                    "direction": direction_name,
                                                    "injection": args.injection,
                                                }
                                            )
            else:
                for layer, payload in directions_by_layer.items():
                    block_idx = args.block_override if args.block_override is not None else int(payload.get("block", layer - 1))
                    if block_idx < 0 or block_idx >= len(encoder.block):
                        raise ValueError(f"block index {block_idx} out of range")

                    for direction_name, u in payload["directions"].items():
                        if direction_name not in direction_names:
                            continue
                        u = u.to(device)
                        for mask_name, mask in masks.items():
                            mask_f = mask.unsqueeze(-1)
                            for alpha in alphas:
                                key = (layer, direction_name, mask_name, alpha)
                                if alpha == 0.0:
                                    shift = torch.zeros_like(score_base)
                                    pred = pred_base
                                    score_alpha = score_base
                                else:
                                    handle = encoder.block[block_idx].register_forward_hook(
                                        make_hook(u, alpha, mask, args.block_scale)
                                    )
                                    enc_out_mod = encoder(**inputs, return_dict=True)
                                    handle.remove()
                                    score_alpha = score_from_encoder(
                                        model, enc_out_mod.last_hidden_state, inputs["attention_mask"], target_intro, target_apply
                                    )
                                    shift = score_alpha - score_base
                                    pred = score_alpha > 0

                                stats[key]["sum_shift"] += float(shift.sum().item())
                                stats[key]["sumsq_shift"] += float((shift * shift).sum().item())
                                stats[key]["count"] += label_batch.numel()
                                stats[key]["correct"] += int((pred == label_batch).sum().item())
                                stats[key]["flip"] += int((pred != pred_base).sum().item()) if alpha != 0.0 else 0

                                if alpha in (alpha_pos, alpha_neg):
                                    boot_shifts[key].extend(shift.detach().cpu().tolist())

                                if args.flips_output and direction_name == "u" and abs(alpha - args.flips_alpha) < 1e-9:
                                    for idx in range(label_batch.numel()):
                                        if pred[idx] != pred_base[idx]:
                                            flips[(layer, mask_name)].append(
                                                {
                                                    "state": states[idx],
                                                    "label": "intro" if label_batch[idx].item() == 1 else "apply",
                                                    "pred_base": "intro" if pred_base[idx].item() else "apply",
                                                    "pred_alpha": "intro" if pred[idx].item() else "apply",
                                                    "score_base": float(score_base[idx].item()),
                                                    "score_alpha": float(score_alpha[idx].item()),
                                                    "alpha": float(alpha),
                                                    "layer": layer,
                                                    "mask": mask_name,
                                                    "direction": direction_name,
                                                    "injection": args.injection,
                                                    "block": int(block_idx),
                                                }
                                            )

    results = []
    for (layer, direction, mask, alpha), s in stats.items():
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
                "mask": mask,
                "alpha": alpha,
                "mean_shift": mean_shift,
                "std_shift": std_shift,
                "accuracy": acc,
                "flip_rate": flip_rate,
                "n": count,
                "injection": args.injection,
            }
        )

    bootstrap = []
    rng = np.random.default_rng(args.seed)
    for layer in directions_by_layer:
        for direction in direction_names:
            for mask in mask_modes:
                shifts_pos = np.array(boot_shifts[(layer, direction, mask, alpha_pos)], dtype=np.float64)
                shifts_neg = np.array(boot_shifts[(layer, direction, mask, alpha_neg)], dtype=np.float64)
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
                        "mask": mask,
                        "slope": float(slope),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "alpha_pos": alpha_pos,
                        "alpha_neg": alpha_neg,
                        "reps": args.bootstrap_reps,
                        "injection": args.injection,
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
                    "mask_modes": mask_modes,
                    "direction_names": direction_names,
                    "goal_head_tokens": args.goal_head_tokens,
                    "injection": args.injection,
                    "block_override": args.block_override,
                    "block_scale": args.block_scale,
                },
                "results": results,
                "bootstrap": bootstrap,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    if args.flips_output:
        flips_path = Path(args.flips_output)
        flips_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flips_path, "w", encoding="utf-8") as f:
            for (layer, mask), items in flips.items():
                items.sort(key=lambda x: abs(x["score_alpha"] - x["score_base"]), reverse=True)
                for item in items[: args.max_flips]:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_direction = "u" if "u" in direction_names else direction_names[0]
        for layer in sorted(directions_by_layer.keys()):
            layer_rows = [r for r in results if r["layer"] == layer and r["direction"] == plot_direction]

            plt.figure(figsize=(6, 4))
            for mask in mask_modes:
                rows = [r for r in layer_rows if r["mask"] == mask]
                rows.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in rows]
                ys = [r["mean_shift"] for r in rows]
                plt.plot(xs, ys, marker="o", label=mask)
            plt.axhline(0.0, color="black", linewidth=0.5)
            plt.xlabel("alpha")
            plt.ylabel("mean score shift")
            plt.title(f"Layer {layer} mean score shift vs alpha ({args.injection}, {plot_direction})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"steer_shift_layer{layer}_{args.injection}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6, 4))
            for mask in mask_modes:
                rows = [r for r in layer_rows if r["mask"] == mask]
                rows.sort(key=lambda r: r["alpha"])
                xs = [r["alpha"] for r in rows]
                ys = [r["accuracy"] for r in rows]
                plt.plot(xs, ys, marker="o", label=mask)
            plt.axhline(0.5, color="black", linewidth=0.5)
            plt.xlabel("alpha")
            plt.ylabel("accuracy")
            plt.title(f"Layer {layer} accuracy vs alpha ({args.injection}, {plot_direction})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"steer_accuracy_layer{layer}_{args.injection}.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
