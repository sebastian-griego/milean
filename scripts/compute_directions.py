#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_layers(arg, num_layers):
    if arg == "all":
        return list(range(num_layers))
    out = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Train JSONL with state/label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="kaiyuy/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layers", default="11,12")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalize(vec):
    return vec / (vec.norm() + 1e-8)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = [ex for ex in iter_jsonl(args.train) if ex.get("label") in ("intro", "apply")]
    if not data:
        raise ValueError("no labeled examples")

    labels = [ex["label"] for ex in data]
    label_ids = [1 if lbl == "intro" else 0 for lbl in labels]
    n = len(label_ids)

    # shuffled labels
    label_ids_shuf = label_ids[:]
    rng.shuffle(label_ids_shuf)

    # within-class split (intro only)
    intro_split = [False] * n
    for i, lbl in enumerate(labels):
        if lbl == "intro":
            intro_split[i] = rng.random() < 0.5

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    encoder = model.get_encoder()

    # get layer ids from a warmup forward
    warm = tokenizer(data[0]["state"], return_tensors="pt", truncation=True, max_length=args.max_length)
    warm = {k: v.to(device) for k, v in warm.items()}
    with torch.inference_mode():
        warm_out = encoder(**warm, output_hidden_states=True, return_dict=True)
    num_layers = len(warm_out.hidden_states)
    layer_ids = parse_layers(args.layers, num_layers)

    if any(layer < 1 for layer in layer_ids):
        raise ValueError("layers must be >=1 to correspond to encoder blocks")

    sums = {}
    for layer in layer_ids:
        sums[layer] = {
            "sum_intro": torch.zeros(model.config.d_model, dtype=torch.float64),
            "sum_apply": torch.zeros(model.config.d_model, dtype=torch.float64),
            "sum_intro_shuf": torch.zeros(model.config.d_model, dtype=torch.float64),
            "sum_apply_shuf": torch.zeros(model.config.d_model, dtype=torch.float64),
            "sum_intro_a": torch.zeros(model.config.d_model, dtype=torch.float64),
            "sum_intro_b": torch.zeros(model.config.d_model, dtype=torch.float64),
            "count_intro": 0,
            "count_apply": 0,
            "count_intro_shuf": 0,
            "count_apply_shuf": 0,
            "count_intro_a": 0,
            "count_intro_b": 0,
        }

    with torch.inference_mode():
        for i in tqdm(range(0, n, args.batch_size)):
            batch = data[i : i + args.batch_size]
            states = [ex["state"] for ex in batch]
            batch_ids = label_ids[i : i + args.batch_size]
            batch_ids_shuf = label_ids_shuf[i : i + args.batch_size]
            batch_intro_split = intro_split[i : i + args.batch_size]

            inputs = tokenizer(
                states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            enc_out = encoder(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = enc_out.hidden_states

            mask = inputs["attention_mask"].unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1)

            label_tensor = torch.tensor(batch_ids, device=inputs["input_ids"].device)
            label_shuf_tensor = torch.tensor(batch_ids_shuf, device=inputs["input_ids"].device)
            intro_split_tensor = torch.tensor(batch_intro_split, device=inputs["input_ids"].device)

            for layer in layer_ids:
                h = hidden_states[layer]
                pooled = (h * mask).sum(dim=1) / denom
                pooled = pooled.to(torch.float64)

                mask_intro = label_tensor == 1
                mask_apply = label_tensor == 0
                if mask_intro.any():
                    sums[layer]["sum_intro"] += pooled[mask_intro].sum(dim=0).cpu()
                    sums[layer]["count_intro"] += int(mask_intro.sum().item())
                if mask_apply.any():
                    sums[layer]["sum_apply"] += pooled[mask_apply].sum(dim=0).cpu()
                    sums[layer]["count_apply"] += int(mask_apply.sum().item())

                mask_intro_shuf = label_shuf_tensor == 1
                mask_apply_shuf = label_shuf_tensor == 0
                if mask_intro_shuf.any():
                    sums[layer]["sum_intro_shuf"] += pooled[mask_intro_shuf].sum(dim=0).cpu()
                    sums[layer]["count_intro_shuf"] += int(mask_intro_shuf.sum().item())
                if mask_apply_shuf.any():
                    sums[layer]["sum_apply_shuf"] += pooled[mask_apply_shuf].sum(dim=0).cpu()
                    sums[layer]["count_apply_shuf"] += int(mask_apply_shuf.sum().item())

                mask_intro_a = mask_intro & intro_split_tensor
                mask_intro_b = mask_intro & (~intro_split_tensor)
                if mask_intro_a.any():
                    sums[layer]["sum_intro_a"] += pooled[mask_intro_a].sum(dim=0).cpu()
                    sums[layer]["count_intro_a"] += int(mask_intro_a.sum().item())
                if mask_intro_b.any():
                    sums[layer]["sum_intro_b"] += pooled[mask_intro_b].sum(dim=0).cpu()
                    sums[layer]["count_intro_b"] += int(mask_intro_b.sum().item())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in layer_ids:
        s = sums[layer]
        if s["count_intro"] == 0 or s["count_apply"] == 0:
            raise ValueError("missing intro/apply examples for direction")
        mean_intro = s["sum_intro"] / s["count_intro"]
        mean_apply = s["sum_apply"] / s["count_apply"]
        u = normalize(mean_intro - mean_apply)

        mean_intro_shuf = s["sum_intro_shuf"] / max(1, s["count_intro_shuf"])
        mean_apply_shuf = s["sum_apply_shuf"] / max(1, s["count_apply_shuf"])
        u_shuf = normalize(mean_intro_shuf - mean_apply_shuf)

        mean_intro_a = s["sum_intro_a"] / max(1, s["count_intro_a"])
        mean_intro_b = s["sum_intro_b"] / max(1, s["count_intro_b"])
        u_within = normalize(mean_intro_a - mean_intro_b)

        u_rand = torch.randn_like(u)
        u_rand = normalize(u_rand)

        payload = {
            "layer": layer,
            "block": layer - 1,
            "model": args.model,
            "max_length": args.max_length,
            "seed": args.seed,
            "train_path": str(args.train),
            "counts": {
                "intro": s["count_intro"],
                "apply": s["count_apply"],
                "intro_shuf": s["count_intro_shuf"],
                "apply_shuf": s["count_apply_shuf"],
                "intro_a": s["count_intro_a"],
                "intro_b": s["count_intro_b"],
            },
            "directions": {
                "u": u.float(),
                "random": u_rand.float(),
                "shuffled": u_shuf.float(),
                "within_class": u_within.float(),
            },
        }

        out_path = out_dir / f"directions_layer{layer}.pt"
        torch.save(payload, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
