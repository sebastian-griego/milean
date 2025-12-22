#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with state/label")
    parser.add_argument("--activations", required=True, help=".pt file from save_activations.py")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output JSON summary")
    parser.add_argument("--model", default="kaiyuy/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--borderline-quantile", type=float, default=0.1)
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--shuffle-control", action="store_true")
    parser.add_argument("--flips-output", default=None, help="Optional JSONL with flipped examples")
    parser.add_argument("--alpha-for-flips", type=float, default=None)
    parser.add_argument("--max-flips", type=int, default=10)
    return parser.parse_args()

def logp_from_encoder(model, attention_mask, encoder_hidden, target_ids):
    labels = torch.tensor(target_ids, dtype=torch.long, device=encoder_hidden.device)
    labels = labels.unsqueeze(0).expand(encoder_hidden.size(0), -1).contiguous()
    enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)
    outputs = model(encoder_outputs=enc_out, attention_mask=attention_mask, labels=labels, return_dict=True)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum(dim=-1)

def compute_direction(acts, labels, label_names):
    intro_id = label_names.index("intro")
    apply_id = label_names.index("apply")
    mask = (labels == intro_id) | (labels == apply_id)
    labels = labels[mask]
    acts = acts[mask]
    mean_intro = acts[labels == intro_id].mean(dim=0)
    mean_apply = acts[labels == apply_id].mean(dim=0)
    u = mean_intro - mean_apply
    u = u / (u.norm() + 1e-8)
    return u

def evaluate_direction(data, model, tokenizer, u, alphas, batch_size, max_length, device, target_intro, target_apply):
    scores_by_alpha = {alpha: [] for alpha in alphas}
    labels_out = []
    states_out = []

    encoder = model.get_encoder()
    u = u.to(device)

    with torch.inference_mode():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            states = [ex["state"] for ex in batch]
            labels = [ex["label"] for ex in batch]

            inputs = tokenizer(
                states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            enc_out = encoder(**inputs, output_hidden_states=False, return_dict=True)
            enc_hidden = enc_out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            for alpha in alphas:
                delta = alpha * u.view(1, 1, -1) * mask
                enc_mod = enc_hidden + delta
                logp_intro = logp_from_encoder(model, inputs["attention_mask"], enc_mod, target_intro)
                logp_apply = logp_from_encoder(model, inputs["attention_mask"], enc_mod, target_apply)
                score = (logp_intro - logp_apply).detach().cpu().tolist()
                scores_by_alpha[alpha].extend(score)

            labels_out.extend(labels)
            states_out.extend(states)

    return scores_by_alpha, labels_out, states_out

def summarize(scores_by_alpha, labels, alphas, borderline_quantile):
    label_arr = np.array(labels)
    base_scores = np.array(scores_by_alpha[0.0])
    base_preds = np.where(base_scores > 0, "intro", "apply")

    abs_base = np.abs(base_scores)
    thresh = np.quantile(abs_base, borderline_quantile)
    borderline_mask = abs_base <= thresh

    summary = []
    for alpha in alphas:
        scores = np.array(scores_by_alpha[alpha])
        preds = np.where(scores > 0, "intro", "apply")
        acc = float(np.mean(preds == label_arr))
        flip = float(np.mean(preds != base_preds))
        shift = float(np.mean(scores - base_scores))
        if borderline_mask.any():
            flip_borderline = float(np.mean(preds[borderline_mask] != base_preds[borderline_mask]))
        else:
            flip_borderline = 0.0
        summary.append(
            {
                "alpha": alpha,
                "accuracy": acc,
                "flip_rate": flip,
                "mean_score_shift": shift,
                "borderline_flip_rate": flip_borderline,
            }
        )
    return summary

def pick_alpha_for_flips(summary, explicit_alpha):
    if explicit_alpha is not None:
        return explicit_alpha
    if not summary:
        return 0.0
    best = max(summary, key=lambda x: abs(x["mean_score_shift"]))
    return best["alpha"]

if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data = [ex for ex in iter_jsonl(args.data) if ex.get("label") in ("intro", "apply")]

    acts = torch.load(args.activations, map_location="cpu")
    layer_key = f"layer_{args.layer}"
    if layer_key not in acts["layers"]:
        raise ValueError(f"layer {args.layer} not found in activations")

    labels = acts["labels"]
    label_names = acts["label_names"]
    layer_acts = acts["layers"][layer_key]
    u = compute_direction(layer_acts, labels, label_names)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    target_intro = tokenizer("intro", add_special_tokens=False).input_ids
    target_apply = tokenizer("apply", add_special_tokens=False).input_ids

    alphas = sorted(set([0.0] + [float(a) for a in args.alphas]))

    results = []

    scores_by_alpha_u, labels_out_u, states_out_u = evaluate_direction(
        data,
        model,
        tokenizer,
        u,
        alphas,
        args.batch_size,
        args.max_length,
        device,
        target_intro,
        target_apply,
    )
    summary_u = summarize(scores_by_alpha_u, labels_out_u, alphas, args.borderline_quantile)
    for row in summary_u:
        row["direction"] = "u"
        results.append(row)

    if args.random_control:
        rand = torch.randn_like(u)
        rand = rand / (rand.norm() + 1e-8)
        scores_by_alpha, labels_out, _ = evaluate_direction(
            data,
            model,
            tokenizer,
            rand,
            alphas,
            args.batch_size,
            args.max_length,
            device,
            target_intro,
            target_apply,
        )
        summary = summarize(scores_by_alpha, labels_out, alphas, args.borderline_quantile)
        for row in summary:
            row["direction"] = "random"
            results.append(row)

    if args.shuffle_control:
        perm = torch.randperm(labels.numel())
        labels_shuf = labels[perm]
        u_shuf = compute_direction(layer_acts, labels_shuf, label_names)
        scores_by_alpha, labels_out, _ = evaluate_direction(
            data,
            model,
            tokenizer,
            u_shuf,
            alphas,
            args.batch_size,
            args.max_length,
            device,
            target_intro,
            target_apply,
        )
        summary = summarize(scores_by_alpha, labels_out, alphas, args.borderline_quantile)
        for row in summary:
            row["direction"] = "shuffled"
            results.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    if args.flips_output:
        alpha_pick = pick_alpha_for_flips(summary_u, args.alpha_for_flips)
        scores = np.array(scores_by_alpha_u[alpha_pick])
        base_scores = np.array(scores_by_alpha_u[0.0])
        base_preds = np.where(base_scores > 0, "intro", "apply")
        preds = np.where(scores > 0, "intro", "apply")
        flipped = np.where(preds != base_preds)[0]
        flipped = flipped[: args.max_flips]
        flips_path = Path(args.flips_output)
        flips_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flips_path, "w", encoding="utf-8") as f:
            for idx in flipped:
                out = {
                    "state": states_out_u[idx],
                    "label": labels_out_u[idx],
                    "pred_base": base_preds[idx],
                    "pred_alpha": preds[idx],
                    "score_base": float(base_scores[idx]),
                    "score_alpha": float(scores[idx]),
                    "alpha": alpha_pick,
                }
                f.write(json.dumps(out, ensure_ascii=True) + "\n")

    print(f"saved results to {output_path}")
