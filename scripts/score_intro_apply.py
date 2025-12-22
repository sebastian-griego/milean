#!/usr/bin/env python3
import argparse
import json
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with state/label")
    parser.add_argument("--output", required=True, help="Output JSONL with scores")
    parser.add_argument("--model", default="kaiyuy/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()

def logp_for_target(model, inputs, target_ids):
    labels = torch.tensor(target_ids, dtype=torch.long, device=inputs["input_ids"].device)
    labels = labels.unsqueeze(0).expand(inputs["input_ids"].size(0), -1).contiguous()
    outputs = model(**inputs, labels=labels, return_dict=True)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum(dim=-1)

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    target_intro = tokenizer("intro", add_special_tokens=False).input_ids
    target_apply = tokenizer("apply", add_special_tokens=False).input_ids

    data = list(iter_jsonl(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0

    with open(output_path, "w", encoding="utf-8") as f_out, torch.inference_mode():
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch = data[i : i + args.batch_size]
            states = [ex["state"] for ex in batch]
            labels = [ex.get("label") for ex in batch]

            inputs = tokenizer(
                states,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logp_intro = logp_for_target(model, inputs, target_intro)
            logp_apply = logp_for_target(model, inputs, target_apply)
            score = logp_intro - logp_apply

            for j, ex in enumerate(batch):
                pred = "intro" if score[j].item() > 0 else "apply"
                label = labels[j]
                if label in ("intro", "apply"):
                    correct += 1 if pred == label else 0
                    total += 1
                out = {
                    "state": ex["state"],
                    "label": label,
                    "logp_intro": logp_intro[j].item(),
                    "logp_apply": logp_apply[j].item(),
                    "score": score[j].item(),
                    "pred": pred,
                }
                f_out.write(json.dumps(out, ensure_ascii=True) + "\n")

    if total:
        acc = correct / total
        print(f"accuracy: {acc:.4f} ({correct}/{total})")
    else:
        print("no labeled examples found")

if __name__ == "__main__":
    main()
