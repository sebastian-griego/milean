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

def parse_layers(arg, num_layers, skip_embedding):
    if arg == "all":
        start = 1 if skip_embedding else 0
        return list(range(start, num_layers))
    out = []
    for part in arg.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL with state/label")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--model", default="kaiyuy/leandojo-lean4-tacgen-byt5-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--layers", default="all", help="Comma-separated layer indices or 'all'")
    parser.add_argument("--include-embedding", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    raw = list(iter_jsonl(args.data))
    label_names = ["apply", "intro"]
    label_to_id = {name: i for i, name in enumerate(label_names)}
    data = []
    label_ids = []
    for ex in raw:
        lbl = ex.get("label")
        if lbl in label_to_id:
            data.append(ex)
            label_ids.append(label_to_id[lbl])
    print(f"using {len(data)} labeled examples")

    encoder = model.get_encoder()

    example_pools = {}
    layer_ids = None

    skip_embedding = not args.include_embedding

    with torch.inference_mode():
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch = data[i : i + args.batch_size]
            states = [ex["state"] for ex in batch]

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

            if layer_ids is None:
                layer_ids = parse_layers(args.layers, len(hidden_states), skip_embedding)
                example_pools = {layer: [] for layer in layer_ids}

            mask = inputs["attention_mask"].unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1)

            for layer in layer_ids:
                h = hidden_states[layer]
                pooled = (h * mask).sum(dim=1) / denom
                example_pools[layer].append(pooled.float().cpu())

    layers_out = {f"layer_{layer}": torch.cat(example_pools[layer], dim=0) for layer in layer_ids}
    labels_tensor = torch.tensor(label_ids, dtype=torch.long)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layers": layers_out,
            "layer_ids": layer_ids,
            "labels": labels_tensor,
            "label_names": label_names,
            "model": args.model,
        },
        output_path,
    )

    print(f"saved activations to {output_path}")

if __name__ == "__main__":
    main()
