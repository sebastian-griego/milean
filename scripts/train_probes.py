#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True, help=".pt file from save_activations.py")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--plot", default=None, help="Optional output plot path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    data = torch.load(args.activations, map_location="cpu")

    layers = data["layers"]
    labels = data["labels"].numpy()

    results = []
    for layer_name in sorted(layers.keys(), key=lambda x: int(x.split("_")[1])):
        X = layers[layer_name].numpy().astype(np.float32)
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="liblinear"),
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        layer_idx = int(layer_name.split("_")[1])
        results.append((layer_idx, acc, len(y_train), len(y_test)))

    results.sort(key=lambda x: x[0])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "accuracy", "n_train", "n_test"])
        for row in results:
            writer.writerow(row)

    best = max(results, key=lambda x: x[1]) if results else None
    if best:
        print(f"best layer: {best[0]} acc={best[1]:.4f}")

    if args.plot:
        layers_idx = [r[0] for r in results]
        accs = [r[1] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(layers_idx, accs, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Probe accuracy")
        plt.title("Intro vs apply probe accuracy by layer")
        plt.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)

if __name__ == "__main__":
    main()
