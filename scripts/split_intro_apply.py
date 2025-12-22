#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with state/label")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance", action="store_true")
    return parser.parse_args()


def write_jsonl(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    data = [ex for ex in iter_jsonl(args.input) if ex.get("label") in ("intro", "apply")]
    if args.balance:
        by_label = {"intro": [], "apply": []}
        for ex in data:
            by_label[ex["label"]].append(ex)
        for k in by_label:
            rng.shuffle(by_label[k])
        train_per = args.train_size // 2
        test_per = args.test_size // 2
        train = by_label["intro"][:train_per] + by_label["apply"][:train_per]
        test = (
            by_label["intro"][train_per : train_per + test_per]
            + by_label["apply"][train_per : train_per + test_per]
        )
        rng.shuffle(train)
        rng.shuffle(test)
    else:
        rng.shuffle(data)
        train = data[: args.train_size]
        test = data[args.train_size : args.train_size + args.test_size]

    write_jsonl(args.train_output, train)
    write_jsonl(args.test_output, test)

    def counts(items):
        intro = sum(1 for ex in items if ex.get("label") == "intro")
        apply = sum(1 for ex in items if ex.get("label") == "apply")
        return intro, apply

    train_intro, train_apply = counts(train)
    test_intro, test_apply = counts(test)
    print(f"train: {len(train)} intro={train_intro} apply={train_apply}")
    print(f"test: {len(test)} intro={test_intro} apply={test_apply}")


if __name__ == "__main__":
    main()
