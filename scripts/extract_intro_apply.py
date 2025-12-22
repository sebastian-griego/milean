#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

def iter_files(input_path, split=None):
    path = Path(input_path)
    if path.is_file():
        yield path
        return
    exts = {".jsonl", ".json"}
    files = [p for p in path.rglob("*") if p.suffix in exts]
    if split:
        files = [p for p in files if split in p.name]
    for p in files:
        yield p

def first_non_ws_char(text):
    for ch in text:
        if not ch.isspace():
            return ch
    return ""

def iter_json_objects(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        first = first_non_ws_char(head)
        if first == "[":
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj

def find_first_str(dct, keys):
    for key in keys:
        val = dct.get(key)
        if isinstance(val, str):
            return val
    return None

def extract_state_tactic(obj, state_keys, tactic_keys):
    state = find_first_str(obj, state_keys)
    tactic = find_first_str(obj, tactic_keys)
    if state and tactic:
        return state, tactic
    for nest_key in ["data", "datum", "example", "sample", "info", "item", "step", "proofstep"]:
        nested = obj.get(nest_key)
        if isinstance(nested, dict):
            state = state or find_first_str(nested, state_keys)
            tactic = tactic or find_first_str(nested, tactic_keys)
    return state, tactic


def iter_traced_steps(obj):
    traced = obj.get("traced_tactics")
    if isinstance(traced, list):
        for step in traced:
            if isinstance(step, dict):
                yield step

def label_from_tactic(tactic):
    t = tactic.strip()
    if not t:
        return None
    if t.startswith("by"):
        return None
    if t.startswith("intro?") or t.startswith("apply?"):
        return None
    if t.startswith("intro"):
        return "intro"
    if t.startswith("apply"):
        return "apply"
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="File or directory containing JSON/JSONL data")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--split", default=None, help="Only read files whose name contains this substring")
    parser.add_argument("--state-key", action="append", default=[], help="Override/extend state keys")
    parser.add_argument("--tactic-key", action="append", default=[], help="Override/extend tactic keys")
    parser.add_argument("--max-state-chars", type=int, default=2000)
    parser.add_argument("--min-state-chars", type=int, default=0)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--exact-per-class", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle-files", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    default_state_keys = [
        "state",
        "proof_state",
        "goal",
        "goal_state",
        "pretty_printed_state",
        "state_before",
    ]
    default_tactic_keys = ["tactic", "tactic_str", "tactic_string", "next_tactic"]
    state_keys = args.state_key + default_state_keys
    tactic_keys = args.tactic_key + default_tactic_keys

    files = list(iter_files(args.input, args.split))
    if not args.no_shuffle_files:
        random.shuffle(files)
    else:
        files.sort()

    by_label = {"intro": [], "apply": []}
    for path in files:
        for obj in iter_json_objects(path):
            steps = list(iter_traced_steps(obj))
            if steps:
                for step in steps:
                    state, tactic = extract_state_tactic(step, state_keys, tactic_keys)
                    if not state or not tactic:
                        continue
                    state = state.strip()
                    if args.max_state_chars and len(state) > args.max_state_chars:
                        continue
                    if args.min_state_chars and len(state) < args.min_state_chars:
                        continue
                    label = label_from_tactic(tactic)
                    if label is None:
                        continue
                    by_label[label].append(
                        {
                            "state": state,
                            "label": label,
                            "tactic": tactic.strip(),
                            "source": str(path),
                            "full_name": obj.get("full_name"),
                        }
                    )
            else:
                state, tactic = extract_state_tactic(obj, state_keys, tactic_keys)
                if not state or not tactic:
                    continue
                state = state.strip()
                if args.max_state_chars and len(state) > args.max_state_chars:
                    continue
                if args.min_state_chars and len(state) < args.min_state_chars:
                    continue
                label = label_from_tactic(tactic)
                if label is None:
                    continue
                by_label[label].append(
                    {
                        "state": state,
                        "label": label,
                        "tactic": tactic.strip(),
                        "source": str(path),
                    }
                )
        if args.max_per_class is not None:
            if all(len(by_label[k]) >= args.max_per_class for k in by_label):
                break

    if args.balance:
        n = min(len(by_label["intro"]), len(by_label["apply"]))
        by_label["intro"] = random.sample(by_label["intro"], n) if len(by_label["intro"]) > n else by_label["intro"]
        by_label["apply"] = random.sample(by_label["apply"], n) if len(by_label["apply"]) > n else by_label["apply"]

    if args.exact_per_class and args.max_per_class is not None:
        for label in ["intro", "apply"]:
            if len(by_label[label]) < args.max_per_class:
                raise ValueError(f"not enough {label} examples for exact-per-class")
            by_label[label] = random.sample(by_label[label], args.max_per_class)

    examples = by_label["intro"] + by_label["apply"]
    random.shuffle(examples)

    if args.max_total is not None and len(examples) > args.max_total and not args.exact_per_class:
        examples = random.sample(examples, args.max_total)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")

    counts = {k: len(v) for k, v in by_label.items()}
    print(f"wrote {len(examples)} examples to {output_path}")
    print(f"counts: intro={counts['intro']} apply={counts['apply']}")

if __name__ == "__main__":
    main()
