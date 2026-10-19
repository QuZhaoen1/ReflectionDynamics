import argparse
import collections
import json
import random
from pathlib import Path

from .io import flatten, load_jsonl, save_jsonl, write_json, digest, stable_int


def inspect_drafts(rows):
    grouped = collections.defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["id"], row["sample"])
        if key in seen:
            raise ValueError(f"Duplicate draft {key}")
        seen.add(key)
        grouped[row["id"]].append(row)
    return grouped


def build_states(rows, mode, seed=41, minimum_samples=8, max_states_per_problem=2,
                 min_valid_rate=0.75, easy_keep=0.30, hard_keep=0.50):
    if mode not in {"train", "validation", "test", "ood"}:
        raise ValueError(f"Unknown split: {mode}")
    if minimum_samples < 2 or not 0 < min_valid_rate <= 1 or not 0 <= easy_keep <= 1 or not 0 <= hard_keep <= 1:
        raise ValueError("Invalid sampling thresholds")
    groups = inspect_drafts(rows)
    for problem_id, entries in groups.items():
        identity = {(r.get("split"), r.get("model"), str(r.get("gold")), str(r.get("question")))
                    for r in entries}
        if len(identity) != 1:
            raise ValueError(f"Inconsistent draft source for {problem_id}")
    states, audit, bins = [], [], collections.Counter()
    for problem_id in sorted(groups):
        rng = random.Random(stable_int(f"selection:{seed}:{problem_id}"))
        samples = groups[problem_id]
        if any(row.get("split") != mode for row in samples):
            raise ValueError(f"Wrong split in {mode}: {problem_id}")
        if len(samples) < minimum_samples:
            audit.append({"id": problem_id, "reason": "insufficient_samples"})
            continue
        valid = [row for row in samples if row.get("valid") is True and isinstance(row.get("correct"), bool)]
        if len(valid) / len(samples) < min_valid_rate:
            audit.append({"id": problem_id, "reason": "low_parse_rate", "valid": len(valid), "total": len(samples)})
            continue
        accuracy = sum(row["correct"] for row in valid) / len(samples)
        difficulty = "hard" if accuracy < 0.25 else "middle" if accuracy <= 0.75 else "easy"
        bins[difficulty] += 1
        if mode == "train" and difficulty == "easy" and rng.random() > easy_keep:
            audit.append({"id": problem_id, "reason": "easy_subsample", "pass_fraction": accuracy})
            continue
        if mode == "train" and difficulty == "hard" and rng.random() > hard_keep:
            audit.append({"id": problem_id, "reason": "hard_subsample", "pass_fraction": accuracy})
            continue
        correct = [r for r in valid if r["correct"]]
        wrong = [r for r in valid if not r["correct"]]
        rng.shuffle(correct)
        rng.shuffle(wrong)
        selection = (wrong[:1] + correct[:1])[:max_states_per_problem]
        if not selection:
            audit.append({"id": problem_id, "reason": "empty_selection"})
            continue
        for row in selection:
            state = {
                "id": f"{problem_id}-s{row['sample']}", "problem_id": problem_id,
                "split": mode, "question": row["question"], "gold": row["gold"],
                "initial_reasoning": row["trace"], "initial_answer": row["initial_answer"],
                "initial_correct": row["correct"], "source": row["source"],
                "level": row.get("level", "unknown"), "topic": row.get("topic", "unknown"),
                "baseline_pass_fraction": accuracy, "baseline_parse_fraction": len(valid) / len(samples),
                "baseline_draws": len(samples), "baseline_bin": difficulty, "draft_model": row.get("model"),
                "draft_adapter": row.get("adapter"), "fixture": row.get("fixture", False),
                "original_seed": row.get("seed"), "sample_id": row["sample"],
            }
            states.append(state)
    if not states:
        raise ValueError(f"No valid states for {mode}; inspect the draft audit")
    return states, audit, {"states": len(states), "problems": len(set(x["problem_id"] for x in states)),
                           "difficulty_bins_before_filter": dict(bins),
                           "correct_states": sum(r["initial_correct"] for r in states),
                           "min_valid_rate": min_valid_rate, "minimum_samples": minimum_samples,
                           "easy_keep": easy_keep, "hard_keep": hard_keep}


def ensure_isolation(paths):
    all_problem_ids = {}
    for split, path in paths.items():
        for row in load_jsonl(path):
            if row["split"] != split:
                raise ValueError(f"Incorrect split annotation in {path}")
            problem_id = row["problem_id"]
            if problem_id in all_problem_ids and all_problem_ids[problem_id] != split:
                raise ValueError(f"Leakage: {problem_id} in {all_problem_ids[problem_id]} and {split}")
            all_problem_ids[problem_id] = split
    return {"unique_problems": len(all_problem_ids), "splits": list(paths)}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "ood"], required=True)
    parser.add_argument("--output-dir", default="data/states")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--min-valid-rate", type=float, default=0.75)
    parser.add_argument("--easy-keep", type=float, default=0.30)
    parser.add_argument("--hard-keep", type=float, default=0.50)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    rows = flatten(args.drafts)
    states, audit, report = build_states(rows, args.split, args.seed, args.minimum_samples,
                                         min_valid_rate=args.min_valid_rate,
                                         easy_keep=args.easy_keep, hard_keep=args.hard_keep)
    output = Path(args.output_dir)
    destination = Path(args.output_file) if args.output_file else output / f"{args.split}.jsonl"
    save_jsonl(destination, states)
    save_jsonl(output / f"{args.split}.audit.jsonl", audit)
    from .io import checksum_file
    report.update({"states_file": str(destination), "states_sha256": checksum_file(destination)})
    write_json(output / f"{args.split}.manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
