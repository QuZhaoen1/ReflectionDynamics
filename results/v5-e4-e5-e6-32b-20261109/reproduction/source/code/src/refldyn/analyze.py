import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .io import flatten, write_json
from .metrics import cluster_bootstrap, summarize, validate_results


def match_checkpoints(groups, require_exact=True):
    comparisons = {}
    for (split, checkpoint), rows in groups.items():
        comparisons[(split, checkpoint)] = {r["id"] for r in rows}
    for split in set(k[0] for k in comparisons):
        others = [(name, ids) for (s, name), ids in comparisons.items() if s == split]
        if others:
            anchor = others[0][1]
            for name, ids in others[1:]:
                if require_exact and ids != anchor:
                    raise ValueError(f"Incomplete checkpoint panel for {split}: {name}")
    return True


def summarize_results(rows, bootstraps=400, seed=41, cost_weight=0.0):
    valid = [row for row in rows if not row.get("skip_reason")]
    validate_results(valid)
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["split"], row["checkpoint"])].append(row)
    match_checkpoints(grouped)
    summaries = []
    for (split, checkpoint), values in sorted(grouped.items()):
        report = summarize(values, cost_weight)
        report.update({"split": split, "checkpoint": checkpoint})
        intervals = cluster_bootstrap(values, bootstrap=bootstraps, seed=seed, cost_weight=cost_weight)
        for key, bounds in intervals.items():
            report[f"{key}_ci_low"], report[f"{key}_ci_high"] = bounds
        summaries.append(report)
    return summaries


def cross_checkpoint_matrix(rows, cost_weight=0.0):
    from .metrics import per_state
    groups = defaultdict(dict)
    for row in rows:
        if row.get("skip_reason"):
            continue
        state = per_state(row, cost_weight)
        groups[row["checkpoint"]][row["id"]] = state
    names = sorted(groups)
    if not names:
        raise ValueError("Empty cross-checkpoint panel")
    common = set(groups[names[0]])
    if any(set(groups[name]) != common for name in names):
        raise ValueError("Cross-checkpoint swap requires identical state IDs")
    matrix = []
    for decision_checkpoint in names:
        for execution_checkpoint in names:
            pairs = [(groups[decision_checkpoint][key]["p_reflect"],
                      groups[execution_checkpoint][key]["u"]) for key in sorted(common)]
            policy_gain = sum(p * u for p, u in pairs) / len(pairs)
            matrix.append({"decision_checkpoint": decision_checkpoint,
                           "execution_checkpoint": execution_checkpoint,
                           "states": len(pairs), "expected_gain": policy_gain})
    return matrix


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cost-weight", type=float, default=0.0)
    args = parser.parse_args()
    records = flatten(args.input)
    summaries = summarize_results(records, args.bootstrap, args.seed, args.cost_weight)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "checkpoints.csv", index=False)
    write_json(destination / "checkpoints.json", summaries)
    for split in set(row["split"] for row in records if not row.get("skip_reason")):
        subset = [row for row in records if row.get("split") == split and not row.get("skip_reason")]
        matrix = cross_checkpoint_matrix(subset, args.cost_weight)
        pd.DataFrame(matrix).to_csv(destination / f"cross_checkpoint_{split}.csv", index=False)
    print(f"Saved {len(summaries)} checkpoint summaries to {destination}")


if __name__ == "__main__":
    run()
