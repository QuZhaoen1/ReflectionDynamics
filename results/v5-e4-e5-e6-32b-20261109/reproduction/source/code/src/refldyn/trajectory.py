import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def checkpoint_step(text):
    match = re.search(r"(?:step|checkpoint)[_-]?(\d+)", str(text))
    if match:
        return int(match.group(1))
    return 0 if str(text).lower() == "base" else None


def fit_transition(steps, values):
    x = np.asarray(steps, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 8 or len(x) != len(np.unique(x)):
        raise ValueError("Require at least eight distinct checkpoints")
    indices = np.argsort(x)
    x, y = x[indices], y[indices]
    scale = max(x[-1] - x[0], 1)
    u = (x - x[0]) / scale

    def fit(columns, parameters):
        design = np.column_stack(columns)
        coeff, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        rss = float(np.sum((y - design @ coeff) ** 2))
        bic = len(y) * np.log(max(rss / len(y), 1e-12)) + parameters * np.log(len(y))
        return float(bic), rss

    linear = fit([np.ones(len(u)), u], 2)
    smooth = fit([np.ones(len(u)), u, u * u], 3)
    segmented = []
    for knot in u[3:-3]:
        bic, rss = fit([np.ones(len(u)), u, np.maximum(0, u - knot)], 4)
        segmented.append({"knot": float(knot), "bic": bic, "rss": rss})
    best = min(segmented, key=lambda row: row["bic"])
    return {"steps": x.tolist(), "linear_bic": linear[0], "quadratic_bic": smooth[0],
            "segmented_bic": best["bic"], "relative_knot": best["knot"],
            "delta_bic_segmented_minus_smooth": best["bic"] - min(linear[0], smooth[0]),
            "interpretation": "exploratory change-point diagnostic; not evidence of a physical phase transition"}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--metric", choices=["reflection_propensity", "selectivity", "repair_rate", "damage_rate"], default="selectivity")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .io import write_json
    rows = pd.read_csv(args.checkpoints)
    rows = rows[rows["split"] == args.split].copy()
    rows["step"] = rows["checkpoint"].map(checkpoint_step)
    rows = rows.dropna(subset=["step", args.metric])
    result = fit_transition(rows["step"], rows[args.metric])
    result.update({"metric": args.metric, "split": args.split})
    write_json(args.output, result)
    print(f"Exploratory BIC difference: {result['delta_bic_segmented_minus_smooth']:.3f}")


if __name__ == "__main__":
    run()
