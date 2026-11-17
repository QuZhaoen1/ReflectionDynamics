import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .trajectory import checkpoint_step


def plot_summary(table, split, output_dir):
    frame = table[table["split"] == split].copy()
    frame["step"] = frame["checkpoint"].map(checkpoint_step)
    frame = frame.dropna(subset=["step"]).sort_values("step")
    if frame.empty:
        raise ValueError(f"No checkpoints in split: {split}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for key in ["reflection_propensity", "selectivity", "repair_rate", "damage_rate", "natural_accuracy"]:
        panel = frame.dropna(subset=[key])
        if panel.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(panel["step"], panel[key], marker="o", linewidth=1.5)
        low, high = key + "_ci_low", key + "_ci_high"
        if low in panel.columns and high in panel.columns:
            ax.fill_between(panel["step"].to_numpy(dtype=float),
                            panel[low].to_numpy(dtype=float), panel[high].to_numpy(dtype=float), alpha=0.2)
        ax.set(xlabel="RL training step", ylabel=key.replace("_", " ").title(), title=f"{key.replace('_', ' ').title()} | {split}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{split}_{key}.png", dpi=180)
        plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    plot_summary(pd.read_csv(args.checkpoints), args.split, args.output_dir)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    run()
