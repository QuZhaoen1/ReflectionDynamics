import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .io import flatten, save_jsonl, write_json
from .selection import inspect_drafts


def wilson_interval(correct, total, z=1.96):
    if total <= 0:
        return [None, None]
    p = correct / total
    z2 = z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / (1 + z2 / total)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def screen_drafts(rows, split, minimum_samples=8, min_valid_rate=0.75, middle=(0.15, 0.85)):
    if split not in {"train", "validation", "test", "ood"}:
        raise ValueError("Unsupported data partition")
    if minimum_samples < 2 or not 0 < min_valid_rate <= 1:
        raise ValueError("Invalid screening thresholds")
    if not (0 < middle[0] < middle[1] < 1):
        raise ValueError("Difficulty boundaries are invalid")
    groups = inspect_drafts(rows)
    details = []
    for problem_id, samples in sorted(groups.items()):
        annotations = {(item.get("split"), item.get("model"), str(item.get("gold")),
                        str(item.get("question"))) for item in samples}
        if len(annotations) != 1:
            raise ValueError(f"Inconsistent source, model, or answer for {problem_id}")
        if samples[0].get("split") != split:
            raise ValueError(f"Encountered {samples[0].get('split')} in {split} screening")
        attempted = len(samples)
        valid = sum(item.get("valid") is True and isinstance(item.get("correct"), bool) for item in samples)
        correct = sum(item.get("correct") is True and item.get("valid") is True for item in samples)
        parse_rate = valid / attempted
        accuracy = correct / attempted
        eligible = attempted >= minimum_samples and parse_rate >= min_valid_rate
        band = "too_easy" if accuracy > middle[1] else ("too_hard" if accuracy < middle[0] else "informative")
        details.append({"problem_id": problem_id, "split": split,
                        "model": samples[0].get("model"), "level": samples[0].get("level", "unknown"),
                        "topic": samples[0].get("topic", "unknown"), "samples": attempted,
                        "valid": valid, "correct": correct, "parse_rate": parse_rate,
                        "pass_at_1_all_attempts": accuracy,
                        "pass_at_1_valid_only": correct / valid if valid else None,
                        "wilson_95_all_attempts": wilson_interval(correct, attempted),
                        "eligible": eligible, "difficulty_band": band if eligible else "insufficient_evidence",
                        "reason": None if eligible else ("too_few_samples" if attempted < minimum_samples
                                                       else "too_many_unparsed")})
    if not details:
        raise ValueError("No drafts available for screening")
    sample_counts = Counter(r["samples"] for r in details)
    categories = Counter(r["difficulty_band"] for r in details)
    level_summary = {}
    for level in sorted({r["level"] for r in details}):
        section = [r for r in details if r["level"] == level and r["eligible"]]
        if section:
            level_summary[level] = {"problems": len(section),
                                    "observed_accuracy": sum(r["correct"] for r in section) /
                                    sum(r["samples"] for r in section),
                                    "informative": sum(r["difficulty_band"] == "informative" for r in section)}
    report = {"split": split, "problems": len(details), "eligible_problems": sum(r["eligible"] for r in details),
              "draft_samples": sum(r["samples"] for r in details),
              "observed_answer_accuracy": sum(r["correct"] for r in details) /
              sum(r["samples"] for r in details),
              "observed_parse_rate": sum(r["valid"] for r in details) /
              sum(r["samples"] for r in details),
              "difficulty_bands": dict(categories), "samples_per_problem": dict(sample_counts),
              "by_level": level_summary, "thresholds": {"minimum_samples": minimum_samples,
                            "min_parse_rate": min_valid_rate, "informative_accuracy": list(middle)},
              "interpretation": "Descriptive calibration of sampled model behavior; not ground-truth problem difficulty."
              }
    return details, report


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", nargs="+", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test", "ood"])
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--min-parse-rate", type=float, default=0.75)
    parser.add_argument("--low", type=float, default=0.15)
    parser.add_argument("--high", type=float, default=0.85)
    parser.add_argument("--output-dir", default="data/calibration")
    args = parser.parse_args()
    details, report = screen_drafts(flatten(args.drafts), args.split, args.minimum_samples,
                                    args.min_parse_rate, (args.low, args.high))
    output = Path(args.output_dir)
    save_jsonl(output / f"{args.split}.jsonl", details)
    write_json(output / f"{args.split}.summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
