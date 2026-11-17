import argparse
import collections
from pathlib import Path

from .answers import gold_from_gsm8k, gold_from_math, validate_gold
from .io import canonical_question, digest, load_jsonl, save_jsonl, stable_int, write_json


def adapt_math(record, original_split):
    question = str(record.get("problem", "")).strip()
    answer = gold_from_math(record.get("solution", ""), record.get("extracted_solution"))
    if not question or not answer:
        return None
    level = str(record.get("level", "unknown"))
    return {"question": question, "gold": answer, "source": "math", "source_split": original_split,
            "level": level, "topic": str(record.get("type", "unknown"))}


def adapt_gsm8k(record, original_split):
    question = str(record.get("question", "")).strip()
    answer = gold_from_gsm8k(record.get("answer", ""))
    if not question or answer is None:
        return None
    return {"question": question, "gold": answer, "source": "gsm8k", "source_split": original_split,
            "level": "word_problem", "topic": "arithmetic_word_problem"}


def adapt_aime(record):
    question = str(record.get("problem") or record.get("prompt") or "").strip()
    answer = str(record.get("answer", "")).strip()
    if not question or not answer:
        return None
    return {"question": question, "gold": answer, "source": "aime", "source_split": "benchmark",
            "level": "competition", "topic": "olympiad"}


def designate(source, source_split, question):
    if source == "aime":
        return "ood"
    if source_split == "test":
        return "test"
    return "validation" if stable_int(canonical_question(question)) % 10 == 0 else "train"


def build_corpus(records, audit_limit=50, validate=True):
    priority = {"ood": 4, "test": 3, "validation": 2, "train": 1}
    accepted, rejected = {}, []
    for row in records:
        question = row["question"].strip()
        candidate = dict(row)
        candidate["split"] = designate(row["source"], row["source_split"], question)
        candidate["id"] = digest(canonical_question(question))[:20]
        if validate and not validate_gold(candidate["gold"]):
            rejected.append({"id": candidate["id"], "reason": "unverifiable_gold", "source": row["source"]})
            continue
        existing = accepted.get(candidate["id"])
        if existing is None:
            accepted[candidate["id"]] = candidate
        elif priority[candidate["split"]] > priority[existing["split"]]:
            accepted[candidate["id"]] = candidate
        elif str(existing["gold"]) != str(candidate["gold"]):
            rejected.append({"id": candidate["id"], "reason": "duplicate_conflicting_gold", "source": row["source"]})
    result = sorted(accepted.values(), key=lambda row: row["id"])
    if not result:
        raise ValueError("Corpus is empty after grading audit")
    counts = dict(collections.Counter(row["split"] for row in result))
    return result, rejected[:audit_limit], {"splits": counts, "rejected_total": len(rejected), "accepted": len(result)}


def download_sources(include_gsm8k=False, include_aime=False, revision=None):
    from datasets import load_dataset
    rows = []
    kwargs = {"revision": revision} if revision else {}
    for split in ("train", "test"):
        for raw in load_dataset("jeggers/competition_math", "original", split=split, **kwargs):
            item = adapt_math(raw, split)
            if item:
                rows.append(item)
    if include_gsm8k:
        for split in ("train", "test"):
            for raw in load_dataset("openai/gsm8k", "main", split=split):
                item = adapt_gsm8k(raw, split)
                if item:
                    rows.append(item)
    if include_aime:
        for raw in load_dataset("AI-MO/aimo-validation-aime", split="train"):
            item = adapt_aime(raw)
            if item:
                rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/corpus")
    parser.add_argument("--input-jsonl", help="Offline normalized rows for testing; not a full benchmark")
    parser.add_argument("--include-gsm8k", action="store_true")
    parser.add_argument("--include-aime", action="store_true")
    parser.add_argument("--math-revision", default=None)
    parser.add_argument("--preview-limit", type=int, default=0)
    args = parser.parse_args()
    records = (list(load_jsonl(args.input_jsonl)) if args.input_jsonl else
               download_sources(args.include_gsm8k, args.include_aime, args.math_revision))
    if args.preview_limit:
        records = sorted(records, key=lambda x: digest(x["question"]))[:args.preview_limit]
    corpus, rejected, summary = build_corpus(records)
    for row in corpus:
        row["fixture"] = bool(args.preview_limit or args.input_jsonl)
    destination = Path(args.output)
    for split in ("train", "validation", "test", "ood"):
        save_jsonl(destination / f"{split}.jsonl", (r for r in corpus if r["split"] == split))
    save_jsonl(destination / "rejected.jsonl", rejected)
    summary.update({"dataset": "jeggers/competition_math:original", "math_revision": args.math_revision or "main (unlocked)",
                    "includes_gsm8k": args.include_gsm8k, "includes_aime": args.include_aime,
                    "preview": bool(args.preview_limit or args.input_jsonl), "strict_gold_check": True})
    write_json(destination / "manifest.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
