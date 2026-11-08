import argparse
from pathlib import Path

from .answers import extract_final, verify_answer
from .backend import InferenceBackend
from .io import digest, load_jsonl, save_jsonl, stable_int, write_json, checksum_file


def draft_prompt(question):
    return (
        "Solve the following competition mathematics problem. Explain your calculation and put "
        "the final answer in \\boxed{...}. Use a single boxed final answer.\n\n"
        f"Problem:\n{question}\n\nSolution:\n"
    )


def collect_draft(backend, row, seed, max_new_tokens, prompt_limit=1536):
    prompt = draft_prompt(row["question"])
    if len(backend.tokenizer.encode(prompt, add_special_tokens=False)) > prompt_limit:
        return {"id": row["id"], "split": row["split"], "question": row["question"],
                "gold": row["gold"], "source": row["source"], "level": row.get("level", "unknown"),
                "topic": row.get("topic", "unknown"), "trace": "", "initial_answer": None,
                "correct": None, "generated_tokens": 0, "seed": seed, "fixture": row.get("fixture", False),
                "reason": "prompt_too_long", "valid": False}
    text, count = backend.generate(prompt, seed, max_new_tokens, sample=True)
    answer = extract_final(text)
    correctness = verify_answer(row["gold"], answer)
    return {"id": row["id"], "split": row["split"], "question": row["question"],
            "gold": row["gold"], "source": row["source"], "level": row.get("level", "unknown"),
            "topic": row.get("topic", "unknown"), "trace": text, "initial_answer": answer,
            "correct": correctness, "valid": correctness is not None, "generated_tokens": count,
            "seed": seed, "fixture": row.get("fixture", False)}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 2 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require >=2 samples and valid shard indices")
    questions = list(load_jsonl(args.questions))
    if args.limit:
        questions = sorted(questions, key=lambda x: digest(x["id"]))[:args.limit]
    questions = [row for row in questions if stable_int(row["id"]) % args.num_shards == args.shard_index]
    if not questions:
        raise ValueError("Shard contains no questions")
    backend = InferenceBackend(args.model, args.adapter, args.max_memory_gib, args.temperature)
    path = Path(args.output)
    manifest_path = Path(str(path) + ".manifest.json")
    fingerprint = {"model": args.model, "adapter": args.adapter,
                   "split_file_sha256": checksum_file(args.questions),
                   "samples_per_question": args.samples, "temperature": args.temperature,
                   "seed": args.seed, "tokens_max": args.max_new_tokens,
                   "prompt_tokens_max": args.max_prompt_tokens, "limit": args.limit,
                   "num_shards": args.num_shards, "shard_index": args.shard_index}
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are incompatible")
    prior = {}
    if path.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(f"Output already exists: {path}. Use --resume or --overwrite")
        if not manifest_path.exists():
            raise ValueError("Cannot resume without drafting manifest")
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != value for k, value in fingerprint.items() if k != "limit"):
            raise ValueError("Cannot resume with modified drafting parameters or corpus")
        old_limit = int(manifest.get("limit", 0))
        if (old_limit == 0 and args.limit != 0) or (old_limit and args.limit and args.limit < old_limit):
            raise ValueError("Cannot shrink a resumed drafting panel")
        prior = {(r["id"], r["sample"]): r for r in load_jsonl(path)}
        allowed = {r["id"] for r in questions}
        for (problem_id, sample_id), item in prior.items():
            expected_seed = stable_int(f"{args.seed}:{problem_id}:{sample_id}") % (2**31 - 1)
            if (item.get("model") != args.model or item.get("adapter") != args.adapter
                    or item.get("seed") != expected_seed or problem_id not in allowed):
                raise ValueError("Resume contains a different model, problem panel, or sampling seed")
    manifest = {"stage": "baseline_drafts", "split_file": args.questions, **fingerprint,
                "rows": len(prior), "questions": len(questions), "invalid": 0}
    write_json(manifest_path, manifest)
    records = list(prior.values())
    for index, row in enumerate(questions, 1):
        for sample in range(args.samples):
            if (row["id"], sample) in prior:
                continue
            sampling_seed = stable_int(f"{args.seed}:{row['id']}:{sample}") % (2**31 - 1)
            draft = collect_draft(backend, row, sampling_seed, args.max_new_tokens, args.max_prompt_tokens)
            draft["sample"] = sample
            draft["model"] = args.model
            draft["adapter"] = args.adapter
            records.append(draft)
        if index % 20 == 0:
            save_jsonl(path, records)
            manifest["rows"] = len(records)
            write_json(manifest_path, manifest)
            print(f"Drafts: {index}/{len(questions)}", flush=True)
    records.sort(key=lambda row: (row["id"], row["sample"]))
    save_jsonl(path, records)
    manifest.update({"rows": len(records), "invalid": sum(1 for row in records if not row.get("valid", False))})
    write_json(manifest_path, manifest)
    print(f"Saved {len(records)} drafts to {path}")


if __name__ == "__main__":
    run()
