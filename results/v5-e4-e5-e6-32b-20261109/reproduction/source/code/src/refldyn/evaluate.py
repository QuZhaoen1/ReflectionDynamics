import argparse
import json
from pathlib import Path

from .answers import extract_final, verify_answer
from .backend import InferenceBackend
from .io import load_jsonl, save_jsonl, stable_int, write_json, checksum_file
from .protocol import forced_reflection_prefix, independent_retry_prompt, parse_decision, prompt_for_state


def evaluate_state(backend, row, checkpoint, seed, repetitions, max_new_tokens, retry=False,
                   max_prompt_tokens=3200):
    prompt = prompt_for_state(row)
    if len(backend.tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
        return {"id": row["id"], "problem_id": row["problem_id"],
                "split": row["split"], "checkpoint": checkpoint, "skip_reason": "prompt_too_long"}
    p_reflect, likelihoods = backend.action_probability(prompt)
    natural_seed = stable_int(f"{seed}:natural:{row['id']}") % (2**31 - 1)
    natural_text, natural_tokens = backend.generate(prompt, natural_seed, max_new_tokens, sample=True)
    natural = parse_decision(natural_text, row["initial_answer"])
    natural_score = verify_answer(row["gold"], natural.answer) if natural.valid else None
    reflect_samples, retry_samples = [], []
    for number in range(repetitions):
        sample_seed = stable_int(f"{seed}:intervene:{row['id']}:{number}") % (2**31 - 1)
        forced = forced_reflection_prefix()
        continuation, generated_tokens = backend.generate(prompt + forced, sample_seed, max_new_tokens, sample=True)
        decision = parse_decision(forced + continuation, row["initial_answer"])
        score = verify_answer(row["gold"], decision.answer) if decision.valid else None
        reflect_samples.append({"sample": number, "seed": sample_seed, "tokens": generated_tokens,
                                "valid": decision.valid, "correct": score is True, "answer": decision.answer,
                                "completion": forced + continuation})
        if retry:
            text, tokens = backend.generate(independent_retry_prompt(row), sample_seed, max_new_tokens, sample=True)
            answer = extract_final(text)
            grade = verify_answer(row["gold"], answer)
            retry_samples.append({"sample": number, "seed": sample_seed, "tokens": tokens,
                                  "valid": grade is not None, "correct": grade is True, "answer": answer,
                                  "completion": text})
    return {"id": row["id"], "problem_id": row["problem_id"], "split": row["split"],
            "source": row["source"], "level": row.get("level", "unknown"),
            "checkpoint": checkpoint, "initial_correct": row["initial_correct"],
            "p_reflect": p_reflect, "log_likelihood_commit": likelihoods[0],
            "log_likelihood_reflect": likelihoods[1], "natural_action": natural.action,
            "natural_valid": natural.valid, "natural_correct": natural_score is True,
            "natural_tokens": natural_tokens, "natural_completion": natural_text,
            "reflect_samples": reflect_samples, "retry_samples": retry_samples}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--adapter")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=3200)
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--retry-control", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid evaluation settings")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are incompatible")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    entire = sorted(load_jsonl(args.states), key=lambda row: stable_int(row["id"]))
    if args.limit:
        entire = entire[:args.limit]
    states = [row for row in entire if stable_int(row["id"]) % args.num_shards == args.shard_index]
    if not states:
        raise ValueError("No states in evaluation shard")
    output = Path(args.output)
    manifest_path = Path(str(output) + ".manifest.json")
    fingerprint = {"states_sha256": checksum_file(args.states), "model": args.model,
                   "adapter": args.adapter, "checkpoint": args.checkpoint,
                   "repetitions": args.repetitions, "seed": args.seed,
                   "max_new_tokens": args.max_new_tokens, "max_prompt_tokens": args.max_prompt_tokens,
                   "retry_control": args.retry_control,
                   "num_shards": args.num_shards, "shard_index": args.shard_index,
                   "limit": args.limit}
    previous = {}
    if output.exists() and not args.overwrite:
        if not args.resume:
            raise FileExistsError(f"Output exists: {output}; set --resume or --overwrite")
        if not manifest_path.exists():
            raise ValueError("Cannot resume without an evaluation manifest")
        last = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(last.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Evaluation resume settings do not match the original run")
        previous = {r["id"]: r for r in load_jsonl(output)}
        if not set(previous).issubset({r["id"] for r in states}):
            raise ValueError("Previous evaluation contains IDs outside the requested panel")
    manifest = {"stage": "reflection_intervention", "states": args.states, **fingerprint,
                "requested_count": len(states), "count": len(previous),
                "probability_interpretation": "normalized string log likelihood over two actions"}
    write_json(manifest_path, manifest)
    backend = InferenceBackend(args.model, args.adapter, args.max_memory_gib)
    results = list(previous.values())
    for idx, row in enumerate(states, 1):
        if row["id"] in previous:
            continue
        observation = evaluate_state(backend, row, args.checkpoint, args.seed, args.repetitions,
                                     args.max_new_tokens, args.retry_control,
                                     args.max_prompt_tokens)
        observation["model"] = args.model
        observation["adapter"] = args.adapter
        results.append(observation)
        if idx % 10 == 0:
            results.sort(key=lambda record: record["id"])
            save_jsonl(output, results)
            manifest["count"] = len(results)
            write_json(manifest_path, manifest)
            print(f"Evaluated {idx}/{len(states)} states", flush=True)
    results.sort(key=lambda record: record["id"])
    save_jsonl(output, results)
    manifest["count"] = len(results)
    write_json(manifest_path, manifest)
    print(f"Saved {len(results)} evaluation records")


if __name__ == "__main__":
    run()
