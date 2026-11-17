import argparse
from pathlib import Path

from .gates import assess_train, verify_gate
from .io import load_jsonl, read_yaml, write_json, checksum_file
from .protocol import prompt_for_state, terminal_score


def prepare_training_records(states, model, pilot=False):
    report = assess_train(states, model, pilot)
    if not report["size_pass"] and not pilot:
        raise ValueError("Insufficient real reasoning states for a full RL experiment")
    if not report["balanced"] and not pilot:
        raise ValueError("Insufficient correct/wrong draft balance")
    if not pilot and any(r.get("curated_with") != "base_forced_reflection" for r in states):
        raise ValueError("Full RL training requires actual base-model reflection screening")
    return [{"prompt": prompt_for_state(r), "initial_answer": str(r["initial_answer"]),
             "gold": str(r["gold"]), "state_id": r["id"]} for r in states]


def outcome_reward(completions, initial_answer, gold, **kwargs):
    outputs = []
    for result, initial, target in zip(completions, initial_answer, gold):
        if isinstance(result, list):
            result = "".join(str(x.get("content", "")) for x in result)
        outputs.append(terminal_score(str(result), initial, target))
    return outputs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gate-report", default="data/selection/train_gate.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--output-dir-override", default=None)
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    model_id, train_path = cfg["model"], cfg["data"]
    if args.seed_override is not None:
        cfg["training"]["seed"] = args.seed_override
    if args.output_dir_override is not None:
        cfg["training"]["output_dir"] = args.output_dir_override
    if not args.pilot:
        verify_gate(args.gate_report, model_id, train_path)
    train_states = list(load_jsonl(train_path))
    records = prepare_training_records(train_states, model_id, args.pilot)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tuning = cfg["training"]
    maximum = int(tuning.get("max_prompt_tokens", 2450))
    ready = [item for item in records if len(tokenizer.encode(item["prompt"], add_special_tokens=False)) <= maximum]
    if args.pilot:
        ready = ready[:32]
    if len(ready) < 8:
        raise ValueError("Too few correctly sized prompts")
    settings = GRPOConfig(
        output_dir=str(tuning["output_dir"]),
        max_steps=8 if args.pilot else int(tuning["max_steps"]),
        per_device_train_batch_size=int(tuning.get("micro_batch_size", 1)),
        gradient_accumulation_steps=int(tuning.get("accumulate", 4)),
        num_generations=int(tuning.get("num_generations", 4)),
        learning_rate=float(tuning.get("learning_rate", 2e-6)),
        warmup_ratio=float(tuning.get("warmup_ratio", 0.03)),
        max_completion_length=int(tuning.get("max_completion_length", 896)),
        max_prompt_length=maximum,
        beta=float(tuning.get("beta", 0.0)),
        bf16=True,
        seed=int(tuning.get("seed", 41)),
        logging_steps=int(tuning.get("logging_steps", 5)),
        save_steps=int(tuning.get("save_steps", 50)),
        save_total_limit=int(tuning.get("save_total_limit", 20)),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        use_vllm=False,
        ds3_gather_for_generation=False,
        deepspeed=str(tuning.get("deepspeed", "configs/zero3_cpu.json")),
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    lora = tuning["lora"]
    adapter = LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]), lora_dropout=0.0,
                         target_modules=lora["target_modules"], task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=model_id, args=settings, reward_funcs=outcome_reward,
                          train_dataset=Dataset.from_list(ready), processing_class=tokenizer,
                          peft_config=adapter)
    if trainer.is_world_process_zero():
        write_json(Path(settings.output_dir) / "experiment.json", {
            "model": model_id, "data": train_path, "data_sha256": checksum_file(train_path),
            "sample_count": len(ready), "pilot": args.pilot, "config": cfg,
            "outcome_reward_only": True, "init_adapter": None})
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(Path(settings.output_dir) / "final_adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(Path(settings.output_dir) / "final_adapter"))


if __name__ == "__main__":
    run()
