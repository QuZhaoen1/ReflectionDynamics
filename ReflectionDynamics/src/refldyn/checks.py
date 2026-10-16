import argparse
import json
import subprocess
from pathlib import Path

from .io import read_yaml


def inspect_config(path):
    cfg = read_yaml(path)
    if cfg.get("model") not in {"Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B"}:
        raise ValueError("Unexpected model ID")
    if "data/generated" in str(cfg.get("data")):
        raise ValueError("Synthetic arithmetic cannot be used for main RL training")
    tune = cfg["training"]
    ds_path = Path(tune["deepspeed"])
    if not ds_path.is_absolute():
        ds_path = Path(path).resolve().parent.parent / ds_path
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    if ds["zero_optimization"]["stage"] != 3:
        raise ValueError("32B profile requires ZeRO-3")
    if tune.get("use_vllm"):
        raise ValueError("This sharded generation profile does not use vLLM")
    if tune.get("ds3_gather_for_generation", False):
        raise ValueError("Do not gather the 32B model onto one A100 during generation")
    if not ds["zero_optimization"].get("offload_param"):
        raise ValueError("CPU parameter offloading missing")
    global_batch = int(tune.get("micro_batch_size", 1)) * 8 * int(tune.get("accumulate", 4))
    if global_batch % int(tune.get("num_generations", 4)):
        raise ValueError("Effective batch must be divisible by GRPO num_generations")
    return {"configuration": path, "model": cfg["model"], "global_batch": global_batch,
            "zero_stage": 3, "cpu_offload": True, "required_gpus": 8}


def preflight_gpu(minimum_gib=36):
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total",
                                          "--format=csv,noheader,nounits"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi unavailable") from exc
    rows = [line.rsplit(",", 1) for line in output.strip().splitlines() if "," in line]
    if len(rows) < 8:
        raise RuntimeError(f"Expected 8 GPUs; detected {len(rows)}")
    for name, memory_mib in rows[:8]:
        if "A100" not in name or int(memory_mib.strip()) < minimum_gib * 1024:
            raise RuntimeError("Expected 8 NVIDIA A100 with at least 36 GiB per GPU")
    return [dict(name=name.strip(), memory_mib=int(m.strip())) for name, m in rows[:8]]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_32b_a10040.yaml")
    parser.add_argument("--check-gpu", action="store_true")
    args = parser.parse_args()
    result = inspect_config(args.config)
    if args.check_gpu:
        result["gpus"] = preflight_gpu()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
