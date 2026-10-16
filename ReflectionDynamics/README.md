# ReflectionDynamics

Research code for measuring the emergence of selective self-reflection during outcome-reward reinforcement learning. The primary experiment compares model-level reflection choice, conditional correction value, and checkpoint-to-checkpoint learning dynamics on verified mathematical reasoning tasks.

## Research setting

- Primary model: `Qwen/Qwen2.5-32B` (base pretrained checkpoint).
- Scaling controls: `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`.
- Reference hardware: eight A100 GPUs, 40 GiB each.
- Training: outcome-only GRPO with PEFT adapters and ZeRO-3 CPU parameter offload.
- Main source: original MATH training partition, with a protected validation subset.
- Final evaluation: original MATH test partition; optional AIME as external stress test.

No pretrained weights, trajectories, or empirical results are bundled. Model downloads and distributed training require validation on the target GPU server.

## Installation

Linux and Python 3.11 are recommended. Install a CUDA-compatible PyTorch distribution supported by the server drivers, then install the project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,test]'
pytest
bash scripts/00_preflight.sh
```

## Experiment sequence

1. Load the MATH corpus and protect evaluation partitions.
2. Independently solve each candidate problem several times with the 32B base model.
3. Report observed pass rates, unparsed answers, Wilson intervals, and difficulty groups.
4. Build real initial reasoning states and probe forced reflection on training candidates only.
5. Curate a balanced mixture of recoverable errors, unresolved errors, fragile correct responses, and stable correct responses.
6. Evaluate the untrained base policy on the untouched validation states.
7. Pass the pre-training quality gate, smoke-test distributed training, and begin GRPO.
8. Evaluate identical fixed states across checkpoints and compare behavioral propensity with correction-value selectivity.

### Initial calibration

```bash
bash scripts/01_ingest.sh
bash scripts/02_draft_32b.sh train 240
bash scripts/02_draft_32b.sh validation 100
cat data/calibration/train.summary.json
bash scripts/03_probe_train_32b.sh 400
bash scripts/03_evaluate_base_32b.sh validation base
```

**Do not launch full GRPO after the preview.** The calibration samples are intended to establish whether the tasks are genuinely nontrivial and whether initial errors can be repaired. The base checkpoint must demonstrate adequate output validity on both independent solutions and reflection decisions.

### Full data preparation

```bash
RESUME=1 bash scripts/02_draft_32b.sh train 1600
RESUME=1 bash scripts/02_draft_32b.sh validation 300
bash scripts/03_probe_train_32b.sh 1200
bash scripts/04_curate.sh 600
bash scripts/03_evaluate_base_32b.sh validation base
bash scripts/04_gate.sh
```

If the 1,600-question training panel lacks enough recoverable errors, expand it in another stage with `RESUME=1` and a larger limit or `0` for the entire source. The draft sampler reuses existing draws only when model, source checksum, random seed and decoding settings match. After changing candidate states, rerun the training probe so its recorded panel checksum matches. The curation gate rejects source mismatches, synthetic fixtures, held-out leakage, deficient format validity, insufficient reparable errors, and inadequate correct-answer controls. If the training set fails its minimum size or balance requirements, examine the calibration outputs and expand the **training-only** probe. Never use test results to choose training examples.

### Training and longitudinal evaluation

```bash
bash scripts/05_pilot_32b.sh
bash scripts/05_train_32b.sh
bash scripts/03_evaluate_base_32b.sh validation step40 runs/grpo_32b_seed41/checkpoint-40
bash scripts/03_evaluate_base_32b.sh validation step80 runs/grpo_32b_seed41/checkpoint-80
bash scripts/06_analyze.sh outputs/validation \
  data/evaluations/base/validation_0.jsonl \
  data/evaluations/base/validation_1.jsonl \
  data/evaluations/step40/validation_0.jsonl \
  data/evaluations/step40/validation_1.jsonl \
  data/evaluations/step80/validation_0.jsonl \
  data/evaluations/step80/validation_1.jsonl
```

Adapter checkpoints must exist before attempting evaluation. Full runs with repeated seeds should be conducted only after the first distributed pilot succeeds. Scripts never auto-download published benchmark questions into Git; generated files and model weights remain ignored.

## Measurements

Reflection propensity is the normalized likelihood of choosing `REFLECT` over `COMMIT` under a fixed two-option prompt. The causal comparison of forced reflection versus direct submission estimates conditional correction gain. The primary selectivity statistic is:

\[
S_t = \mathbb{E}_{s\sim D}[p_t(s)U_t(s)]-\mathbb{E}_{s\sim D}[p_t(s)]\mathbb{E}_{s\sim D}[U_t(s)].
\]

It measures improvement over frequency-matched random reflection on a fixed initial-state distribution. The evaluator also records sampled decisions, verification validity, correction of initially wrong answers, damage to initially correct answers, independent retries, and generated-token costs. Cross-checkpoint policy/execution swaps are interface-level interventions; they do not by themselves identify internal neural mechanisms.

## Reproducibility

All sampling uses explicit seeds; draft/evaluation manifests record model, checkpoint, input hashes, settings and shard information. Dataset splits are assigned at the original-question level, and training curation can use only training-source probes. Confidence intervals and exploratory change-point diagnostics are provided. A smooth training curve or an apparent breakpoint is not sufficient evidence of a phase transition.

See `docs/RUNBOOK_CN.md` for the full execution instructions, `docs/PROTOCOL.md` for the experiment design, `docs/DATA.md` for source policies, `docs/SOURCES.md` for upstream repositories, and `docs/STATUS.md` for unverified hardware-dependent components, and `docs/RELEASE.md` for this version.

## GitHub transfer

Commit the unpacked source directory. The `.gitignore` file excludes the raw benchmark data, generated trajectories, cached weights and run outputs.

```bash
git init
git add .
git commit -m "Initial research pipeline"
gh repo create ReflectionDynamics --private --source=. --remote=origin --push
```
