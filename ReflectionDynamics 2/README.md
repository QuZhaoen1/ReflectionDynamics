# ReflectionDynamics

**Think Again, or Don't: Learning the Counterfactual Value of Reflection**

Version 0.5 (compact distribution) provides CAVR value learning, selective reflection, checkpoint interventions, and reinforcement-learning experiments. It continues the completed Qwen2.5-32B seed41 study without replacing its checkpoints or historical results.

## Method

CAVR estimates initial correctness, conditional repair probability, conditional damage probability, and the cost of an additional reflection:

\[
\widehat U(s)=(1-\widehat p_c(s))\widehat p_r(s)-\widehat p_c(s)\widehat p_h(s)-\lambda\widehat C(s)/1000.
\]

A lightweight ensemble receives frozen decoder features from the problem and proposed solution. Its score determines whether the executor preserves the proposed answer or performs one reflection. Confidence-only and direct-value controllers are trained separately on the same data. An alternating recipe refreshes value estimates as the executor changes.

## Start from the completed run

Use the existing Python 3.11 training environment, or install the appropriate PyTorch CUDA wheel before installing this project.

```bash
python -m pip install -e '.[train,test]'
python -m pytest
cp configs/runtime.env.example configs/runtime.env
```

Set `LEGACY_RUN` to the completed server run directory containing `data/states/` and `adapters/`. Set `WORKSPACE` to a new persistent directory. Choose the GPU profile in the same file.

```bash
bash scripts/v5/00_init.sh
bash scripts/v5/01_frozen.sh
```

The second command collects new independent intervention banks, extracts frozen features, fits the controllers, and evaluates the held-out controller panel and the historical validation panel. It does **not** restart the 400-step backbone training.

[Chinese runbook](docs/RUNBOOK_CN.md) · [Algorithm](docs/ALGORITHM.md) · [Experiment matrix](docs/EXPERIMENTS.md) · [Resume](docs/RESUME.md) · [Upgrade](docs/UPGRADE.md)

## Experiments

| Experiment | Entry point | Main artifacts |
|---|---|---|
| E1: action measurement and reflection dynamics | `02_measurement.sh` | Two independent action banks, selection metrics, trajectory fits |
| E2: frozen-executor CAVR and ablations | `01_frozen.sh` | Separate confidence/direct-value models, policy–cost tables |
| E3: actual decision/execution crossing | `03_crossed.sh` | Fresh continuation outputs for each checkpoint pair |
| E4: alternating executor and controller training | `04_online.sh` | Round checkpoints, refreshed probes, updated controllers |
| E4 controls | `05_no_refresh.sh`, `06_uniform_executor.sh` | Stale-controller and uniform-executor runs |
| E5: task transfer | `07_ood.sh math`, `07_ood.sh aime`, `07_ood.sh humaneval` | Original test/transfer tasks, fresh drafts, policy tables |
| E6: independent standard training seeds | `08_standard_seeds.sh` | New standard-GRPO runs for seeds 41, 73, 107 |
| Infrastructure: interruption and recovery | `09_resume_test.sh` | Stop at step 2, resume the same job to step 8, export verification |

All scripts are under `scripts/v5/`. The complete low-level CLI is available with `rd5 --help`. Use `10_plan_all.sh` to generate command plans without launching model work.

## Algorithm implementation

- `src/cavr/controller.py`: multi-head losses, ensemble fitting, calibration, checkpointed optimizer state.
- `src/cavr/policy.py`: cost-aware scores, risk adjustment, exact reflection-call budgets.
- `src/cavr/route.py`: exploration-preserving training-state selection.
- `src/cavr/train.py`: standard outcome GRPO and forced-reflection executor GRPO.
- `src/cavr/workflow.py`: frozen, measurement, crossed, alternating, and transfer workflows.
- `src/cavr/interventions.py`: actual executor generation from recorded decisions.

The historical `refldyn` modules are retained for data compatibility. The old two-string `action_probability` implementation now raises an error. New action probabilities are estimated from explicitly defined sampled events, including invalid and censored outputs in the denominator.

## Hardware profiles

`configs/hardware/a100_8x80.yaml` is the primary profile. `a100_4x80.yaml` retains an effective batch of 32 generated completions through gradient accumulation. `a100_8x40.yaml` disables gathering the full model during generation. Additional 7B and 14B profiles are included for separate scale controls.

Inference defaults to two GPUs per replica. Controller fitting defaults to CPU. The GPU count for inference need not equal the GPU count used for training.

## Data and storage

The importer reads the completed run and creates a separate workspace. Original training-source problems are split into controller train/development/holdout panels. The controller holdout excludes controller fitting, but its problems may have been seen by the historical backbone. The previously inspected validation panel remains exploratory. Original MATH test and external transfer tasks are used for independent evaluation.

Intervention outputs have immutable manifests, content hashes, completion receipts, and per-generation resume keys. The `fit` and `score` banks use independent seeds. The predictor receives no answer labels, intervention outcomes, or hidden code tests.

Code tasks execute only in a restricted Docker container:

```bash
bash scripts/v5/build_code_runner.sh
bash scripts/v5/07_ood.sh humaneval
```

## Reporting

Policy tables include correctness, paired original-problem cluster intervals, reflection frequency, continuation tokens, executor prefill tokens, feature-prefill tokens, and sequential latency measurements. A matched number of reflection calls is not labeled as equal total compute. The label-informed oracle is reported separately from learned policies.

Trajectory fitting compares smooth and segmented descriptions. It does not automatically label a curve as a phase transition. Multi-seed aggregation is a separate command.

## Verification

```bash
bash scripts/v5/test_cpu.sh
```

The CPU suite covers action boundaries, label isolation, data partitions, resume integrity, controller training, inference checksums, intervention plans, and the fixture integration pipeline. Fixtures are rejected by the research training interface. GPU-dependent checks are listed in [VALIDATION.md](docs/VALIDATION.md).

Model weights, training outputs, benchmark problem text, and local credentials are not bundled in the release.

## Compact distribution

This release keeps the current training and evaluation code, hardware profiles, server launch scripts, and regression tests in under 100 repository files. Earlier v4 convenience wrappers and duplicate archived documentation are available in prior repository revisions. Legacy regression tests are grouped in `tests/test_legacy_regression.py`; `python -m pytest` executes the same test cases.
