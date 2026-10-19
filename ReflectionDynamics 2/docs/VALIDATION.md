# Release validation

Release: 0.5.0. Validation date: 2026-10-19.

## Executed locally

- 134 tests passed, zero failures, zero errors, zero skips. This includes the 47 historical tests and 87 additional cases.
- Editable installation and the installed `rd5` command completed successfully.
- Python source compilation, shell syntax, YAML, JSON and TOML parsing completed successfully.
- The CPU fixture pipeline fitted real PyTorch controller ensembles, exported and reloaded safetensors, scored nine policies, audited independent action banks, and executed the modular intervention path with a deterministic fixture backend.
- Running the same fixture pipeline again reused completed samples and the controller export without changing the result.
- Command plans for frozen control, eleven-checkpoint measurement, actual crossing, online/no-refresh/uniform training, and four transfer workflows were generated. All 567 model/controller CLI commands parsed successfully; the nine distributed-training command forms were inspected. This was a dry run, not model execution.

The local Python version is 3.13.5. Server experiments target the documented Python 3.11 environment. Package versions and a structured check list are recorded in `validation.json`.

## Server integration checks

The following require the collaborator's environment and have not been executed in this release validation: real Qwen tokenizer loading, model inference, public dataset downloads, 32B distributed GRPO, real GPU checkpoint restoration, and Docker code grading. The repository supplies their implementation and launch commands, but no GPU pass certificate or model-performance result is bundled.

Run `00_init.sh`, inspect `preflight.json`, execute the frozen-controller workflow, and use `09_resume_test.sh` before new long actor-training runs. Audit action invalid/censored frequencies and paired reports on actual saved states.

## Reproduce the CPU check

```bash
python -m pip install -e '.[test]'
bash scripts/v5/test_cpu.sh /tmp/reflection-v5-smoke
```

The fixture corpus is restricted to software tests and is rejected by the research training interface. No synthetic performance number is presented as a result from Qwen.

## Compact packaging

The v4 regression tests are consolidated into `tests/test_legacy_regression.py` without removing test cases. The v5 test modules and current experiment launch scripts remain separate.
