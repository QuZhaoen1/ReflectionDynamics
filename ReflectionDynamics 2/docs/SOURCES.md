# Implementation references

The starting project is QuZhaoen1/ReflectionDynamics, commit `aafee68f6bd7866c668c0542a65f4e6c4def5e77`; the seed41 archive records the model revision and actual historical training configuration. v5 preserves that experiment as a separate study.

- Repository: https://github.com/QuZhaoen1/ReflectionDynamics
- Historical archive: https://github.com/QuZhaoen1/ReflectionDynamics/tree/aafee68f6bd7866c668c0542a65f4e6c4def5e77/results/seed41-32b-20261019
- TRL 0.27.1 GRPO interface: https://huggingface.co/docs/trl/v0.27.1/en/grpo_trainer
- Transformers generation interface: https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation
- PEFT checkpoint format: https://huggingface.co/docs/peft/en/developer_guides/checkpoint
- DeepSpeed checkpoint export: https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
- MATH source project: https://github.com/hendrycks/math
- MATH dataset mirror used by the existing importer: https://huggingface.co/datasets/jeggers/competition_math
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- AIME collection: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
- HumanEval: https://huggingface.co/datasets/openai/openai_humaneval
- Math-Verify: https://github.com/huggingface/Math-Verify

Dataset loaders record the resolved revision at runtime. Dataset availability and licensing remain properties of the source repositories; benchmark text is not distributed in this package. Software versions are pinned where the training API requires it. The code is an implementation of the stated CAVR research design, not a claim of reproducing external named methods.
