# Data design and provenance

| Split | Source | Allowed use |
|---|---|---|
| train | Original MATH training split | Candidate generation, calibration, forced-reflection probes, curation and RL training |
| validation | Deterministic held-out subset of original MATH training | Gate decisions, checkpoint selection and development |
| test | Original MATH test split | Final evaluation, after development is frozen |
| ood | AIME external benchmark | Final stress testing only |

The importer strips reference solutions but retains final reference answers for executable grading. Reference answers are used by the grader, never inserted into inference prompts. Deduplication is performed on normalized full question text; semantically equivalent restatements may remain and should be checked with an external similarity audit for publication-quality results.

Independent draft sampling uses eight trials per source problem by default. `rd screen` reports an observed pass fraction and a Wilson interval, as well as failed parsing frequency. These measurements describe a **specific model and decoding policy**, not an immutable property of the mathematics problem. A very low pass fraction does not imply that the model cannot benefit from reflection. A high pass fraction is useful for measuring unnecessary reflection and damage.

Training candidate states consist of model-generated initial reasoning, not arithmetic errors invented by editing ground-truth answers. The train-only forced-reflection probe then estimates empirical correction outcomes before a balanced training set is selected. Validation/test problems are neither probed for curation nor used as RL training prompts.

The correction categories are empirical labels based on few repeated samples. Results are stochastic and have selection bias, so checkpoint metrics must be computed on untouched fixed panels. The MATH test set is not guaranteed to have been absent from the base model's original pretraining corpus.

Review a sample of symbolic grading decisions manually. Keep a published manifest with dataset revisions, the exact tokenizer/model commit, temperature, seeds, maximum tokens, software versions, and split hashes.
