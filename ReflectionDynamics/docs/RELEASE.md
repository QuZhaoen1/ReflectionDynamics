# Release 0.4.0

This release replaces automatic selection by competition level with model-specific calibration and train-only correction probing.

- Eight independent initial solutions per problem by default, with parsing coverage, empirical pass rate and Wilson intervals.
- Training-candidate screening that retains informative problems and subsamples both easy and hard controls.
- Base-model forced-reflection and independent-retry probes before training-set curation.
- Four empirical training categories: recoverable/unresolved errors and fragile/stable correct answers.
- Minimum curation integrity requirements bound to source-file and probing-manifest hashes.
- Strict validation-panel isolation and checksum binding before any full RL experiment.
- Safe resumable draft extension to a larger question panel without discarding matching samples.
- Interruption-safe, parameter-bound reflection evaluation and adapter-checkpoint comparison.
- CPU-only tests for stage interfaces, mathematical metrics, split integrity, gate decisions and resume paths.

No GPU training was carried out while preparing this release. It remains necessary to validate Qwen2.5-32B loading, format adherence, answer verification coverage, distributed GRPO resource usage, checkpoint recovery and scientific hypotheses on the intended server.
