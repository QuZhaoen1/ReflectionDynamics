# Experiment matrix

## Primary tests

| ID | Question | Controlled factors | Outputs |
|---|---|---|---|
| E1a | Do measured action events agree across independent samples? | Same state, checkpoint, decoding and action window | actionA/actionB, invalid/censored rates |
| E1b | Does selection improve during outcome RL? | Fixed visible states, independent value and action banks | selection trajectory, fixed-executor decomposition |
| E2a | Can reflection value be predicted before acting? | Frozen executor, grouped split, frozen representation | held-out net gain, call-budget curves |
| E2b | Is decomposition useful beyond correctness? | Independently fitted confidence and direct-value models | matched-call comparisons |
| E2c | What do damage, cost, uncertainty and refresh contribute? | Score and training ablations labeled separately | no_damage/no_cost/risk/no_refresh |
| E3 | Which checkpoint controls versus executes reflection? | Canonical action bridge, fresh executor generations | actual crossed matrix |
| E4 | Does alternating control/execution optimization help? | Same actor update budget and data sources | CAVR-online, uniform executor, no-refresh |
| E5 | Does a frozen policy transfer? | Locked controller, new source tasks, no test fitting | original MATH test, AIME, GSM8K, HumanEval |
| E6 | Are dynamics stable across runs/scales? | Independent training seeds and explicit source model | standard seeds; separate 7B/14B training profiles |

## Default settings

Primary executor: Qwen2.5-32B Base plus the exported step400 adapter. Frozen feature encoder: Qwen2.5-32B Base. Intervention repetitions: 16 for reflection and 16 for retry. Action measurements: two independent banks of 64 prefix samples. Controller ensemble seeds: 41, 73, 107. Standard GRPO seeds: 41, 73, 107. Primary cost: 0.05 correctness units per 1,000 continuation tokens. Primary budget: 0.3 of states; additional budgets 0.1, 0.5, 0.7. Three alternating rounds, 40 executor updates per round.

The controller ensemble seeds are not independent backbone training seeds. They must not be counted as three replications of a 32B learning-dynamics experiment.

## Baselines included in the implementation

Commit; always reflect; independent retry; deterministic hash-random selection; independently trained confidence; independently trained direct value; decomposed CAVR; separately trained no-damage CAVR; inference-only damage/cost removal; ensemble risk adjustment; stale/no-refresh CAVR; uniform executor GRPO; standard natural-action outcome GRPO.

These are explicitly implemented baselines, not claimed reproductions of unrelated named papers. External methods should be added using their actual public implementations and matched settings before a final literature comparison.

## Compute accounting

Policy reports separate continuation tokens, executor input tokens, feature input tokens, and measured sequential latency. Calls at equal reflection frequency can have different generated lengths and input lengths. Use the complete reported cost/accuracy curves, rather than relabeling a call budget as a FLOP budget.

Hidden-feature extraction is charged once per state in deployment estimates, even if cached during a laboratory sweep. Initial-draft cost is shared among policies on the fixed-state panel and reported separately by the draft records. GRPO logs, probe manifests and workflow command plans retain costs of the full study.

## Statistical units

The default estimand is state-weighted performance on a declared panel. Confidence intervals resample original problems and retain all their states. Current cluster intervals condition on the recorded generations. Additional action/outcome banks and training seeds assess other sources of uncertainty.

The label-informed correctness oracle is not a trained model. The same samples are not used to choose a state-specific empirical oracle and then claim its measured gain. No experiment script chooses its best checkpoint on the original test set.

## Cross-checkpoint interpretation

The expected matrix reuses independently estimated action probabilities and executor values. The actual matrix samples new continuations under recorded decisions. Both use a canonical text bridge. Neither alone identifies neuron-level mechanisms or establishes a physical phase transition.

Smooth-versus-segmented curve fits are exploratory and retain both models. A small number of independent seeds yields coarse uncertainty. The `seeds` command aggregates explicitly supplied training-seed summaries rather than treating repeated states as independent training runs.
