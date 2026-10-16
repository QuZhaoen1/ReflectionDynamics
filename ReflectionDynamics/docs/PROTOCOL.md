# Experimental protocol

## Main question

Does outcome-only reinforcement learning teach a pretrained model to reflect **where it is useful**, rather than merely to reflect more frequently?

## Models and partitions

Primary model: Qwen2.5-32B Base with PEFT/GRPO. Smaller Base checkpoints are scaling controls only. MATH original training problems are separated into training and validation at the question level. Original test and optional AIME remain protected until the method and its hyperparameters are frozen.

## Study A: base model calibration

Generate eight independent solutions per candidate problem. Retain both solved and unsolved cases, record unparseable answers separately, and estimate pass fractions with Wilson intervals. Exclude low-parse-rate problems from automatic calibration rather than labeling parsing problems as difficult reasoning. The screening thresholds are heuristic and must be reported. Public competition level is not a proxy for actual model-level difficulty.

## Study B: initial-state and reflection quality

On training candidate states only, sample forced-reflection continuations. Categorize states as recoverable errors, unresolved errors, fragile correct answers and stable correct answers. Allocate training examples across all categories so the policy must learn not only correction but also when to stop. Track true reflection outcomes and independent retry controls. Data-based curation changes the training distribution; no evaluation claim is based on that selected sample.

## Study C: outcome-only RL

Use the same two-action reflection protocol for every checkpoint. The model sees a proposed reasoning trajectory and chooses `COMMIT` or `REFLECT`, followed by a final answer. The reward is solely correctness of the final verified answer when output is valid; no privileged reflection-value label enters the training prompt or primary reward. Hold training source states fixed and vary training seeds. Make any necessary format-init interventions symmetrical across controls.

## Study D: fixed-state counterfactual evaluation

For the identical baseline-generated initial reasoning states, obtain (a) sampled natural decisions, (b) logits for the two permitted decision strings, (c) forced reflection with multiple independent random seeds, (d) direct commit and (e) independent retry. Invalid outputs count as task failures and are always reported separately. Match token limits and report actual generated tokens; the retry baseline has a different prompt length and is not a perfect compute-matched control.

Let \(p_t(s)\) be the normalized two-option reflection likelihood at checkpoint \(t\). Let \(U_t(s)\) be the conditional forced-reflection correctness gain relative to submission of the initial candidate, less optional token cost. Define:

\[
S_t=\mathbb E_s[p_t(s)U_t(s)]-\mathbb E_s[p_t(s)]\mathbb E_s[U_t(s)].
\]

This quantity is excess expected utility versus a frequency-matched random reflection policy on the same fixed state distribution. It is a **functional measurement**, not a direct test for consciousness. It changes with decision quality **and** with correction competence; use cross-checkpoint policy/execution swaps to separate these interface-level effects.

## Hypotheses and failure criteria

- H1: A higher reflection rate does not necessarily imply higher value selectivity.
- H2: Correction competence can constrain the usefulness of reflection selection.
- H3: Selection may not generalize to a new task family or difficulty distribution.

Stop the full 32B training when calibration/format quality fails or when there are too few recoverable mistakes and correct-answer controls. Do not retrospectively adjust the held-out test split to make the results look favorable. Any phase-transition claim requires repeated training seeds, sufficient checkpoint density, comparisons against smooth learning curves, and controls for schedule and token-length changes.
