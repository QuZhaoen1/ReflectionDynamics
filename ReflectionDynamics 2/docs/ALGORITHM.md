# CAVR: Counterfactual Advantage for Reflection

## 1. State, actions and executor

A state is `s = (question, initial_reasoning, initial_answer)`. The two deployment actions are COMMIT and REFLECT. COMMIT returns the proposed answer without asking the model to copy it again. REFLECT runs a canonical continuation with the selected executor checkpoint. Independent retry is a separate strong baseline, not a hidden third action in CAVR.

The action interface is deliberately modular. A natural-language decision prefix is not transferred as hidden state or arbitrary reasoning text to another executor. Crossing checkpoints identifies component-level effects under this interface.

## 2. Conditional value

Let `y0` be verified initial correctness and `R1` the final correctness after reflection. The measured value at executor checkpoint `j` is

```
U_j(s) = E[R1 | s, executor_j, canonical_REFLECT] - y0 - lambda * E[new_tokens] / 1000.
```

The predictor uses

```
p_c = P(initial correctness | visible state)
p_r = P(reflected answer correct | visible state, initially wrong)
p_h = P(reflected answer incorrect | visible state, initially correct)
U_hat = (1 - p_c) * p_r - p_c * p_h - lambda * predicted_tokens / 1000
```

The correctness label is used to construct training targets and masks, never as a predictor input. Conditional heads are trained only on their supported initial-correctness stratum. Their use outside that stratum is part of the learned mixture approximation, not an assertion that counterfactual correctness labels have been observed twice.

At zero cost, knowing exact initial correctness is sufficient for the label-informed optimum: retain a correct answer and reflect on an incorrect answer whenever repair has positive probability. Consequently, CAVR must compete against a strong correctness predictor. Its extra repair/damage modeling is most relevant under imperfect correctness prediction and nonzero or limited reflection budgets.

## 3. Features and architecture

The default representation concatenates the last-token hidden vector and the mean hidden vector from the final decoder layer of the frozen Base model. A two-layer MLP with LayerNorm, GELU and dropout predicts seven scalar outputs: initial correctness, repair, damage, direct value, reflection cost, retry correctness, and retry cost.

The three-member ensemble uses seeds 41, 73, 107. Standardization is fitted only on controller training features. Calibration temperatures are selected only on controller development problems. Head fitting uses original-problem weights, so multiple states from the same problem do not automatically give that problem more training weight.

The hash-feature path is an executable lexical baseline and a CPU test path. The main server workflow uses decoder features.

## 4. Training objectives

For CAVR, the loss is the sum of:

1. Initial-correctness binary cross entropy.
2. Sample-count-weighted repair cross entropy on initially wrong states.
3. Sample-count-weighted damage cross entropy on initially correct states.
4. A 0.5-weight consistency loss between decomposed outcome gain and observed gain.
5. A 0.05-weight squared loss on `log1p(reflection_tokens)`.
6. A 0.2-weight retry-correctness loss and a 0.05-weight retry-cost loss.

The direct-value controller is fitted separately with gain regression and cost regression. The confidence controller is fitted separately with correctness and cost losses. The no-damage controller omits damage training and damage subtraction. Inference-only score ablations are also available and remain distinguishable from separately trained baselines through the `controller_mode` field.

The default score is mean predicted net value. A positive `risk` coefficient subtracts ensemble dispersion. This is risk adjustment, not a statistically guaranteed lower confidence bound.

## 5. Action selection

The unconstrained CAVR policy reflects when its net score exceeds zero. A budgeted policy ranks states using only pre-action predictions and chooses exactly `round(budget * number_of_states)` states. The default budgets are 0.1, 0.3, 0.5 and 0.7. Equal-call-budget comparisons do not imply equal token counts or latency.

Confidence selection ranks `1 - p_c`; its unconstrained default threshold is 0.5. Random selection uses reproducible state hashes. Learned policies do not use the oracle labels.

## 6. Alternating training

The online recipe starts from an exported executor adapter such as step400.

```
for each round:
    probe executor on controller-train/development states
    fit or refresh CAVR
    include each training state with probability:
        epsilon + (1 - epsilon) * sigmoid(U_hat / temperature)
    update the reflection executor with GRPO for 40 optimizer steps
    export the complete adapter
```

The default exploration floor is 0.15. The state-selection distribution is deliberately gate-weighted; it is not presented as an importance-corrected estimate of uniform training. Executor reward is verified final correctness minus a declared continuation-token cost.

The default online study uses three rounds. Each round warm-starts adapter weights but starts a new optimizer schedule. Within a round, interruption recovery restores the entire GRPO checkpoint. This is block-coordinate alternating training, not one uninterrupted optimizer trajectory.

`no_refresh` retains the initial controller and evaluates it explicitly under executor drift. `uniform_executor` trains the executor on the uniform training panel and still fits an evaluation controller. The standard outcome-GRPO baseline trains natural COMMIT/REFLECT output instead of forcing the reflection action.

Compare matched optimizer/rollout budgets and also report the additional probe and feature-extraction cost of CAVR. Do not treat equal optimizer steps as equal total study cost.

## 7. Independent evaluation

Training targets come from the `fit` bank. Controller holdout/validation/transfer evaluation comes from a separately seeded `score` bank. Trial identity includes the executor and decoding configuration. A controller trained for one executor is not silently evaluated as current for another executor; explicit drift mode is required.

Historical seed41 counterfactual samples are not silently pooled into the new banks. They used a different measurement pipeline and fewer repetitions. They remain available for descriptive historical comparison.
