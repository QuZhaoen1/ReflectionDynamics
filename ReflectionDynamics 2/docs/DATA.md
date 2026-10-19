# Data schema

A state has `id`, `problem_id`, `split`, `question`, `initial_reasoning`, `initial_answer` and a verified boolean `initial_correct`. Mathematics states also have `gold`. Code states have `task="code"`, `hidden_tests` and `entry_point`.

The predictor's visible state consists only of `question`, `initial_reasoning`, `initial_answer`. Model labels, reference solutions, source category labels and intervention results are not input features. Hidden code tests are used only by the isolated evaluator.

Every generated event includes `event_id`, state ID/hash, original problem ID, executor identity, checkpoint label, bank, arm, repetition, random seed, token counts, timing, output text and outcome when applicable. Complete event logs have matching manifest and receipt hashes. CLI fitting and evaluation reject incomplete or modified logs.

`fit` is reserved for controller training/development labels. `score` is used for held-out policy evaluation. `actionA` and `actionB` are independent action-prefix banks. Data from different executors or decoding settings cannot be pooled into one target bank.

Imported training-source problems split 60/20/20 into gate training, development and holdout. The legacy validation remains separate. This gate holdout is not a guarantee that the historical executor has never seen the problem. Original test and external tasks are kept outside controller fitting.

Raw generated text and hidden tests remain in the workspace. The release ZIP contains only small fixture examples, marked as fixtures and rejected by the research training interface.
