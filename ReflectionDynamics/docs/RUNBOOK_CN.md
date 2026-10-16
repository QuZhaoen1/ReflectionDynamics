# ReflectionDynamics v4：8×A100 40GB 实验手册

## 1. 研究任务

主实验模型为 `Qwen/Qwen2.5-32B` 的 Base 权重。目标不是训练模型做简单算术，而是在真实数学推理状态上研究：结果奖励的强化学习是否形成能够根据预期纠错收益选择反思时机的能力。

完整实验的执行顺序：**数据集分区 → 模型实际难度校准 → 真实轨迹采集 → 训练分区反思探测 → 训练集构建 → 验证集质量门槛 → 分布式 RL → 固定状态纵向分析**。

## 2. 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[train,test]'
pytest
bash scripts/00_preflight.sh
```

需要 Linux、合适的 NVIDIA 驱动、CUDA PyTorch、8 张 A100（按每张 40GB 规划），建议至少 256GiB 主机内存以及较大高速存储空间。32B 采用 ZeRO-3 + CPU 参数卸载 + LoRA，配置存在但还未在目标机器实测。若首次加载、单步训练或恢复失败，应记录错误、调整软件版本或并行策略，不应跳过测试。

## 3. 导入原始题目

```bash
bash scripts/01_ingest.sh
```

数据源：`jeggers/competition_math` 的 `original` 配置。数据集原始 train 进行稳定哈希分区，形成 train 与 validation；原始 test 保持独立，AIME 保留为 ood。数据集不是程序新编的简单整数题。原始题库可能出现在预训练语料中，因此“未参与本项目 RL 训练”不等于“基础模型从未见过”。若要研究严格的新颖任务，还需要额外的独立或新构造评测。

查看 `data/corpus/manifest.json` 和 `data/corpus/rejected.jsonl`。答案由 Math-Verify 等价验证；复杂表达式可能发生解析失败。正式报告必须抽样人工复核答案判定。建议将数据仓库固定到确切版本哈希（而非 main），并保存环境依赖锁定信息。

## 4. 模型难度校准：先验证题目有没有挑战性

```bash
bash scripts/02_draft_32b.sh train 240
bash scripts/02_draft_32b.sh validation 100
cat data/calibration/train.summary.json
cat data/calibration/validation.summary.json
```

默认每道题独立采样 8 条原始推理轨迹，记录每次回答是否正确、解析是否有效。`rd screen` 输出：

- `pass_at_1_all_attempts`：所有采样中的可验证正确比例；无效输出计为失败。
- `pass_at_1_valid_only`：成功解析的解答中的正确比例。
- `wilson_95_all_attempts`：该题正确率的近似二项置信区间（8 次采样时仍然很宽）。
- `difficulty_band`：`too_easy`、`informative`、`too_hard` 或 `insufficient_evidence`。该等级**是目标模型的经验难度，不是题目的固有竞赛等级**。

默认 `informative` 区间为观测正确率 15%–85%。这个区间是候选过滤阈值而不是最优训练原则。保留部分过难题与简单题，前者可检验能力边界，后者可检验反思损害。

如果模型大部分答案无效，应先统一输出协议、提高 token 限制或做跨组一致的格式初始化，不能把所有无效当作真正的推理错误。需要逐题检查难度与输出长度分布。

## 5. 训练分区的反思收益探测

```bash
bash scripts/03_probe_train_32b.sh 400
```

它读取 `data/states/train_candidates.jsonl`（**仅 train 来源**），对同一初始推理执行：自然选择、强制反思，以及从题目重新独立求解的对照。默认每状态强制反思 4 次；`PROBE_REPETITIONS=8` 可增加探测重复数。输出到 `data/evaluations/train_probe/`。

初步确认模型反思行为格式正常、错误状态存在正向纠错机会，并观察正确答案被错误修改的比例。这些训练分区上的探测结果可以用于挑选训练样本，**不能用于评估正式方法的泛化能力**。

## 6. 正式构建训练集

```bash
RESUME=1 bash scripts/02_draft_32b.sh train 1600
RESUME=1 bash scripts/02_draft_32b.sh validation 300
bash scripts/03_probe_train_32b.sh 1200
bash scripts/04_curate.sh 600
bash scripts/03_evaluate_base_32b.sh validation base
bash scripts/04_gate.sh
```

`rd curate` 使用训练分区的真实初始轨迹和反思探测，将状态分类为：

| 分类 | 含义 | 建议训练配比 |
|---|---|---:|
| recoverable_error | 初始错误，至少一次强制反思修复 | 35% |
| unresolved_error | 初始错误，探测中未能修复 | 20% |
| fragile_correct | 初始正确，至少一次反思改错或产生无效结果 | 25% |
| stable_correct | 初始正确，探测中均保持正确 | 20% |

优先采用逐批扩展的候选数据，避免直接对全部题目进行大量 32B 推理。若 1,600 道训练候选题不足以提供可修复错误，可使用 `RESUME=1 bash scripts/02_draft_32b.sh train 3000` 扩展；`0` 代表全量。保留原先题目的采样结果，但状态候选集改变后必须重新运行 train 侧反思探测，以满足源文件哈希校验。

配比是**目标比例，不保证每类都有足量样本**。不足时从其他可用类别填充，并记录实际比例；默认每道原题只保留一个训练状态，避免重复样本支配奖励。每类标签基于有限次随机采样，不能视为永远正确的“真值”。

结果：`data/states/train.jsonl`，以及 `data/states/train.curation.json`。正式训练不得使用 `train_candidates.jsonl` 或 `data/fixtures/` 直接代替已筛选训练集。

`04_gate.sh` 核对分区隔离、训练样本规模、可修复错误和正确对照数量、验证集完整性、Base 反思有效率。未通过就停；先看 gate JSON 的具体失败原因，必要时扩大 train 侧探测数、修正解析协议或更换任务，禁止从原始 test 挖题补训练。

## 7. 32B 训练与多种子机制实验

```bash
bash scripts/05_pilot_32b.sh
bash scripts/05_train_32b.sh
```

仅当 pilot 成功且 gate 通过才跑完整训练。`training.output_dir` 下保存 LoRA adapter checkpoints、训练状态和实验参数。建议在确定首个 seed 的资源占用与稳定性后做三个不同训练随机种子的独立重复；这些重复应使用同一固定验证集以及同一基线初始状态集。

不要用 LoRA 32B 的结果无条件推广到全参数 RL，也不要因为保存了 checkpoint 就认定发生了元认知涌现。

## 8. 跨 checkpoint 评估

```bash
bash scripts/03_evaluate_base_32b.sh validation step40 runs/grpo_32b_seed41/checkpoint-40
bash scripts/03_evaluate_base_32b.sh validation step80 runs/grpo_32b_seed41/checkpoint-80
bash scripts/06_analyze.sh outputs/validation \
  data/evaluations/base/validation_0.jsonl data/evaluations/base/validation_1.jsonl \
  data/evaluations/step40/validation_0.jsonl data/evaluations/step40/validation_1.jsonl \
  data/evaluations/step80/validation_0.jsonl data/evaluations/step80/validation_1.jsonl
```

`refldyn.analyze` 要求不同 checkpoint 具有一致的固定状态 ID。主要指标是反思倾向、条件性反思收益、选择性、成功纠错概率、正确被改错风险、独立重试正确率和 token 预算。`rd transitions` 的变点指标属于探索性诊断，不等同于证明物理相变。

## 9. 中断恢复与数据安全

```bash
RESUME=1 bash scripts/02_draft_32b.sh train 1600
RESUME=1 bash scripts/03_probe_train_32b.sh 1200
RESUME=1 bash scripts/03_evaluate_base_32b.sh validation base
```

必须与原始运行的模型、题目、分片、重复次数、随机种子和输出上限一致；改变任一参数应新建输出路径或显式覆盖。Github 只传输源代码和配置；缓存权重、下载的题目、轨迹、日志及运行数据都由 `.gitignore` 排除。

## 10. 不可将其解释为论文结果的事项

1. 目前没有在目标 8×A100 服务器完成正式 GRPO 训练；无法保证首次启动即成功。
2. 单测通过意味着本地代码逻辑得到测试，不意味着真实 32B 计算稳定或模型有效纠错。
3. MATH 中“Level 5”不等于模型一定失败，`8` 次采样估计不精确，正式难度报告应汇总到题目群体并增加复测。
4. 数据筛选条件基于训练侧的随机探测结果，训练集具有选择偏差，不能把在同一批训练候选上的结果当成最终论文结论。
5. 对同一模型重复观察到的阶段性跃迁，不足以独立证明相变，更不能推断出现了主观意识。
