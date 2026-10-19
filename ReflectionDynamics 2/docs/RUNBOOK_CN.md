# v5 服务器执行手册

## 1. 本次运行目标

这轮分成两个层次。先复用首轮的 32B Base、step40 至 step400 适配器和原始推理状态，完成动作测量修复、CAVR 冻结执行器实验及跨阶段干预。随后运行 CAVR 交替训练、对照训练和独立测试任务。

不要覆盖已经完成的 seed41 目录。新工作区必须位于持久存储，模型缓存、完整优化器 checkpoint 和报告都不应放在每周清空的临时目录。

## 2. 更新代码

压缩包里的 `ReflectionDynamics/` 对应现有 GitHub 仓库中的同名代码子目录。从仓库根目录更新到独立分支。原来的根目录 `results/` 不属于本次替换范围。

```bash
git switch -c cavr-v5
# 将新版 ReflectionDynamics/ 中的文件覆盖同名代码子目录
cd ReflectionDynamics
python -m pip install -e '.[train,test]'
python -m pytest
```

优先复用首轮已经跑通的 Python 3.11 环境。另建环境时，先安装与驱动兼容的 PyTorch CUDA wheel，再安装项目。`requirements-server.txt` 固定了关键框架版本；它不是包含全部 wheel 哈希的跨平台锁文件。

## 3. 只需要配置的入口文件

```bash
cp configs/runtime.env.example configs/runtime.env
```

编辑：

```bash
LEGACY_RUN=/实际的首轮运行根目录
WORKSPACE=/新建的持久目录/cavr_v5
GPU_IDS=0,1,2,3,4,5,6,7
GPUS_PER_REPLICA=2
MEMORY_GIB=72
TRAIN_PROFILE=configs/hardware/a100_8x80.yaml
```

`LEGACY_RUN` 必须能找到：

```text
data/states/train.jsonl
data/states/validation.jsonl
adapters/step40/adapter_model.safetensors
...
adapters/step400/adapter_model.safetensors
```

这里需要服务器上的完整原始推理状态，不是 GitHub 结果归档里的去文本数值文件。后者可用于统计重算，不能代替控制器训练输入。

只有四张 80GB 卡时，将 `GPU_IDS` 改为 `0,1,2,3`，训练配置改为 `configs/hardware/a100_4x80.yaml`。推理仍采用每个副本两张卡。一次正在恢复的 GRPO 作业不要改变卡数。

## 4. 初始化和基础检查

```bash
bash scripts/v5/00_init.sh
bash scripts/v5/test_cpu.sh
```

初始化会复制小型状态文件到新目录，并登记旧适配器的只读路径和哈希，不复制几十 GB 的基础模型，不改变旧结果。新目录的 `experiment.json` 是后续各阶段的统一索引。

旧训练集按原题划分为 `gate_train`、`gate_dev`、`gate_holdout`，约为 60%、20%、20%。同题的多个初始状态不会跨分区。这里的 holdout 只表示未参与控制器拟合，不表示历史 backbone 没见过这些题。已查看过的 validation 仍作为探索性验证，独立结论看后面的原始 test 和迁移任务。

## 5. 第一优先：冻结执行器的 CAVR

```bash
bash scripts/v5/01_frozen.sh
```

脚本默认针对 step400 完成：

- 对训练/开发状态建立 16 次反思和 16 次重试的 `fit` 样本库。
- 对 holdout/validation 建立独立随机种子的 `score` 样本库。
- 提取 Base 模型的冻结隐藏特征并缓存。
- 分别训练 CAVR、置信度门控、直接价值门控、去除损害项的控制器。
- 输出不同反思调用预算和成本系数下的正确率、配对区间、token 数及耗时。

这一步不会重跑 400 步 32B GRPO。主要 GPU 工作是推理和隐藏状态提取，控制器默认在 CPU 上训练。

结果位于：

```text
reports/frozen/step400/cavr/gate_holdout/policies.csv
reports/frozen/step400/cavr/validation/policies.csv
reports/frozen/step400/confidence/validation/policies.csv
reports/frozen/step400/direct/validation/policies.csv
controllers/step400/*/history.json
```

主要比较固定成本系数 0.05、预算 0.3 的预设点，同时完整保留预算曲线。不要在 test 表里挑出最好的某个点再当作预先设定的结果。

## 6. 修复动作概率测量并分析学习过程

```bash
bash scripts/v5/02_measurement.sh
```

这一步评估 Base 和全部十个 checkpoint。每个状态使用两套独立的 64 次动作前缀采样，分别称为 actionA、actionB。动作分为 COMMIT、REFLECT、INVALID、CENSORED，不对前两项重新归一化后冒充完整自然概率。

默认动作前缀上限 24 tokens，生成分布显式指定 temperature=0.7、top_p=0.95、top_k=0。若截断率较高，应对整个实验声明更大的动作窗口并使用新输出目录，不能直接覆盖旧测量。

```text
reports/measurement/step400-audit.json
reports/measurement/step400-selection.json
reports/dynamics/trajectory.csv
reports/dynamics/modular_expected_matrix.csv
reports/dynamics/decomposition.json
reports/dynamics/*.png
```

`modular_expected_matrix.csv` 是动作概率和强制执行价值构成的期望矩阵。下一步才是真正重新生成的交叉执行实验，两者文件名和用途已经分开。

## 7. 实际跨 checkpoint 干预

```bash
bash scripts/v5/03_crossed.sh
```

默认交叉 Base、step160、step240、step400，共 16 个决策/执行组合。决策来自相应 checkpoint 的已记录动作样本，执行阶段由另一个 checkpoint 真实重新生成；COMMIT 直接保留原答案。默认每个组合每状态执行 4 次。

```text
interventions/base--step400.jsonl
reports/crossed_actual/actual_interventions.csv
```

这是统一文本接口上的组件干预，不是拼接不同 checkpoint 的神经网络参数。无效决策的后备动作为 COMMIT，会保留其原始动作分类供审计。

## 8. CAVR 交替训练和消融

先验证完整保存和恢复：

```bash
bash scripts/v5/09_resume_test.sh
```

它使用同一个 8-step 目标，在第 2 步保存停止，再从完整 checkpoint 恢复到第 8 步。测试目录存在时会拒绝覆盖。

随后分别运行：

```bash
bash scripts/v5/04_online.sh
bash scripts/v5/05_no_refresh.sh
bash scripts/v5/06_uniform_executor.sh
```

默认每轮 40 步、三轮，执行器从 step400 开始。新运行目录互不覆盖。`TRAIN_SEED` 决定外层运行种子，切换种子会产生新的算法实验目录。

交替训练不是把旧 400 步接成同一个优化器轨迹。每轮继承上一轮 LoRA 权重、重新建立本轮优化器；轮内中断则恢复该轮完整优化器和训练进度。报告时分别列出历史训练成本和新增算法成本。

`no_refresh` 使用初始控制器到最后，不在末尾偷偷重训控制器。它的测试明确标记执行器漂移。`uniform_executor` 是均匀状态采样的执行器训练对照。

## 9. 独立测试和跨领域泛化

先锁定控制器、阈值、预算和成本系数，再执行：

```bash
bash scripts/v5/07_ood.sh math
bash scripts/v5/07_ood.sh aime
bash scripts/v5/07_ood.sh gsm8k
bash scripts/v5/build_code_runner.sh
bash scripts/v5/07_ood.sh humaneval
```

MATH 使用原始 test。其余为任务/难度迁移。所有题目重新生成初始状态，默认每题 2 次，保留所有可解析的初始解答，不按反思结果筛选最终测试集。源数据在下载时解析并记录精确仓库 revision。

代码任务的测试文本不会进入模型提示或控制器特征。代码只在受限 Docker 容器中执行；没有 Docker 时不会退回宿主机 `exec`。容器运行会记录镜像 ID、超时和异常，仍需抽查奖励被投机利用的样例。

## 10. 独立随机种子与尺度对照

```bash
bash scripts/v5/08_standard_seeds.sh
```

该脚本是新的 v5 标准 GRPO 对照，会训练 41、73、107 三个独立种子。它不会把旧 seed41 结果覆盖为新协议结果。完成后可将新适配器路径添加到工作区的 checkpoint 索引，使用相同的 probe / selection / dynamics 入口评估。

7B、14B 的硬件训练配置位于 `configs/hardware/qwen_7b_8x80.yaml` 与 `qwen_14b_8x80.yaml`。按原题控制的外部状态实验可以复用同一题目/初始状态；研究各模型自身的反思形成则须用对应模型重新执行 `rd5 draft`，不能把 32B 草稿悄悄标成 7B 自生成轨迹。不同隐藏维度的控制器分别训练。

## 11. 中断与恢复

推理输出每条生成立即落盘并 fsync，任务重新执行时跳过已完成 event。配置或模型哈希变化时拒绝混跑。只有完成且哈希一致的样本库才能被 CLI 的 fit/evaluate 读取。

```bash
rd5 repair-tail --path /实际路径/被中断的分片.jsonl
```

该命令只修复最后一条被截断的 JSON 记录，先备份原文件，不会吞掉中间损坏。

GRPO 的完整恢复命令见 `RESUME.md`。每 20 步保存，默认保留 24 个 checkpoint。提前按第一次保存的实际体积核算整轮磁盘需求；150 GiB 只是启动下限，不是整轮存储预算。

## 12. 第一批回传

回传 `reports/frozen/`、`reports/measurement/`、`reports/dynamics/`、`reports/crossed_actual/`、控制器的 `model.json`/`history.json`、运行配置和训练 `status.json`/`metrics.jsonl` 即可先做判断。基础权重和完整优化器 checkpoint 留在持久盘，不提交 Git。

完整生成文本用于案例抽检，在服务器保留。论文汇总优先从数值报告重算，不手工抄图上的数字。
