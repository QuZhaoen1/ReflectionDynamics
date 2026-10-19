# 服务器文件索引

本次完成运行根目录：

`/personal_fast_storage/reflectiondynamics/runs/seed41-20261017-formal`

| 相对路径/位置 | 内容 |
|---|---|
| `runs/grpo_32b_seed41/checkpoint-{40,80,...,400}` | 十个完整 ZeRO 训练 checkpoint，含训练状态和优化器分片 |
| `adapters/step{40,80,...,400}` | 导出的独立 LoRA adapter，用于加载和评估 |
| `data/evaluations/{base,step40,...,step400}` | 完整验证生成及 manifests |
| `data/states/train.jsonl`、`validation.jsonl` | 固定训练/验证初始状态 |
| `data/drafts`、`data/evaluations/train_probe` | 校准草稿和训练侧探测 |
| `outputs/validation` | 历史原始汇总；包含已知有缺陷的概率派生指标，不可直接用于论文结论 |
| `logs`、`completed_stages`、`status.json` | 运行日志、输出哈希收据和完成标记 |
| `/project_c20250508/262301122/reflectiondynamics-assets/models/Qwen2.5-32B` | 约 62 GiB 基础模型；不是个人 100 GiB 内的第二份模型 |
| `/project_c20250508/262301122/reflectiondynamics-assets/datasets` | 下载数据源 |
| `/personal_fast_storage/reflectiondynamics/envs/py311-torch291` | Python 环境 |
| `/project_c20250508/262301122/reflectiondynamics-runs/seed41-20261017-formal` | 存储修复前数据保留副本，非当前最终输出目录 |
| `/project_c20250508/262301122/reflectiondynamics-runs/seed41-20261016` | 之前的 preview，与正式运行分开 |

最终流水线日志报告个人存储占用约 7.54 GiB（100 GiB 配额）。这是当时检查值，不是未来容量保证。本次归档未删除或迁移任何原始模型、数据或 checkpoint。GitHub 归档不替代这些大文件的存储备份；继续实验仍需正确挂载个人存储与项目存储。
