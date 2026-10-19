# 历史执行环境与代码

这里记录的是实际跑完 seed41 的配置和脚本，**包括尚未修复的动作概率测量实现**。这是一份复现资料，不代表推荐直接重跑错误口径分析。

## 代码恢复

从原 commit `6f2e91eb7d22bd65ab44b4815be823fa03878e1f` 的 `ReflectionDynamics/` 子目录取出项目源码，在独立目录中应用 `source_delta.patch`（路径前缀为 `a/src/…` 和 `b/src/…`，可用 `patch -p1 < source_delta.patch`），将恢复后的 `src/` 放在运行目录的 `source/src/` 下，再放入这里的 Python 脚本与 `configs/`。核对 `../provenance/source_sha256.json` 中的文件哈希。补丁包含运行时训练兼容性/存储容量检查等改动；它不改动仓库主源码。

`run.sh` 是原平台入口记录，依赖原绝对运行路径及 `activate.sh`，不是可在任意电脑直接执行的安装脚本。重新运行需要设置新的持久运行根目录、环境与缓存路径，将 `PYTHONPATH` 指向运行目录和 `source/src`，并修改 `storage_guard.py` 的存储根目录/配额。不要在本次已完成目录覆盖输入或输出。`pipeline.py` 会验证阶段签名和输出哈希以支持续跑。

## 依赖与数据

- Python 3.11，PyTorch 2.9.1（CUDA 12.9 构建）、Transformers 4.57.6、TRL 0.27.1、PEFT 0.19.1、DeepSpeed 0.19.2。具体包名/版本以 `packages-observed-20261019.json` 为准。
- 包清单是在归档当天读取开发机上的原环境所得，不是训练启动时创建的带 wheel 哈希锁文件。不能保证跨驱动、操作系统和 GPU 的位级复现；`requirements-observed.txt` 只记录版本，PyTorch CUDA wheel 仍需正确的官方源。
- 基础模型 revision 与原代码 commit 见 `../provenance/provenance.json`。数据/状态哈希见 `../provenance/data/`，评估分片哈希见 `../provenance/evaluation-source-sha256.json`。
- 数据仓库 revision 未在现有 manifest 中完整固定；此归档不补造不存在的版本证据。完整轨迹和状态仍需从原持久存储取得。仅下载当前上游数据不能保证重建相同状态。
- 训练过程的 checkpoint 需要使用 `export_adapter.py` 导出完整 LoRA 权重，再运行 `verify_adapter.py` 验证。不要把 ZeRO checkpoint 根目录的小占位 adapter 文件直接当作有效权重。

下一次实验先修复动作概率估计并通过针对 tokenizer 边界和解码分布的验证，再启动昂贵的模型计算。归档中的数值统计可直接用上一级 `recompute.py` 重算，不需要 GPU。
