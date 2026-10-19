# 保存、停止与恢复

## 推理和控制器

`probe`、`draft`、`execute-plan` 使用单条 event 落盘，生成完成后立即 flush/fsync。重复同一命令即续跑。增加同一采样库的 repetitions 可以扩展已保存前缀；减少样本数或改变模型/协议应使用新目录。

隐藏状态提取按状态缓存，再原子合并为 NPZ。控制器保存模型、AdamW 状态、随机数状态和最优开发集参数；`fit --resume` 恢复，已完整完成且签名一致的训练直接复用。

完整样本库附带 manifest 和 completion receipt。单条末尾截断可通过 `rd5 repair-tail` 备份并修复。不要编辑事件中的问题 ID 或把其他设置的分片手工拼到同一个库。

## GRPO

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m cavr.train \
  --config configs/hardware/a100_8x80.yaml \
  --states /持久目录/cavr_v5/states/gate_train.jsonl \
  --output /持久目录/cavr_v5/runs/standard_v5_seed41 \
  --mode standard --seed 41 --resume auto
```

保持与原作业完全相同的模型、输入状态、目标步数、随机种子、GPU 数量和配置。恢复从完整 checkpoint 开始，未保存部分重新计算。独立 LoRA 导出仅用于推理或下一轮权重初始化，不等于完整续训。

训练接收 SIGTERM 和 SIGUSR1 后，在所有 rank 的下一个优化步边界协调保存并停止。SIGKILL、节点消失和平台立即回收不会给进程保存机会，此时恢复最近的周期性 checkpoint。长生成期间收到信号也需要等待该步返回；平台应提供足够的终止宽限时间。

```bash
bash scripts/v5/09_resume_test.sh
```

测试保持同一个 8-step 目标，先在 step2 保存停止，再恢复到 step8。它同时检查完整优化器分片数量和最终适配器导出。GPU 服务器上先完成此测试，再依赖跨周恢复。

## 存储

将模型缓存、完整 optimizer checkpoint 和数据放在资源刷新后仍保留的存储。启动下限不等于完整容量规划；按首个完整 checkpoint 的实际大小乘以保留数量，再加基础权重、导出适配器和日志空间。

ZeRO 分片恢复保持原 world size。跨 GPU 数量恢复需要单独转换/验证，不由这套自动续跑流程隐式完成。
