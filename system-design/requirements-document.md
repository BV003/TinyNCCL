# TinyNCCL

## 做什么
基于 CUDA 从零实现 All‑Reduce 通信原语，能力覆盖：
- 经典 **Ring All‑Reduce**（必做）
- 可选扩展：**Tree All‑Reduce**
- GPU P2P（GPUDirect）跨卡数据传输
- 正确性校验：对标 PyTorch `torch.distributed` 验证结果
- 性能基准：带宽对比自研实现 vs 原生 NCCL

## 为什么选这个项目
1. **AI Infra 面试高频考点**：All‑Reduce 是分布式训练的核心原语，NCCL 相关几乎是必问题目。
2. **技术栈补齐**：从单机推理拓展至多卡通信，补齐分布式训练最关键模块。
3. **代码体量可控**：一套干净可运行的 Ring All‑Reduce CUDA 内核，代码量约 500‑800 行。
4. **面试故事素材充足**：可延伸讲解：为什么 Ring 相比 Tree 在 NCCL 中更常用、NVSwitch 拓扑下的优化思路。
5. **可继续深度迭代**：后续可扩展基于 NVLink / NVSwitch 的拓扑感知调度。


