# Cloud Setup — TinyNCCL 开发环境

> 目标：一台 Linux 主机 + **2 张 CUDA GPU（同一台机器）**，15 分钟到达 Phase 0（环境验证）。
> 推荐 **Vast.ai**（最便宜）；备选 **GCP**（最稳定）。两者选一即可，后面步骤完全一样。

## 0. 我们的硬性要求（回顾）

| 要求 | 值 |
|---|---|
| GPU | **≥ 2 张** NVIDIA（同一台机器，单机多卡） |
| 显存 | ≥ 8 GB/张（24 GB 更舒服） |
| P2P | 可选（有则走 GPUDirect，没有则走 host-staging fallback，代码都支持） |
| NVLink | **不需要** |
| 网络 | 不需要（v1 单机范围） |
| OS | Linux（Ubuntu 20.04/22.04） |

---

## 方案 A：Vast.ai（推荐 · 最便宜）

### A1. 注册并充值
1. 打开 https://vast.ai → 注册账号 → 绑定支付方式（信用卡/PayPal）充值 $10–20 即可起步。

### A2. 搜索实例（关键过滤条件）
点击 **Rent** 进入搜索页，设置过滤：
- `GPU count` → **2**（或 ≥2）
- `GPU model` → 优先 `RTX 4090` / `A10G` / `L4` / `T4`（建议 24 GB 显存型号）
- `RAM` → **≥ 32 GB**（vCPU 数量高的更好）
- `Storage` → **≥ 100 GB**
- `On-demand / Interruptible` → 两者都看，**Interruptible 便宜 50%+**

按 `$/hr` 排序，挑一个价格低、评分高（listings 右上角有 rate/reliability）的机器。

### A3. 配置并启动
1. 点进一个 listing → **Edit Configuration**：
   - **Template / Image** → 选 **`Pytorch`**（自带 CUDA + PyTorch，省去装环境）
   - Disk → 选 ≥ 100 GB
2. 点 **Rent** → 等待开机（一般 1–5 分钟）。
3. 开机后页面会显示 **SSH 命令**，形如：
   ```bash
   ssh -p 22XXX root@12.34.56.78 -L 8080:localhost:8080
   ```
   复制它，在本机终端执行（密码会显示在页面上，或用你上传的 SSH key）。

### A4. 常用操作
- **关机/续租**：Vast.ai 控制台直接 Stop / Rent Again（按小时计费，不用就停）。
- **保存镜像**：配置好环境后，可以点实例的 **Save** 生成自己的模板，下次一键复现。
- **Vast.ai 的坑**：
  - 它是第三方市场，机器质量/网络参差不齐 → 挑评分高的，先用 `nvidia-smi` 验证
  - 有的 listing 写「2× GPU」但实际共享带宽 → 单机开发不受影响
  - 没有 24/7 客服 → 出问题换一台即可，数据用磁盘镜像/`rsync` 备份

---

## 方案 B：GCP（备选 · 最稳定）

1. 注册 Google Cloud → 创建项目 → 开启结算。
2. Compute Engine → **Create Instance**：
   - Name: `tinynccl`
   - Region: `us-central1-a` / `us-west1-b` / `europe-west4`（选有 L4 spot 库存的）
   - Machine: `g2-standard-48`（= 2× L4, 48 vCPU, 192 GB RAM）
   - GPU: `NVIDIA L4` × 2，勾选 **Spot**（便宜 50–70%）
   - Boot disk: **Ubuntu 22.04, 200 GB**
   - **Deep Learning VM** 勾选（或选 Deep Learning 镜像）→ CUDA + PyTorch 预装
3. 创建后 SSH（浏览器内 SSH 或 gcloud 命令行）。

---

## 方案通用：装完之后的 Phase 0 验证（A/B 都要做）

### 1. 确认 GPU + 驱动
```bash
nvidia-smi          # 应显示 2 张 GPU，driver 版本正常
nvidia-smi topo -m  # 查看两卡间拓扑（NVLink / PCIe），记下来
```

### 2. 确认 PyTorch 能看到两张卡
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```
预期输出：`2` + 两张卡的名字（如 `NVIDIA GeForce RTX 4090`）。

### 3. 检查 P2P 是否可用（关键信息，写进文档）
```python
import torch
print("cuda:0 -> cuda:1:", torch.cuda.can_device_access_peer(0, 1))
print("cuda:1 -> cuda:0:", torch.cuda.can_device_access_peer(1, 0))
```
- 结果 **True** → 你的主路径将是 GPUDirect P2P（L4/T4/A10G 数据中心卡大概率 True）
- 结果 **False** → 走 host-staging fallback（4090 基本是 False），代码两条路径都支持，不用慌

### 4. 冒烟测试：torch NCCL 在 2 卡上能跑通（我们的 baseline 基础设施）
```python
# smoke_test.py —— 跑通后我们后面所有正确性验证都靠它
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
t = torch.full((4,), rank + 1, device="cuda")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expected = sum(range(1, dist.get_world_size() + 1))
assert t.cpu().tolist() == [expected] * 4, t
print(f"rank {rank}: NCCL all_reduce OK -> {t.cpu().tolist()}")
dist.destroy_process_group()
```
运行：
```bash
torchrun --nproc_per_node=2 smoke_test.py
```
预期每张卡输出 `NCCL all_reduce OK -> [6, 6, 6, 6]`（2 卡时 1+2+3=6，实际 N 卡是 sum(1..N)）。

### 5. 日常开发连接（可选，体验最好）
- **VS Code Remote-SSH**：本地 VS Code 装 `Remote - SSH` 插件 → 连接上述 SSH 地址 → 直接在云主机上编辑/运行，文件放云主机 `/root/tiny-nccl`。
- 或者本地编辑，`scp` / `rsync` 同步到云主机。

---

## 成本控制清单

- [ ] 用 **Interruptible（Vast.ai）/ Spot（GCP）**，省 50–70%
- [ ] 不写代码就**关机**（按小时计费）
- [ ] 环境配好就**做镜像/快照**，下次 2 分钟恢复
- [ ] 预期总成本：$30–80（整个 v1 项目）

---

## Phase 0 完成标准（做到这些就可以开始写代码了）

- [ ] `nvidia-smi` 显示 2 张 GPU
- [ ] `nvidia-smi topo -m` 记录了两卡拓扑（PCIe 几代 / 是否 NVLink）
- [ ] PyTorch 看到 2 张卡
- [ ] P2P 探测结果（True/False）已记录
- [ ] `torchrun --nproc_per_node=2 smoke_test.py` 跑通
- [ ] SSH / VS Code Remote 连接顺畅

完成 → 进入 Phase 1：写 `src/topology.py` + torch 基准测试 harness（我帮你写）。
