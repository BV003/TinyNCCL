# TinyNCCL

## Learning Purpose

I do this project for learning about NCCL and Distributed system.

## Hardware Requirements

2* A5000,一台机器上

connect to terminal
```
ssh -i ~/.ssh/vastai_id -p 15598 root@88.183.139.106 -L 8080:localhost:8080    
```

## Environment

| Component      | Version         |
|---------------|-----------------|
| NVIDIA Driver  | ≥ 13.0          |
| nvcc          | 13.0            |
| Python         | 3.12            |
| PyTorch        | 2.5.1+cu130     |

## Build & Install

`--no-build-isolation` is required: `pip install -e .` creates a temporary build
environment that fetches a mismatched torch from PyPI, producing ABI-incompatible
binaries. `--no-build-isolation` uses the system torch instead.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e . --no-build-isolation
```

## Tests 

```bash
torchrun --nproc_per_node=2 tests/smoke_test.py
python tests/test_kernels.py
python tests/test_transport.py
python tests/test_single_process.py
torchrun --nproc_per_node=2 tests/test_fixed_data.py
torchrun --nproc_per_node=2 tests/test_vs_torch_distributed.py 
```
## Benchmark

```bash
torchrun --nproc_per_node=2 scripts/benchmark.py
```

Optional arguments:

```bash
torchrun --nproc_per_node=2 scripts/benchmark.py \
  --counts 2048 32768 524288 \
  --warmup 2 \
  --iters 10
```

The benchmark reports end-to-end latency and effective bus bandwidth for the
TinyNCCL CUDA IPC path and the PyTorch NCCL baseline. TinyNCCL latency includes
the current per-call IPC handle and event bootstrap overhead.


## Current Implementation

当前已实现单节点多进程 CUDA IPC Ring AllReduce：

- 每个进程管理一张 GPU
- 使用 CUDA IPC memory handle 交换通信 buffer
- 使用 CUDA IPC event 同步 Reduce-Scatter 和 All-Gather
- 已在一台机器的两张 NVIDIA L4 上通过多尺寸测试

当前不支持跨节点通信。跨节点需要 TCP、InfiniBand、RoCE 或 RDMA transport。
