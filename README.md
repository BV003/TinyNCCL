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

### Measured Results

Test environment: one host `iad0`, two NVIDIA L4 GPUs, `--warmup 2 --iters 10`.

| Implementation | count | bytes/rank | latency (ms) | busbw (GB/s) |
|---------------|-------|-----------|--------------|--------------|
| Tiny-IPC      |  2048 |   0.01 MB |       33.505 |         0.00 |
| NCCL          |  2048 |   0.01 MB |        0.080 |         0.10 |
| Tiny-IPC      | 32768 |   0.13 MB |       33.219 |         0.00 |
| NCCL          | 32768 |   0.13 MB |        0.079 |         1.65 |
| Tiny-IPC      |524288 |   2.10 MB |       33.485 |         0.06 |
| NCCL          |524288 |   2.10 MB |        0.447 |         4.69 |

### Analysis

- Correctness passes for all sizes; results match the NCCL reference.
- TinyNCCL latency is roughly constant (~33 ms) and dominated by the per-call
  file-based handle rendezvous and CUDA IPC event exchange, not by the GPU data
  transfer itself.
- NCCL uses a persistent process group and a single handshake at startup, so its
  reported latency is the steady-state collective cost.
- These numbers are an end-to-end comparison of two different architectures and
  should not be read as a GPU bandwidth comparison. A persistent communicator
  (setup once, reuse many times) is the main next-step optimization.

## Future Improvements

- **Persistent communicator**: exchange IPC handles once and reuse them, removing
  the ~33 ms per-call bootstrap and making benchmark measure steady-state latency.
- **Reduce sync overhead**: drop the redundant per-step file barrier once the IPC
  event path is trusted.
- **Validate on 4/8 GPUs**: the generalized Ring logic is only tested on 2 GPUs.
- **Tree AllReduce**: add as a comparison algorithm.
- **Multi-node transport**: TCP, then InfiniBand/RoCE with GPUDirect RDMA.
- **Code cleanup**: remove unused `IpcHandle` fields, unused includes, and the
  per-step unused init event; unify error checks in the single-process path.


## Current Implementation

当前已实现单节点多进程 CUDA IPC Ring AllReduce：

- 每个进程管理一张 GPU
- 使用 CUDA IPC memory handle 交换通信 buffer
- 使用 CUDA IPC event 同步 Reduce-Scatter 和 All-Gather
- 已在一台机器的两张 NVIDIA L4 上通过多尺寸测试

当前不支持跨节点通信。跨节点需要 TCP、InfiniBand、RoCE 或 RDMA transport。
