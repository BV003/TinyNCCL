# TinyNCCL

## Learning Purpose

I do this project for learning about NCCL and Distributed system.

## Hardware Requirements

2* A5000

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

## Run Tests and Benchmark

```bash
torchrun --nproc_per_node=2 tests/smoke_test.py
python tests/test_kernels.py
torchrun --nproc_per_node=2 tests/test_vs_torch_distributed.py > Logs/test.log 2>&1 
python scripts/benchmark.py   
```



## Now


/Users/michael/Documents/codehome/TinyNCCL/tests/test_fixed_data_two.py
测试出来是遗留缓冲的问题，是transport_copy的问题，是cudamemcpy不支持跨进程
NCCL不是这么做的CUDA‑IPC或者RDMA/TCP 网络