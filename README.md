# TinyNCCL

## Learning Purpose

I do this project for learning about NCCL and Distributed system.

## Hardware Requirements

2x A5000 on a single machine

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

Run the whole suite (builds first, then runs every test that fits this
machine — multi-process tests run with one rank per visible GPU, or `NPROC=N`,
and are skipped when fewer than 2 GPUs are visible):

```bash
scripts/run_tests.sh              # add --no-build to skip the re-install
```

If the environment that has torch is not the default interpreter, point the
script at it (torchrun is picked up from the same directory):

```bash
PYTHON=/path/to/venv/bin/python scripts/run_tests.sh
```

Or run them individually:

```bash
c++ -std=c++17 -O2 -Isrc tests/test_ring_schedule.cpp -o /tmp/ring_schedule && /tmp/ring_schedule
torchrun --nproc_per_node=2 tests/smoke_test.py
python tests/test_kernels.py
python tests/test_transport.py
python tests/test_single_process.py
torchrun --nproc_per_node=2 tests/test_fixed_data.py
torchrun --nproc_per_node=2 tests/test_vs_torch_distributed.py
torchrun --nproc_per_node=2 tests/test_persistent_communicator.py
```

`tests/test_ring_schedule.cpp` needs no GPU: it runs the real chunk-index and
signal math from `src/ring_schedule.h` for 1–8 ranks under random interleavings
of the GPU-side dependencies. The multi-process tests work for any
`--nproc_per_node`.

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

The multi-process API is the persistent communicator:

```python
comm = tiny.RingCommunicator(rank, world, max_count, ipc_dir)  # set up once
comm.allreduce(send, recv)                                     # reuse many times
```

## Measured Results

Test environment: one host, two NVIDIA H800 PCIe GPUs (MIG disabled),
`--warmup 2 --iters 10`.

| Implementation | count | bytes/rank | latency (ms) | busbw (GB/s) |
|---------------|-------|-----------|--------------|--------------|
| Tiny          |  2048 |   0.01 MB |        0.064 |         0.13 |
| NCCL          |  2048 |   0.01 MB |        0.060 |         0.14 |
| Tiny          | 32768 |   0.13 MB |        0.078 |         1.67 |
| NCCL          | 32768 |   0.13 MB |        0.061 |         2.13 |
| Tiny          |524288 |   2.10 MB |        0.210 |         9.97 |
| NCCL          |524288 |   2.10 MB |        0.176 |        11.90 |

TinyNCCL is within ~1.1–1.3x of NCCL across all sizes. For reference, the
original per-call bootstrap design measured ~33 ms on two L4 GPUs — constant
across sizes, i.e. entirely host-side overhead, not GPU time.

## Analysis

- Profiling the original per-call design showed the cost was dominated by
  `cudaIpcOpenMemHandle` (~2.35 ms) plus the file-based handle rendezvous.
- `RingCommunicator` sets up the stream, buffers, memory mapping and GPU-side
  synchronization once. Each `allreduce` then only submits work to the stream.
- Synchronization uses a per-rank monotonic 32-bit counter in device memory,
  signalled and waited on in-stream via `cuStreamWriteValue32` /
  `cuStreamWaitValue32` (resolved with `cudaGetDriverEntryPoint`). The
  steady-state path touches no files and makes no host round-trips. The same
  counter doubles as the cross-call barrier that protects the reused buffer.

## Future Improvements

- **Validate on 4/8 GPUs**: the ring schedule for n > 2 is checked by the CPU
  simulation in `tests/test_ring_schedule.cpp` (an all-gather indexing bug that
  only appeared with 3+ ranks was fixed this way), but has not yet been run on
  real hardware with more than 2 GPUs.
- **Vectorize the kernel / tune chunk size**: the reduce kernel is a plain
  element-wise add.
- **Use the caller's CUDA stream**: run on the torch current stream to drop the
  final `cudaStreamSynchronize`.
- **Tree AllReduce**: add as a comparison algorithm.
- **Multi-node transport**: TCP, then InfiniBand/RoCE with GPUDirect RDMA.

## Current Implementation

Currently implemented: single-node, multi-process CUDA IPC Ring AllReduce:

- Each process manages one GPU
- Communication buffers are exchanged via CUDA IPC memory handles (once, by
  `RingCommunicator`)
- Reduce-Scatter and All-Gather are synchronized with GPU-side memory counters,
  driven entirely in-stream

Cross-node communication is not supported yet. It requires TCP, InfiniBand, RoCE, or RDMA transport.
