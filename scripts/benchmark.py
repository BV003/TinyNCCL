#=============================================================================
# benchmark.py
#
# TinyNCCL — Ring AllReduce bandwidth benchmark.
#
# Measures the effective (bus) bandwidth of our Ring AllReduce:
#     busbw = 2 * (N-1) / N * S / t
# where S is the total bytes per rank and t is the average time.
#
# Usage:
#   python benchmark.py [count_elements] [num_iters]
#=============================================================================
from __future__ import annotations

import sys

import torch
import tiny


def run(count: int, iters: int) -> None:
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print("benchmark requires at least 2 GPUs")
        return

    bytes_per_rank = count * 4  # float32

    sendbuffs = [torch.randn(count, device=f"cuda:{i}") for i in range(n_gpus)]
    recvbuffs = [torch.empty(count, device=f"cuda:{i}") for i in range(n_gpus)]

    # warmup
    for _ in range(3):
        tiny.tiny_ring_allreduce_sum(sendbuffs, recvbuffs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        tiny.tiny_ring_allreduce_sum(sendbuffs, recvbuffs)
    end.record()
    torch.cuda.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0
    avg_ms = elapsed_s / iters * 1000.0

    busbw_gbps = 2.0 * (n_gpus - 1) / n_gpus * bytes_per_rank / (elapsed_s / iters) / 1e9

    print(f"GPUs        : {n_gpus}")
    print(f"count       : {count} ({bytes_per_rank / 1e6:.1f} MB per rank)")
    print(f"iters       : {iters}")
    print(f"avg time    : {avg_ms:.3f} ms")
    print(f"busbw       : {busbw_gbps:.2f} GB/s")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else (1 << 22)   # 16 MB
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run(count, iters)
