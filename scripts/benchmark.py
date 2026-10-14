"""End-to-end benchmark for TinyNCCL CUDA IPC AllReduce.

Run with torchrun, for example:

    torchrun --nproc_per_node=2 scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

import tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=None,
        help="element counts; defaults to sizes divisible by world size",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip-nccl", action="store_true")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="use the persistent RingCommunicator instead of the "
        "stateless API",
    )
    parser.add_argument(
        "--ipc-root",
        default="/tmp/tinynccl_benchmark",
        help="shared directory for IPC bootstrap files",
    )
    return parser.parse_args()


def barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def check_result(result: torch.Tensor, reference: torch.Tensor) -> bool:
    local_ok = torch.allclose(result, reference, atol=1e-5, rtol=1e-5)
    flag = torch.tensor([int(local_ok)], device=result.device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _time(run_once, warmup: int, iters: int, local_rank: int) -> float:
    """Warm up, then return the mean seconds per ``run_once()`` call."""
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(local_rank)
    barrier(local_rank)

    start = time.perf_counter()
    for _ in range(iters):
        run_once()
    torch.cuda.synchronize(local_rank)
    barrier(local_rank)
    return (time.perf_counter() - start) / iters


def run_tiny(
    send: torch.Tensor,
    rank: int,
    world: int,
    local_rank: int,
    root: Path,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    recv = torch.empty_like(send)

    warmup_dirs = [root / f"tiny_warmup_{i}" for i in range(warmup)]
    iter_dirs = [root / f"tiny_iter_{i}" for i in range(iters)]
    if rank == 0:
        for directory in warmup_dirs + iter_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    barrier(local_rank)

    call = {"n": 0}

    def run_once() -> None:
        # The stateless API bootstraps IPC handles on every invocation and
        # needs a fresh shared directory for each one.
        i = call["n"]
        call["n"] += 1
        directory = warmup_dirs[i] if i < warmup else iter_dirs[i - warmup]
        tiny.tiny_ring_allreduce_sum_ipc(rank, world, send, recv, str(directory))

    seconds = _time(run_once, warmup, iters, local_rank)
    return seconds, recv


def run_tiny_persistent(
    send: torch.Tensor,
    rank: int,
    world: int,
    local_rank: int,
    root: Path,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    recv = torch.empty_like(send)

    comm_dir = root / "comm"
    if rank == 0:
        comm_dir.mkdir(parents=True, exist_ok=True)
    barrier(local_rank)

    comm = tiny.RingCommunicator(rank, world, send.numel(), str(comm_dir))
    seconds = _time(
        lambda: comm.allreduce(send, recv), warmup, iters, local_rank
    )
    return seconds, recv


def run_nccl(
    send: torch.Tensor,
    local_rank: int,
    warmup: int,
    iters: int,
) -> tuple[float, torch.Tensor]:
    result = torch.empty_like(send)

    def run_once() -> None:
        result.copy_(send)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)

    seconds = _time(run_once, warmup, iters, local_rank)
    return seconds, result


def format_result(name: str, count: int, seconds: float, world: int) -> str:
    bytes_per_rank = count * 4
    bus_bandwidth = (
        2.0 * (world - 1) / world * bytes_per_rank / seconds / 1e9
    )
    return (
        f"{name:<10} count={count:>10} "
        f"bytes={bytes_per_rank / 1e6:>8.2f} MB "
        f"latency={seconds * 1e3:>10.3f} ms "
        f"busbw={bus_bandwidth:>8.2f} GB/s"
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is unavailable", file=sys.stderr)
        return 1
    if "RANK" not in os.environ:
        print(
            "Run this benchmark with torchrun, for example: "
            "torchrun --nproc_per_node=2 scripts/benchmark.py",
            file=sys.stderr,
        )
        return 1

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        if rank == 0:
            print("benchmark requires at least 2 ranks", file=sys.stderr)
        dist.destroy_process_group()
        return 1

    counts = args.counts or [world * 1024, world * 16384, world * 262144]
    invalid = [count for count in counts if count <= 0 or count % world != 0]
    if invalid:
        if rank == 0:
            print(f"counts must be positive and divisible by {world}: {invalid}",
                  file=sys.stderr)
        dist.destroy_process_group()
        return 1

    run_id = f"{os.getppid()}_{os.environ.get('MASTER_PORT', 'default')}"
    root = Path(args.ipc_root) / run_id
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
    barrier(local_rank)

    try:
        for count in counts:
            torch.manual_seed(2026 + rank)
            send = torch.randn(count, device="cuda")
            count_root = root / f"count_{count}"
            if rank == 0:
                count_root.mkdir(parents=True, exist_ok=True)
            barrier(local_rank)

            if args.persistent:
                tiny_seconds, tiny_result = run_tiny_persistent(
                    send, rank, world, local_rank, count_root,
                    args.warmup, args.iters,
                )
            else:
                tiny_seconds, tiny_result = run_tiny(
                    send, rank, world, local_rank, count_root,
                    args.warmup, args.iters,
                )

            if not args.skip_nccl:
                nccl_seconds, nccl_result = run_nccl(
                    send, local_rank, args.warmup, args.iters
                )
                tiny_ok = check_result(tiny_result, nccl_result)
            else:
                nccl_seconds = None
                tiny_ok = True

            if rank == 0:
                label = "Tiny-Persist" if args.persistent else "Tiny-IPC"
                print(format_result(label, count, tiny_seconds, world))
                if nccl_seconds is not None:
                    print(format_result("NCCL", count, nccl_seconds, world))
                    print(f"  correctness: {'OK' if tiny_ok else 'FAIL'}")
            barrier(local_rank)

            if not tiny_ok:
                return 1
    finally:
        barrier(local_rank)
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
        barrier(local_rank)
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
