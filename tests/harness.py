"""Shared scaffolding for TinyNCCL multi-process tests.

Run the tests with torchrun, e.g.:

    torchrun --nproc_per_node=2 tests/test_fixed_data.py
"""
from __future__ import annotations

import os
import shutil

import torch
import torch.distributed as dist


def barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def init_distributed() -> tuple[int, int, int]:
    """Initialize NCCL and return ``(rank, world_size, local_rank)``."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def make_ipc_root(tag: str, rank: int, local_rank: int) -> str:
    """Create a per-run shared IPC directory, cleaned up once by rank 0."""
    run_id = f"{os.getppid()}_{os.environ.get('MASTER_PORT', 'default')}"
    root = f"/tmp/tinynccl_{tag}_{run_id}"
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root, exist_ok=True)
    barrier(local_rank)
    return root


def cleanup_ipc_root(root: str, rank: int, local_rank: int) -> None:
    barrier(local_rank)
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)


def all_ranks_ok(local_ok: bool, local_rank: int) -> bool:
    """AND a per-rank result across all ranks (all_reduce MIN)."""
    flag = torch.tensor([int(local_ok)], device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    barrier(local_rank)
    return bool(flag.item())
