#=============================================================================
# test_vs_torch_distributed.py
#
# TinyNCCL — correctness check for Ring AllReduce vs torch.distributed (NCCL).
#
# Runs under torchrun: one process per GPU. Each rank contributes a random
# tensor on its local GPU, then we compare two ways of computing the global
# element-wise sum:
#   * reference: torch.distributed.all_reduce (NCCL)
#   * ours:      tiny.tiny_ring_allreduce_sum (single-process, multi-GPU)
#
# Because tiny's API is single-process (it takes pointers to every GPU's
# tensor at once) while NCCL is multi-process, rank 0 reconstructs every
# rank's input locally using a deterministically broadcast seed:
#   torch.manual_seed(seed) seeds every CUDA device's generator identically,
#   so torch.randn(count, device="cuda:i") produces the same values in any
#   process.
#
# Usage:
#   torchrun --nproc_per_node=<N> tests/test_vs_torch_distributed.py
#=============================================================================
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

import tiny


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if world < 2:
        if rank == 0:
            print("skip: needs at least 2 GPUs")
        dist.destroy_process_group()
        return 0

    # Seed: rank 0 picks a random one and broadcasts it so every rank
    # generates the same tensor values (but on its own GPU).
    if rank == 0:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item()
    else:
        seed = 0
    seed_t = torch.tensor([seed], dtype=torch.int64, device="cuda")
    dist.broadcast(seed_t, src=0)
    seed = seed_t.item()

    # Test a few sizes: small, 1M, and a non-power-of-two still divisible by N.
    counts = [1 << 12, 1 << 20, world * 3000]

    all_ok = True

    for count in counts:
        torch.manual_seed(seed)

        # Each rank's contribution lives on its own GPU.
        t = torch.randn(count, device=f"cuda:{rank}")

        # Reference: NCCL all_reduce (SUM).
        ncc = t.clone()
        dist.all_reduce(ncc, op=dist.ReduceOp.SUM)

        # Ours: rank 0 reconstructs every rank's input on its GPU and runs the
        # single-process ring allreduce.
        if rank == 0:
            torch.manual_seed(seed)
            send = [torch.randn(count, device=f"cuda:{i}") for i in range(world)]
            recv = [torch.empty(count, device=f"cuda:{i}") for i in range(world)]
            tiny.tiny_ring_allreduce_sum(send, recv)

        # Gather every rank's NCCL result to rank 0 for comparison.
        if rank == 0:
            gather_list = [torch.empty(count, device="cuda:0") for _ in range(world)]
        else:
            gather_list = None
        dist.gather(ncc, gather_list=gather_list, dst=0)

        if rank == 0:
            ok = True
            for i in range(world):
                got = recv[i].cpu()
                ref = gather_list[i].cpu()
                max_err = (got - ref).abs().max().item()
                match = torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
                ok = ok and match
                status = "OK " if match else "FAIL"
                print(f"  [count={count:>10}] gpu {i}: {status}  max_abs_err={max_err:.3e}")
            if not ok:
                all_ok = False

        dist.barrier()

    if rank == 0:
        print("PASS: ring allreduce matches NCCL" if all_ok else "FAIL: mismatch detected")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
