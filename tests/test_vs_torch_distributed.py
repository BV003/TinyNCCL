from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

import tiny
from harness import (
    all_ranks_ok,
    barrier,
    cleanup_ipc_root,
    init_distributed,
    make_ipc_root,
)


def main() -> int:
    rank, world, local_rank = init_distributed()
    if world < 2:
        if rank == 0:
            print("skip: needs at least 2 ranks")
        dist.destroy_process_group()
        return 0

    ipc_root = make_ipc_root("random", rank, local_rank)
    comm_dir = os.path.join(ipc_root, "comm")
    if rank == 0:
        os.makedirs(comm_dir, exist_ok=True)
    barrier(local_rank)

    base_seed = 12345
    counts = [world, world * 4, world * 16, world * 1024]
    comm = tiny.RingCommunicator(rank, world, max(counts), comm_dir)

    all_ok = True
    for count in counts:
        torch.manual_seed(base_seed + rank)
        send = torch.randn(count, device="cuda")

        reference = send.clone()
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        barrier(local_rank)

        recv = torch.empty_like(send)
        torch.cuda.synchronize()
        comm.allreduce(send, recv)
        barrier(local_rank)

        local_ok = torch.allclose(recv, reference, atol=1e-5, rtol=1e-5)
        status = "OK" if local_ok else "FAIL"
        print(f"[NCCL compare count={count:>5}] rank={rank} {status}")
        if not local_ok:
            bad = torch.nonzero((recv - reference).abs() > 1e-5).flatten()[:5]
            for index in bad.tolist():
                print(
                    f"  idx={index}: tiny={recv[index].item():.6f} "
                    f"nccl={reference[index].item():.6f}"
                )
        all_ok = all_ranks_ok(all_ok and local_ok, local_rank)

    cleanup_ipc_root(ipc_root, rank, local_rank)
    if rank == 0:
        print("PASS: TinyNCCL matches NCCL" if all_ok else "FAIL: mismatch detected")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
