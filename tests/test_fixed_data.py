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

    ipc_root = make_ipc_root("fixed", rank, local_rank)

    all_ok = True
    counts = [world, world * 4, world * 16]
    for count in counts:
        ipc_dir = os.path.join(ipc_root, f"count_{count}")
        if rank == 0:
            os.makedirs(ipc_dir, exist_ok=True)
        barrier(local_rank)

        # Rank r contributes r * 1000 + [0, 1, ..., count - 1].
        values = torch.arange(count, dtype=torch.float32, device="cuda")
        send = values + rank * 1000.0
        recv = torch.empty_like(send)
        expected = values * world + 1000.0 * sum(range(world))

        tiny.tiny_ring_allreduce_sum_ipc(rank, world, send, recv, ipc_dir)
        barrier(local_rank)

        local_ok = torch.allclose(recv, expected, atol=1e-5, rtol=1e-5)
        status = "OK" if local_ok else "FAIL"
        print(f"[fixed count={count:>5}] rank={rank} {status}")
        if not local_ok:
            bad = torch.nonzero((recv - expected).abs() > 1e-5).flatten()[:5]
            for index in bad.tolist():
                print(
                    f"  idx={index}: got={recv[index].item():.6f} "
                    f"expected={expected[index].item():.6f}"
                )
        all_ok = all_ranks_ok(all_ok and local_ok, local_rank)

    cleanup_ipc_root(ipc_root, rank, local_rank)
    if rank == 0:
        print("PASS: fixed-data IPC tests" if all_ok else "FAIL: fixed-data IPC tests")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
