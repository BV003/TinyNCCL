"""Exercise the persistent RingCommunicator across many reuses."""
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

    ipc_root = make_ipc_root("persistent", rank, local_rank)
    ipc_dir = os.path.join(ipc_root, "comm")
    if rank == 0:
        os.makedirs(ipc_dir, exist_ok=True)
    barrier(local_rank)

    max_count = 4096
    comm = tiny.RingCommunicator(rank, world, max_count, ipc_dir)

    all_ok = True
    counts = [world, world * 4, world * 16, world * 1024]
    for iteration in range(5):
        for count in counts:
            # Rank r contributes r * 1000 + [0, 1, ..., count - 1].
            values = torch.arange(count, dtype=torch.float32, device="cuda")
            send = values + rank * 1000.0
            recv = torch.empty_like(send)
            expected = values * world + 1000.0 * sum(range(world))

            comm.allreduce(send, recv)
            torch.cuda.synchronize()

            local_ok = torch.allclose(recv, expected, atol=1e-5, rtol=1e-5)
            if not local_ok:
                print(
                    f"[iter={iteration} count={count:>5}] rank={rank} FAIL",
                    file=sys.stderr,
                )
            all_ok = all_ok and local_ok
        if rank == 0:
            print(f"iteration {iteration}: ok")

    ok = all_ranks_ok(all_ok, local_rank)
    cleanup_ipc_root(ipc_root, rank, local_rank)
    if rank == 0:
        print("PASS: persistent communicator" if ok else "FAIL: persistent")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
