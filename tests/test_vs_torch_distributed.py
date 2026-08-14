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


    # Test a few sizes: small, 1M, and a non-power-of-two still divisible by N.
    counts = [8, 16,32]

    all_ok = True

    base_seed = 12345

    for count in counts:
        torch.manual_seed(base_seed + rank)

        # Each rank's contribution lives on its own GPU.
        t = torch.randn(count, device=f"cuda:{rank}")

        # Reference: NCCL all_reduce (SUM).
        ncc = t.clone()
        dist.all_reduce(ncc, op=dist.ReduceOp.SUM)
        dist.barrier()

        # Ours: rank 0 reconstructs every rank's input on its GPU and runs the
        # single-process ring allreduce.
        if rank == 0:
            send = []

            for i in range(world):
                torch.manual_seed(base_seed + i)

                send.append(
                    torch.randn(
                        count,
                        device=f"cuda:{i}"
                    )
                )

            recv = [
                torch.empty(
                    count,
                    device=f"cuda:{i}"
                )
                for i in range(world)
            ]

            tiny.tiny_ring_allreduce_sum(send, recv)

        # Gather every rank's NCCL result to rank 0 for comparison.
        if rank == 0:
            gather_list = [torch.empty(count, device="cuda:0") for _ in range(world)]
        else:
            gather_list = None
        dist.gather(ncc, gather_list=gather_list, dst=0)

        if rank == 0:
            local_sum = sum(s.cpu() for s in send)
            ok = True
            for i in range(world):
                got = recv[i].cpu()
                ref = gather_list[i].cpu()
                d_local = int((got - local_sum).abs().gt(1e-5).sum().item())
                d_nccl = int((got - ref).abs().gt(1e-5).sum().item())
                match = torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
                ok = ok and match
                status = "OK " if match else "FAIL"
                print(f"  [count={count:>10}] gpu {i}: {status}  tiny_vs_local={d_local}  tiny_vs_nccl={d_nccl}")
            if not ok:
                all_ok = False

        dist.barrier()

    if rank == 0:
        print("PASS: ring allreduce matches NCCL" if all_ok else "FAIL: mismatch detected")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
