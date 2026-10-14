from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()

    tensor = torch.full((4,), rank + 1, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = sum(range(1, world + 1))
    actual = tensor.cpu().tolist()
    if actual != [expected] * 4:
        print(f"rank {rank}: NCCL all_reduce FAILED -> {actual}", file=sys.stderr)
        dist.destroy_process_group()
        return 1

    print(f"rank {rank}: NCCL all_reduce OK -> {actual}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
