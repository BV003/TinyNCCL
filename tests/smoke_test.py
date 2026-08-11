import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
t = torch.full((4,), rank + 1, device="cuda")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expected = sum(range(1, dist.get_world_size() + 1))
assert t.cpu().tolist() == [expected] * 4, t
print(f"rank {rank}: NCCL all_reduce OK -> {t.cpu().tolist()}")
dist.destroy_process_group()