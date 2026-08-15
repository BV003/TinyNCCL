from __future__ import annotations

import os
import sys
import shutil

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

    # 清理上一次运行遗留的 IPC handle，并创建本次运行的共享目录。
    ipc_root = "/tmp/tinynccl_ipc_shared"
    if rank == 0:
        shutil.rmtree(ipc_root, ignore_errors=True)
        os.makedirs(ipc_root, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    counts = [8, 16, 32]
    all_ok = True
    base_seed = 12345

    for count in counts:
        # 每一轮使用独立目录，避免复用上一轮的 CUDA IPC handle。
        ipc_dir = os.path.join(ipc_root, f"count_{count}")
        if rank == 0:
            os.makedirs(ipc_dir, exist_ok=True)
        dist.barrier(device_ids=[local_rank])

        torch.manual_seed(base_seed + rank)

        # 每个 rank 在自己的 GPU 上创建输入
        sendbuff = torch.randn(count, device=f"cuda:{local_rank}")
        recvbuff = torch.empty(count, device=f"cuda:{local_rank}")

        # 参考值：NCCL all_reduce (SUM)
        ncc_ref = sendbuff.clone()
        dist.all_reduce(ncc_ref, op=dist.ReduceOp.SUM)
        dist.barrier(device_ids=[local_rank])

        # 我们的 IPC AllReduce
        tiny.tiny_ring_allreduce_sum_ipc(
            rank, world,
            sendbuff, recvbuff,
            ipc_dir
        )

        dist.barrier(device_ids=[local_rank])

        # 对比结果
        err_mask = (recvbuff - ncc_ref).abs() > 1e-5
        err_cnt = int(err_mask.sum().item())
        match = torch.allclose(recvbuff, ncc_ref, atol=1e-5, rtol=1e-5)

        if not match:
            all_ok = False
            bad_idx = torch.nonzero(err_mask).squeeze(-1)[:5].tolist()
            print(f"[count={count:>5}] rank={rank} FAIL err_cnt={err_cnt} bad_idx={bad_idx}")
            for idx in bad_idx:
                print(f"  idx={idx}: tiny={recvbuff[idx]:.6f} nccl={ncc_ref[idx]:.6f}")
        else:
            print(f"[count={count:>5}] rank={rank} OK")

        dist.barrier(device_ids=[local_rank])

    dist.barrier(device_ids=[local_rank])

    # 清理 IPC 目录
    if rank == 0:
        shutil.rmtree(ipc_root, ignore_errors=True)

    if rank == 0:
        print("PASS: ring allreduce matches NCCL" if all_ok else "FAIL: mismatch detected")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
