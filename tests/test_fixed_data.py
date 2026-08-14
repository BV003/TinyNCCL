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

    count = 16

    all_ok = True

    # ========== 写死固定输入，不随机 ==========
    if rank == 0:
        send = []
        # GPU0 输入：[0,1,2,...,15]
        gpu0_arr = torch.arange(0, count, dtype=torch.float32, device="cuda:0")
        # GPU1 输入：[100,101,...,115]
        gpu1_arr = torch.arange(100, 100+count, dtype=torch.float32, device="cuda:1")
        send.append(gpu0_arr)
        send.append(gpu1_arr)

        recv = [
            torch.empty(count, device=f"cuda:{i}")
            for i in range(world)
        ]
        # 调用原有完整allreduce（RS + AG）
        tiny.tiny_ring_allreduce_sum(send, recv)

    # NCCL参考，每个rank自己的tensor，这里随便造，gather拿参考
    t_dummy = torch.empty(count, device=f"cuda:{rank}")
    ncc = t_dummy.clone()
    dist.all_reduce(ncc, op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0:
        gather_list = [torch.empty(count, device="cuda:0") for _ in range(world)]
    else:
        gather_list = None
    dist.gather(ncc, gather_list=gather_list, dst=0)

    if rank == 0:
        print("\n==== FIXED INPUT TEST count=16 ====")
        print(f"GPU0 input: {send[0].cpu()}")
        print(f"GPU1 input: {send[1].cpu()}")

        # 手工真值：每个元素 = gpu0[x] + gpu1[x]
        local_sum = send[0].cpu() + send[1].cpu()
        print(f"Expected sum (cpu ground truth): {local_sum}\n")

        ref0 = gather_list[0].cpu()
        ref_err = int((local_sum - ref0).abs().gt(1e-5).sum().item())
        print(f"[count={count}] CHECK_REF: cpu_sum vs nccl_ref errors = {ref_err}")

        for i in range(world):
            got = recv[i].cpu()
            ref = local_sum

            err_mask_local = (got - ref).abs() > 1e-5
            d_local = int(err_mask_local.sum().item())
            bad_idx_local = torch.nonzero(err_mask_local).squeeze(-1)

            match_local = (d_local == 0)
            status = "OK " if match_local else "FAIL"
            print(f"[count={count}] gpu {i}: {status} err_cnt={d_local}")

            if d_local > 0:
                show_idx = bad_idx_local[:10].tolist()
                print(f"      -> bad indices: {show_idx}")
                print("      ===== DUMP BAD =====")
                for idx in show_idx:
                    v_tiny = float(got[idx])
                    v_ref = float(ref[idx])
                    delta = v_tiny - v_ref
                    print(f"      idx={idx:2d} | tiny={v_tiny:8.4f} | ref={v_ref:8.4f} | delta={delta:8.4f}")
                print("      ====================\n")

            if not match_local:
                all_ok = False

    dist.barrier()

    if rank == 0:
        print("\n" + ("PASS" if all_ok else "FAIL: mismatch detected"))

    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())