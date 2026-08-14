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

    # 只在rank0分配send/recv；recv只分配1次！全程复用显存
    recv = None
    if rank == 0:
        recv = [
            torch.empty(count, device=f"cuda:{i}")
            for i in range(world)
        ]

    def run_one_round(round_name: str, gpu0_start: float, gpu1_start: float):
        nonlocal all_ok
        # NCCL参考
        t_dummy = torch.empty(count, device=f"cuda:{rank}")
        ncc = t_dummy.clone()
        dist.all_reduce(ncc, op=dist.ReduceOp.SUM)
        dist.barrier()

        if rank == 0:
            send = []
            gpu0_arr = torch.arange(gpu0_start, gpu0_start + count, dtype=torch.float32, device="cuda:0")
            gpu1_arr = torch.arange(gpu1_start, gpu1_start + count, dtype=torch.float32, device="cuda:1")
            send.append(gpu0_arr)
            send.append(gpu1_arr)

            # 调用allreduce，recv复用旧显存！
            tiny.tiny_ring_allreduce_sum(send, recv)

        # gather参考值
        if rank == 0:
            gather_list = [torch.empty(count, device="cuda:0") for _ in range(world)]
        else:
            gather_list = None
        dist.gather(ncc, gather_list=gather_list, dst=0)

        if rank == 0:
            print(f"\n==== {round_name} count={count} ====")
            print(f"GPU0 input start={gpu0_start}, GPU1 input start={gpu1_start}")
            local_sum = send[0].cpu() + send[1].cpu()
            print(f"Expected cpu ground truth: {local_sum}\n")

            ref0 = gather_list[0].cpu()
            ref_err = int((local_sum - ref0).abs().gt(1e-5).sum().item())
            print(f"CHECK_REF: cpu_sum vs nccl_ref errors = {ref_err}")

            for i in range(world):
                got = recv[i].cpu()
                ref = local_sum
                err_mask_local = (got - ref).abs() > 1e-5
                d_local = int(err_mask_local.sum().item())
                bad_idx_local = torch.nonzero(err_mask_local).squeeze(-1)
                match_local = (d_local == 0)
                status = "OK " if match_local else "FAIL"
                print(f"gpu {i}: {status} err_cnt={d_local}")

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

    # -------- Round1 --------
    run_one_round("ROUND 1: fixed 0~15 / 100~115", gpu0_start=0.0, gpu1_start=100.0)

    # -------- Round2：复用recv底层显存，完全不重建recv列表 --------
    run_one_round("ROUND 2: fixed 1000~1015 / 2000~2015", gpu0_start=1000.0, gpu1_start=2000.0)

    if rank == 0:
        print("\n" + ("[FINAL] PASS" if all_ok else "[FINAL] FAIL: mismatch detected"))

    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())