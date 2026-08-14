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

            # 先校验：CPU求和 和 gather过来的NCCL参考本身是否一致
            ref0 = gather_list[0].cpu()
            ref_err = int((local_sum - ref0).abs().gt(1e-5).sum().item())
            print(f"  [count={count:>10}] CHECK_REF: cpu_sum vs nccl_ref errors = {ref_err}")
            if ref_err > 0:
                print("  !!! WARNING: CPU sum and NCCL reference themselves differ! seed/gather problem!")

            for i in range(world):
                got = recv[i].cpu()
                ref = gather_list[i].cpu()

                # tiny输出 vs CPU直接求和
                err_mask_local = (got - local_sum).abs() > 1e-5
                d_local = int(err_mask_local.sum().item())
                bad_idx_local = torch.nonzero(err_mask_local).squeeze(-1)

                # tiny输出 vs NCCL参考
                err_mask_nccl = (got - ref).abs() > 1e-5
                d_nccl = int(err_mask_nccl.sum().item())
                bad_idx_nccl = torch.nonzero(err_mask_nccl).squeeze(-1)

                match_local = (d_local == 0)
                match_nccl = (d_nccl == 0)
                match = torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
                ok = ok and match

                status = "OK " if match else "FAIL"
                print(f"  [count={count:>10}] gpu {i}: {status}")
                print(f"      tiny_vs_cpu_sum: err_cnt={d_local}, match={match_local}")
                print(f"      tiny_vs_nccl_ref: err_cnt={d_nccl}, match={match_nccl}")

                if d_local > 0:
                    show_idx = bad_idx_local[:10].tolist()
                    print(f"      -> bad indices (vs cpu sum): {show_idx}{' ...' if len(bad_idx_local)>10 else ''}")

                    # 打印：tiny输出、CPU真值、NCCL真值、差值，只打印出错下标
                    print("      ===== DUMP BAD REGION =====")
                    for idx in show_idx:
                        v_tiny = float(got[idx])
                        v_cpu = float(local_sum[idx])
                        v_nccl = float(ref[idx])
                        delta = v_tiny - v_cpu
                        print(f"      idx={idx:2d} | tiny={v_tiny:8.4f} | cpu_sum={v_cpu:8.4f} | nccl_ref={v_nccl:8.4f} | delta={delta:8.4f}")
                    print("      ===========================")

            if not ok:
                all_ok = False

        dist.barrier()
        
    if rank == 0:
        print("PASS: ring allreduce matches NCCL" if all_ok else "FAIL: mismatch detected")
    dist.destroy_process_group()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
