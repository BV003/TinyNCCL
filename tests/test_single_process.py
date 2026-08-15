from __future__ import annotations

import torch
import tiny


def run_case(count: int, n_gpus: int) -> None:
    torch.manual_seed(2026 + count)
    send = [torch.randn(count, device=f"cuda:{i}") for i in range(n_gpus)]
    recv = [torch.empty_like(send[i]) for i in range(n_gpus)]

    expected = sum(value.cpu() for value in send)
    tiny.tiny_ring_allreduce_sum(send, recv)

    for device, value in enumerate(recv):
        assert torch.allclose(value.cpu(), expected, atol=1e-5, rtol=1e-5), (
            f"Ring AllReduce mismatch on GPU {device}, count={count}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("skip: CUDA is unavailable")
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print("skip: need at least 2 GPUs")
        return

    counts = [n_gpus, n_gpus * 4, n_gpus * 1024]
    for count in counts:
        for iteration in range(3):
            run_case(count, n_gpus)
            print(f"single-process count={count} iteration={iteration}: OK")

    print("Single-process Ring AllReduce tests passed")


if __name__ == "__main__":
    main()
