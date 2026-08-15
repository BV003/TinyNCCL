from __future__ import annotations

import torch
import tiny


def check_copy(src_device: int, dst_device: int, count: int) -> None:
    src = torch.arange(count, dtype=torch.float32, device=f"cuda:{src_device}")
    dst = torch.empty(count, dtype=torch.float32, device=f"cuda:{dst_device}")

    tiny.tiny_copy_peer(dst, src)
    torch.cuda.synchronize(dst_device)

    assert torch.equal(dst.cpu(), src.cpu()), (
        f"copy mismatch: cuda:{src_device} -> cuda:{dst_device}, count={count}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("skip: CUDA is unavailable")
        return
    if torch.cuda.device_count() < 2:
        print("skip: need at least 2 GPUs")
        return

    for count in (1, 17, 1 << 20):
        check_copy(0, 1, count)
        check_copy(1, 0, count)
        check_copy(0, 0, count)

    print("Transport tests passed")


if __name__ == "__main__":
    main()
