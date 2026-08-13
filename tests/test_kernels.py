import torch
import tiny

def test_reduce_sum_scalar():
    a = torch.tensor([1.0], device="cuda")
    b = torch.tensor([2.0], device="cuda")
    dst = torch.empty(1, device="cuda")
    tiny.tiny_reduce_sum(dst, a, b)
    assert dst[0].item() == 3.0

def test_reduce_sum_small():
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    dst = torch.empty(3, device="cuda")
    tiny.tiny_reduce_sum(dst, a, b)
    assert torch.allclose(dst, torch.tensor([5.0, 7.0, 9.0], device="cuda"))

def test_reduce_sum_large():
    n = 1 << 20  # 1M elements
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    dst = torch.empty(n, device="cuda")
    tiny.tiny_reduce_sum(dst, a, b)
    assert torch.allclose(dst, a + b)

def test_reduce_sum_2d():
    a = torch.randn(32, 64, device="cuda")
    b = torch.randn(32, 64, device="cuda")
    dst = torch.empty(32, 64, device="cuda")
    tiny.tiny_reduce_sum(dst, a, b)
    assert torch.allclose(dst, a + b)

def test_copy_peer():
    if torch.cuda.device_count() < 2:
        print("skip: need 2 GPUs")
        return
    n = 1 << 20
    # cuda:0 -> cuda:1
    src = torch.randn(n, device="cuda:0")
    dst = torch.empty_like(src, device="cuda:1")
    tiny.tiny_copy_peer(dst, src)
    torch.cuda.synchronize()
    assert torch.allclose(dst.cpu(), src.cpu())

    # cuda:1 -> cuda:0
    src = torch.randn(n, device="cuda:1")
    dst = torch.empty_like(src, device="cuda:0")
    tiny.tiny_copy_peer(dst, src)
    torch.cuda.synchronize()
    assert torch.allclose(dst.cpu(), src.cpu())

    # same device
    src = torch.randn(n, device="cuda:0")
    dst = torch.empty_like(src, device="cuda:0")
    tiny.tiny_copy_peer(dst, src)
    torch.cuda.synchronize()
    assert torch.allclose(dst, src)
    
if __name__ == "__main__":
    test_reduce_sum_scalar()
    test_reduce_sum_small()
    test_reduce_sum_large()
    test_reduce_sum_2d()
    test_copy_peer()
    print("All tests passed!")
