import torch
from build import tiny

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
    
if __name__ == "__main__":
    test_reduce_sum_scalar()
    test_reduce_sum_small()
    test_reduce_sum_large()
    test_reduce_sum_2d()
    print("All tests passed!")