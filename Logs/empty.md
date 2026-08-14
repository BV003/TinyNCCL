No empty

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 tests/test_vs_torch_distributed.py 2>&1 | tee Logs/err.log