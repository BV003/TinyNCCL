1. Phase 0 — Environment Verification
Follow docs/cloud-setup.md Phases 0 checklist — verify your 2× A5000 setup:
```
nvidia-smi                     # both GPUs visible?
nvidia-smi topo -m             # record topology
python -c "import torch; print(torch.cuda.device_count())"  # should print 2
python scripts/topology.py     # already written — run it to check P2P matrix
torchrun --nproc_per_node=2 smoke_test.py   # baseline NCCL all_reduce
```

2. After Phase 0 is green, start on the core: src/ring_allreduce.cu
Your build script (scripts/build.py) already expects this file — it's the heart of the project and everything else layers on top of it. In order within this file:
Step	What to build
- [x] 2a	[reduce_kernel](Layer 4) — element-wise sum/max/min CUDA kernel on float32 tensors


- [x] 2b	[copy_kernel](Layer 4) — peer-to-peer data movement kernel (cudaMemcpyPeerAsync)

- [x] 2e	Transport selection (Layer 5) — GPUDirect P2P vs host-staging fallback based on topology.py P2P results

- [x] 2c	Ring AllReduce algorithm (Layer 3) — ReduceScatter (N-1 steps) + AllGather (N-1 steps), orchestrating send/recv + reduce on chunks (serial/synchronous version)


- [x] 2d	CUDA streams + events for async pipeline
让通信和计算同步进行




3. tests/test_vs_torch_distributed.py (correctness)
Once ring_allreduce.cu compiles, write this next — validate against torch.distributed.all_reduce on random tensors.


4. src/benchmark.py (performance)
Bandwidth comparison: your implementation vs NCCL.



5. Optional: src/tree_allreduce.cu + algorithm selector (if you want both)