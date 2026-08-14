## Phase 1
Environment Verification
Follow docs/cloud-setup.md Phases 0 checklist 
```
nvidia-smi                    
nvidia-smi topo -m            
python -c "import torch; print(torch.cuda.device_count())"  
python scripts/topology.py    
torchrun --nproc_per_node=2 smoke_test.py  
```

## Phase 2


- [x] 2a	[reduce_kernel](Layer 4) — element-wise sum/max/min CUDA kernel on float32 tensors


- [x] 2b	[copy_kernel](Layer 5) — peer-to-peer data movement kernel (cudaMemcpyPeerAsync)

- [x] 2c	Transport selection (Layer 5) — GPUDirect P2P vs host-staging fallback based on topology.py P2P results

- [x] 2d	Ring AllReduce algorithm (Layer 3) — ReduceScatter (N-1 steps) + AllGather (N-1 steps), orchestrating send/recv + reduce on chunks (serial/synchronous version)





## Phase 3
Correctness

validate against torch.distributed.all_reduce on random tensors.


## Phase 4
Performance

Bandwidth comparison: your implementation vs NCCL.



## Phase 5
src/tree_allreduce.cu + algorithm selector (if you want both)


## TODO IN THE Future

- [ ] CUDA streams + events for async pipeline, 让通信和计算同步进行
（未完成，目前由于设备限制，是每轮step后同步一次）
- [ ] 4台设备进行之间的通信