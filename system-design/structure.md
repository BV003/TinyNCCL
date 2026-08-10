## 项目文件结构
```
tiny‑nccl/
├── src/
│   ├── ring_allreduce.cu      # Ring 算法：send/recv + reduce 计算逻辑
│   ├── tree_allreduce.cu      # 可选：Tree 算法实现
│   ├── topology.py            # PCIe / NVLink 硬件拓扑检测
│   └── benchmark.py           # 性能测试：自研实现 vs NCCL 带宽对比
├── tests/
│   └── test_vs_torch_distributed.py  # 正确性测试，对标 PyTorch distributed
└── README.md
```