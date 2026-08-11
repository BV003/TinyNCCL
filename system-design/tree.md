TinyNCCL/
├── src/                    # C++/CUDA 源码
│   ├── kernels.cu / .h
│   ├── transport.cu / .h
│   ├── ring_allreduce.cu / .h
│   └── bindings.cpp
│
├── tiny_nccl/              # Python 包（被 import 的库）
│   ├── __init__.py         # 暴露 all_reduce, CommContext
│   └── context.py          # CommContext 类
│
├── scripts/                # 可执行工具
│   ├── build.py
│   ├── topology.py
│   └── benchmark.py
│
├── tests/
│   ├── smoke_test.py
│   └── test_vs_torch_distributed.py
│
└── README.md