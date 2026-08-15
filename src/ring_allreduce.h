#ifndef TINYNCCL_RING_ALLREDUCE_H
#define TINYNCCL_RING_ALLREDUCE_H

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

// 单进程多GPU版本（保留用于单进程测试）
void tiny_ring_allreduce_sum_impl(
    float** send, float** recv,
    size_t count, int n_gpus);

// 多进程 IPC 版本：每个进程只管理自己的 GPU
// my_rank: 当前进程 rank
// world_size: 总进程数
// sendbuff: 当前 rank 的输入 buffer（device pointer）
// recvbuff: 当前 rank 的输出 buffer（device pointer）
// count: 每个 rank 的元素数量
// ipc_dir: 用于交换 IPC handle 的共享目录
void tiny_ring_allreduce_sum_ipc(
    int my_rank, int world_size,
    float* sendbuff, float* recvbuff,
    size_t count,
    const std::string& ipc_dir);

#endif
