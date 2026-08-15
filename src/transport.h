#ifndef TINYNCCL_TRANSPORT_H
#define TINYNCCL_TRANSPORT_H

#include <cuda_runtime.h>
#include <cstddef>

void transport_copy(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream);

// 跨进程拷贝：dst 和 src 是通过 IPC 映射的本地 pointer
// 两个 pointer 都在当前进程的地址空间中有效
void transport_copy_ipc(
    void* dst, const void* src,
    size_t nbytes, cudaStream_t stream);

#endif
