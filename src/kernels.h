#ifndef TINYNCCL_KERNELS_H
#define TINYNCCL_KERNELS_H

#include <cuda_runtime.h>
#include <cstddef> 

void tiny_reduce_sum_kernel(
    float* dst, const float* a, const float* b,
    int n, cudaStream_t stream);

void tiny_copy_peer_kernel(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream);
    
#endif