#ifndef TINYNCCL_KERNELS_H
#define TINYNCCL_KERNELS_H

#include <cuda_runtime.h>

void tiny_reduce_sum_kernel(
    float* dst, const float* a, const float* b,
    int n, cudaStream_t stream);

#endif
