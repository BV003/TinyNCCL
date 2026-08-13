#include "kernels.h"

__global__ void reduce_sum_kernel(
    float* __restrict__ dst,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = a[idx] + b[idx];
    }
}

void tiny_reduce_sum_kernel(
    float* dst, const float* a, const float* b,
    int n, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    reduce_sum_kernel<<<blocks, threads, 0, stream>>>(dst, a, b, n);
}
