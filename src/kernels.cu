#include "kernels.h"

#include <algorithm>
#include <cstddef>

namespace {

constexpr std::size_t kThreadsPerBlock = 256;
// Cap the grid so very large inputs use a grid-stride loop instead of an
// unbounded block count.
constexpr std::size_t kMaxBlocks = 65536;

}  // namespace

__global__ void reduce_sum_kernel(float* __restrict__ dst,
                                  const float* __restrict__ a,
                                  const float* __restrict__ b, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {
        dst[idx] = a[idx] + b[idx];
    }
}

void tiny_reduce_sum_kernel(float* dst, const float* a, const float* b,
                            std::size_t n, cudaStream_t stream) {
    if (n == 0)
        return;

    const std::size_t needed = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const std::size_t blocks = std::min(needed, kMaxBlocks);
    reduce_sum_kernel<<<static_cast<unsigned int>(blocks),
                        static_cast<unsigned int>(kThreadsPerBlock), 0,
                        stream>>>(dst, a, b, n);
}
