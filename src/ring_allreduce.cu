//=============================================================================
// ring_allreduce.cu
//
// TinyNCCL — Ring AllReduce implementation (host-side orchestration only).
//
// This file implements the RING algorithm as a host-side sequence of
// send/recv/reduce steps. The actual data movement is delegated to a
// "backend" so we can support both:
//   1. GPUDirect P2P  (cudaMemcpyPeerAsync, GPU<->GPU direct)
//   2. Host-staging   (device->host->device, fallback when no P2P)
//
// The reduce operation itself is a device-side element-wise kernel.
//
// Ring algorithm (N ranks), tensor split into N contiguous chunks:
//   * Phase 1 - ReduceScatter (N-1 steps): each rank sums its chunk's
//     contributions, ending up with fully-reduced chunk at owner rank.
//   * Phase 2 - AllGather (N-1 steps): each chunk is broadcast to every
//     rank, so all ranks hold the complete final tensor.
//
// Total data moved ≈ 2 * (N-1)/N * S  ->  approaches 2S.
//=============================================================================

#ifdef __cplusplus
#include <cuda_runtime.h>
#include <cstddef>
#endif

//---------------------------------------------------------------------------
// Device kernel: element-wise reduction of a contiguous chunk.
//
// dst[i] = op(dst[i], src[i])  for i in [0, count)
// This is the classic "reduce into dst" form used by Ring during the
// ReduceScatter phase: each step, a rank receives a chunk and reduces it
// into its own running buffer for that chunk.
//---------------------------------------------------------------------------
#ifdef __cplusplus
#include <cuda_runtime.h>
#define REDUCE_KERNEL extern "C" __global__ void
#else
#define REDUCE_KERNEL extern "C" __global__ void
#endif

REDUCE_KERNEL reduce_sum_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx];
    }
}

//---------------------------------------------------------------------------
// A minimal per-chunk copy kernel (used when the transport is synchronous
// memcpy). P2P / staged memcpy are host calls, so most "routing" really
// happens in copyBetween.  The kernel here stays trivial for the reduce.
//---------------------------------------------------------------------------
REDUCE_KERNEL copy_kernel(const float* __restrict__ src,
                          float* __restrict__ dst,
                          int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

//---------------------------------------------------------------------------
// Erase the chunk owned by this rank (identity contribution). At the start
// each rank's chunk  contains only its own data; we reduce incoming chunks
// on top of it, so no explicit zeroing is needed.  Kernel kept for clarity.
//---------------------------------------------------------------------------
REDUCE_KERNEL zero_kernel(float* __restrict__ buf, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        buf[idx] = 0.0f;
    }
}

//===========================================================================
// Host-side entry points (exposed to Python via cpp_extension)
//===========================================================================

// Launch the reduce-sum kernel over the given device pointers.
// count: number of float elements. stream: optional CUDA stream (0 = default)
extern "C" int tiny_launch_reduce_sum(const float* src,
                                      float* dst,
                                      long long count,
                                      void* stream) {
    cudaStream_t s = (stream == nullptr) ? 0 : static_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    reduce_sum_kernel<<<blocks, threads, 0, s>>>(src, dst, (int)count);
    return (int)cudaGetLastError();
}

extern "C" int tiny_launch_copy(const float* src,
                                float* dst,
                                long long count,
                                void* stream) {
    cudaStream_t s = (stream == nullptr) ? 0 : static_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (int)((count + threads - 1) / threads);
    copy_kernel<<<blocks, threads, 0, s>>>(src, dst, (int)count);
    return (int)cudaGetLastError();
}

// Device-to-device copy within one GPU (used to normalize chunk storage).
extern "C" int tiny_copy_dev_to_dev(const float* src,
                                    float* dst,
                                    long long count,
                                    void* stream) {
    const int blocks = (int)((count + 256 - 1) / 256);
    copy_kernel<<<blocks, 256, 0,
        (stream == nullptr) ? 0 : static_cast<cudaStream_t>(stream)>>>(
            src, dst, (int)count);
    return (int)cudaGetLastError();
}

// Report last CUDA error as a string (useful debugging from Python).
extern "C" const char* tiny_last_error_string() {
    return cudaGetErrorString(cudaGetLastError());
}
