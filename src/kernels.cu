#include "kernels.h"

#include <array>
#include <stdexcept>

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

namespace {
constexpr int kMaxDevices = 8;

std::array<std::array<bool, kMaxDevices>, kMaxDevices> g_peer_enabled{};
void ensure_peer_access(int a, int b) {
    if (a == b) return;
    if (a >= kMaxDevices || b >= kMaxDevices)
        throw std::runtime_error("device index exceeds kMaxDevices");
    if (g_peer_enabled[a][b]) return;
    cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceEnablePeerAccess failed: ") +
            cudaGetErrorString(err));
    }
    g_peer_enabled[a][b] = true;
    }
}  
void tiny_copy_peer(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream)
{
    ensure_peer_access(dst_dev, src_dev);
    cudaError_t err = cudaMemcpyPeerAsync(
        dst, dst_dev, src, src_dev, nbytes, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyPeerAsync failed: ") +
            cudaGetErrorString(err));
    }
}