#include "kernels.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

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

// peer access enabled once per ordered pair (a -> b)
std::array<std::array<bool, kMaxDevices>, kMaxDevices> g_peer_enabled{};

// P2P availability matrix: 0 = unknown, 1 = supported, 2 = unsupported
std::array<std::array<int, kMaxDevices>, kMaxDevices> g_p2p_supported{};

// cached pinned host buffer used by the host-staging fallback
void* g_staging = nullptr;
size_t g_staging_size = 0;

void check_dev_index(int a, int b) {
    if (a >= kMaxDevices || b >= kMaxDevices)
        throw std::runtime_error("device index exceeds kMaxDevices");
}

void ensure_peer_access(int a, int b) {
    if (a == b) return;
    check_dev_index(a, b);
    if (g_peer_enabled[a][b]) return;
    cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaDeviceEnablePeerAccess failed: ") +
            cudaGetErrorString(err));
    }
    g_peer_enabled[a][b] = true;
}

bool supports_p2p(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) return true;
    check_dev_index(src_dev, dst_dev);
    if (g_p2p_supported[src_dev][dst_dev] == 0) {
        int can = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&can, src_dev, dst_dev);
        g_p2p_supported[src_dev][dst_dev] =
            (err == cudaSuccess && can) ? 1 : 2;
    }
    return g_p2p_supported[src_dev][dst_dev] == 1;
}

void* get_staging(size_t nbytes) {
    if (nbytes <= g_staging_size) return g_staging;
    if (g_staging) cudaFreeHost(g_staging);
    cudaError_t err = cudaHostAlloc(&g_staging, nbytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        g_staging = nullptr;
        g_staging_size = 0;
        throw std::runtime_error(
            std::string("cudaHostAlloc failed: ") + cudaGetErrorString(err));
    }
    g_staging_size = nbytes;
    return g_staging;
}

void copy_via_host(
    void* dst, const void* src, size_t nbytes, cudaStream_t stream)
{
    void* host = get_staging(nbytes);
    cudaError_t err = cudaMemcpyAsync(
        host, src, nbytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyAsync (D2H) failed: ") +
            cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(dst, host, nbytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyAsync (H2D) failed: ") +
            cudaGetErrorString(err));
    }
}

}  // namespace

void tiny_copy_peer_kernel(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream)
{
    if (src_dev == dst_dev) {
        cudaError_t err = cudaMemcpyAsync(
            dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpyAsync (D2D) failed: ") +
                cudaGetErrorString(err));
        }
        return;
    }

    if (supports_p2p(src_dev, dst_dev)) {
        ensure_peer_access(dst_dev, src_dev);
        cudaError_t err = cudaMemcpyPeerAsync(
            dst, dst_dev, src, src_dev, nbytes, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemcpyPeerAsync failed: ") +
                cudaGetErrorString(err));
        }
    } else {
        copy_via_host(dst, src, nbytes, stream);
    }
}