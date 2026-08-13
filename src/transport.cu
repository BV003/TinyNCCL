#include "transport.h"

#include <array>
#include <stdexcept>
#include <string>

namespace {
constexpr int kMaxDevices = 8;

// P2P availability matrix: 0 = unknown, 1 = supported, 2 = unsupported
std::array<std::array<int, kMaxDevices>, kMaxDevices> g_p2p_supported{};
// peer access enabled once per ordered pair (a -> b)
std::array<std::array<bool, kMaxDevices>, kMaxDevices> g_peer_enabled{};

// cached pinned host buffer used by the host-staging fallback
void* g_staging = nullptr;
size_t g_staging_size = 0;

void check_dev_index(int a, int b) {
    if (a >= kMaxDevices || b >= kMaxDevices)
        throw std::runtime_error("device index exceeds kMaxDevices");
}

// Probe once + enable peer access once. Returns true if the P2P path is usable.
bool p2p_available(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) return true;
    check_dev_index(src_dev, dst_dev);
    if (g_p2p_supported[src_dev][dst_dev] == 0) {
        int can = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&can, src_dev, dst_dev);
        if (err == cudaSuccess && can) {
            cudaError_t e2 = cudaDeviceEnablePeerAccess(dst_dev, 0);
            if (e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled) {
                g_p2p_supported[src_dev][dst_dev] = 1;
                g_peer_enabled[src_dev][dst_dev] = true;
            } else {
                g_p2p_supported[src_dev][dst_dev] = 2;
            }
        } else {
            g_p2p_supported[src_dev][dst_dev] = 2;
        }
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

void transport_copy(
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

    if (p2p_available(src_dev, dst_dev)) {
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
