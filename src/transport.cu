#include "transport.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
constexpr int kMaxDevices = 8;

std::array<std::array<int, kMaxDevices>, kMaxDevices> g_p2p_supported{};
std::array<void*, kMaxDevices> g_staging{};
std::array<size_t, kMaxDevices> g_staging_size{};

void check_dev_index(int a, int b) {
    if (a >= kMaxDevices || b >= kMaxDevices)
        throw std::runtime_error("device index exceeds kMaxDevices");
}

bool p2p_available(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) return true;
    check_dev_index(src_dev, dst_dev);
    if (g_p2p_supported[src_dev][dst_dev] == 0) {
        int old_dev;
        cudaGetDevice(&old_dev);

        cudaSetDevice(dst_dev);
        cudaError_t err = cudaDeviceEnablePeerAccess(src_dev, 0);
        bool ok = (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
        cudaGetLastError();
        g_p2p_supported[src_dev][dst_dev] = ok ? 1 : 2;
        fprintf(stderr, "[transport] P2P %d -> %d: %s\n",
                src_dev, dst_dev,
                ok ? "GPUDirect P2P" : "host-staging fallback");

        cudaSetDevice(old_dev);
    }
    return g_p2p_supported[src_dev][dst_dev] == 1;
}

void* get_staging(int dst_dev, size_t nbytes) {
    if (nbytes <= g_staging_size[dst_dev]) return g_staging[dst_dev];
    if (g_staging[dst_dev]) cudaFreeHost(g_staging[dst_dev]);
    cudaError_t err = cudaHostAlloc(&g_staging[dst_dev], nbytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        g_staging[dst_dev] = nullptr;
        g_staging_size[dst_dev] = 0;
        throw std::runtime_error(
            std::string("cudaHostAlloc failed: ") + cudaGetErrorString(err));
    }
    g_staging_size[dst_dev] = nbytes;
    return g_staging[dst_dev];
}

void copy_via_host(
    void* dst, int dst_dev, const void* src, size_t nbytes, cudaStream_t stream)
{
    void* host = get_staging(dst_dev, nbytes);
    cudaError_t err = cudaMemcpyAsync(
        host, src, nbytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyAsync (D2H) failed: ") +
            cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(dst, host, nbytes, cudaMemcpyDefault, stream);
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
    int old_dev;
    cudaGetDevice(&old_dev);

    if (src_dev == dst_dev) {
        cudaSetDevice(dst_dev);
        cudaError_t err = cudaMemcpyAsync(
            dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            cudaSetDevice(old_dev);
            throw std::runtime_error(
                std::string("cudaMemcpyAsync (D2D) failed: ") +
                cudaGetErrorString(err));
        }
    } else {
        if (p2p_available(src_dev, dst_dev)) {
            cudaSetDevice(dst_dev);
            cudaError_t err = cudaMemcpyPeerAsync(
                dst, dst_dev, src, src_dev, nbytes, stream);
            if (err != cudaSuccess) {
                cudaSetDevice(old_dev);
                throw std::runtime_error(
                    std::string("cudaMemcpyPeerAsync failed: ") +
                    cudaGetErrorString(err));
            }
        } else {
            cudaSetDevice(dst_dev);
            copy_via_host(dst, dst_dev, src, nbytes, stream);
        }
    }

    cudaSetDevice(old_dev);
}