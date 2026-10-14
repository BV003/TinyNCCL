#include "transport.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "cuda_check.h"

namespace {

// Number of devices tracked in the P2P / staging caches below. The ring
// collective itself supports any world size; this only bounds the caches.
constexpr int kMaxTrackedDevices = 8;

enum class P2pState : int { kUnknown = 0, kSupported = 1, kUnsupported = 2 };

// These caches are not thread-safe: the transport is called from a single
// collective thread (one copy at a time), so they only exist to avoid repeated
// peer probing and re-allocating the host staging buffer.
std::array<std::array<P2pState, kMaxTrackedDevices>, kMaxTrackedDevices>
    g_p2p_state{};
std::array<void*, kMaxTrackedDevices> g_staging{};
std::array<size_t, kMaxTrackedDevices> g_staging_size{};

// Save the current device and restore it on scope exit, so an exception in the
// middle of a copy no longer leaves the caller on the wrong device.
class DeviceGuard {
public:
    DeviceGuard() { check_cuda(cudaGetDevice(&saved_), "cudaGetDevice"); }
    ~DeviceGuard() { cudaSetDevice(saved_); }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int saved_ = -1;
};

void check_dev_index(int src_dev, int dst_dev) {
    if (src_dev < 0 || dst_dev < 0 || src_dev >= kMaxTrackedDevices ||
        dst_dev >= kMaxTrackedDevices) {
        throw std::runtime_error("device index exceeds kMaxTrackedDevices");
    }
}

bool p2p_available(int src_dev, int dst_dev) {
    if (src_dev == dst_dev)
        return true;
    check_dev_index(src_dev, dst_dev);

    if (g_p2p_state[src_dev][dst_dev] == P2pState::kUnknown) {
        DeviceGuard guard;
        check_cuda(cudaSetDevice(dst_dev), "cudaSetDevice");

        const cudaError_t err = cudaDeviceEnablePeerAccess(src_dev, 0);
        const bool supported =
            err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled;
        cudaGetLastError();
        g_p2p_state[src_dev][dst_dev] =
            supported ? P2pState::kSupported : P2pState::kUnsupported;
    }
    return g_p2p_state[src_dev][dst_dev] == P2pState::kSupported;
}

void* get_staging(int dst_dev, size_t nbytes) {
    if (nbytes <= g_staging_size[dst_dev])
        return g_staging[dst_dev];

    if (g_staging[dst_dev])
        cudaFreeHost(g_staging[dst_dev]);
    const cudaError_t err =
        cudaHostAlloc(&g_staging[dst_dev], nbytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        g_staging[dst_dev] = nullptr;
        g_staging_size[dst_dev] = 0;
        throw std::runtime_error(std::string("cudaHostAlloc failed: ") +
                                 cudaGetErrorString(err));
    }
    g_staging_size[dst_dev] = nbytes;
    return g_staging[dst_dev];
}

void copy_via_host(void* dst, int dst_dev, const void* src, size_t nbytes,
                   cudaStream_t stream) {
    void* host = get_staging(dst_dev, nbytes);
    check_cuda(cudaMemcpyAsync(host, src, nbytes, cudaMemcpyDefault, stream),
               "cudaMemcpyAsync D2H failed");
    check_cuda(cudaMemcpyAsync(dst, host, nbytes, cudaMemcpyDefault, stream),
               "cudaMemcpyAsync H2D failed");
}

}  // namespace

void transport_copy(void* dst, int dst_dev, const void* src, int src_dev,
                    size_t nbytes, cudaStream_t stream) {
    DeviceGuard guard;
    check_cuda(cudaSetDevice(dst_dev), "cudaSetDevice");

    cudaError_t err = cudaSuccess;
    if (src_dev == dst_dev) {
        err =
            cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
    } else if (p2p_available(src_dev, dst_dev)) {
        err = cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, nbytes, stream);
    } else {
        copy_via_host(dst, dst_dev, src, nbytes, stream);
    }
    check_cuda(err, "transport copy failed");
}

void transport_copy_ipc(void* dst, int dst_dev, const void* src, int src_dev,
                        size_t nbytes, cudaStream_t stream) {
    DeviceGuard guard;
    check_cuda(cudaSetDevice(dst_dev), "cudaSetDevice IPC copy failed");
    check_cuda(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, nbytes, stream),
               "cudaMemcpyPeerAsync IPC copy failed");
}
