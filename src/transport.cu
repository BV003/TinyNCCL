#include "transport.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
constexpr int kMaxDevices = 8;

// P2P能力矩阵: g_p2p_supported[src][dst]  src->dst拷贝是否支持P2P
// 0=未探测,1=支持P2P,2=回退host拷贝
std::array<std::array<int, kMaxDevices>, kMaxDevices> g_p2p_supported{};
// host‑staging中转页锁内存，每个dst_dev独立一块
std::array<void*, kMaxDevices> g_staging{};
std::array<size_t, kMaxDevices> g_staging_size{};

void check_dev_index(int a, int b) {
    if (a >= kMaxDevices || b >= kMaxDevices)
        throw std::runtime_error("device index exceeds kMaxDevices");
}

bool p2p_available(int src_dev, int dst_dev) {
    if (src_dev == dst_dev) {
        fprintf(stderr, "[DBG p2p_available] src==dst %d, skip\n", src_dev);
        return true;
    }
    check_dev_index(src_dev, dst_dev);

    if (g_p2p_supported[src_dev][dst_dev] == 0) {
        int old_dev;
        cudaGetDevice(&old_dev);
        fprintf(stderr, "[DBG p2p_available] first probe %d->%d, old_dev=%d\n", src_dev, dst_dev, old_dev);

        cudaSetDevice(dst_dev);
        int cur_after_set;
        cudaGetDevice(&cur_after_set);
        fprintf(stderr, "[DBG p2p_available] cudaSetDevice(%d), current_dev=%d\n", dst_dev, cur_after_set);

        cudaError_t err = cudaDeviceEnablePeerAccess(src_dev, 0);
        bool ok = (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled);
        cudaGetLastError();
        g_p2p_supported[src_dev][dst_dev] = ok ? 1 : 2;
        fprintf(stderr, "[transport] P2P %d -> %d: %s\n",
                src_dev, dst_dev,
                ok ? "GPUDirect P2P" : "host‑staging fallback");

        cudaSetDevice(old_dev);
        int cur_restore;
        cudaGetDevice(&cur_restore);
        fprintf(stderr, "[DBG p2p_available] restore old_dev=%d, current_dev=%d\n", old_dev, cur_restore);
    } else {
        // 缓存命中，不再做设备切换
        fprintf(stderr, "[DBG p2p_available] cache hit %d->%d, status=%d\n",
                src_dev, dst_dev, g_p2p_supported[src_dev][dst_dev]);
    }
    return g_p2p_supported[src_dev][dst_dev] == 1;
}

void* get_staging(int dst_dev, size_t nbytes) {
    if (nbytes <= g_staging_size[dst_dev]) {
        fprintf(stderr, "[DBG get_staging] reuse buffer dev=%d size=%zu\n", dst_dev, nbytes);
        return g_staging[dst_dev];
    }
    if (g_staging[dst_dev]) cudaFreeHost(g_staging[dst_dev]);
    cudaError_t err = cudaHostAlloc(&g_staging[dst_dev], nbytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        g_staging[dst_dev] = nullptr;
        g_staging_size[dst_dev] = 0;
        throw std::runtime_error(
            std::string("cudaHostAlloc failed: ") + cudaGetErrorString(err));
    }
    g_staging_size[dst_dev] = nbytes;
    fprintf(stderr, "[DBG get_staging] alloc new host staging dev=%d size=%zu\n", dst_dev, nbytes);
    return g_staging[dst_dev];
}

void copy_via_host(
    void* dst, int dst_dev, const void* src, size_t nbytes, cudaStream_t stream)
{
    fprintf(stderr, "[DBG copy_via_host] enter dst_dev=%d nbytes=%zu\n", dst_dev, nbytes);
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
    fprintf(stderr, "[DBG copy_via_host] submit done\n");
}

}  // namespace

void transport_copy(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream)
{
    int old_dev;
    cudaGetDevice(&old_dev);
    fprintf(stderr, "\n[DBG transport_copy] ENTER old_dev=%d | dst_dev=%d src_dev=%d nbytes=%zu\n",
            old_dev, dst_dev, src_dev, nbytes);

    if (src_dev == dst_dev) {
        fprintf(stderr, "[DBG transport_copy] same‑device copy path\n");
        cudaSetDevice(dst_dev);
        int cur;
        cudaGetDevice(&cur);
        fprintf(stderr, "[DBG transport_copy] same dev set current=%d\n", cur);

        cudaError_t err = cudaMemcpyAsync(
            dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            cudaSetDevice(old_dev);
            throw std::runtime_error(
                std::string("cudaMemcpyAsync (D2D) failed: ") + cudaGetErrorString(err));
        }
    } else {
        if (p2p_available(src_dev, dst_dev)) {
            fprintf(stderr, "[DBG transport_copy] P2P path selected\n");
            cudaSetDevice(dst_dev);
            int cur;
            cudaGetDevice(&cur);
            // 重点！这里打印提交cudaMemcpyPeerAsync那一刻的当前CUDA设备
            fprintf(stderr, "[DBG transport_copy] BEFORE cudaMemcpyPeerAsync current_dev=%d expected_dst_dev=%d\n", cur, dst_dev);

            cudaError_t err = cudaMemcpyPeerAsync(
                dst, dst_dev, src, src_dev, nbytes, stream);
            if (err != cudaSuccess) {
                cudaSetDevice(old_dev);
                throw std::runtime_error(
                    std::string("cudaMemcpyPeerAsync failed: ") + cudaGetErrorString(err));
            }
            fprintf(stderr, "[DBG transport_copy] cudaMemcpyPeerAsync submitted, err=%d\n", err);
        } else {
            fprintf(stderr, "[DBG transport_copy] host‑staging path selected\n");
            cudaSetDevice(dst_dev);
            copy_via_host(dst, dst_dev, src, nbytes, stream);
        }
    }

    // 恢复调用方原始设备
    cudaSetDevice(old_dev);
    int cur_final;
    cudaGetDevice(&cur_final);
    fprintf(stderr, "[DBG transport_copy] EXIT restore to old_dev=%d, current_dev=%d\n", old_dev, cur_final);
}

void transport_copy_ipc(
    void* dst, const void* src,
    size_t nbytes, cudaStream_t stream)
{
    int current_device = -1;
    cudaError_t err = cudaGetDevice(&current_device);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice (IPC copy) failed: ") +
                                 cudaGetErrorString(err));
    }

    cudaPointerAttributes dst_attr{};
    cudaPointerAttributes src_attr{};
    cudaError_t dst_attr_err = cudaPointerGetAttributes(&dst_attr, dst);
    cudaError_t src_attr_err = cudaPointerGetAttributes(&src_attr, src);
    fprintf(stderr,
            "[IPC COPY] current_dev=%d dst=%p dst_dev=%d dst_attr_err=%s "
            "src=%p src_dev=%d src_attr_err=%s nbytes=%zu\n",
            current_device, dst,
            dst_attr_err == cudaSuccess ? dst_attr.device : -1,
            cudaGetErrorString(dst_attr_err), src,
            src_attr_err == cudaSuccess ? src_attr.device : -1,
            cudaGetErrorString(src_attr_err), nbytes);

    // IPC 映射后的 pointer 已经属于当前进程地址空间；这里使用
    // cudaMemcpyDefault，让 CUDA 根据 UVA pointer attributes 判断方向。
    err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDefault, stream);
    fprintf(stderr, "[IPC COPY] submitted err=%s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpyAsync (IPC copy) failed: ") +
            cudaGetErrorString(err));
    }
}
