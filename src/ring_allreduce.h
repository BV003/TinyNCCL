#ifndef TINYNCCL_RING_ALLREDUCE_H
#define TINYNCCL_RING_ALLREDUCE_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ipc.h"

// 单进程多GPU版本（保留用于单进程测试）
void tiny_ring_allreduce_sum_impl(float** send, float** recv, size_t count,
                                  int n_gpus);

// 无状态多进程版本：每次调用都重新建立 CUDA IPC 通信。
void tiny_ring_allreduce_sum_ipc(int my_rank, int world_size, float* sendbuff,
                                 float* recvbuff, size_t count,
                                 const std::string& ipc_dir);

// 持久化 communicator：通信建立一次，之后可重复调用 allreduce。
//
// 同步完全走 GPU 侧：每个 rank 有一个单调递增的显存计数器，rank 在 stream
// 上用 cuStreamWriteValue32 递增它、用 cuStreamWaitValue32 等待邻居的计数器。
// 因此 steady-state 每次 allreduce 不涉及文件、事件创建或 CPU 侧往返。
class RingCommunicator {
public:
    // max_count 是每次 allreduce 允许的最大元素数，必须是 world_size 的倍数。
    RingCommunicator(int my_rank, int world_size, size_t max_count,
                     const std::string& ipc_dir);
    ~RingCommunicator();

    RingCommunicator(const RingCommunicator&) = delete;
    RingCommunicator& operator=(const RingCommunicator&) = delete;

    // 对 count 个 float 做 sum AllReduce（count <= max_count 且能被
    // world_size 整除）。
    void allreduce(float* sendbuff, float* recvbuff, size_t count);

    int world_size() const { return world_size_; }
    size_t max_count() const { return max_count_; }

private:
    void initialize();
    void setup_stream_and_buffers();
    void setup_remote_memory();
    void reduce_scatter(size_t count, uint32_t base);
    void all_gather(size_t count, uint32_t base);
    void signal(uint32_t value);
    void wait_for(int rank, uint32_t value, const char* what);
    void teardown();

    int signals_per_call() const { return 1 + 2 * steps_; }

    const int my_rank_;
    const int world_size_;
    const size_t max_count_;
    const std::string ipc_dir_;
    int steps_ = 0;
    int prev_ = 0;
    size_t epoch_ = 0;

    int local_dev_ = -1;
    cudaStream_t stream_ = nullptr;
    float* tmp_ = nullptr;
    float* commbuff_ = nullptr;
    uint32_t* counter_ = nullptr;

    std::vector<IpcHandle> handles_;
    std::vector<float*> remote_ptrs_;
    std::vector<uint32_t*> remote_counters_;
};

#endif  // TINYNCCL_RING_ALLREDUCE_H
