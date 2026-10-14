#ifndef TINYNCCL_RING_ALLREDUCE_H
#define TINYNCCL_RING_ALLREDUCE_H

#include <cuda_runtime.h>

#include <cstddef>
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
// 与无状态版本的差别：显存句柄 / 事件句柄只在构造时交换、且远端指针与
// 远端事件只在构造时打开，因此每次 allreduce 省掉了昂贵的
// cudaIpcOpenMemHandle 与文件 rendezvous。
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
    void reduce_scatter(size_t count);
    void all_gather(size_t count);
    void release_call_events();
    void teardown();

    const int my_rank_;
    const int world_size_;
    const size_t max_count_;
    const std::string ipc_dir_;
    std::string call_dir_;
    int steps_ = 0;
    int prev_ = 0;
    size_t epoch_ = 0;

    int local_dev_ = -1;
    cudaStream_t stream_ = nullptr;
    float* tmp_ = nullptr;
    float* commbuff_ = nullptr;

    IpcEventSet init_events_{};
    bool init_events_created_ = false;
    cudaEvent_t remote_init_ready_ = nullptr;
    std::vector<IpcEventSet> rs_events_;
    std::vector<IpcEventSet> ag_events_;
    std::vector<cudaEvent_t> remote_rs_events_;
    std::vector<cudaEvent_t> remote_ag_events_;

    std::vector<IpcHandle> handles_;
    std::vector<float*> remote_ptrs_;
};

#endif  // TINYNCCL_RING_ALLREDUCE_H
