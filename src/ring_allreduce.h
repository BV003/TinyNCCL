#ifndef TINYNCCL_RING_ALLREDUCE_H
#define TINYNCCL_RING_ALLREDUCE_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ipc.h"

// Single-process, multi-GPU variant (kept for the single-process test).
void tiny_ring_allreduce_sum_impl(float** send, float** recv, size_t count,
                                  int n_gpus);

// Persistent communicator: set up once, then call allreduce repeatedly.
//
// Synchronization is entirely GPU-side: each rank owns a monotonically
// increasing 32-bit counter in device memory, bumps it with
// cuStreamWriteValue32 and waits on its neighbour's counter with
// cuStreamWaitValue32. The steady-state allreduce touches no files, creates no
// events and makes no host round-trips.
class RingCommunicator {
public:
    // max_count is the largest element count accepted by allreduce; it must be
    // a multiple of world_size.
    RingCommunicator(int my_rank, int world_size, size_t max_count,
                     const std::string& ipc_dir);
    ~RingCommunicator();

    RingCommunicator(const RingCommunicator&) = delete;
    RingCommunicator& operator=(const RingCommunicator&) = delete;

    // Sum AllReduce of count floats (count <= max_count and divisible by
    // world_size).
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
    float* scratch_ = nullptr;
    float* commbuff_ = nullptr;
    uint32_t* counter_ = nullptr;

    std::vector<IpcHandle> handles_;
    std::vector<float*> remote_ptrs_;
    std::vector<uint32_t*> remote_counters_;
};

#endif  // TINYNCCL_RING_ALLREDUCE_H
