#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>

#include "transport.h"
#include "kernels.h"

// Ring AllReduce = ReduceScatter (N-1 steps) + AllGather (N-1 steps).
//
// Data of each rank is split into n_gpus chunks. After ReduceScatter, rank i
// holds the fully-reduced chunk (i+1)%n_gpus. AllGather then broadcasts every
// reduced chunk around the ring so all ranks end up with the full result.
//
// This is the serial (synchronous) version: it uses the default stream
// (nullptr) so every transport copy and reduce completes before the next
// operation starts.
void tiny_ring_allreduce_sum(
    float** send, float** recv, size_t count, int n_gpus)
{
    if (n_gpus <= 0)
        throw std::runtime_error("n_gpus must be >= 1");
    if (count % static_cast<size_t>(n_gpus) != 0)
        throw std::runtime_error("count must be divisible by n_gpus");

    const size_t chunk = count / static_cast<size_t>(n_gpus);
    cudaStream_t stream = nullptr;  // default stream -> synchronous

    // 1. Initialize: recv = send on each device.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaMemcpyAsync(recv[i], send[i], count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // Per-device temporary buffer to receive one chunk from the previous rank.
    std::vector<float*> tmp(n_gpus, nullptr);
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaMalloc(&tmp[i], chunk * sizeof(float));
    }

    // 2. ReduceScatter: n_gpus - 1 steps.
    //    At step s, rank i sends chunk (i - s) % n_gpus to rank (i+1), which
    //    accumulates it into its local copy of that chunk.
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i - step + n_gpus) % n_gpus;

            transport_copy(tmp[next], next,
                           recv[i] + idx * chunk, i,
                           chunk * sizeof(float), stream);

            cudaSetDevice(next);
            tiny_reduce_sum_kernel(recv[next] + idx * chunk,
                                   recv[next] + idx * chunk,
                                   tmp[next],
                                   static_cast<int>(chunk), stream);
        }
    }

    // 3. AllGather: n_gpus - 1 steps.
    //    At step s, rank i sends the reduced chunk (i + 1 - s) % n_gpus to
    //    rank (i+1), filling in the missing chunks.
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i + 1 - step + n_gpus) % n_gpus;

            transport_copy(recv[next] + idx * chunk, next,
                           recv[i] + idx * chunk, i,
                           chunk * sizeof(float), stream);
        }
    }

    // Free per-device temporary buffers.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaFree(tmp[i]);
    }
}
