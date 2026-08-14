#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>

#include "transport.h"
#include "kernels.h"

// Ring AllReduce = ReduceScatter (N-1 steps) + AllGather (N-1 steps).
//
// Each rank owns one dedicated copy stream. The copy and the reduce are issued
// on the SAME stream so that stream ordering (a hard CUDA guarantee) makes the
// reduce wait for the copy to finish. An earlier version used a separate
// compute stream + cudaEvent to express that dependency, but cudaMemcpyPeerAsync
// completion on this topology was not reliably captured by the event, causing a
// copy/reduce race (non-deterministic wrong results).
//
// Steps are separated by cudaDeviceSynchronize to satisfy the cross-rank
// dependency (rank i's receive in step s needs prev's reduce from step s-1).
// NOTE: this does NOT implement communication/computation overlap via double
// buffering; that is a separate, later optimization.
void tiny_ring_allreduce_sum_impl(
    float** send, float** recv, size_t count, int n_gpus)
{
    if (n_gpus <= 0)
        throw std::runtime_error("n_gpus must be >= 1");
    if (count % static_cast<size_t>(n_gpus) != 0)
        throw std::runtime_error("count must be divisible by n_gpus");

    const size_t chunk = count / static_cast<size_t>(n_gpus);

    // Per-rank async resources: one stream + one temp buffer per device.
    std::vector<cudaStream_t> stream(n_gpus);
    std::vector<float*> tmp(n_gpus, nullptr);

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&tmp[i], chunk * sizeof(float));
    }

    // 1. Initialize: recv = send on each device (synchronous).
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(recv[i], send[i], count * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    // 2. ReduceScatter: n_gpus - 1 steps.
    for (int step = 0; step < n_gpus - 1; step++) {
        // For each rank: copy neighbor's chunk into tmp, then reduce it into
        // recv. Both on the same stream so the reduce strictly follows the copy.
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i - step + n_gpus) % n_gpus;

            cudaSetDevice(next);
            transport_copy(tmp[next], next,
                           recv[i] + idx * chunk, i,
                           chunk * sizeof(float), stream[next]);
            tiny_reduce_sum_kernel(recv[next] + idx * chunk,
                                   recv[next] + idx * chunk,
                                   tmp[next],
                                   static_cast<int>(chunk), stream[next]);
        }

        // synchronize all devices before the next step (cross-rank dep).
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    // 3. AllGather: n_gpus - 1 steps (copy only).
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i + 1 - step + n_gpus) % n_gpus;

            cudaSetDevice(next);
            transport_copy(recv[next] + idx * chunk, next,
                           recv[i] + idx * chunk, i,
                           chunk * sizeof(float), stream[next]);
        }
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    // Cleanup.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(stream[i]);
        cudaFree(tmp[i]);
    }
}
