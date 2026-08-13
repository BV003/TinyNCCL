#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>

#include "transport.h"
#include "kernels.h"

// Ring AllReduce = ReduceScatter (N-1 steps) + AllGather (N-1 steps).
//
// Async version (Layer 4): each rank owns a dedicated copy stream (DMA) and a
// compute stream (reduce kernel). cudaEvent expresses the "reduce waits for
// copy" dependency, so different ranks' copies and reduces can overlap. Steps
// are separated by cudaDeviceSynchronize to satisfy the cross-rank dependency
// (rank i's receive in step s needs prev's reduce from step s-1).
//
// NOTE: this does NOT implement single-rank step-to-step overlap via double
// buffering — that only pays off for N>2 and additionally needs cross-device
// event sync (limited when P2P is unavailable). See the per-step synchronize
// comments below.
void tiny_ring_allreduce_sum_impl(
    float** send, float** recv, size_t count, int n_gpus)
{
    if (n_gpus <= 0)
        throw std::runtime_error("n_gpus must be >= 1");
    if (count % static_cast<size_t>(n_gpus) != 0)
        throw std::runtime_error("count must be divisible by n_gpus");

    const size_t chunk = count / static_cast<size_t>(n_gpus);

    // Per-rank async resources.
    std::vector<cudaStream_t> copy_stream(n_gpus);
    std::vector<cudaStream_t> compute_stream(n_gpus);
    std::vector<float*> tmp(n_gpus, nullptr);
    std::vector<cudaEvent_t> copy_event(n_gpus);

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&copy_stream[i]);
        cudaStreamCreate(&compute_stream[i]);
        cudaMalloc(&tmp[i], chunk * sizeof(float));
        cudaEventCreateWithFlags(&copy_event[i], cudaEventDisableTiming);
    }

    // 1. Initialize: recv = send on each device (synchronous).
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(recv[i], send[i], count * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    // 2. ReduceScatter: n_gpus - 1 steps.
    for (int step = 0; step < n_gpus - 1; step++) {
        // (a) launch all copies on each rank's copy stream.
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i - step + n_gpus) % n_gpus;

            cudaSetDevice(next);
            transport_copy(tmp[next], next,
                           recv[i] + idx * chunk, i,
                           chunk * sizeof(float), copy_stream[next]);
            cudaEventRecord(copy_event[next], copy_stream[next]);
        }

        // (b) launch all reduces on each rank's compute stream, waiting on
        //     the corresponding copy.
        for (int i = 0; i < n_gpus; i++) {
            int next = (i + 1) % n_gpus;
            int idx = (i - step + n_gpus) % n_gpus;

            cudaSetDevice(next);
            cudaStreamWaitEvent(compute_stream[next], copy_event[next], 0);
            tiny_reduce_sum_kernel(recv[next] + idx * chunk,
                                   recv[next] + idx * chunk,
                                   tmp[next],
                                   static_cast<int>(chunk), compute_stream[next]);
        }

        // (c) synchronize all devices before the next step (cross-rank dep).
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
                           chunk * sizeof(float), copy_stream[next]);
        }
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    // Cleanup.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaEventDestroy(copy_event[i]);
        cudaStreamDestroy(copy_stream[i]);
        cudaStreamDestroy(compute_stream[i]);
        cudaFree(tmp[i]);
    }
}
