#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>
#include <cstdio>

#include "transport.h"
#include "kernels.h"

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
        //debug
            if (n_gpus == 2 && count == 16) {
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);

            std::vector<float> debug(count);

            cudaMemcpy(
                debug.data(),
                recv[i],
                count * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            printf("[DEBUG RS] GPU %d\n", i);

            printf("  chunk0: ");
            for (int j = 0; j < static_cast<int>(chunk); j++) {
                printf("%.2f ", debug[j]);
            }

            printf("\n  chunk1: ");
            for (int j = 0; j < static_cast<int>(chunk); j++) {
                printf("%.2f ", debug[chunk + j]);
            }

            printf("\n");
        }
    }
    //debug
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
        //debug
        if (n_gpus == 2 && count == 16) {
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);

        std::vector<float> debug(count);

        cudaMemcpy(
            debug.data(),
            recv[i],
            count * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        printf("[DEBUG AG] GPU %d\n", i);

        printf("  chunk0: ");
        for (int j = 0; j < static_cast<int>(chunk); j++) {
            printf("%.2f ", debug[j]);
        }

        printf("\n  chunk1: ");
        for (int j = 0; j < static_cast<int>(chunk); j++) {
            printf("%.2f ", debug[chunk + j]);
        }

        printf("\n");
    }
}
//debug
    }

    // Cleanup.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(stream[i]);
        cudaFree(tmp[i]);
    }
}
