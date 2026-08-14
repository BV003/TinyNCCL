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
    // For each rank i: receive chunk FROM prev(i), reduce into i's own buffer
    for (int i = 0; i < n_gpus; i++) {
        int prev = (i - 1 + n_gpus) % n_gpus;   // 上一个邻居，数据源
        int idx = (i - step + n_gpus) % n_gpus; // 需要接收的chunk编号

        // 目标设备是 i：数据从 prev 拷贝到 i 的 tmp[i]
        cudaSetDevice(i);
        transport_copy(tmp[i], i,
                       recv[prev] + idx * chunk, prev,
                       chunk * sizeof(float), stream[i]);

        // 在i自己的设备上，把收到的块累加到 recv[i][idx]
        tiny_reduce_sum_kernel(recv[i] + idx * chunk,
                               recv[i] + idx * chunk,
                               tmp[i],
                               static_cast<int>(chunk), stream[i]);
    }

    // synchronize all devices before the next step (cross‑rank dep).
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // ----- 你的debug打印，原样保留 -----
    if (n_gpus == 2 && count == 16) {
        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            std::vector<float> debug(count);
            cudaMemcpy(debug.data(), recv[i], count * sizeof(float), cudaMemcpyDeviceToHost);
            printf("[DEBUG RS] GPU %d\n", i);
            printf("  chunk0: ");
            for (int j = 0; j < static_cast<int>(chunk); j++) printf("%.2f ", debug[j]);
            printf("\n  chunk1: ");
            for (int j = 0; j < static_cast<int>(chunk); j++) printf("%.2f ", debug[chunk + j]);
            printf("\n");
        }
    }
    // ----- debug结束 -----
}


    // 3. AllGather: n_gpus -1 steps
    // After Reduce‑Scatter: device k holds complete sum at chunk index = k
    for (int step = 0; step < n_gpus - 1; step++)
    {
        printf("\n==== AllGather step = %d ====\n", step);
        for(int i = 0; i < n_gpus; i++)
        {
            int own_chunk = i;          // i设备自己拥有sum的chunk编号
            int send_dst  = (i + 1) % n_gpus; //发给下一个GPU

            printf("AG step=%d dev=%d send own_chunk=%d to dev=%d, src recv+%zu -> dst recv+%zu\n",
                   step, i, own_chunk, send_dst,
                   own_chunk*chunk, own_chunk*chunk);

            cudaSetDevice(send_dst);
            transport_copy(
                recv[send_dst] + own_chunk * chunk, send_dst,
                recv[i]        + own_chunk * chunk, i,
                chunk * sizeof(float), stream[send_dst]);
        }

        // 每一步结束，同步全部GPU
        for(int dev = 0; dev < n_gpus; dev++)
        {
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
        }

        //debug
        if (n_gpus == 2 && count == 16) {
            for (int i = 0; i < n_gpus; i++) {
                cudaSetDevice(i);
                std::vector<float> debug(count);
                cudaMemcpy(debug.data(), recv[i], count * sizeof(float), cudaMemcpyDeviceToHost);
                printf("[DEBUG AG] GPU %d\n", i);
                printf("  chunk0: ");
                for (int j = 0; j < static_cast<int>(chunk); j++) printf("%.2f ", debug[j]);
                printf("\n  chunk1: ");
                for (int j = 0; j < static_cast<int>(chunk); j++) printf("%.2f ", debug[chunk + j]);
                printf("\n");
            }
        }
        //debug
    }

    // 全局最终同步（保留）
    for (int dev = 0; dev < n_gpus; dev++)
    {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    // Cleanup.
    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(stream[i]);
        cudaFree(tmp[i]);
    }
}
