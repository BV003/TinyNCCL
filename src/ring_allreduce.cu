#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>
#include <cstdio>

#include "transport.h"
#include "kernels.h"
#include "ipc.h"

// ============================================================
// 单进程多GPU版本（保留用于单进程测试）
// ============================================================

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

// ============================================================
// 多进程 IPC 版本
// ============================================================

void tiny_ring_allreduce_sum_ipc(
    int my_rank, int world_size,
    float* sendbuff, float* recvbuff,
    size_t count,
    const std::string& ipc_dir)
{
    if (world_size <= 0)
        throw std::runtime_error("world_size must be >= 1");
    if (count % static_cast<size_t>(world_size) != 0)
        throw std::runtime_error("count must be divisible by world_size");

    const size_t chunk = count / static_cast<size_t>(world_size);
    const size_t chunk_bytes = chunk * sizeof(float);

    int local_dev;
    cudaGetDevice(&local_dev);

    // 1. 创建 stream 和临时 buffer
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float* tmp = nullptr;
    cudaMalloc(&tmp, chunk_bytes);

    // 2. 初始化 recvbuff = sendbuff
    cudaMemcpy(recvbuff, sendbuff, count * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // 3. 交换 IPC handles：每个 rank 把自己的 recvbuff 暴露给其他 rank
    //    这样其他 rank 可以通过 IPC 读取我们的 recvbuff
    std::vector<IpcHandle> handles = ipc_exchange_handles(
        my_rank, world_size,
        recvbuff, count * sizeof(float), local_dev,
        ipc_dir);

    // 4. 打开所有远程 handles，得到本地可访问的 pointer
    std::vector<float*> remote_ptrs(world_size, nullptr);
    for (int r = 0; r < world_size; r++) {
        if (r == my_rank) {
            remote_ptrs[r] = recvbuff; // 自己的 buffer 直接用
        } else {
            remote_ptrs[r] = static_cast<float*>(
                ipc_open_handle(handles[r], local_dev));
        }
    }

    // 5. Reduce-Scatter: world_size - 1 steps
    for (int step = 0; step < world_size - 1; step++) {
        int prev = (my_rank - 1 + world_size) % world_size;
        int idx = (my_rank - step + world_size) % world_size;

        // 从 prev rank 的 recvbuff 中读取 chunk idx，拷贝到本地 tmp
        // 通过 IPC，remote_ptrs[prev] 是 prev rank 的 recvbuff 的本地映射
        transport_copy_ipc(
            tmp,
            remote_ptrs[prev] + idx * chunk,
            chunk_bytes, stream);

        // 在本地设备上，把收到的块累加到 recvbuff[idx]
        tiny_reduce_sum_kernel(
            recvbuff + idx * chunk,
            recvbuff + idx * chunk,
            tmp,
            static_cast<int>(chunk), stream);

        cudaStreamSynchronize(stream);

        // debug
        if (world_size == 2 && count == 16) {
            std::vector<float> debug(count);
            cudaMemcpy(debug.data(), recvbuff, count * sizeof(float),
                       cudaMemcpyDeviceToHost);
            printf("[DEBUG RS] Rank %d (step=%d)\n", my_rank, step);
            printf("  chunk0: ");
            for (int j = 0; j < static_cast<int>(chunk); j++)
                printf("%.2f ", debug[j]);
            printf("\n  chunk1: ");
            for (int j = 0; j < static_cast<int>(chunk); j++)
                printf("%.2f ", debug[chunk + j]);
            printf("\n");
        }
    }

    // 6. All-Gather: world_size - 1 steps
    for (int step = 0; step < world_size - 1; step++) {
        int own_chunk = my_rank;
        int send_dst = (my_rank + 1) % world_size;

        printf("[AG] step=%d rank=%d send own_chunk=%d to rank=%d\n",
               step, my_rank, own_chunk, send_dst);

        // 把自己的 chunk 拷贝到 send_dst rank 的 recvbuff
        // 通过 IPC，remote_ptrs[send_dst] 是 send_dst rank 的 recvbuff 的本地映射
        transport_copy_ipc(
            remote_ptrs[send_dst] + own_chunk * chunk,
            recvbuff + own_chunk * chunk,
            chunk_bytes, stream);

        cudaStreamSynchronize(stream);

        // debug
        if (world_size == 2 && count == 16) {
            std::vector<float> debug(count);
            cudaMemcpy(debug.data(), recvbuff, count * sizeof(float),
                       cudaMemcpyDeviceToHost);
            printf("[DEBUG AG] Rank %d (step=%d)\n", my_rank, step);
            printf("  chunk0: ");
            for (int j = 0; j < static_cast<int>(chunk); j++)
                printf("%.2f ", debug[j]);
            printf("\n  chunk1: ");
            for (int j = 0; j < static_cast<int>(chunk); j++)
                printf("%.2f ", debug[chunk + j]);
            printf("\n");
        }
    }

    // 7. 最终同步
    cudaStreamSynchronize(stream);

    // 8. 关闭远程 IPC handles
    for (int r = 0; r < world_size; r++) {
        if (r != my_rank && remote_ptrs[r]) {
            ipc_close_handle(remote_ptrs[r]);
        }
    }

    // 9. 清理
    cudaStreamDestroy(stream);
    cudaFree(tmp);

    printf("[IPC AllReduce] Rank %d done\n", my_rank);
}
