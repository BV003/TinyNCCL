#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>
#include <cstdio>

#include "transport.h"
#include "kernels.h"
#include "ipc.h"

namespace {

void check_cuda_ipc(cudaError_t err, const char* operation,
                    int rank, const char* phase, int step) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[rank=") + std::to_string(rank) + "] " +
            phase + " step=" + std::to_string(step) + " " + operation +
            " failed: " + cudaGetErrorString(err));
    }
}

void log_event_result(cudaError_t err, const char* operation,
                      int rank, int step, cudaEvent_t event) {
    fprintf(stderr,
            "[IPC EVENT] rank=%d step=%d op=%s event=%p result=%s\n",
            rank, step, operation, static_cast<void*>(event),
            cudaGetErrorString(err));
    check_cuda_ipc(err, operation, rank, "EVENT", step);
}

void dump_chunk(const char* phase, int rank, int step, int chunk_idx,
                const float* device_ptr, size_t chunk) {
    std::vector<float> values(chunk);
    cudaError_t err = cudaMemcpy(values.data(), device_ptr,
                                 chunk * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    check_cuda_ipc(err, "cudaMemcpy(debug chunk)", rank, phase, step);
    fprintf(stderr, "[IPC DATA] phase=%s rank=%d step=%d chunk=%d:",
            phase, rank, step, chunk_idx);
    for (float value : values) fprintf(stderr, " %.6f", value);
    fprintf(stderr, "\n");
}

}  // namespace

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
    if (my_rank < 0 || my_rank >= world_size)
        throw std::runtime_error("my_rank is outside world_size");
    if (count % static_cast<size_t>(world_size) != 0)
        throw std::runtime_error("count must be divisible by world_size");

    const size_t chunk = count / static_cast<size_t>(world_size);
    const size_t chunk_bytes = chunk * sizeof(float);

    int local_dev;
    cudaGetDevice(&local_dev);

    // 1. 创建 stream 和临时 buffer
    cudaStream_t stream;
    check_cuda_ipc(cudaStreamCreate(&stream), "cudaStreamCreate",
                   my_rank, "init", -1);
    float* tmp = nullptr;
    float* commbuff = nullptr;
    check_cuda_ipc(cudaMalloc(&tmp, chunk_bytes), "cudaMalloc(tmp)",
                   my_rank, "init", -1);
    check_cuda_ipc(cudaMalloc(&commbuff, count * sizeof(float)),
                   "cudaMalloc(commbuff)", my_rank, "init", -1);

    // 2. 初始化独立的 CUDA IPC communication buffer。
    //    不直接导出 PyTorch allocator 管理的 recvbuff。
    check_cuda_ipc(cudaMemcpy(commbuff, sendbuff, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(init)", my_rank, "init", -1);
    check_cuda_ipc(cudaGetLastError(), "cudaGetLastError(init)",
                   my_rank, "init", -1);

    // 3. 创建并交换初始化 event。
    IpcEventSet init_events = ipc_create_events(local_dev);
    // 必须先 record，再让另一个进程打开并等待这个 init event。
    log_event_result(cudaEventRecord(init_events.init_ready, stream),
                     "record init_ready", my_rank, -1,
                     init_events.init_ready);
    std::vector<IpcEventHandleSet> event_handles = ipc_exchange_event_handles(
        my_rank, world_size, init_events, ipc_dir + "/init_events");
    cudaEvent_t remote_init_ready = ipc_open_event(
        event_handles[(my_rank - 1 + world_size) % world_size]
            .init_ready_handle, local_dev);

    // 4. 交换 IPC handles：每个 rank 暴露独立的 cudaMalloc communication buffer。
    std::vector<IpcHandle> handles = ipc_exchange_handles(
        my_rank, world_size,
        commbuff, count * sizeof(float), local_dev,
        ipc_dir);

    // 5. 打开所有远程 handles，得到本地可访问的 pointer
    std::vector<float*> remote_ptrs(world_size, nullptr);
    for (int r = 0; r < world_size; r++) {
        if (r == my_rank) {
            remote_ptrs[r] = commbuff; // 自己的 communication buffer
        } else {
            remote_ptrs[r] = static_cast<float*>(
                ipc_open_handle(handles[r], local_dev));
        }
    }

    const int steps = world_size - 1;
    const int prev = (my_rank - 1 + world_size) % world_size;

    // 每个 step 使用独立 event，避免复用 event 时覆盖远程 wait 的状态。
    std::vector<IpcEventSet> rs_events(steps);
    std::vector<cudaEvent_t> remote_rs_events(steps, nullptr);

    // 6. Reduce-Scatter: world_size - 1 steps
    for (int step = 0; step < world_size - 1; step++) {
        int idx = (my_rank - step + world_size) % world_size;

        // 从 prev rank 的 communication buffer 中读取 chunk idx。
        if (step == 0) {
            log_event_result(cudaStreamWaitEvent(stream, remote_init_ready, 0),
                             "wait init_ready", my_rank, step,
                             remote_init_ready);
        } else {
            log_event_result(cudaStreamWaitEvent(
                                 stream, remote_rs_events[step - 1], 0),
                             "wait reduce_scatter_done", my_rank, step,
                             remote_rs_events[step - 1]);
        }
        transport_copy_ipc(
            tmp,
            local_dev,
            remote_ptrs[prev] + idx * chunk,
            handles[prev].device,
            chunk_bytes, stream);

        // 在本地设备上，把收到的块累加到 communication buffer[idx]
        tiny_reduce_sum_kernel(
            commbuff + idx * chunk,
            commbuff + idx * chunk,
            tmp,
            static_cast<int>(chunk), stream);

        check_cuda_ipc(cudaGetLastError(), "kernel launch", my_rank, "RS", step);
        rs_events[step] = ipc_create_events(local_dev);
        log_event_result(cudaEventRecord(
                             rs_events[step].reduce_scatter_done, stream),
                         "record reduce_scatter_done", my_rank, step,
                         rs_events[step].reduce_scatter_done);
        check_cuda_ipc(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                       my_rank, "RS", step);
        fprintf(stderr, "[IPC RS] rank=%d step=%d received rank=%d chunk=%d\n",
                my_rank, step, prev, idx);
        dump_chunk("RS_LOCAL", my_rank, step, idx,
                   commbuff + idx * chunk, chunk);
        ipc_barrier(my_rank, world_size, ipc_dir, "rs", step);

        std::vector<IpcEventHandleSet> rs_handles =
            ipc_exchange_event_handles(
                my_rank, world_size, rs_events[step],
                ipc_dir + "/rs_done_step_" + std::to_string(step));
        remote_rs_events[step] = ipc_open_event(
            rs_handles[prev].reduce_scatter_done_handle, local_dev);
        fprintf(stderr,
                "[IPC EVENT] rank=%d opened RS step=%d event from rank=%d "
                "event=%p\n",
                my_rank, step, prev,
                static_cast<void*>(remote_rs_events[step]));

        // debug
        if (world_size == 2 && count == 16) {
            std::vector<float> debug(count);
            cudaMemcpy(debug.data(), commbuff, count * sizeof(float),
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

    // 7. All-Gather: world_size - 1 steps
    std::vector<IpcEventSet> ag_events(steps);
    std::vector<cudaEvent_t> remote_ag_events(steps, nullptr);
    for (int step = 0; step < world_size - 1; step++) {
        int chunk_idx = (my_rank - step - 1 + world_size) % world_size;

        fprintf(stderr,
                "[IPC AG] step=%d rank=%d pull chunk=%d from rank=%d\n",
                step, my_rank, chunk_idx, prev);

        // Pull 模式：从前驱 rank 的 IPC 映射内存读取 chunk，
        // 写入当前 rank 自己的 communication buffer，避免直接写远程进程内存。
        cudaEvent_t ready_event = step == 0
            ? remote_rs_events[steps - 1]
            : remote_ag_events[step - 1];
        log_event_result(cudaStreamWaitEvent(stream, ready_event, 0),
                         step == 0 ? "wait reduce_scatter_done"
                                   : "wait all_gather_done",
                         my_rank, step, ready_event);

        // 先把远程 chunk 读到本地临时 buffer，便于验证 IPC 读取结果；
        // 再进行本地 D2D copy，避免直接把远程 IPC 指针作为最终目标。
        transport_copy_ipc(
            tmp,
            local_dev,
            remote_ptrs[prev] + chunk_idx * chunk,
            handles[prev].device,
            chunk_bytes, stream);

        check_cuda_ipc(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                       my_rank, "AG", step);
        dump_chunk("AG_REMOTE", my_rank, step, chunk_idx, tmp, chunk);
        check_cuda_ipc(cudaMemcpy(commbuff + chunk_idx * chunk, tmp,
                                  chunk_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(AG local)", my_rank, "AG", step);
        ag_events[step] = ipc_create_events(local_dev);
        log_event_result(cudaEventRecord(ag_events[step].reduce_scatter_done,
                                         stream),
                         "record all_gather_done", my_rank, step,
                         ag_events[step].reduce_scatter_done);
        check_cuda_ipc(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize(AG local)", my_rank, "AG", step);
        dump_chunk("AG_LOCAL", my_rank, step, chunk_idx,
                   commbuff + chunk_idx * chunk, chunk);
        ipc_barrier(my_rank, world_size, ipc_dir, "ag", step);

        std::vector<IpcEventHandleSet> ag_handles =
            ipc_exchange_event_handles(
                my_rank, world_size, ag_events[step],
                ipc_dir + "/ag_done_step_" + std::to_string(step));
        remote_ag_events[step] = ipc_open_event(
            ag_handles[prev].reduce_scatter_done_handle, local_dev);
        fprintf(stderr,
                "[IPC EVENT] rank=%d opened AG step=%d event from rank=%d "
                "event=%p\n",
                my_rank, step, prev,
                static_cast<void*>(remote_ag_events[step]));

        // debug
        if (world_size == 2 && count == 16) {
            std::vector<float> debug(count);
            cudaMemcpy(debug.data(), commbuff, count * sizeof(float),
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

    // 8. 最终同步
    check_cuda_ipc(cudaStreamSynchronize(stream), "final cudaStreamSynchronize",
                   my_rank, "final", -1);
    check_cuda_ipc(cudaMemcpy(recvbuff, commbuff, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(final recvbuff)", my_rank, "final", -1);

    // 9. 关闭远程 IPC handles 和 events
    for (int r = 0; r < world_size; r++) {
        if (r != my_rank && remote_ptrs[r]) {
            ipc_close_handle(remote_ptrs[r]);
        }
    }
    ipc_close_event(remote_init_ready);
    for (cudaEvent_t event : remote_rs_events) ipc_close_event(event);
    for (cudaEvent_t event : remote_ag_events) ipc_close_event(event);
    ipc_destroy_events(init_events);
    for (IpcEventSet& events : rs_events) ipc_destroy_events(events);
    for (IpcEventSet& events : ag_events) ipc_destroy_events(events);

    // 10. 清理
    cudaStreamDestroy(stream);
    cudaFree(tmp);
    cudaFree(commbuff);

    printf("[IPC AllReduce] Rank %d done\n", my_rank);
}
