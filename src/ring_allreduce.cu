#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>

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
    (void)event;
    check_cuda_ipc(err, operation, rank, "EVENT", step);
}

}  // namespace

void tiny_ring_allreduce_sum_impl(
    float** send, float** recv, size_t count, int n_gpus)
{
    if (n_gpus <= 0)
        throw std::runtime_error("n_gpus must be >= 1");
    if (count % static_cast<size_t>(n_gpus) != 0)
        throw std::runtime_error("count must be divisible by n_gpus");

    const size_t chunk = count / static_cast<size_t>(n_gpus);

    std::vector<cudaStream_t> stream(n_gpus);
    std::vector<float*> tmp(n_gpus, nullptr);

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&tmp[i], chunk * sizeof(float));
    }

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaMemcpy(recv[i], send[i], count * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    for (int step = 0; step < n_gpus - 1; step++) {
    for (int i = 0; i < n_gpus; i++) {
        int prev = (i - 1 + n_gpus) % n_gpus;
        int idx = (i - step + n_gpus) % n_gpus;

        cudaSetDevice(i);
        transport_copy(tmp[i], i,
                       recv[prev] + idx * chunk, prev,
                       chunk * sizeof(float), stream[i]);

        tiny_reduce_sum_kernel(recv[i] + idx * chunk,
                               recv[i] + idx * chunk,
                               tmp[i],
                               static_cast<int>(chunk), stream[i]);
    }

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    }
    for (int step = 0; step < n_gpus - 1; step++) {
        for(int i = 0; i < n_gpus; i++)
        {
            int own_chunk = i;
            int send_dst  = (i + 1) % n_gpus;

            cudaSetDevice(send_dst);
            transport_copy(
                recv[send_dst] + own_chunk * chunk, send_dst,
                recv[i]        + own_chunk * chunk, i,
                chunk * sizeof(float), stream[send_dst]);
        }

        for(int dev = 0; dev < n_gpus; dev++)
        {
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
        }

    }

    for (int dev = 0; dev < n_gpus; dev++)
    {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(stream[i]);
        cudaFree(tmp[i]);
    }
}

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

    cudaStream_t stream;
    check_cuda_ipc(cudaStreamCreate(&stream), "cudaStreamCreate",
                   my_rank, "init", -1);
    float* tmp = nullptr;
    float* commbuff = nullptr;
    check_cuda_ipc(cudaMalloc(&tmp, chunk_bytes), "cudaMalloc(tmp)",
                   my_rank, "init", -1);
    check_cuda_ipc(cudaMalloc(&commbuff, count * sizeof(float)),
                   "cudaMalloc(commbuff)", my_rank, "init", -1);

    check_cuda_ipc(cudaMemcpy(commbuff, sendbuff, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(init)", my_rank, "init", -1);
    check_cuda_ipc(cudaGetLastError(), "cudaGetLastError(init)",
                   my_rank, "init", -1);

    IpcEventSet init_events = ipc_create_events(local_dev);
    log_event_result(cudaEventRecord(init_events.init_ready, stream),
                     "record init_ready", my_rank, -1,
                     init_events.init_ready);
    std::vector<IpcEventHandleSet> event_handles = ipc_exchange_event_handles(
        my_rank, world_size, init_events, ipc_dir + "/init_events");
    cudaEvent_t remote_init_ready = ipc_open_event(
        event_handles[(my_rank - 1 + world_size) % world_size]
            .init_ready_handle, local_dev);

    std::vector<IpcHandle> handles = ipc_exchange_handles(
        my_rank, world_size,
        commbuff, count * sizeof(float), local_dev,
        ipc_dir);

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

    std::vector<IpcEventSet> rs_events(steps);
    std::vector<cudaEvent_t> remote_rs_events(steps, nullptr);

    for (int step = 0; step < world_size - 1; step++) {
        int idx = (my_rank - step + world_size) % world_size;

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
        ipc_barrier(my_rank, world_size, ipc_dir, "rs", step);

        std::vector<IpcEventHandleSet> rs_handles =
            ipc_exchange_event_handles(
                my_rank, world_size, rs_events[step],
                ipc_dir + "/rs_done_step_" + std::to_string(step));
        remote_rs_events[step] = ipc_open_event(
            rs_handles[prev].reduce_scatter_done_handle, local_dev);
    }

    std::vector<IpcEventSet> ag_events(steps);
    std::vector<cudaEvent_t> remote_ag_events(steps, nullptr);
    for (int step = 0; step < world_size - 1; step++) {
        int chunk_idx = (my_rank - step - 1 + world_size) % world_size;

        cudaEvent_t ready_event = step == 0
            ? remote_rs_events[steps - 1]
            : remote_ag_events[step - 1];
        log_event_result(cudaStreamWaitEvent(stream, ready_event, 0),
                         step == 0 ? "wait reduce_scatter_done"
                                   : "wait all_gather_done",
                         my_rank, step, ready_event);

        transport_copy_ipc(
            tmp,
            local_dev,
            remote_ptrs[prev] + chunk_idx * chunk,
            handles[prev].device,
            chunk_bytes, stream);

        check_cuda_ipc(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                       my_rank, "AG", step);
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
        ipc_barrier(my_rank, world_size, ipc_dir, "ag", step);

        std::vector<IpcEventHandleSet> ag_handles =
            ipc_exchange_event_handles(
                my_rank, world_size, ag_events[step],
                ipc_dir + "/ag_done_step_" + std::to_string(step));
        remote_ag_events[step] = ipc_open_event(
            ag_handles[prev].reduce_scatter_done_handle, local_dev);
    }

    check_cuda_ipc(cudaStreamSynchronize(stream), "final cudaStreamSynchronize",
                   my_rank, "final", -1);
    check_cuda_ipc(cudaMemcpy(recvbuff, commbuff, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(final recvbuff)", my_rank, "final", -1);

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

    cudaStreamDestroy(stream);
    cudaFree(tmp);
    cudaFree(commbuff);
}
