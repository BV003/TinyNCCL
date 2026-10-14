#include "ring_allreduce.h"

#include <sys/stat.h>

#include <stdexcept>
#include <vector>

#include "cuda_check.h"
#include "ipc.h"
#include "kernels.h"
#include "transport.h"

namespace {

void check_cuda_ipc(cudaError_t err, const char* operation, int rank,
                    const char* phase, int step) {
    check_cuda(err, std::string("[rank=") + std::to_string(rank) + "] " +
                        phase + " step=" + std::to_string(step) + " " +
                        operation + " failed");
}

constexpr char kReduceScatterPhase[] = "rs";
constexpr char kAllGatherPhase[] = "ag";
constexpr char kInitEventsSubdir[] = "/init_events";

std::string step_event_dir(const std::string& ipc_dir, const char* phase,
                           int step) {
    return ipc_dir + "/" + phase + "_done_step_" + std::to_string(step);
}

void synchronized_copy(float* dst, float* src, size_t count, int device) {
    cudaSetDevice(device);
    cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice);
}

}  // namespace

// ---------------------------------------------------------------------------
// RingCommunicator
//
// Persistent state (buffers, memory mapping, stream) is set up once. The IPC
// event objects are intentionally recreated per allreduce: cudaStreamWaitEvent
// captures the event's latest record at call time, so a reused event would let
// a rank proceed on the *previous* call's record before the peer re-records it.
// ---------------------------------------------------------------------------

RingCommunicator::RingCommunicator(int my_rank, int world_size,
                                   size_t max_count, const std::string& ipc_dir)
    : my_rank_(my_rank),
      world_size_(world_size),
      max_count_(max_count),
      ipc_dir_(ipc_dir) {
    if (world_size <= 0)
        throw std::runtime_error("world_size must be >= 1");
    if (my_rank < 0 || my_rank >= world_size)
        throw std::runtime_error("my_rank is outside world_size");
    if (max_count == 0 || max_count % static_cast<size_t>(world_size) != 0)
        throw std::runtime_error(
            "max_count must be positive and divisible by world_size");

    steps_ = world_size - 1;
    prev_ = (my_rank - 1 + world_size) % world_size;

    // If setup fails partway, still release whatever was acquired.
    try {
        initialize();
    } catch (...) {
        teardown();
        throw;
    }
}

RingCommunicator::~RingCommunicator() {
    teardown();
}

void RingCommunicator::initialize() {
    setup_stream_and_buffers();
    setup_remote_memory();
}

void RingCommunicator::setup_stream_and_buffers() {
    const size_t max_chunk = max_count_ / static_cast<size_t>(world_size_);

    check_cuda_ipc(cudaGetDevice(&local_dev_), "cudaGetDevice", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaStreamCreate(&stream_), "cudaStreamCreate", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaMalloc(&tmp_, max_chunk * sizeof(float)),
                   "cudaMalloc(tmp)", my_rank_, "init", -1);
    check_cuda_ipc(cudaMalloc(&commbuff_, max_count_ * sizeof(float)),
                   "cudaMalloc(commbuff)", my_rank_, "init", -1);
}

void RingCommunicator::setup_remote_memory() {
    handles_ =
        ipc_exchange_handles(my_rank_, world_size_, commbuff_,
                             max_count_ * sizeof(float), local_dev_, ipc_dir_);

    remote_ptrs_.assign(world_size_, nullptr);
    for (int r = 0; r < world_size_; r++) {
        remote_ptrs_[r] =
            (r == my_rank_)
                ? commbuff_  // our own buffer, not IPC-mapped
                : static_cast<float*>(ipc_open_handle(handles_[r], local_dev_));
    }
}

void RingCommunicator::allreduce(float* sendbuff, float* recvbuff,
                                 size_t count) {
    if (count == 0 || count > max_count_ ||
        count % static_cast<size_t>(world_size_) != 0) {
        throw std::runtime_error(
            "count must be positive, <= max_count and divisible by world_size");
    }

    try {
        // The comm buffer is reused across calls, so make sure every rank has
        // finished the previous allreduce before we overwrite it. Each call
        // also gets a fresh rendezvous directory, keyed by epoch, because the
        // per-call event handles differ every time.
        const size_t this_epoch = epoch_;
        ipc_barrier(my_rank_, world_size_, ipc_dir_, "epoch",
                    static_cast<int>(this_epoch));
        epoch_++;
        call_dir_ = ipc_dir_ + "/epoch_" + std::to_string(this_epoch);
        mkdir(call_dir_.c_str(), 0755);

        init_events_ = ipc_create_events(local_dev_);
        init_events_created_ = true;

        check_cuda_ipc(
            cudaMemcpyAsync(commbuff_, sendbuff, count * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream_),
            "cudaMemcpy(init)", my_rank_, "init", -1);
        check_cuda_ipc(cudaEventRecord(init_events_.init_ready, stream_),
                       "record init_ready", my_rank_, "EVENT", -1);

        std::vector<IpcEventHandleSet> init_handles =
            ipc_exchange_event_handles(my_rank_, world_size_, init_events_,
                                       call_dir_ + kInitEventsSubdir);
        remote_init_ready_ =
            ipc_open_event(init_handles[prev_].init_ready_handle, local_dev_);

        reduce_scatter(count);
        all_gather(count);

        check_cuda_ipc(cudaStreamSynchronize(stream_),
                       "final cudaStreamSynchronize", my_rank_, "final", -1);
        check_cuda_ipc(cudaMemcpy(recvbuff, commbuff_, count * sizeof(float),
                                  cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(final recvbuff)", my_rank_, "final", -1);
    } catch (...) {
        release_call_events();
        throw;
    }

    release_call_events();
}

void RingCommunicator::reduce_scatter(size_t count) {
    const size_t chunk = count / static_cast<size_t>(world_size_);
    const size_t chunk_bytes = chunk * sizeof(float);

    rs_events_.resize(steps_);
    remote_rs_events_.assign(steps_, nullptr);

    for (int step = 0; step < steps_; step++) {
        const int idx = (my_rank_ - step + world_size_) % world_size_;

        const cudaEvent_t wait_event =
            (step == 0) ? remote_init_ready_ : remote_rs_events_[step - 1];
        check_cuda_ipc(
            cudaStreamWaitEvent(stream_, wait_event, 0),
            (step == 0) ? "wait init_ready" : "wait reduce_scatter_done",
            my_rank_, "EVENT", step);

        transport_copy_ipc(tmp_, local_dev_, remote_ptrs_[prev_] + idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);

        tiny_reduce_sum_kernel(commbuff_ + idx * chunk, commbuff_ + idx * chunk,
                               tmp_, static_cast<int>(chunk), stream_);
        check_cuda_ipc(cudaGetLastError(), "kernel launch", my_rank_, "RS",
                       step);

        rs_events_[step] = ipc_create_events(local_dev_);
        check_cuda_ipc(cudaEventRecord(rs_events_[step].step_done, stream_),
                       "record reduce_scatter_done", my_rank_, "EVENT", step);
        check_cuda_ipc(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
                       my_rank_, "RS", step);

        std::vector<IpcEventHandleSet> rs_handles = ipc_exchange_event_handles(
            my_rank_, world_size_, rs_events_[step],
            step_event_dir(call_dir_, kReduceScatterPhase, step));
        remote_rs_events_[step] =
            ipc_open_event(rs_handles[prev_].step_done_handle, local_dev_);
    }
}

void RingCommunicator::all_gather(size_t count) {
    const size_t chunk = count / static_cast<size_t>(world_size_);
    const size_t chunk_bytes = chunk * sizeof(float);

    ag_events_.resize(steps_);
    remote_ag_events_.assign(steps_, nullptr);

    for (int step = 0; step < steps_; step++) {
        const int chunk_idx = (my_rank_ - step - 1 + world_size_) % world_size_;

        const cudaEvent_t ready_event = (step == 0)
                                            ? remote_rs_events_[steps_ - 1]
                                            : remote_ag_events_[step - 1];
        check_cuda_ipc(
            cudaStreamWaitEvent(stream_, ready_event, 0),
            (step == 0) ? "wait reduce_scatter_done" : "wait all_gather_done",
            my_rank_, "EVENT", step);

        transport_copy_ipc(tmp_, local_dev_,
                           remote_ptrs_[prev_] + chunk_idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);

        check_cuda_ipc(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
                       my_rank_, "AG", step);
        check_cuda_ipc(cudaMemcpy(commbuff_ + chunk_idx * chunk, tmp_,
                                  chunk_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(AG local)", my_rank_, "AG", step);

        ag_events_[step] = ipc_create_events(local_dev_);
        check_cuda_ipc(cudaEventRecord(ag_events_[step].step_done, stream_),
                       "record all_gather_done", my_rank_, "EVENT", step);
        check_cuda_ipc(cudaStreamSynchronize(stream_),
                       "cudaStreamSynchronize(AG local)", my_rank_, "AG", step);

        std::vector<IpcEventHandleSet> ag_handles = ipc_exchange_event_handles(
            my_rank_, world_size_, ag_events_[step],
            step_event_dir(call_dir_, kAllGatherPhase, step));
        remote_ag_events_[step] =
            ipc_open_event(ag_handles[prev_].step_done_handle, local_dev_);
    }
}

void RingCommunicator::release_call_events() {
    ipc_close_event(remote_init_ready_);
    remote_init_ready_ = nullptr;
    for (cudaEvent_t event : remote_rs_events_)
        ipc_close_event(event);
    for (cudaEvent_t event : remote_ag_events_)
        ipc_close_event(event);
    remote_rs_events_.clear();
    remote_ag_events_.clear();

    if (init_events_created_) {
        ipc_destroy_events(init_events_);
        init_events_created_ = false;
    }
    for (IpcEventSet& events : rs_events_)
        ipc_destroy_events(events);
    for (IpcEventSet& events : ag_events_)
        ipc_destroy_events(events);
    rs_events_.clear();
    ag_events_.clear();
}

void RingCommunicator::teardown() {
    release_call_events();
    for (int r = 0;
         r < world_size_ && r < static_cast<int>(remote_ptrs_.size()); r++) {
        if (r != my_rank_ && remote_ptrs_[r]) {
            ipc_close_handle(remote_ptrs_[r]);
        }
    }
    if (stream_)
        cudaStreamDestroy(stream_);
    if (tmp_)
        cudaFree(tmp_);
    if (commbuff_)
        cudaFree(commbuff_);
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

void tiny_ring_allreduce_sum_impl(float** send, float** recv, size_t count,
                                  int n_gpus) {
    if (n_gpus <= 0)
        throw std::runtime_error("n_gpus must be >= 1");
    if (count % static_cast<size_t>(n_gpus) != 0)
        throw std::runtime_error("count must be divisible by n_gpus");

    const size_t chunk = count / static_cast<size_t>(n_gpus);
    const size_t chunk_bytes = chunk * sizeof(float);

    std::vector<cudaStream_t> stream(n_gpus);
    std::vector<float*> tmp(n_gpus, nullptr);

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&stream[i]);
        cudaMalloc(&tmp[i], chunk_bytes);
    }

    for (int i = 0; i < n_gpus; i++) {
        synchronized_copy(recv[i], send[i], count, i);
    }

    // Reduce-Scatter
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            const int prev = (i - 1 + n_gpus) % n_gpus;
            const int idx = (i - step + n_gpus) % n_gpus;

            cudaSetDevice(i);
            transport_copy(tmp[i], i, recv[prev] + idx * chunk, prev,
                           chunk_bytes, stream[i]);

            tiny_reduce_sum_kernel(recv[i] + idx * chunk, recv[i] + idx * chunk,
                                   tmp[i], static_cast<int>(chunk), stream[i]);
        }

        for (int i = 0; i < n_gpus; i++) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
    }

    // All-Gather
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            const int own_chunk = i;
            const int send_dst = (i + 1) % n_gpus;

            cudaSetDevice(send_dst);
            transport_copy(recv[send_dst] + own_chunk * chunk, send_dst,
                           recv[i] + own_chunk * chunk, i, chunk_bytes,
                           stream[send_dst]);
        }

        for (int dev = 0; dev < n_gpus; dev++) {
            cudaSetDevice(dev);
            cudaDeviceSynchronize();
        }
    }

    for (int i = 0; i < n_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(stream[i]);
        cudaFree(tmp[i]);
    }
}

void tiny_ring_allreduce_sum_ipc(int my_rank, int world_size, float* sendbuff,
                                 float* recvbuff, size_t count,
                                 const std::string& ipc_dir) {
    RingCommunicator communicator(my_rank, world_size, count, ipc_dir);
    communicator.allreduce(sendbuff, recvbuff, count);
}
