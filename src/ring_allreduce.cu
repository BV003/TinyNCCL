#include "ring_allreduce.h"

#include <vector>
#include <stdexcept>

#include "transport.h"
#include "kernels.h"
#include "ipc.h"
#include "cuda_check.h"

namespace {

void check_cuda_ipc(cudaError_t err, const char* operation, int rank,
                    const char* phase, int step) {
    check_cuda(err, std::string("[rank=") + std::to_string(rank) + "] " +
                        phase + " step=" + std::to_string(step) + " " +
                        operation + " failed");
}

// Barrier phases and the directory suffix used to exchange each step's events.
constexpr char kReduceScatterPhase[] = "rs";
constexpr char kAllGatherPhase[] = "ag";
constexpr char kInitEventsSubdir[] = "/init_events";

std::string step_event_dir(const std::string& ipc_dir, const char* phase,
                           int step) {
    return ipc_dir + "/" + phase + "_done_step_" + std::to_string(step);
}

// One multi-process Ring AllReduce over CUDA IPC.
//
// Owns every CUDA/IPC resource it acquires so that an exception thrown in any
// phase still releases them in the destructor (the public entry point
// constructs the object, then calls initialize()/run(), so the destructor is
// guaranteed to run even when setup fails).
class RingSession {
public:
    RingSession(int my_rank, int world_size, float* sendbuff, float* recvbuff,
                size_t count, const std::string& ipc_dir)
        : my_rank_(my_rank),
          world_size_(world_size),
          sendbuff_(sendbuff),
          recvbuff_(recvbuff),
          count_(count),
          chunk_(count / static_cast<size_t>(world_size)),
          chunk_bytes_(chunk_ * sizeof(float)),
          steps_(world_size - 1),
          prev_((my_rank - 1 + world_size) % world_size),
          ipc_dir_(ipc_dir) {}

    ~RingSession() { teardown(); }

    RingSession(const RingSession&) = delete;
    RingSession& operator=(const RingSession&) = delete;

    void initialize();
    void run();

private:
    void setup_init_event();
    void setup_remote_buffers();
    void reduce_scatter();
    void all_gather();
    void copy_result_out();
    void teardown();

    const int my_rank_;
    const int world_size_;
    float* const sendbuff_;
    float* const recvbuff_;
    const size_t count_;
    const size_t chunk_;
    const size_t chunk_bytes_;
    const int steps_;
    const int prev_;
    const std::string ipc_dir_;

    int local_dev_ = -1;
    cudaStream_t stream_ = nullptr;
    float* tmp_ = nullptr;
    float* commbuff_ = nullptr;

    IpcEventSet init_events_{};
    bool init_events_created_ = false;
    cudaEvent_t remote_init_ready_ = nullptr;

    std::vector<IpcHandle> handles_;
    std::vector<float*> remote_ptrs_;
    std::vector<IpcEventSet> rs_events_;
    std::vector<IpcEventSet> ag_events_;
    std::vector<cudaEvent_t> remote_rs_events_;
    std::vector<cudaEvent_t> remote_ag_events_;
};

void RingSession::initialize() {
    check_cuda_ipc(cudaGetDevice(&local_dev_), "cudaGetDevice", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaStreamCreate(&stream_), "cudaStreamCreate", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaMalloc(&tmp_, chunk_bytes_), "cudaMalloc(tmp)", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaMalloc(&commbuff_, count_ * sizeof(float)),
                   "cudaMalloc(commbuff)", my_rank_, "init", -1);
    check_cuda_ipc(cudaMemcpy(commbuff_, sendbuff_, count_ * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(init)", my_rank_, "init", -1);
    check_cuda_ipc(cudaGetLastError(), "cudaGetLastError(init)", my_rank_,
                   "init", -1);

    setup_init_event();
    setup_remote_buffers();

    rs_events_.resize(steps_);
    ag_events_.resize(steps_);
    remote_rs_events_.assign(steps_, nullptr);
    remote_ag_events_.assign(steps_, nullptr);
}

void RingSession::setup_init_event() {
    init_events_ = ipc_create_events(local_dev_);
    init_events_created_ = true;
    check_cuda_ipc(cudaEventRecord(init_events_.init_ready, stream_),
                   "record init_ready", my_rank_, "EVENT", -1);

    std::vector<IpcEventHandleSet> event_handles = ipc_exchange_event_handles(
        my_rank_, world_size_, init_events_, ipc_dir_ + kInitEventsSubdir);
    remote_init_ready_ =
        ipc_open_event(event_handles[prev_].init_ready_handle, local_dev_);
}

void RingSession::setup_remote_buffers() {
    handles_ =
        ipc_exchange_handles(my_rank_, world_size_, commbuff_,
                             count_ * sizeof(float), local_dev_, ipc_dir_);

    remote_ptrs_.assign(world_size_, nullptr);
    for (int r = 0; r < world_size_; r++) {
        remote_ptrs_[r] =
            (r == my_rank_)
                ? commbuff_  // our own communication buffer, not IPC-mapped
                : static_cast<float*>(ipc_open_handle(handles_[r], local_dev_));
    }
}

void RingSession::reduce_scatter() {
    for (int step = 0; step < steps_; step++) {
        const int idx = (my_rank_ - step + world_size_) % world_size_;

        const cudaEvent_t wait_event =
            (step == 0) ? remote_init_ready_ : remote_rs_events_[step - 1];
        check_cuda_ipc(
            cudaStreamWaitEvent(stream_, wait_event, 0),
            (step == 0) ? "wait init_ready" : "wait reduce_scatter_done",
            my_rank_, "EVENT", step);

        transport_copy_ipc(tmp_, local_dev_, remote_ptrs_[prev_] + idx * chunk_,
                           handles_[prev_].device, chunk_bytes_, stream_);

        tiny_reduce_sum_kernel(commbuff_ + idx * chunk_,
                               commbuff_ + idx * chunk_, tmp_,
                               static_cast<int>(chunk_), stream_);
        check_cuda_ipc(cudaGetLastError(), "kernel launch", my_rank_, "RS",
                       step);

        rs_events_[step] = ipc_create_events(local_dev_);
        check_cuda_ipc(
            cudaEventRecord(rs_events_[step].reduce_scatter_done, stream_),
            "record reduce_scatter_done", my_rank_, "EVENT", step);
        check_cuda_ipc(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
                       my_rank_, "RS", step);
        ipc_barrier(my_rank_, world_size_, ipc_dir_, kReduceScatterPhase, step);

        std::vector<IpcEventHandleSet> rs_handles = ipc_exchange_event_handles(
            my_rank_, world_size_, rs_events_[step],
            step_event_dir(ipc_dir_, kReduceScatterPhase, step));
        remote_rs_events_[step] = ipc_open_event(
            rs_handles[prev_].reduce_scatter_done_handle, local_dev_);
    }
}

void RingSession::all_gather() {
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
                           remote_ptrs_[prev_] + chunk_idx * chunk_,
                           handles_[prev_].device, chunk_bytes_, stream_);

        check_cuda_ipc(cudaStreamSynchronize(stream_), "cudaStreamSynchronize",
                       my_rank_, "AG", step);
        check_cuda_ipc(cudaMemcpy(commbuff_ + chunk_idx * chunk_, tmp_,
                                  chunk_bytes_, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(AG local)", my_rank_, "AG", step);

        ag_events_[step] = ipc_create_events(local_dev_);
        check_cuda_ipc(
            cudaEventRecord(ag_events_[step].reduce_scatter_done, stream_),
            "record all_gather_done", my_rank_, "EVENT", step);
        check_cuda_ipc(cudaStreamSynchronize(stream_),
                       "cudaStreamSynchronize(AG local)", my_rank_, "AG", step);
        ipc_barrier(my_rank_, world_size_, ipc_dir_, kAllGatherPhase, step);

        std::vector<IpcEventHandleSet> ag_handles = ipc_exchange_event_handles(
            my_rank_, world_size_, ag_events_[step],
            step_event_dir(ipc_dir_, kAllGatherPhase, step));
        remote_ag_events_[step] = ipc_open_event(
            ag_handles[prev_].reduce_scatter_done_handle, local_dev_);
    }
}

void RingSession::copy_result_out() {
    check_cuda_ipc(cudaStreamSynchronize(stream_),
                   "final cudaStreamSynchronize", my_rank_, "final", -1);
    check_cuda_ipc(cudaMemcpy(recvbuff_, commbuff_, count_ * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(final recvbuff)", my_rank_, "final", -1);
}

void RingSession::run() {
    reduce_scatter();
    all_gather();
    copy_result_out();
}

void RingSession::teardown() {
    for (int r = 0;
         r < world_size_ && r < static_cast<int>(remote_ptrs_.size()); r++) {
        if (r != my_rank_ && remote_ptrs_[r]) {
            ipc_close_handle(remote_ptrs_[r]);
        }
    }
    ipc_close_event(remote_init_ready_);
    for (cudaEvent_t event : remote_rs_events_)
        ipc_close_event(event);
    for (cudaEvent_t event : remote_ag_events_)
        ipc_close_event(event);
    if (init_events_created_)
        ipc_destroy_events(init_events_);
    for (IpcEventSet& events : rs_events_)
        ipc_destroy_events(events);
    for (IpcEventSet& events : ag_events_)
        ipc_destroy_events(events);
    if (stream_)
        cudaStreamDestroy(stream_);
    if (tmp_)
        cudaFree(tmp_);
    if (commbuff_)
        cudaFree(commbuff_);
}

void synchronized_copy(float* dst, float* src, size_t count, int device) {
    cudaSetDevice(device);
    cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice);
}

}  // namespace

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
    if (world_size <= 0)
        throw std::runtime_error("world_size must be >= 1");
    if (my_rank < 0 || my_rank >= world_size)
        throw std::runtime_error("my_rank is outside world_size");
    if (count % static_cast<size_t>(world_size) != 0)
        throw std::runtime_error("count must be divisible by world_size");

    RingSession session(my_rank, world_size, sendbuff, recvbuff, count,
                        ipc_dir);
    session.initialize();
    session.run();
}
