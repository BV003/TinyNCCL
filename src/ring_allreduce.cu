#include "ring_allreduce.h"

#include <cuda.h>

#include <cstddef>
#include <limits>
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

void synchronized_copy(float* dst, float* src, size_t count, int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    check_cuda(
        cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice),
        "cudaMemcpy");
}

using StreamWaitValue32 = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t,
                                       unsigned int);
using StreamWriteValue32 = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t,
                                        unsigned int);

struct MemOps {
    StreamWaitValue32 wait32 = nullptr;
    StreamWriteValue32 write32 = nullptr;
};

// Stream memory ops live in the driver API; resolve them at runtime so the
// extension does not need to link against libcuda directly.
const MemOps& mem_ops() {
    static const MemOps ops = [] {
        MemOps result;
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSuccess;
        if (cudaGetDriverEntryPoint("cuStreamWaitValue32", &fn,
                                    cudaEnableDefault,
                                    &status) != cudaSuccess ||
            fn == nullptr) {
            throw std::runtime_error("failed to resolve cuStreamWaitValue32");
        }
        result.wait32 = reinterpret_cast<StreamWaitValue32>(fn);
        fn = nullptr;
        if (cudaGetDriverEntryPoint("cuStreamWriteValue32", &fn,
                                    cudaEnableDefault,
                                    &status) != cudaSuccess ||
            fn == nullptr) {
            throw std::runtime_error("failed to resolve cuStreamWriteValue32");
        }
        result.write32 = reinterpret_cast<StreamWriteValue32>(fn);
        return result;
    }();
    return ops;
}

void ensure_mem_ops_available() {
    (void)mem_ops();
}

// A call emits 1 + 2*steps signals in this order: input copied, then the
// reduce-scatter steps, then the all-gather steps. After signal k the counter
// reaches base + k + 1, so a rank waits for the done-signal of the previous
// dependency.
uint32_t input_ready_signal(uint32_t base) {
    return base + 1;
}

uint32_t rs_step_signal(uint32_t base, int step) {
    return base + step + 2;
}

uint32_t ag_step_signal(uint32_t base, int steps, int step) {
    return base + steps + step + 2;
}

uint32_t rs_step_dependency(uint32_t base, int step) {
    return step == 0 ? input_ready_signal(base)
                     : rs_step_signal(base, step - 1);
}

uint32_t ag_step_dependency(uint32_t base, int steps, int step) {
    return step == 0 ? rs_step_signal(base, steps - 1)
                     : ag_step_signal(base, steps, step - 1);
}

}  // namespace

// ---------------------------------------------------------------------------
// RingCommunicator
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
    ensure_mem_ops_available();

    const size_t max_chunk = max_count_ / static_cast<size_t>(world_size_);
    check_cuda_ipc(cudaGetDevice(&local_dev_), "cudaGetDevice", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaStreamCreate(&stream_), "cudaStreamCreate", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaMalloc(&scratch_, max_chunk * sizeof(float)),
                   "cudaMalloc(scratch)", my_rank_, "init", -1);
    check_cuda_ipc(cudaMalloc(&commbuff_, max_count_ * sizeof(float)),
                   "cudaMalloc(commbuff)", my_rank_, "init", -1);
    check_cuda_ipc(
        cudaMalloc(reinterpret_cast<void**>(&counter_), sizeof(uint32_t)),
        "cudaMalloc(counter)", my_rank_, "init", -1);
    check_cuda_ipc(cudaMemset(counter_, 0, sizeof(uint32_t)),
                   "cudaMemset(counter)", my_rank_, "init", -1);
}

void RingCommunicator::setup_remote_memory() {
    handles_ = ipc_exchange_handles(my_rank_, world_size_, commbuff_,
                                    max_count_ * sizeof(float), local_dev_,
                                    ipc_dir_ + "/comm");
    std::vector<IpcHandle> counter_handles =
        ipc_exchange_handles(my_rank_, world_size_, counter_, sizeof(uint32_t),
                             local_dev_, ipc_dir_ + "/counter");

    remote_ptrs_.assign(world_size_, nullptr);
    remote_counters_.assign(world_size_, nullptr);
    for (int r = 0; r < world_size_; r++) {
        remote_ptrs_[r] =
            (r == my_rank_)
                ? commbuff_  // our own buffer, not IPC-mapped
                : static_cast<float*>(ipc_open_handle(handles_[r], local_dev_));
        remote_counters_[r] = (r == my_rank_)
                                  ? counter_
                                  : static_cast<uint32_t*>(ipc_open_handle(
                                        counter_handles[r], local_dev_));
    }
}

void RingCommunicator::signal(uint32_t value) {
    const CUresult result =
        mem_ops().write32(reinterpret_cast<CUstream>(stream_),
                          reinterpret_cast<CUdeviceptr>(counter_), value,
                          CU_STREAM_WRITE_VALUE_DEFAULT);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("cuStreamWriteValue32 failed with code " +
                                 std::to_string(static_cast<int>(result)));
    }
}

void RingCommunicator::wait_for(int rank, uint32_t value, const char* what) {
    const CUresult result =
        mem_ops().wait32(reinterpret_cast<CUstream>(stream_),
                         reinterpret_cast<CUdeviceptr>(remote_counters_[rank]),
                         value, CU_STREAM_WAIT_VALUE_GEQ);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("cuStreamWaitValue32 (") + what +
                                 ") failed with code " +
                                 std::to_string(static_cast<int>(result)));
    }
}

void RingCommunicator::allreduce(float* sendbuff, float* recvbuff,
                                 size_t count) {
    if (count == 0 || count > max_count_ ||
        count % static_cast<size_t>(world_size_) != 0) {
        throw std::runtime_error(
            "count must be positive, <= max_count and divisible by world_size");
    }

    const size_t per_call = static_cast<size_t>(signals_per_call());
    if (epoch_ >= (std::numeric_limits<uint32_t>::max)() / per_call) {
        throw std::runtime_error(
            "signal counter would overflow; create a new communicator");
    }
    const uint32_t base = static_cast<uint32_t>(epoch_ * per_call);
    epoch_++;

    // Wait until every rank has finished the previous call before we reuse the
    // communication buffer. (For the first call the counters start at 0.)
    for (int r = 0; r < world_size_; r++) {
        if (r != my_rank_)
            wait_for(r, base, "epoch barrier");
    }

    check_cuda_ipc(cudaMemcpyAsync(commbuff_, sendbuff, count * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_),
                   "cudaMemcpy(init)", my_rank_, "init", -1);
    signal(input_ready_signal(base));

    reduce_scatter(count, base);
    all_gather(count, base);

    check_cuda_ipc(cudaStreamSynchronize(stream_),
                   "final cudaStreamSynchronize", my_rank_, "final", -1);
    check_cuda_ipc(cudaMemcpy(recvbuff, commbuff_, count * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cudaMemcpy(final recvbuff)", my_rank_, "final", -1);
}

void RingCommunicator::reduce_scatter(size_t count, uint32_t base) {
    const size_t chunk = count / static_cast<size_t>(world_size_);
    const size_t chunk_bytes = chunk * sizeof(float);

    for (int step = 0; step < steps_; step++) {
        const int idx = (my_rank_ - step + world_size_) % world_size_;

        wait_for(prev_, rs_step_dependency(base, step), "reduce-scatter");
        transport_copy_ipc(scratch_, local_dev_,
                           remote_ptrs_[prev_] + idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);

        tiny_reduce_sum_kernel(commbuff_ + idx * chunk, commbuff_ + idx * chunk,
                               scratch_, chunk, stream_);
        check_cuda_ipc(cudaGetLastError(), "kernel launch", my_rank_, "RS",
                       step);

        signal(rs_step_signal(base, step));
    }
}

void RingCommunicator::all_gather(size_t count, uint32_t base) {
    const size_t chunk = count / static_cast<size_t>(world_size_);
    const size_t chunk_bytes = chunk * sizeof(float);

    for (int step = 0; step < steps_; step++) {
        const int chunk_idx = (my_rank_ - step - 1 + world_size_) % world_size_;

        wait_for(prev_, ag_step_dependency(base, steps_, step), "all-gather");
        transport_copy_ipc(scratch_, local_dev_,
                           remote_ptrs_[prev_] + chunk_idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);
        check_cuda_ipc(
            cudaMemcpyAsync(commbuff_ + chunk_idx * chunk, scratch_,
                            chunk_bytes, cudaMemcpyDeviceToDevice, stream_),
            "cudaMemcpy(AG local)", my_rank_, "AG", step);

        signal(ag_step_signal(base, steps_, step));
    }
}

void RingCommunicator::teardown() {
    // Best-effort: make sure no kernel still touches the buffers before we free
    // them (a failed allreduce can leave work queued on the stream).
    if (stream_)
        cudaStreamSynchronize(stream_);

    for (int r = 0;
         r < world_size_ && r < static_cast<int>(remote_ptrs_.size()); r++) {
        if (r != my_rank_ && remote_ptrs_[r]) {
            ipc_close_handle(remote_ptrs_[r]);
        }
    }
    for (int r = 0;
         r < world_size_ && r < static_cast<int>(remote_counters_.size());
         r++) {
        if (r != my_rank_ && remote_counters_[r]) {
            ipc_close_handle(remote_counters_[r]);
        }
    }
    remote_ptrs_.clear();
    remote_counters_.clear();
    handles_.clear();

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (scratch_) {
        cudaFree(scratch_);
        scratch_ = nullptr;
    }
    if (commbuff_) {
        cudaFree(commbuff_);
        commbuff_ = nullptr;
    }
    if (counter_) {
        cudaFree(counter_);
        counter_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

namespace {

// Owns the per-device streams and scratch buffers of the single-process path so
// that a partial setup failure still releases what was already acquired.
class DeviceStreamBuffers {
public:
    DeviceStreamBuffers(int n_gpus, size_t chunk_bytes)
        : streams_(static_cast<size_t>(n_gpus), nullptr),
          scratch_(static_cast<size_t>(n_gpus), nullptr) {
        try {
            for (int i = 0; i < n_gpus; i++) {
                check_cuda(cudaSetDevice(i), "cudaSetDevice");
                check_cuda(cudaStreamCreate(&streams_[i]), "cudaStreamCreate");
                check_cuda(cudaMalloc(&scratch_[i], chunk_bytes), "cudaMalloc");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~DeviceStreamBuffers() { release(); }

    DeviceStreamBuffers(const DeviceStreamBuffers&) = delete;
    DeviceStreamBuffers& operator=(const DeviceStreamBuffers&) = delete;

    cudaStream_t stream(int i) const { return streams_[i]; }
    float* scratch(int i) const { return scratch_[i]; }

private:
    void release() {
        for (size_t i = 0; i < streams_.size(); i++) {
            if (streams_[i])
                cudaStreamDestroy(streams_[i]);
            if (scratch_[i])
                cudaFree(scratch_[i]);
        }
    }

    std::vector<cudaStream_t> streams_;
    std::vector<float*> scratch_;
};

void impl_reduce_scatter(float** recv, int n_gpus, size_t chunk,
                         size_t chunk_bytes,
                         const DeviceStreamBuffers& buffers) {
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            const int prev = (i - 1 + n_gpus) % n_gpus;
            const int idx = (i - step + n_gpus) % n_gpus;

            check_cuda(cudaSetDevice(i), "cudaSetDevice");
            transport_copy(buffers.scratch(i), i, recv[prev] + idx * chunk,
                           prev, chunk_bytes, buffers.stream(i));

            tiny_reduce_sum_kernel(recv[i] + idx * chunk, recv[i] + idx * chunk,
                                   buffers.scratch(i), chunk,
                                   buffers.stream(i));
        }

        for (int i = 0; i < n_gpus; i++) {
            check_cuda(cudaSetDevice(i), "cudaSetDevice");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }
}

void impl_all_gather(float** recv, int n_gpus, size_t chunk, size_t chunk_bytes,
                     const DeviceStreamBuffers& buffers) {
    for (int step = 0; step < n_gpus - 1; step++) {
        for (int i = 0; i < n_gpus; i++) {
            const int own_chunk = i;
            const int send_dst = (i + 1) % n_gpus;

            check_cuda(cudaSetDevice(send_dst), "cudaSetDevice");
            transport_copy(recv[send_dst] + own_chunk * chunk, send_dst,
                           recv[i] + own_chunk * chunk, i, chunk_bytes,
                           buffers.stream(send_dst));
        }

        for (int dev = 0; dev < n_gpus; dev++) {
            check_cuda(cudaSetDevice(dev), "cudaSetDevice");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }
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

    DeviceStreamBuffers buffers(n_gpus, chunk_bytes);

    for (int i = 0; i < n_gpus; i++) {
        synchronized_copy(recv[i], send[i], count, i);
    }

    impl_reduce_scatter(recv, n_gpus, chunk, chunk_bytes, buffers);
    impl_all_gather(recv, n_gpus, chunk, chunk_bytes, buffers);
}
