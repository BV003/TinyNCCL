#include "ring_allreduce.h"

#include <cuda.h>

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
    cudaSetDevice(device);
    cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice);
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

}  // namespace

// ---------------------------------------------------------------------------
// RingCommunicator
//
// Everything persistent is set up once. Synchronization is GPU-side: each rank
// owns a monotonically increasing 32-bit counter in device memory; a rank
// signals progress with cuStreamWriteValue32 and waits on its neighbour with
// cuStreamWaitValue32. No files, events or host round-trips remain in the
// steady-state path.
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
    const size_t max_chunk = max_count_ / static_cast<size_t>(world_size_);

    (void)mem_ops();  // fail fast if the driver entry points are unavailable
    check_cuda_ipc(cudaGetDevice(&local_dev_), "cudaGetDevice", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaStreamCreate(&stream_), "cudaStreamCreate", my_rank_,
                   "init", -1);
    check_cuda_ipc(cudaMalloc(&tmp_, max_chunk * sizeof(float)),
                   "cudaMalloc(tmp)", my_rank_, "init", -1);
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

    const uint32_t base =
        static_cast<uint32_t>(epoch_ * static_cast<size_t>(signals_per_call()));
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
    signal(base + 1);

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

        wait_for(prev_, base + step + 1, "reduce-scatter");
        transport_copy_ipc(tmp_, local_dev_, remote_ptrs_[prev_] + idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);

        tiny_reduce_sum_kernel(commbuff_ + idx * chunk, commbuff_ + idx * chunk,
                               tmp_, static_cast<int>(chunk), stream_);
        check_cuda_ipc(cudaGetLastError(), "kernel launch", my_rank_, "RS",
                       step);

        signal(base + step + 2);
    }
}

void RingCommunicator::all_gather(size_t count, uint32_t base) {
    const size_t chunk = count / static_cast<size_t>(world_size_);
    const size_t chunk_bytes = chunk * sizeof(float);

    for (int step = 0; step < steps_; step++) {
        const int chunk_idx = (my_rank_ - step - 1 + world_size_) % world_size_;

        wait_for(prev_, base + steps_ + step + 1, "all-gather");
        transport_copy_ipc(tmp_, local_dev_,
                           remote_ptrs_[prev_] + chunk_idx * chunk,
                           handles_[prev_].device, chunk_bytes, stream_);
        check_cuda_ipc(
            cudaMemcpyAsync(commbuff_ + chunk_idx * chunk, tmp_, chunk_bytes,
                            cudaMemcpyDeviceToDevice, stream_),
            "cudaMemcpy(AG local)", my_rank_, "AG", step);

        signal(base + steps_ + step + 2);
    }
}

void RingCommunicator::teardown() {
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
    if (stream_)
        cudaStreamDestroy(stream_);
    if (tmp_)
        cudaFree(tmp_);
    if (commbuff_)
        cudaFree(commbuff_);
    if (counter_)
        cudaFree(counter_);
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
