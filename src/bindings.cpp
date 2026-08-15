#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <string>
#include "kernels.h"
#include "transport.h"
#include "ring_allreduce.h"
#include "ipc.h"

void tiny_reduce_sum(torch::Tensor dst, torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes().vec() == b.sizes().vec(), "a and b must have same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "only float32 supported");
    dst.resize_(a.sizes());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int n = a.numel();
    tiny_reduce_sum_kernel(
        dst.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        n, stream);
}

void tiny_copy_peer(torch::Tensor dst, torch::Tensor src) {
    TORCH_CHECK(src.sizes().vec() == dst.sizes().vec(), "src and dst must have same shape");
    TORCH_CHECK(src.dtype() == dst.dtype(), "src and dst must have same dtype");
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "both tensors must be CUDA");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int src_dev = src.get_device();
    int dst_dev = dst.get_device();
    transport_copy(
        dst.data_ptr(),
        dst_dev,
        src.data_ptr(),
        src_dev,
        src.numel() * src.element_size(),
        stream);
}

void tiny_ring_allreduce_sum(std::vector<torch::Tensor> sendbuffs,
                             std::vector<torch::Tensor> recvbuffs) {
    int n = static_cast<int>(sendbuffs.size());
    TORCH_CHECK(n >= 1, "need at least 1 GPU");
    TORCH_CHECK(static_cast<int>(recvbuffs.size()) == n,
                "sendbuffs and recvbuffs must have the same size");
    TORCH_CHECK(sendbuffs[0].dtype() == torch::kFloat32, "only float32 supported");
    size_t count = sendbuffs[0].numel();
    TORCH_CHECK(count % static_cast<size_t>(n) == 0,
                "numel must be divisible by the number of GPUs");

    std::vector<float*> send(n), recv(n);
    for (int i = 0; i < n; i++) {
        TORCH_CHECK(sendbuffs[i].is_cuda() && recvbuffs[i].is_cuda(),
                    "tensors must be CUDA");
        TORCH_CHECK(sendbuffs[i].get_device() == i,
                    "sendbuffs must be ordered by device index");
        TORCH_CHECK(recvbuffs[i].get_device() == i,
                    "recvbuffs must be ordered by device index");
        TORCH_CHECK(sendbuffs[i].numel() == static_cast<int64_t>(count) &&
                    recvbuffs[i].numel() == static_cast<int64_t>(count),
                    "all tensors must have the same size");
        send[i] = sendbuffs[i].data_ptr<float>();
        recv[i] = recvbuffs[i].data_ptr<float>();
    }

    tiny_ring_allreduce_sum_impl(send.data(), recv.data(), count, n);
}

void ipc_allreduce_wrapper(
    int64_t my_rank, int64_t world_size,
    torch::Tensor sendbuff, torch::Tensor recvbuff,
    std::string ipc_dir)
{
    TORCH_CHECK(sendbuff.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(recvbuff.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(sendbuff.is_cuda() && recvbuff.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(sendbuff.sizes() == recvbuff.sizes(), "sendbuff and recvbuff must have same shape");

    size_t count = sendbuff.numel();
    TORCH_CHECK(count % static_cast<size_t>(world_size) == 0,
                "numel must be divisible by world_size");

    tiny_ring_allreduce_sum_ipc(
        static_cast<int>(my_rank),
        static_cast<int>(world_size),
        sendbuff.data_ptr<float>(),
        recvbuff.data_ptr<float>(),
        count,
        ipc_dir);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
    m.def("tiny_copy_peer", &tiny_copy_peer,
          "Copy data between CUDA devices: dst = src");
    m.def("tiny_ring_allreduce_sum", &tiny_ring_allreduce_sum,
          "Ring AllReduce (sum) across GPUs in a single process");
    m.def("tiny_ring_allreduce_sum_ipc", &ipc_allreduce_wrapper,
          "Ring AllReduce (sum) across GPUs in multiple processes via CUDA IPC");
}
