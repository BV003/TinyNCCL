#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <string>
#include <vector>

#include "kernels.h"
#include "transport.h"
#include "ring_allreduce.h"
#include "ipc.h"

namespace {

void require_same_shape(const torch::Tensor& a, const torch::Tensor& b,
                        const char* message) {
    TORCH_CHECK(a.sizes().vec() == b.sizes().vec(), message);
}

void require_float32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must be float32");
}

void require_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void require_divisible(size_t count, size_t divisor, const char* name) {
    TORCH_CHECK(count % divisor == 0, name, " must be divisible by ", divisor);
}

}  // namespace

void tiny_reduce_sum(torch::Tensor dst, torch::Tensor a, torch::Tensor b) {
    require_same_shape(a, b, "a and b must have same shape");
    require_float32(a, "a");
    dst.resize_(a.sizes());  // output shape follows the inputs
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int n = a.numel();
    tiny_reduce_sum_kernel(dst.data_ptr<float>(), a.data_ptr<float>(),
                           b.data_ptr<float>(), n, stream);
}

void tiny_copy_peer(torch::Tensor dst, torch::Tensor src) {
    require_same_shape(src, dst, "src and dst must have same shape");
    TORCH_CHECK(src.dtype() == dst.dtype(), "src and dst must have same dtype");
    require_cuda(src, "src");
    require_cuda(dst, "dst");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int src_dev = src.get_device();
    int dst_dev = dst.get_device();
    transport_copy(dst.data_ptr(), dst_dev, src.data_ptr(), src_dev,
                   src.numel() * src.element_size(), stream);
}

void tiny_ring_allreduce_sum(std::vector<torch::Tensor> sendbuffs,
                             std::vector<torch::Tensor> recvbuffs) {
    const int n = static_cast<int>(sendbuffs.size());
    TORCH_CHECK(n >= 1, "need at least 1 GPU");
    TORCH_CHECK(static_cast<int>(recvbuffs.size()) == n,
                "sendbuffs and recvbuffs must have the same size");
    require_float32(sendbuffs[0], "sendbuffs");

    const size_t count = sendbuffs[0].numel();
    require_divisible(count, static_cast<size_t>(n), "numel");

    std::vector<float*> send(n), recv(n);
    for (int i = 0; i < n; i++) {
        require_cuda(sendbuffs[i], "sendbuffs");
        require_cuda(recvbuffs[i], "recvbuffs");
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

void ipc_allreduce_wrapper(int64_t my_rank, int64_t world_size,
                           torch::Tensor sendbuff, torch::Tensor recvbuff,
                           std::string ipc_dir) {
    require_float32(sendbuff, "sendbuff");
    require_float32(recvbuff, "recvbuff");
    require_cuda(sendbuff, "sendbuff");
    require_cuda(recvbuff, "recvbuff");
    require_same_shape(sendbuff, recvbuff,
                       "sendbuff and recvbuff must have same shape");

    const size_t count = sendbuff.numel();
    require_divisible(count, static_cast<size_t>(world_size), "numel");

    tiny_ring_allreduce_sum_ipc(
        static_cast<int>(my_rank), static_cast<int>(world_size),
        sendbuff.data_ptr<float>(), recvbuff.data_ptr<float>(), count, ipc_dir);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
    m.def("tiny_copy_peer", &tiny_copy_peer,
          "Copy data between CUDA devices: dst = src");
    m.def("tiny_ring_allreduce_sum", &tiny_ring_allreduce_sum,
          "Ring AllReduce (sum) across GPUs in a single process");
    m.def(
        "tiny_ring_allreduce_sum_ipc", &ipc_allreduce_wrapper,
        "Ring AllReduce (sum) across GPUs in multiple processes via CUDA IPC");

    pybind11::class_<RingCommunicator>(m, "RingCommunicator")
        .def(pybind11::init<int, int, size_t, std::string>(),
             pybind11::arg("my_rank"), pybind11::arg("world_size"),
             pybind11::arg("max_count"), pybind11::arg("ipc_dir"),
             "Persistent CUDA IPC communicator: set up once, reuse many "
             "times")
        .def(
            "allreduce",
            [](RingCommunicator& self, torch::Tensor sendbuff,
               torch::Tensor recvbuff) {
                require_float32(sendbuff, "sendbuff");
                require_float32(recvbuff, "recvbuff");
                require_cuda(sendbuff, "sendbuff");
                require_cuda(recvbuff, "recvbuff");
                require_same_shape(sendbuff, recvbuff,
                                   "sendbuff and recvbuff must have same "
                                   "shape");
                const size_t count = sendbuff.numel();
                require_divisible(count, static_cast<size_t>(self.world_size()),
                                  "numel");
                TORCH_CHECK(count <= self.max_count(),
                            "count exceeds communicator max_count");
                self.allreduce(sendbuff.data_ptr<float>(),
                               recvbuff.data_ptr<float>(), count);
            },
            pybind11::arg("sendbuff"), pybind11::arg("recvbuff"))
        .def_property_readonly("world_size", &RingCommunicator::world_size)
        .def_property_readonly("max_count", &RingCommunicator::max_count);
}
