#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <string>
#include <vector>

#include "kernels.h"
#include "transport.h"
#include "ring_allreduce.h"

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

// Shared checks for the two multi-process allreduce entry points.
void require_allreduce_inputs(const torch::Tensor& sendbuff,
                              const torch::Tensor& recvbuff,
                              size_t world_size) {
    require_float32(sendbuff, "sendbuff");
    require_float32(recvbuff, "recvbuff");
    require_cuda(sendbuff, "sendbuff");
    require_cuda(recvbuff, "recvbuff");
    require_same_shape(sendbuff, recvbuff,
                       "sendbuff and recvbuff must have same shape");
    require_divisible(sendbuff.numel(), world_size, "numel");
}

}  // namespace

void tiny_reduce_sum(torch::Tensor dst, torch::Tensor a, torch::Tensor b) {
    require_same_shape(a, b, "a and b must have same shape");
    require_float32(a, "a");
    dst.resize_(a.sizes());  // output shape follows the inputs
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const std::size_t n = static_cast<std::size_t>(a.numel());
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
    m.def("tiny_copy_peer", &tiny_copy_peer,
          "Copy data between CUDA devices: dst = src");

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
                require_allreduce_inputs(
                    sendbuff, recvbuff, static_cast<size_t>(self.world_size()));
                const size_t count = sendbuff.numel();
                TORCH_CHECK(count <= self.max_count(),
                            "count exceeds communicator max_count");
                self.allreduce(sendbuff.data_ptr<float>(),
                               recvbuff.data_ptr<float>(), count);
            },
            pybind11::arg("sendbuff"), pybind11::arg("recvbuff"))
        .def_property_readonly("world_size", &RingCommunicator::world_size)
        .def_property_readonly("max_count", &RingCommunicator::max_count);
}
