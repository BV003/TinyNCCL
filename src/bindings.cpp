#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernels.h"

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
    tiny_copy_peer_kernel(
        dst.data_ptr(),
        dst_dev,
        src.data_ptr(),
        src_dev,
        src.numel() * src.element_size(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
    m.def("tiny_copy_peer", &tiny_copy_peer,
          "Copy data between CUDA devices: dst = src");
}
