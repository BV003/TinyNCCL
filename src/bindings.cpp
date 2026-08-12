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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
}
