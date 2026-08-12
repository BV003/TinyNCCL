#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include "kernels.h"

void tiny_reduce_sum(torch::Tensor dst, torch::Tensor a, torch::Tensor b) {
    std::cout << "[1] checking sizes..." << std::endl;
    TORCH_CHECK(a.sizes().vec() == b.sizes().vec(), "a and b must have same shape");
    std::cout << "[2] checking dtype..." << std::endl;
    TORCH_CHECK(a.dtype() == torch::kFloat32, "only float32 supported");
    std::cout << "[3] resize dst..." << std::endl;
    dst.resize_(a.sizes());
    std::cout << "[4] getting stream..." << std::endl;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    std::cout << "[5] numel..." << std::endl;
    int n = a.numel();
    std::cout << "[6] launching kernel (n=" << n << ")..." << std::endl;
    tiny_reduce_sum_kernel(
        dst.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        n, stream);
    std::cout << "[7] done." << std::endl;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_reduce_sum", &tiny_reduce_sum,
          "Element-wise float32 sum: dst = a + b");
}