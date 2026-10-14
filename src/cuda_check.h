#ifndef TINYNCCL_CUDA_CHECK_H
#define TINYNCCL_CUDA_CHECK_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Throw with `context` and the CUDA error string when `err` is not success.
inline void check_cuda(cudaError_t err, const std::string& context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(context + ": " + cudaGetErrorString(err));
    }
}

#endif  // TINYNCCL_CUDA_CHECK_H
