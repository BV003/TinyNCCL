#ifndef TINYNCCL_CUDA_CHECK_H
#define TINYNCCL_CUDA_CHECK_H

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Throw std::runtime_error with `context` and the CUDA error string when `err`
// indicates failure. Shared by all translation units so error handling and
// message formatting stay uniform.
inline void check_cuda(cudaError_t err, const std::string& context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(context + ": " + cudaGetErrorString(err));
    }
}

#endif  // TINYNCCL_CUDA_CHECK_H
