#ifndef TINYNCCL_TRANSPORT_H
#define TINYNCCL_TRANSPORT_H

#include <cuda_runtime.h>
#include <cstddef>

void transport_copy(
    void* dst, int dst_dev,
    const void* src, int src_dev,
    size_t nbytes, cudaStream_t stream);

#endif
