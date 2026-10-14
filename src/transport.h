#ifndef TINYNCCL_TRANSPORT_H
#define TINYNCCL_TRANSPORT_H

#include <cuda_runtime.h>
#include <cstddef>

void transport_copy(void* dst, int dst_dev, const void* src, int src_dev,
                    size_t nbytes, cudaStream_t stream);

// Copy between pointers that are already mapped into this process (for example
// via CUDA IPC); both src and dst must be valid in the current address space.
void transport_copy_ipc(void* dst, int dst_dev, const void* src, int src_dev,
                        size_t nbytes, cudaStream_t stream);

#endif
