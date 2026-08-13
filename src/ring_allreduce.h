#ifndef TINYNCCL_RING_ALLREDUCE_H
#define TINYNCCL_RING_ALLREDUCE_H

#include <cuda_runtime.h>
#include <cstddef>

// Ring AllReduce (single process, multiple GPUs).
//
// send[i] / recv[i] are device pointers on GPU i (i = 0 .. n_gpus-1).
// Each GPU contributes `count` float32 elements; the result (element-wise sum
// across all GPUs) is written to recv[i] for every i.
//
// Requires count % n_gpus == 0 (chunking). Serial (synchronous) version.
void tiny_ring_allreduce_sum(
    float** send, float** recv,
    size_t count, int n_gpus);

#endif
