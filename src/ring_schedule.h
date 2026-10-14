#ifndef TINYNCCL_RING_SCHEDULE_H
#define TINYNCCL_RING_SCHEDULE_H

// Pure host-side index and signal math for the ring AllReduce schedule.
//
// Kept free of CUDA headers so it can be compiled and exhaustively checked on a
// CPU-only machine (tests/test_ring_schedule.cpp).
//
// Every rank pulls from its predecessor prev = (rank - 1) mod n.
//
// Reduce-scatter step s (s = 0 .. n-2): rank r adds prev's chunk
//   rs_chunk(r, s) = (r - s) mod n
// into its own copy. After the last step rank r holds the fully reduced chunk
//   owned_chunk(r) = rs_chunk(r, n-2) = (r + 2) mod n.
//
// All-gather step s (s = 0 .. n-2): rank r copies prev's chunk
//   ag_chunk(r, s) = (r - s + 1) mod n
// At step 0 that is owned_chunk(prev); at step s it is the chunk prev received
// at step s-1. (For n == 2 this coincides with (r - s - 1) mod n, which is why
// an off-by-two here only shows up with 3 or more ranks.)

#include <cstdint>

namespace tinynccl {
namespace ring {

inline int wrap(int value, int n) {
    const int m = value % n;
    return m < 0 ? m + n : m;
}

inline int prev_rank(int rank, int n) {
    return wrap(rank - 1, n);
}

inline int rs_chunk(int rank, int step, int n) {
    return wrap(rank - step, n);
}

inline int owned_chunk(int rank, int n) {
    return wrap(rank + 2, n);
}

inline int ag_chunk(int rank, int step, int n) {
    return wrap(rank - step + 1, n);
}

// Persistent-communicator signals. A call emits 1 + 2*steps signals in this
// order: input copied, then the reduce-scatter steps, then the all-gather
// steps. After signal k the counter reaches base + k + 1, so a rank waits for
// the done-signal of the step it depends on.
inline int signals_per_call(int steps) {
    return 1 + 2 * steps;
}

inline uint32_t input_ready_signal(uint32_t base) {
    return base + 1;
}

inline uint32_t rs_step_signal(uint32_t base, int step) {
    return base + static_cast<uint32_t>(step) + 2;
}

inline uint32_t ag_step_signal(uint32_t base, int steps, int step) {
    return base + static_cast<uint32_t>(steps + step) + 2;
}

inline uint32_t rs_step_dependency(uint32_t base, int step) {
    return step == 0 ? input_ready_signal(base)
                     : rs_step_signal(base, step - 1);
}

inline uint32_t ag_step_dependency(uint32_t base, int steps, int step) {
    return step == 0 ? rs_step_signal(base, steps - 1)
                     : ag_step_signal(base, steps, step - 1);
}

}  // namespace ring
}  // namespace tinynccl

#endif  // TINYNCCL_RING_SCHEDULE_H
