// CPU-only check of the ring AllReduce schedule in src/ring_schedule.h.
//
// No GPU needed: each rank's work is modelled as the same sequence of stream
// operations RingCommunicator / tiny_ring_allreduce_sum_impl issue (waits,
// copies, reductions, signals), and a random scheduler interleaves the ranks
// in any order the GPU-side dependencies allow. Copies and reductions are
// split into separate read and write operations so write-after-read races
// between neighbours are visible too.
//
// Build and run:
//   c++ -std=c++17 -O2 -Isrc tests/test_ring_schedule.cpp -o /tmp/ring_schedule
//   /tmp/ring_schedule

#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "ring_schedule.h"

namespace ring = tinynccl::ring;

namespace {

using Value = int64_t;

int g_failures = 0;

void fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    g_failures++;
}

// One queued stream operation: runnable() gates it (a cuStreamWaitValue32 or
// the host-side sync), run() performs it.
struct Op {
    std::function<bool()> runnable;
    std::function<void()> run;
};

struct Rank {
    std::vector<Value> send;
    std::vector<Value> comm;  // the IPC-shared communication buffer
    std::vector<Value> recv;
    Value scratch = 0;
    uint32_t counter = 0;
    std::vector<Op> ops;
    size_t next = 0;
};

// Interleave the ranks' op queues at random. Returns false on deadlock.
bool run_random(std::vector<Rank>& ranks, std::mt19937& rng) {
    for (;;) {
        std::vector<int> ready;
        bool done = true;
        for (int r = 0; r < static_cast<int>(ranks.size()); r++) {
            Rank& rk = ranks[r];
            if (rk.next < rk.ops.size()) {
                done = false;
                if (rk.ops[rk.next].runnable())
                    ready.push_back(r);
            }
        }
        if (done)
            return true;
        if (ready.empty())
            return false;
        Rank& rk = ranks[ready[rng() % ready.size()]];
        rk.ops[rk.next++].run();
    }
}

Op always(std::function<void()> run) {
    return Op{[] { return true; }, std::move(run)};
}

// ---------------------------------------------------------------------------
// Persistent communicator: mirrors RingCommunicator::allreduce. Each chunk is
// a single element; that is enough to track which contributions it holds.
// ---------------------------------------------------------------------------

void enqueue_persistent_call(std::vector<Rank>& ranks, int n, uint32_t base) {
    const int steps = n - 1;
    for (int r = 0; r < n; r++) {
        Rank& me = ranks[r];
        Rank& prev = ranks[ring::prev_rank(r, n)];
        auto& ops = me.ops;

        // Epoch barrier: every other rank finished the previous call.
        for (int o = 0; o < n; o++) {
            if (o == r)
                continue;
            Rank* other = &ranks[o];
            ops.push_back(
                Op{[other, base] { return other->counter >= base; }, [] {}});
        }
        ops.push_back(always([&me, n] {
            for (int c = 0; c < n; c++)
                me.comm[c] = me.send[c];
        }));
        ops.push_back(always(
            [&me, base] { me.counter = ring::input_ready_signal(base); }));

        for (int s = 0; s < steps; s++) {
            const int idx = ring::rs_chunk(r, s, n);
            const uint32_t dep = ring::rs_step_dependency(base, s);
            ops.push_back(
                Op{[&prev, dep] { return prev.counter >= dep; }, [] {}});
            ops.push_back(
                always([&me, &prev, idx] { me.scratch = prev.comm[idx]; }));
            ops.push_back(always([&me, idx] { me.comm[idx] += me.scratch; }));
            ops.push_back(always([&me, base, s] {
                me.counter = ring::rs_step_signal(base, s);
            }));
        }
        for (int s = 0; s < steps; s++) {
            const int idx = ring::ag_chunk(r, s, n);
            const uint32_t dep = ring::ag_step_dependency(base, steps, s);
            ops.push_back(
                Op{[&prev, dep] { return prev.counter >= dep; }, [] {}});
            ops.push_back(
                always([&me, &prev, idx] { me.scratch = prev.comm[idx]; }));
            ops.push_back(always([&me, idx] { me.comm[idx] = me.scratch; }));
            ops.push_back(always([&me, base, steps, s] {
                me.counter = ring::ag_step_signal(base, steps, s);
            }));
        }
        // cudaStreamSynchronize, then copy into the caller's recvbuff.
        ops.push_back(always([&me, n] {
            for (int c = 0; c < n; c++)
                me.recv[c] = me.comm[c];
        }));
    }
}

void check_persistent(int n, int calls, std::mt19937& rng) {
    std::uniform_int_distribution<Value> dist(-1000000, 1000000);
    std::vector<Rank> ranks(n);
    for (Rank& rk : ranks) {
        rk.comm.assign(n, 0);
        rk.recv.assign(n, 0);
        rk.send.assign(n, 0);
    }

    // Calls run one after another (as with a host-side barrier between them);
    // counters and buffers carry over, as they do on the GPU.
    const uint32_t per_call =
        static_cast<uint32_t>(ring::signals_per_call(n - 1));
    for (int call = 0; call < calls; call++) {
        std::vector<Value> expected(n, 0);
        for (Rank& rk : ranks) {
            for (int c = 0; c < n; c++) {
                rk.send[c] = dist(rng);
                expected[c] += rk.send[c];
            }
            rk.ops.clear();
            rk.next = 0;
        }
        enqueue_persistent_call(ranks, n,
                                static_cast<uint32_t>(call) * per_call);
        if (!run_random(ranks, rng)) {
            fail("persistent n=" + std::to_string(n) +
                 " call=" + std::to_string(call) + ": deadlock");
            return;
        }
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                if (ranks[r].recv[c] != expected[c]) {
                    fail("persistent n=" + std::to_string(n) + " call=" +
                         std::to_string(call) + " rank=" + std::to_string(r) +
                         " chunk=" + std::to_string(c) + ": got " +
                         std::to_string(ranks[r].recv[c]) + ", expected " +
                         std::to_string(expected[c]));
                    return;
                }
            }
        }
    }
}

// Back-to-back calls with overlap: rank r may race into call k+1 while its
// neighbours are still finishing call k. Only the epoch barrier prevents it
// from overwriting a buffer they are still reading.
void check_persistent_overlapping(int n, int calls, std::mt19937& rng) {
    std::uniform_int_distribution<Value> dist(-1000000, 1000000);
    const uint32_t per_call =
        static_cast<uint32_t>(ring::signals_per_call(n - 1));

    // One shared set of buffers and counters; each call gets its own input
    // and output snapshot via extra ops at the start and end of the call.
    std::vector<Rank> ranks(n);
    std::vector<std::vector<std::vector<Value>>> sends(
        calls, std::vector<std::vector<Value>>(n, std::vector<Value>(n)));
    std::vector<std::vector<std::vector<Value>>> recvs = sends;
    std::vector<std::vector<Value>> expected(calls, std::vector<Value>(n, 0));
    for (int k = 0; k < calls; k++)
        for (int r = 0; r < n; r++)
            for (int c = 0; c < n; c++) {
                sends[k][r][c] = dist(rng);
                expected[k][c] += sends[k][r][c];
            }

    for (Rank& rk : ranks) {
        rk.comm.assign(n, 0);
        rk.recv.assign(n, 0);
        rk.send.assign(n, 0);
    }
    for (int k = 0; k < calls; k++) {
        // Stage this call's input, run its ops, capture its output - all
        // queued, so different ranks can be in different calls.
        for (int r = 0; r < n; r++) {
            Rank* me = &ranks[r];
            const std::vector<Value>* src = &sends[k][r];
            me->ops.push_back(always([me, src] { me->send = *src; }));
        }
        enqueue_persistent_call(ranks, n, static_cast<uint32_t>(k) * per_call);
        for (int r = 0; r < n; r++) {
            Rank* me = &ranks[r];
            std::vector<Value>* dst = &recvs[k][r];
            me->ops.push_back(always([me, dst] { *dst = me->recv; }));
        }
    }
    if (!run_random(ranks, rng)) {
        fail("overlapping n=" + std::to_string(n) + ": deadlock");
        return;
    }
    for (int k = 0; k < calls; k++)
        for (int r = 0; r < n; r++)
            if (recvs[k][r] != expected[k]) {
                fail("overlapping n=" + std::to_string(n) +
                     " call=" + std::to_string(k) +
                     " rank=" + std::to_string(r) + ": wrong result");
                return;
            }
}

// ---------------------------------------------------------------------------
// Single-process path: mirrors tiny_ring_allreduce_sum_impl. Steps are
// separated by cudaDeviceSynchronize on every GPU; inside a step all GPUs run
// concurrently, so their reads and writes are interleaved at random.
// ---------------------------------------------------------------------------

void check_single_process(int n, std::mt19937& rng) {
    std::uniform_int_distribution<Value> dist(-1000000, 1000000);
    std::vector<std::vector<Value>> buf(n, std::vector<Value>(n));
    std::vector<Value> expected(n, 0);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            buf[r][c] = dist(rng);
            expected[c] += buf[r][c];
        }

    auto run_step = [&](const std::function<int(int)>& chunk_of, bool reduce) {
        std::vector<Rank> ranks(n);
        std::vector<Value> scratch(n, 0);
        for (int i = 0; i < n; i++) {
            const int prev = ring::prev_rank(i, n);
            const int idx = chunk_of(i);
            ranks[i].ops.push_back(always([&scratch, &buf, i, prev, idx] {
                scratch[i] = buf[prev][idx];
            }));
            ranks[i].ops.push_back(always([&scratch, &buf, i, idx, reduce] {
                buf[i][idx] = reduce ? buf[i][idx] + scratch[i] : scratch[i];
            }));
        }
        run_random(ranks, rng);
    };

    for (int s = 0; s < n - 1; s++)
        run_step([&](int i) { return ring::rs_chunk(i, s, n); }, true);
    for (int s = 0; s < n - 1; s++)
        run_step([&](int i) { return ring::ag_chunk(i, s, n); }, false);

    for (int r = 0; r < n; r++)
        if (buf[r] != expected) {
            fail("single-process n=" + std::to_string(n) +
                 " rank=" + std::to_string(r) + ": wrong result");
            return;
        }
}

void check_owned_chunk(int n) {
    // owned_chunk documents where reduce-scatter leaves the full sum, and is
    // what all-gather step 0 must read from prev.
    for (int r = 0; r < n; r++) {
        if (n >= 2 && ring::rs_chunk(r, n - 2, n) != ring::owned_chunk(r, n))
            fail("owned_chunk mismatch n=" + std::to_string(n));
        if (n >= 2 && ring::ag_chunk(r, 0, n) !=
                          ring::owned_chunk(ring::prev_rank(r, n), n))
            fail("ag step 0 does not read prev's owned chunk, n=" +
                 std::to_string(n));
    }
}

}  // namespace

int main() {
    std::mt19937 rng(2026);
    constexpr int kMaxRanks = 8;
    constexpr int kTrials = 200;

    for (int n = 1; n <= kMaxRanks; n++) {
        const int before = g_failures;
        check_owned_chunk(n);
        for (int t = 0; t < kTrials && g_failures == before; t++) {
            check_persistent(n, 3, rng);
            check_persistent_overlapping(n, 4, rng);
            check_single_process(n, rng);
        }
        std::printf("n=%d: %s\n", n, g_failures == before ? "OK" : "FAIL");
    }

    if (g_failures) {
        std::printf("Ring schedule tests FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("Ring schedule tests passed\n");
    return 0;
}
