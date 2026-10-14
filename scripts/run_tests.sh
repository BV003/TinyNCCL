#!/usr/bin/env bash
#
# Build TinyNCCL and run the test suite.
#
# Usage:
#   scripts/run_tests.sh              # build, then run every test that fits this machine
#   scripts/run_tests.sh --no-build   # skip the install, use the current build
#
# Override the interpreter / launcher when they are not on PATH:
#   PYTHON=python3.12 TORCHRUN=torchrun scripts/run_tests.sh
#
# Multi-process tests need two GPUs with peer access (CUDA IPC). When fewer are
# visible they are skipped, so single-GPU machines still get the rest.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"

BUILD=1
if [ "${1:-}" = "--no-build" ]; then
    BUILD=0
fi

cd "$REPO_ROOT" || exit 2

if [ "$BUILD" -eq 1 ]; then
    echo "### build"
    if ! "$PYTHON" -m pip install -e . --no-build-isolation; then
        echo "build failed; re-run with --no-build to use an existing build"
        exit 2
    fi
fi

gpu_count=$(
    "$PYTHON" - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)

pass=0
fail=0
run() {
    local name="$1"
    shift
    if "$@" >"/tmp/tinynccl_${name}.log" 2>&1; then
        echo "PASS  $name"
        pass=$((pass + 1))
    else
        echo "FAIL  $name  (see /tmp/tinynccl_${name}.log)"
        fail=$((fail + 1))
    fi
}

run kernels "$PYTHON" tests/test_kernels.py
run transport "$PYTHON" tests/test_transport.py
run single "$PYTHON" tests/test_single_process.py

if [ "$gpu_count" -ge 2 ]; then
    run smoke timeout 180 "$TORCHRUN" --nproc_per_node=2 tests/smoke_test.py
    run fixed_data timeout 300 "$TORCHRUN" --nproc_per_node=2 tests/test_fixed_data.py
    run vs_nccl timeout 300 "$TORCHRUN" --nproc_per_node=2 tests/test_vs_torch_distributed.py
    run persistent timeout 300 "$TORCHRUN" --nproc_per_node=2 tests/test_persistent_communicator.py
else
    echo "SKIP  multi-process tests (need >= 2 GPUs, found ${gpu_count})"
fi

echo "-----------------------------------------"
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]
