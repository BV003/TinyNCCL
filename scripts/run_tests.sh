#!/usr/bin/env bash
#
# Build TinyNCCL and run the test suite.
#
# Usage:
#   scripts/run_tests.sh              # build, then run every test that fits this machine
#   scripts/run_tests.sh --no-build   # skip the install, use the current build
#
# The interpreter is auto-detected (active venv > python3 > python). Point it at
# a specific environment when needed:
#   PYTHON=/path/to/venv/bin/python scripts/run_tests.sh
# torchrun defaults to a sibling of PYTHON when present, else the one on PATH.
#
# Multi-process tests need at least two GPUs with peer access (CUDA IPC) and run
# with one rank per visible GPU; they are skipped when fewer are visible, so
# single-GPU machines still get the rest. The ring schedule test needs no GPU.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 2

BUILD=1
if [ "${1:-}" = "--no-build" ]; then
    BUILD=0
fi

resolve_python() {
    if [ -n "${PYTHON:-}" ]; then
        echo "$PYTHON"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        echo "${VIRTUAL_ENV}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        echo python3
    elif command -v python >/dev/null 2>&1; then
        echo python
    fi
}

PYTHON="$(resolve_python)"
if [ -z "$PYTHON" ]; then
    echo "error: no python interpreter found; set PYTHON=/path/to/python" >&2
    exit 2
fi

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "error: '$PYTHON' cannot import torch." >&2
    echo "Point PYTHON (and optionally TORCHRUN) at the environment that has it:" >&2
    echo "  PYTHON=/path/to/venv/bin/python scripts/run_tests.sh" >&2
    exit 2
fi

if [ -n "${TORCHRUN:-}" ]; then
    :
elif [ -x "$(dirname "$PYTHON")/torchrun" ]; then
    TORCHRUN="$(dirname "$PYTHON")/torchrun"
else
    TORCHRUN=torchrun
fi

echo "### python:   $PYTHON"
echo "### torchrun: $TORCHRUN"

if [ "$BUILD" -eq 1 ]; then
    echo "### build"
    if "$PYTHON" -m pip --version >/dev/null 2>&1; then
        "$PYTHON" -m pip install -e . --no-build-isolation || {
            echo "build failed; re-run with --no-build to keep the current build"
            exit 2
        }
    elif command -v uv >/dev/null 2>&1; then
        uv pip install -e . --no-build-isolation --python "$PYTHON" || {
            echo "build failed; re-run with --no-build to keep the current build"
            exit 2
        }
    else
        echo "error: neither pip nor uv is available; run with --no-build" >&2
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

# CPU-only: the ring schedule (chunk indices + GPU signal dependencies) for
# 1..8 ranks under random interleavings. Needs only a host C++ compiler.
CXX="${CXX:-c++}"
if command -v "$CXX" >/dev/null 2>&1; then
    run schedule sh -c "\"$CXX\" -std=c++17 -O2 -Isrc tests/test_ring_schedule.cpp \
        -o /tmp/tinynccl_ring_schedule && /tmp/tinynccl_ring_schedule"
else
    echo "SKIP  schedule (no C++ compiler; set CXX)"
fi

run kernels "$PYTHON" tests/test_kernels.py
run transport "$PYTHON" tests/test_transport.py
run single "$PYTHON" tests/test_single_process.py

# Multi-process tests use every visible GPU (override with NPROC=N), so 4- and
# 8-GPU machines exercise the ring with more than two ranks.
NPROC="${NPROC:-$gpu_count}"
if [ "$NPROC" -ge 2 ]; then
    echo "### multi-process tests with $NPROC ranks"
    run smoke timeout 180 "$TORCHRUN" --nproc_per_node="$NPROC" tests/smoke_test.py
    run fixed_data timeout 300 "$TORCHRUN" --nproc_per_node="$NPROC" tests/test_fixed_data.py
    run vs_nccl timeout 300 "$TORCHRUN" --nproc_per_node="$NPROC" tests/test_vs_torch_distributed.py
    run persistent timeout 300 "$TORCHRUN" --nproc_per_node="$NPROC" tests/test_persistent_communicator.py
else
    echo "SKIP  multi-process tests (need >= 2 GPUs, found ${gpu_count})"
fi

echo "-----------------------------------------"
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ]
