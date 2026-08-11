#=============================================================================
# build.py
#
# Builds the TinyNCCL CUDA extension using torch's cpp_extension loader.
# This compiles src/ring_allreduce.cu into a loadable Python module exposing
# the device-side C functions (reduce_sum, copy, ...).
#
# Usage:
#   python build.py            # build in place + import check
#
# After building, from Python:
#   from build import tiny     # tiny.tiny_launch_reduce_sum, ...
#=============================================================================
import os
import sys

import torch
from torch.utils.cpp_extension import load, include_paths

_HERE = os.path.dirname(os.path.abspath(__file__))      # scripts/
_PROJECT = os.path.dirname(_HERE)                        # project root
_SRC = os.path.join(_PROJECT, "src")                    # project/src (sibling of scripts)
_SOURCES = [
    os.path.join(_SRC, "kernels.cu"),
    os.path.join(_SRC, "bindings.cpp"),
]


def _build():
    # Compile the .cu into a python extension. Note: torch.cpp_extension
    # requires a source file; we point it at our ring_allreduce.cu.
    return load(
        name="tiny_nccl_ext",
        sources=_SOURCES,
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_include_paths=[_SRC],
        verbose=True,
    )


def main():
    ext = _build()

    # Quick self-check that the module loaded and exports our symbols.
    print("[ok] extension loaded:", ext.__file__)
    exports = [n for n in dir(ext) if n.startswith("tiny_")]
    print("[ok] exported:", ", ".join(sorted(exports)) or "(none)")
    if not exports:
        sys.exit("[err] no tiny_* symbols found")


if __name__ == "__main__":
    main()
