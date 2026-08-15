from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tiny",
    ext_modules=[
        CUDAExtension(
            name="tiny",
            sources=["src/kernels.cu", "src/transport.cu", "src/ring_allreduce.cu", "src/ipc.cu", "src/bindings.cpp"],
            extra_compile_args={
                "cxx": ["-std=c++20"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
