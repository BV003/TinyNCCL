from setuptools import setup
import torch.utils.cpp_extension

torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tiny",
    ext_modules=[
        CUDAExtension(
            name="tiny",
            sources=["src/kernels.cu", "src/bindings.cpp"],
            extra_compile_args={
                "cxx": ["-std=c++20"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
