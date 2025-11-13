"""
Setup script for DeepseekV3 MoE CUDA extension.
This is an alternative to CMakeLists.txt for easier building.

To specify CUDA architectures, set the TORCH_CUDA_ARCH_LIST environment variable:
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
    python setup.py build_ext --inplace

If not set, PyTorch will auto-detect the architecture.
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import torch
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        name='deepseek_moe_cuda',
        sources=[
            'deepseek_moe.cpp',
            'deepseek_moe_cuda.cu',
        ],
        include_dirs=[
            # Include pybind11 headers
            pybind11.get_include(),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-fPIC'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '--expt-extended-lambda',
                '--expt-relaxed-constexpr',
                # Architecture is controlled by TORCH_CUDA_ARCH_LIST env var
                # or auto-detected by PyTorch if not set
            ],
        },
    ),
]

setup(
    name='deepseek_moe',
    version='0.1.0',
    description='DeepseekV3 MoE CUDA Implementation',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
)

