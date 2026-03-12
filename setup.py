"""
Build script for molly-evolution C++/CUDA extension.

Automatically detects CUDA availability:
  - With CUDA: builds CUDAExtension with .cu kernels
  - Without CUDA: builds CppExtension (CPU-only ops)
  - Neither available: installs pure-Python fallback
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup

# ---------------------------------------------------------------------------
# Detect CUDA / C++ toolchain
# ---------------------------------------------------------------------------

def _has_nvcc():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _has_cpp_compiler():
    try:
        import torch
        from torch.utils.cpp_extension import check_compiler_abi_compatibility
        return True
    except Exception:
        return False


def get_extensions():
    """Return list of C++/CUDA extensions to build, or [] for pure-Python."""
    try:
        import torch
        from torch.utils.cpp_extension import CppExtension, CUDAExtension
    except ImportError:
        print("WARNING: PyTorch not found. Installing pure-Python version.")
        return []

    csrc = Path(__file__).parent / "csrc"
    if not csrc.exists():
        return []

    cpp_sources = [
        str(csrc / "molly_ops.cpp"),
        str(csrc / "gene_scoring.cpp"),
        str(csrc / "quantize.cpp"),
        str(csrc / "repair.cpp"),
    ]

    cuda_sources = [
        str(csrc / "gene_scoring_cuda.cu"),
        str(csrc / "quantize_cuda.cu"),
        str(csrc / "repair_cuda.cu"),
    ]

    include_dirs = [str(csrc)]
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    if _has_nvcc() and torch.cuda.is_available():
        extra_compile_args["nvcc"] = [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ]
        ext = CUDAExtension(
            name="molly_evolution._C",
            sources=cpp_sources + cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    else:
        # CPU-only build — skip .cu files
        ext = CppExtension(
            name="molly_evolution._C",
            sources=cpp_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )

    return [ext]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

ext_modules = get_extensions()

cmdclass = {}
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension
    cmdclass["build_ext"] = BuildExtension

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
