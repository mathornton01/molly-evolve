/**
 * utils.h — Shared C++ utilities for Molly Evolution kernels.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

// Macro for CUDA availability
#ifdef WITH_CUDA
#define MOLLY_HAS_CUDA 1
#else
#define MOLLY_HAS_CUDA 0
#endif

// Check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_CONTIGUOUS(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Dispatch to CUDA or raise error
#define DISPATCH_CUDA_OR_THROW(func_name) \
    TORCH_CHECK(false, func_name " requires CUDA but extension was built without CUDA support")

namespace molly {

// Thread block size for CUDA kernels
constexpr int BLOCK_SIZE = 256;

// Warp size
constexpr int WARP_SIZE = 32;

// Maximum number of genes (for stack-allocated buffers)
constexpr int MAX_GENES = 4096;

/**
 * Symmetric quantization helpers (matching Python implementation).
 * n_bits=16 by default.
 */
inline float compute_scale(const torch::Tensor& tensor, int n_bits = 16) {
    float abs_max = tensor.abs().max().item<float>();
    float qmax = static_cast<float>((1 << (n_bits - 1)) - 1);
    return abs_max / qmax;
}

inline torch::Tensor quantize_symmetric(const torch::Tensor& tensor, float scale) {
    if (scale == 0.0f) {
        return torch::zeros_like(tensor, torch::kInt16);
    }
    auto scaled = tensor / scale;
    return scaled.round().clamp(-32767, 32767).to(torch::kInt16);
}

inline torch::Tensor dequantize_symmetric(const torch::Tensor& tensor, float scale) {
    return tensor.to(torch::kFloat32) * scale;
}

}  // namespace molly
