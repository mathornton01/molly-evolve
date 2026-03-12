/**
 * quantize_cuda.cu — CUDA kernels for symmetric quantization/dequantization.
 *
 * Provides GPU-accelerated versions of the quantize/dequantize operations
 * for batch snapshot and sync operations.
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

// ── Quantize kernel ─────────────────────────────────────────────────────────

__global__ void quantize_kernel(
    const float* __restrict__ input,
    int16_t* __restrict__ output,
    float inv_scale,
    float qmax,
    int64_t n) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = input[idx] * inv_scale;
    val = roundf(val);
    val = fminf(fmaxf(val, -qmax), qmax);
    output[idx] = static_cast<int16_t>(val);
}


// ── Dequantize kernel ───────────────────────────────────────────────────────

__global__ void dequantize_kernel(
    const int16_t* __restrict__ input,
    float* __restrict__ output,
    float scale,
    int64_t n) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = static_cast<float>(input[idx]) * scale;
}


// ── Host functions ──────────────────────────────────────────────────────────

std::tuple<torch::Tensor, float> quantize_tensor_cuda(
    const torch::Tensor& tensor, int n_bits) {

    CHECK_CUDA(tensor);
    TORCH_CHECK(n_bits > 0 && n_bits <= 16, "n_bits must be in [1, 16]");

    auto t = tensor.to(torch::kFloat32).contiguous();
    const int64_t n = t.numel();
    float abs_max = t.abs().max().item<float>();
    float qmax = static_cast<float>((1 << (n_bits - 1)) - 1);
    float scale = abs_max / qmax;

    if (scale == 0.0f) {
        return {torch::zeros({n}, torch::TensorOptions()
                    .dtype(torch::kInt16).device(t.device())), 0.0f};
    }

    auto output = torch::empty({n}, torch::TensorOptions()
                    .dtype(torch::kInt16).device(t.device()));

    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    quantize_kernel<<<blocks, threads>>>(
        t.data_ptr<float>(),
        output.data_ptr<int16_t>(),
        1.0f / scale,
        qmax,
        n);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    // Reshape to match input
    output = output.reshape(tensor.sizes());
    return {output, scale};
}

torch::Tensor dequantize_tensor_cuda(
    const torch::Tensor& quantized, float scale) {

    CHECK_CUDA(quantized);
    auto q = quantized.contiguous();
    const int64_t n = q.numel();

    auto output = torch::empty({n}, torch::TensorOptions()
                    .dtype(torch::kFloat32).device(q.device()));

    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;

    dequantize_kernel<<<blocks, threads>>>(
        q.data_ptr<int16_t>(),
        output.data_ptr<float>(),
        scale,
        n);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return output.reshape(quantized.sizes());
}

}  // namespace molly
