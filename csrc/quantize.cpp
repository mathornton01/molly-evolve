/**
 * quantize.cpp — CPU symmetric quantization/dequantization.
 *
 * Matches the Python TransformerGene._quantize / _dequantize exactly:
 *   scale = abs_max / (2^(n_bits-1) - 1)
 *   quantized = round(tensor / scale).clamp(-qmax, qmax).to(int16)
 *   dequantized = quantized.float() * scale
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

std::tuple<torch::Tensor, float> quantize_tensor_cpu(
    const torch::Tensor& tensor, int n_bits) {

    TORCH_CHECK(n_bits > 0 && n_bits <= 16, "n_bits must be in [1, 16]");

    auto t = tensor.to(torch::kFloat32).contiguous();
    float abs_max = t.abs().max().item<float>();
    float qmax = static_cast<float>((1 << (n_bits - 1)) - 1);
    float scale = abs_max / qmax;

    if (scale == 0.0f) {
        return {torch::zeros_like(t, torch::kInt16), 0.0f};
    }

    auto scaled = t / scale;
    auto quantized = scaled.round().clamp(-qmax, qmax).to(torch::kInt16);

    return {quantized, scale};
}

torch::Tensor dequantize_tensor_cpu(
    const torch::Tensor& quantized, float scale) {

    return quantized.to(torch::kFloat32) * scale;
}

}  // namespace molly
