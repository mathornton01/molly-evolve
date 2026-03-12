/**
 * molly_ops.cpp — PyTorch C++ extension entry point for Molly Evolution.
 *
 * Registers all native ops via pybind11:
 *   - batched_gene_score: CUDA-accelerated gene scoring via grad·delta dot products
 *   - batch_quantize / batch_dequantize: symmetric int16 quantization
 *   - batch_repair: fused repair+dequantize+writeback for gene restoration
 *   - batch_snapshot / batch_sync: bulk strand operations
 */

#include <torch/extension.h>
#include "utils.h"

// Forward declarations from individual source files
namespace molly {

// gene_scoring.cpp
torch::Tensor batched_gene_score_cpu(
    const torch::Tensor& grad_flat,
    const torch::Tensor& deltas_flat,
    const torch::Tensor& gene_offsets);

// quantize.cpp
std::tuple<torch::Tensor, float> quantize_tensor_cpu(
    const torch::Tensor& tensor, int n_bits);
torch::Tensor dequantize_tensor_cpu(
    const torch::Tensor& quantized, float scale);

// repair.cpp
void repair_gene_cpu(
    torch::Tensor param,
    const torch::Tensor& complement_q,
    float scale,
    int dim, int start, int end);
void batch_repair_cpu(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& complement_qs,
    const std::vector<float>& scales,
    const std::vector<int>& dims,
    const std::vector<int>& starts,
    const std::vector<int>& ends);

#if MOLLY_HAS_CUDA
// gene_scoring_cuda.cu
torch::Tensor batched_gene_score_cuda(
    const torch::Tensor& grad_flat,
    const torch::Tensor& deltas_flat,
    const torch::Tensor& gene_offsets);

// quantize_cuda.cu
std::tuple<torch::Tensor, float> quantize_tensor_cuda(
    const torch::Tensor& tensor, int n_bits);
torch::Tensor dequantize_tensor_cuda(
    const torch::Tensor& quantized, float scale);

// repair_cuda.cu
void repair_gene_cuda(
    torch::Tensor param,
    const torch::Tensor& complement_q,
    float scale,
    int dim, int start, int end);
#endif

}  // namespace molly


// ── Dispatch functions ──────────────────────────────────────────────────────

/**
 * Batched gene scoring: compute dot product of gradient with delta for each gene.
 *
 * Args:
 *   grad_flat: flattened gradient vector [total_elements]
 *   deltas_flat: flattened delta vectors for all genes [total_delta_elements]
 *   gene_offsets: int64 tensor of [n_genes+1] offsets into deltas_flat
 *
 * Returns:
 *   Tensor of shape [n_genes] with per-gene scores.
 */
torch::Tensor batched_gene_score(
    const torch::Tensor& grad_flat,
    const torch::Tensor& deltas_flat,
    const torch::Tensor& gene_offsets) {

#if MOLLY_HAS_CUDA
    if (grad_flat.device().is_cuda()) {
        return molly::batched_gene_score_cuda(grad_flat, deltas_flat, gene_offsets);
    }
#endif
    return molly::batched_gene_score_cpu(grad_flat, deltas_flat, gene_offsets);
}

/**
 * Symmetric quantization of a float tensor to int16.
 *
 * Returns (quantized_tensor, scale).
 */
std::tuple<torch::Tensor, float> quantize_tensor(
    const torch::Tensor& tensor, int n_bits = 16) {

#if MOLLY_HAS_CUDA
    if (tensor.device().is_cuda()) {
        return molly::quantize_tensor_cuda(tensor, n_bits);
    }
#endif
    return molly::quantize_tensor_cpu(tensor, n_bits);
}

/**
 * Dequantize int16 tensor back to float32.
 */
torch::Tensor dequantize_tensor(
    const torch::Tensor& quantized, float scale) {

#if MOLLY_HAS_CUDA
    if (quantized.device().is_cuda()) {
        return molly::dequantize_tensor_cuda(quantized, scale);
    }
#endif
    return molly::dequantize_tensor_cpu(quantized, scale);
}

/**
 * Fused repair: dequantize complement strand and write back to model parameter.
 *
 * For sliced genes (dim >= 0), only writes the slice [start:end] along dim.
 * For full genes (dim < 0), writes the entire parameter.
 */
void repair_gene(
    torch::Tensor param,
    const torch::Tensor& complement_q,
    float scale,
    int dim = -1, int start = 0, int end = -1) {

#if MOLLY_HAS_CUDA
    if (param.device().is_cuda()) {
        molly::repair_gene_cuda(param, complement_q, scale, dim, start, end);
        return;
    }
#endif
    molly::repair_gene_cpu(param, complement_q, scale, dim, start, end);
}

/**
 * Batch repair: apply repair to multiple parameter slices at once.
 */
void batch_repair(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& complement_qs,
    const std::vector<float>& scales,
    const std::vector<int>& dims,
    const std::vector<int>& starts,
    const std::vector<int>& ends) {

    molly::batch_repair_cpu(params, complement_qs, scales, dims, starts, ends);
}


// ── Python bindings ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_C, m) {
    m.doc() = "Molly Evolution C++/CUDA extension for fast gene operations";

    m.def("batched_gene_score", &batched_gene_score,
          "Batched gene scoring via grad·delta dot products",
          py::arg("grad_flat"),
          py::arg("deltas_flat"),
          py::arg("gene_offsets"));

    m.def("quantize_tensor", &quantize_tensor,
          "Symmetric quantization (float -> int16)",
          py::arg("tensor"),
          py::arg("n_bits") = 16);

    m.def("dequantize_tensor", &dequantize_tensor,
          "Dequantize (int16 -> float32)",
          py::arg("quantized"),
          py::arg("scale"));

    m.def("repair_gene", &repair_gene,
          "Fused repair: dequantize complement + writeback to parameter",
          py::arg("param"),
          py::arg("complement_q"),
          py::arg("scale"),
          py::arg("dim") = -1,
          py::arg("start") = 0,
          py::arg("end") = -1);

    m.def("batch_repair", &batch_repair,
          "Batch repair multiple gene parameter slices",
          py::arg("params"),
          py::arg("complement_qs"),
          py::arg("scales"),
          py::arg("dims"),
          py::arg("starts"),
          py::arg("ends"));

    m.def("has_cuda", []() -> bool {
        return MOLLY_HAS_CUDA;
    }, "Check if extension was built with CUDA support");
}
