/**
 * repair_cuda.cu — Fused CUDA kernel for gene repair.
 *
 * Combines dequantization + writeback into a single kernel launch,
 * avoiding intermediate tensor allocation and a second kernel dispatch.
 *
 * For sliced genes: only writes to the target slice of the parameter tensor.
 * For full genes: writes the entire parameter tensor.
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

// ── Fused repair kernel (float32 output) ────────────────────────────────────

__global__ void repair_full_kernel_f32(
    float* __restrict__ param,
    const int16_t* __restrict__ complement_q,
    float scale,
    int64_t n) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    param[idx] = static_cast<float>(complement_q[idx]) * scale;
}

// ── Fused repair kernel (float16 output) ────────────────────────────────────

__global__ void repair_full_kernel_f16(
    at::Half* __restrict__ param,
    const int16_t* __restrict__ complement_q,
    float scale,
    int64_t n) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    param[idx] = __float2half(static_cast<float>(complement_q[idx]) * scale);
}

// ── Fused sliced repair kernel (float32) ────────────────────────────────────

/**
 * Repair a slice of a 2D parameter tensor along dim.
 *
 * For dim=0: rows [start, end) are overwritten
 * For dim=1: columns [start, end) are overwritten
 *
 * complement_q is contiguous with shape matching the slice.
 */
__global__ void repair_slice_2d_kernel_f32(
    float* __restrict__ param,
    const int16_t* __restrict__ complement_q,
    float scale,
    int64_t param_rows,
    int64_t param_cols,
    int dim,
    int64_t slice_start,
    int64_t slice_len) {

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (dim == 1) {
        // Slice along columns: complement_q shape is [param_rows, slice_len]
        const int64_t total = param_rows * slice_len;
        if (idx >= total) return;
        const int64_t row = idx / slice_len;
        const int64_t col = idx % slice_len;
        param[row * param_cols + (slice_start + col)] =
            static_cast<float>(complement_q[idx]) * scale;
    } else {
        // Slice along rows: complement_q shape is [slice_len, param_cols]
        const int64_t total = slice_len * param_cols;
        if (idx >= total) return;
        const int64_t row = idx / param_cols;
        const int64_t col = idx % param_cols;
        param[(slice_start + row) * param_cols + col] =
            static_cast<float>(complement_q[idx]) * scale;
    }
}


// ── Host function ───────────────────────────────────────────────────────────

void repair_gene_cuda(
    torch::Tensor param,
    const torch::Tensor& complement_q,
    float scale,
    int dim, int start, int end) {

    CHECK_CUDA(param);
    CHECK_CUDA(complement_q);

    const int threads = BLOCK_SIZE;

    if (dim >= 0 && end > start) {
        // Sliced repair
        TORCH_CHECK(param.dim() == 2 || param.dim() == 1,
                     "Sliced CUDA repair supports 1-D or 2-D tensors");

        if (param.dim() == 1) {
            // 1-D slice: just a contiguous sub-range
            const int64_t n = end - start;
            const int blocks = (n + threads - 1) / threads;
            repair_full_kernel_f32<<<blocks, threads>>>(
                param.data_ptr<float>() + start,
                complement_q.contiguous().data_ptr<int16_t>(),
                scale, n);
        } else {
            // 2-D slice
            auto q = complement_q.contiguous();
            const int64_t rows = param.size(0);
            const int64_t cols = param.size(1);
            const int64_t slice_len = end - start;

            int64_t total = (dim == 1)
                ? rows * slice_len
                : slice_len * cols;
            const int blocks = (total + threads - 1) / threads;

            if (param.dtype() == torch::kFloat32) {
                repair_slice_2d_kernel_f32<<<blocks, threads>>>(
                    param.data_ptr<float>(),
                    q.data_ptr<int16_t>(),
                    scale, rows, cols, dim, start, slice_len);
            } else {
                // Fallback: dequantize to temp, then copy
                auto restored = q.to(torch::kFloat32) * scale;
                auto target = param.narrow(dim, start, end - start);
                target.copy_(restored.to(param.dtype()));
            }
        }
    } else {
        // Full repair
        const int64_t n = param.numel();
        const int blocks = (n + threads - 1) / threads;
        auto q = complement_q.contiguous();

        if (param.dtype() == torch::kFloat32) {
            repair_full_kernel_f32<<<blocks, threads>>>(
                param.data_ptr<float>(),
                q.data_ptr<int16_t>(),
                scale, n);
        } else if (param.dtype() == torch::kFloat16) {
            repair_full_kernel_f16<<<blocks, threads>>>(
                reinterpret_cast<at::Half*>(param.data_ptr<at::Half>()),
                q.data_ptr<int16_t>(),
                scale, n);
        } else {
            // Generic fallback
            auto restored = q.to(torch::kFloat32) * scale;
            param.copy_(restored.to(param.dtype()));
        }
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));
}

}  // namespace molly
