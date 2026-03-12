/**
 * gene_scoring_cuda.cu — CUDA kernel for batched gene scoring.
 *
 * Each gene gets one thread block. Within the block, threads cooperatively
 * compute the dot product grad[start:end] · delta[start:end] using
 * warp-level reductions + shared memory.
 *
 * This replaces the Python loop over parameters and genes with a single
 * kernel launch, eliminating per-gene synchronization overhead.
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

// ── Warp reduce ─────────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ── Gene scoring kernel ─────────────────────────────────────────────────────

__global__ void gene_score_kernel(
    const float* __restrict__ grad_flat,
    const float* __restrict__ deltas_flat,
    const int64_t* __restrict__ gene_offsets,
    float* __restrict__ scores,
    int64_t n_genes) {

    const int gene_id = blockIdx.x;
    if (gene_id >= n_genes) return;

    const int64_t start = gene_offsets[gene_id];
    const int64_t end = gene_offsets[gene_id + 1];
    const int64_t len = end - start;

    // Each thread accumulates partial dot product
    float partial = 0.0f;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        partial += grad_flat[start + i] * deltas_flat[start + i];
    }

    // Warp-level reduction
    partial = warp_reduce_sum(partial);

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];  // max 32 warps per block (1024 threads)
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        shared[warp_id] = partial;
    }
    __syncthreads();

    // First warp reduces across warps
    const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (warp_id == 0) {
        float val = (lane < n_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            scores[gene_id] = val;
        }
    }
}


// ── Host function ───────────────────────────────────────────────────────────

torch::Tensor batched_gene_score_cuda(
    const torch::Tensor& grad_flat,
    const torch::Tensor& deltas_flat,
    const torch::Tensor& gene_offsets) {

    CHECK_CUDA_CONTIGUOUS(grad_flat);
    CHECK_CUDA_CONTIGUOUS(deltas_flat);
    CHECK_CUDA(gene_offsets);

    const int64_t n_genes = gene_offsets.size(0) - 1;
    auto scores = torch::zeros({n_genes}, grad_flat.options());

    if (n_genes == 0) return scores;

    // One block per gene, BLOCK_SIZE threads per block
    const int threads = BLOCK_SIZE;
    const int blocks = static_cast<int>(n_genes);

    gene_score_kernel<<<blocks, threads>>>(
        grad_flat.data_ptr<float>(),
        deltas_flat.data_ptr<float>(),
        gene_offsets.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        n_genes);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel error: ", cudaGetErrorString(err));

    return scores;
}

}  // namespace molly
