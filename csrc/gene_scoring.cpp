/**
 * gene_scoring.cpp — CPU implementation of batched gene scoring.
 *
 * Computes dot product of gradient with per-gene delta vectors.
 * Each gene owns a contiguous slice of deltas_flat, defined by gene_offsets.
 *
 * Score_g = sum(grad[offset_g : offset_{g+1}] * delta[offset_g : offset_{g+1}])
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

torch::Tensor batched_gene_score_cpu(
    const torch::Tensor& grad_flat,
    const torch::Tensor& deltas_flat,
    const torch::Tensor& gene_offsets) {

    TORCH_CHECK(grad_flat.dim() == 1, "grad_flat must be 1-D");
    TORCH_CHECK(deltas_flat.dim() == 1, "deltas_flat must be 1-D");
    TORCH_CHECK(gene_offsets.dim() == 1, "gene_offsets must be 1-D");

    const int64_t n_genes = gene_offsets.size(0) - 1;
    auto scores = torch::zeros({n_genes}, grad_flat.options());

    auto grad_a = grad_flat.accessor<float, 1>();
    auto delta_a = deltas_flat.accessor<float, 1>();
    auto offset_a = gene_offsets.accessor<int64_t, 1>();
    auto scores_a = scores.accessor<float, 1>();

    // Parallel over genes
    at::parallel_for(0, n_genes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t g = begin; g < end; g++) {
            int64_t start = offset_a[g];
            int64_t stop = offset_a[g + 1];
            float dot = 0.0f;
            for (int64_t i = start; i < stop; i++) {
                dot += grad_a[i] * delta_a[i];
            }
            scores_a[g] = dot;
        }
    });

    return scores;
}

}  // namespace molly
