/**
 * repair.cpp — CPU gene repair: dequantize complement + writeback.
 *
 * Implements purifying selection by restoring model parameters from
 * the complement (reference) strand.
 *
 * For sliced genes (dim >= 0): writes only param.narrow(dim, start, end-start)
 * For full genes (dim < 0): writes the entire parameter tensor
 */

#include <torch/extension.h>
#include "utils.h"

namespace molly {

void repair_gene_cpu(
    torch::Tensor param,
    const torch::Tensor& complement_q,
    float scale,
    int dim, int start, int end) {

    // Dequantize complement
    auto restored = complement_q.to(torch::kFloat32) * scale;

    // Cast to parameter dtype
    restored = restored.to(param.dtype());

    if (dim >= 0 && end > start) {
        // Sliced gene: write only the slice
        auto slice = param.narrow(dim, start, end - start);
        TORCH_CHECK(slice.sizes() == restored.sizes(),
                     "Restored tensor shape mismatch for sliced repair");
        slice.copy_(restored);
    } else {
        // Full gene: write entire parameter
        TORCH_CHECK(param.sizes() == restored.sizes(),
                     "Restored tensor shape mismatch for full repair");
        param.copy_(restored);
    }
}

void batch_repair_cpu(
    const std::vector<torch::Tensor>& params,
    const std::vector<torch::Tensor>& complement_qs,
    const std::vector<float>& scales,
    const std::vector<int>& dims,
    const std::vector<int>& starts,
    const std::vector<int>& ends) {

    const size_t n = params.size();
    TORCH_CHECK(complement_qs.size() == n, "Mismatched batch sizes");
    TORCH_CHECK(scales.size() == n, "Mismatched batch sizes");
    TORCH_CHECK(dims.size() == n, "Mismatched batch sizes");
    TORCH_CHECK(starts.size() == n, "Mismatched batch sizes");
    TORCH_CHECK(ends.size() == n, "Mismatched batch sizes");

    for (size_t i = 0; i < n; i++) {
        repair_gene_cpu(params[i], complement_qs[i], scales[i],
                        dims[i], starts[i], ends[i]);
    }
}

}  // namespace molly
