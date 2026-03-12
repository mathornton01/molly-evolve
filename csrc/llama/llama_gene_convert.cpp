/**
 * llama_gene_convert.cpp — Gene conversion engine for llama.cpp models.
 *
 * Implements repair/fix/snapshot using CPU-only operations on GGUF weights.
 * This is the edge/CPU deployment path — no PyTorch or CUDA required.
 */

#include "llama_gene_convert.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace molly {

bool LlamaGeneConverter::load(const std::string& model_path) {
    model_path_ = model_path;
    return load_genome_gguf(model_path, state_);
}

void LlamaGeneConverter::quantize_to_strand(
    const float* data, size_t n,
    std::vector<int16_t>& out_q, float& out_scale,
    int n_bits) {

    // Find abs max
    float abs_max = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float a = std::fabs(data[i]);
        if (a > abs_max) abs_max = a;
    }

    float qmax = static_cast<float>((1 << (n_bits - 1)) - 1);
    out_scale = abs_max / qmax;

    out_q.resize(n);
    if (out_scale == 0.0f) {
        std::fill(out_q.begin(), out_q.end(), 0);
        return;
    }

    float inv_scale = 1.0f / out_scale;
    for (size_t i = 0; i < n; i++) {
        float val = std::round(data[i] * inv_scale);
        val = std::max(-qmax, std::min(qmax, val));
        out_q[i] = static_cast<int16_t>(val);
    }
}

void LlamaGeneConverter::snapshot(
    std::function<std::pair<const float*, size_t>(const std::string&)> get_tensor) {

    for (auto& gene : state_.genes) {
        if (!gene.slice_defs.empty()) {
            // Sliced gene — snapshot each slice
            gene.complement_data.resize(gene.slice_defs.size());
            gene.scales.resize(gene.slice_defs.size());

            for (size_t j = 0; j < gene.slice_defs.size(); j++) {
                auto [data, n] = get_tensor(gene.slice_defs[j].param_name);
                // For sliced genes, we'd need the slice extraction logic
                // For now, quantize the full parameter and note that the
                // Python bridge should handle slice extraction
                quantize_to_strand(data, n,
                                   gene.complement_data[j], gene.scales[j]);
            }
        } else {
            // Full gene — snapshot each parameter
            gene.complement_data.resize(gene.param_names.size());
            gene.scales.resize(gene.param_names.size());

            for (size_t j = 0; j < gene.param_names.size(); j++) {
                auto [data, n] = get_tensor(gene.param_names[j]);
                quantize_to_strand(data, n,
                                   gene.complement_data[j], gene.scales[j]);
            }
        }
    }
}

void LlamaGeneConverter::repair_genes(
    const std::vector<int>& gene_ids,
    std::function<void(const std::string&, const float*, size_t,
                       int, int, int)> set_tensor) {

    for (int gid : gene_ids) {
        if (gid < 0 || gid >= state_.n_genes) continue;
        const auto& gene = state_.genes[gid];

        if (!gene.slice_defs.empty()) {
            for (size_t j = 0; j < gene.slice_defs.size(); j++) {
                const auto& sd = gene.slice_defs[j];
                const auto& q = gene.complement_data[j];
                float scale = gene.scales[j];

                // Dequantize
                std::vector<float> restored(q.size());
                for (size_t k = 0; k < q.size(); k++) {
                    restored[k] = static_cast<float>(q[k]) * scale;
                }

                set_tensor(sd.param_name, restored.data(), restored.size(),
                           sd.dim, sd.start, sd.end);
            }
        } else {
            for (size_t j = 0; j < gene.param_names.size(); j++) {
                const auto& q = gene.complement_data[j];
                float scale = gene.scales[j];

                std::vector<float> restored(q.size());
                for (size_t k = 0; k < q.size(); k++) {
                    restored[k] = static_cast<float>(q[k]) * scale;
                }

                set_tensor(gene.param_names[j], restored.data(),
                           restored.size(), -1, 0, -1);
            }
        }
    }
}

void LlamaGeneConverter::fix_genes(
    const std::vector<int>& gene_ids,
    std::function<std::pair<const float*, size_t>(const std::string&)> get_tensor) {

    for (int gid : gene_ids) {
        if (gid < 0 || gid >= state_.n_genes) continue;
        auto& gene = state_.genes[gid];

        if (!gene.slice_defs.empty()) {
            for (size_t j = 0; j < gene.slice_defs.size(); j++) {
                auto [data, n] = get_tensor(gene.slice_defs[j].param_name);
                quantize_to_strand(data, n,
                                   gene.complement_data[j], gene.scales[j]);
            }
        } else {
            for (size_t j = 0; j < gene.param_names.size(); j++) {
                auto [data, n] = get_tensor(gene.param_names[j]);
                quantize_to_strand(data, n,
                                   gene.complement_data[j], gene.scales[j]);
            }
        }
    }
}

bool LlamaGeneConverter::save(const std::string& base_gguf_path,
                               const std::string& out_path) {
    return save_genome_gguf(base_gguf_path, out_path, state_);
}

}  // namespace molly
