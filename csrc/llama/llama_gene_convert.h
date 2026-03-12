/**
 * llama_gene_convert.h — Gene conversion on GGUF models via llama.cpp.
 *
 * Provides CPU gene conversion (repair/fix) without requiring PyTorch.
 * Uses llama.cpp's tensor access to read/write model weights directly.
 *
 * Workflow:
 *   1. Load GGUF model with llama.cpp
 *   2. Load genome state from GGUF appendix
 *   3. Fine-tune (externally) or detect drift
 *   4. Call repair_genes() to restore damaged genes
 *   5. Save updated genome state back to GGUF
 */

#pragma once

#include "genome_gguf.h"
#include <vector>
#include <string>
#include <functional>

namespace molly {

/**
 * Gene conversion engine for llama.cpp models.
 *
 * Operates entirely on CPU using the GGUF model's weight tensors.
 * Supports both full-parameter and sliced (per-head) genes.
 */
class LlamaGeneConverter {
public:
    /**
     * Initialize from a GGUF model file with embedded genome data.
     *
     * @param model_path  Path to GGUF model with Molly genome appendix
     * @return true if genome data was found and loaded
     */
    bool load(const std::string& model_path);

    /**
     * Snapshot current model weights as the complement (reference) strand.
     * Call this after confirming the model is in a good state.
     *
     * @param get_tensor  Callback that returns weight data for a given param name.
     *                    Signature: (param_name) -> (data_ptr, n_elements)
     */
    void snapshot(
        std::function<std::pair<const float*, size_t>(const std::string&)> get_tensor);

    /**
     * Repair specified genes: restore model weights from complement strand.
     *
     * @param gene_ids    Indices of genes to repair
     * @param set_tensor  Callback to write repaired weights back to model.
     *                    Signature: (param_name, data_ptr, n_elements, dim, start, end)
     */
    void repair_genes(
        const std::vector<int>& gene_ids,
        std::function<void(const std::string&, const float*, size_t,
                           int, int, int)> set_tensor);

    /**
     * Fix specified genes: copy current (primary) weights to complement.
     * Marks these genes as "adapted" so they won't be repaired later.
     */
    void fix_genes(
        const std::vector<int>& gene_ids,
        std::function<std::pair<const float*, size_t>(const std::string&)> get_tensor);

    /**
     * Save genome state back to GGUF file.
     */
    bool save(const std::string& base_gguf_path,
              const std::string& out_path);

    /** Get number of genes. */
    int num_genes() const { return state_.n_genes; }

    /** Get genome state for inspection. */
    const GenomeState& state() const { return state_; }

private:
    GenomeState state_;
    std::string model_path_;

    void quantize_to_strand(const float* data, size_t n,
                            std::vector<int16_t>& out_q, float& out_scale,
                            int n_bits = 16);
};

}  // namespace molly
