/**
 * genome_gguf.h — GGUF serialization for dual-genome state.
 *
 * Stores primary and complement strands alongside the standard GGUF model,
 * enabling gene conversion on quantized GGUF models without PyTorch.
 *
 * Custom GGUF keys:
 *   molly.gene.count           : uint32  — total number of genes
 *   molly.gene.{i}.name        : string  — gene name
 *   molly.gene.{i}.n_params    : uint32  — number of parameter segments
 *   molly.gene.{i}.complement  : tensor  — quantized complement strand (int16)
 *   molly.gene.{i}.scale       : float32 — quantization scale
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace molly {

/**
 * Represents one gene's complement strand data.
 */
struct GeneStrand {
    std::string name;
    std::vector<std::string> param_names;
    // Per-parameter: quantized int16 data + scale
    std::vector<std::vector<int16_t>> complement_data;
    std::vector<float> scales;
    // Optional slice info (for SlicedTransformerGene)
    struct SliceDef {
        std::string param_name;
        int dim;
        int start;
        int end;
    };
    std::vector<SliceDef> slice_defs;  // empty for full-param genes
};

/**
 * Full genome state for GGUF serialization.
 */
struct GenomeState {
    int n_genes;
    std::vector<GeneStrand> genes;
};

/**
 * Save genome complement strands to a GGUF file.
 * Appends custom metadata tensors to existing GGUF model file.
 *
 * @param gguf_path  Path to base GGUF model file
 * @param out_path   Output path for GGUF with genome data
 * @param state      Genome state to serialize
 * @return true on success
 */
bool save_genome_gguf(const std::string& gguf_path,
                      const std::string& out_path,
                      const GenomeState& state);

/**
 * Load genome complement strands from a GGUF file.
 *
 * @param gguf_path  Path to GGUF file with genome data
 * @param state      Output genome state
 * @return true on success
 */
bool load_genome_gguf(const std::string& gguf_path,
                      GenomeState& state);

}  // namespace molly
