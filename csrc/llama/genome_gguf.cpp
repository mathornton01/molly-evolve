/**
 * genome_gguf.cpp — GGUF serialization for dual-genome state.
 *
 * Implements save/load of gene complement strands in GGUF format.
 * This enables gene conversion on CPU/edge without PyTorch dependency.
 *
 * NOTE: This is a standalone implementation that reads/writes custom
 * GGUF keys. It does NOT require the full llama.cpp model loading —
 * just the GGUF file format parsing.
 */

#include "genome_gguf.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include <algorithm>

namespace molly {

// ── GGUF format constants ───────────────────────────────────────────────────

static constexpr uint32_t GGUF_MAGIC = 0x46475547;  // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

// We use a simpler custom binary format appended after the GGUF data.
// This avoids modifying GGUF internals while staying in the same file.

static constexpr uint32_t MOLLY_MAGIC = 0x4D4F4C4C;  // "MOLL"
static constexpr uint32_t MOLLY_VERSION = 1;


// ── Save ────────────────────────────────────────────────────────────────────

bool save_genome_gguf(const std::string& gguf_path,
                      const std::string& out_path,
                      const GenomeState& state) {
    // Read base GGUF file
    std::ifstream base(gguf_path, std::ios::binary);
    if (!base.is_open()) {
        std::cerr << "Failed to open base GGUF: " << gguf_path << std::endl;
        return false;
    }
    std::vector<char> base_data(
        (std::istreambuf_iterator<char>(base)),
        std::istreambuf_iterator<char>());
    base.close();

    // Write output: base GGUF + Molly genome appendix
    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open output: " << out_path << std::endl;
        return false;
    }

    // Write base GGUF data
    out.write(base_data.data(), base_data.size());

    // Write Molly genome appendix
    auto write_u32 = [&](uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
    auto write_f32 = [&](float v) { out.write(reinterpret_cast<char*>(&v), 4); };
    auto write_str = [&](const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        write_u32(len);
        out.write(s.data(), len);
    };

    write_u32(MOLLY_MAGIC);
    write_u32(MOLLY_VERSION);
    write_u32(static_cast<uint32_t>(state.n_genes));

    for (const auto& gene : state.genes) {
        write_str(gene.name);
        uint32_t n_slices = static_cast<uint32_t>(
            gene.slice_defs.empty() ? gene.param_names.size() : gene.slice_defs.size());
        write_u32(n_slices);

        // Write slice definitions (if any)
        uint8_t has_slices = gene.slice_defs.empty() ? 0 : 1;
        out.write(reinterpret_cast<char*>(&has_slices), 1);

        if (has_slices) {
            for (const auto& sd : gene.slice_defs) {
                write_str(sd.param_name);
                write_u32(static_cast<uint32_t>(sd.dim));
                write_u32(static_cast<uint32_t>(sd.start));
                write_u32(static_cast<uint32_t>(sd.end));
            }
        } else {
            for (const auto& pn : gene.param_names) {
                write_str(pn);
            }
        }

        // Write complement data and scales
        for (size_t j = 0; j < gene.complement_data.size(); j++) {
            write_f32(gene.scales[j]);
            uint32_t n_elem = static_cast<uint32_t>(gene.complement_data[j].size());
            write_u32(n_elem);
            out.write(reinterpret_cast<const char*>(gene.complement_data[j].data()),
                      n_elem * sizeof(int16_t));
        }
    }

    out.close();
    return true;
}


// ── Load ────────────────────────────────────────────────────────────────────

bool load_genome_gguf(const std::string& gguf_path,
                      GenomeState& state) {
    std::ifstream in(gguf_path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        std::cerr << "Failed to open GGUF: " << gguf_path << std::endl;
        return false;
    }

    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    // Read entire file
    std::vector<char> data(file_size);
    in.read(data.data(), file_size);
    in.close();

    // Search for MOLLY_MAGIC from the end
    size_t molly_offset = std::string::npos;
    for (size_t i = data.size() - 8; i >= 4; i--) {
        uint32_t magic;
        std::memcpy(&magic, &data[i], 4);
        if (magic == MOLLY_MAGIC) {
            molly_offset = i;
            break;
        }
        if (i == 0) break;
    }

    if (molly_offset == std::string::npos) {
        std::cerr << "No Molly genome data found in GGUF file" << std::endl;
        return false;
    }

    // Parse Molly appendix
    size_t pos = molly_offset;
    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        std::memcpy(&v, &data[pos], 4);
        pos += 4;
        return v;
    };
    auto read_f32 = [&]() -> float {
        float v;
        std::memcpy(&v, &data[pos], 4);
        pos += 4;
        return v;
    };
    auto read_str = [&]() -> std::string {
        uint32_t len = read_u32();
        std::string s(&data[pos], len);
        pos += len;
        return s;
    };

    uint32_t magic = read_u32();
    uint32_t version = read_u32();
    if (magic != MOLLY_MAGIC || version != MOLLY_VERSION) {
        std::cerr << "Invalid Molly genome header" << std::endl;
        return false;
    }

    state.n_genes = static_cast<int>(read_u32());
    state.genes.resize(state.n_genes);

    for (int i = 0; i < state.n_genes; i++) {
        auto& gene = state.genes[i];
        gene.name = read_str();
        uint32_t n_slices = read_u32();

        uint8_t has_slices = data[pos];
        pos += 1;

        if (has_slices) {
            gene.slice_defs.resize(n_slices);
            for (uint32_t j = 0; j < n_slices; j++) {
                gene.slice_defs[j].param_name = read_str();
                gene.slice_defs[j].dim = static_cast<int>(read_u32());
                gene.slice_defs[j].start = static_cast<int>(read_u32());
                gene.slice_defs[j].end = static_cast<int>(read_u32());
            }
        } else {
            gene.param_names.resize(n_slices);
            for (uint32_t j = 0; j < n_slices; j++) {
                gene.param_names[j] = read_str();
            }
        }

        // Read complement data
        gene.complement_data.resize(n_slices);
        gene.scales.resize(n_slices);
        for (uint32_t j = 0; j < n_slices; j++) {
            gene.scales[j] = read_f32();
            uint32_t n_elem = read_u32();
            gene.complement_data[j].resize(n_elem);
            std::memcpy(gene.complement_data[j].data(),
                        &data[pos], n_elem * sizeof(int16_t));
            pos += n_elem * sizeof(int16_t);
        }
    }

    return true;
}

}  // namespace molly
