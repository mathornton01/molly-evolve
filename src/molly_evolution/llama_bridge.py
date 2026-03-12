"""
llama_bridge.py — Python interface for llama.cpp gene conversion.

Provides a high-level API for running Molly Evolution on GGUF models
without PyTorch, using the compiled C++ llama integration library.

Usage:
    from molly_evolution.llama_bridge import LlamaGenome

    genome = LlamaGenome("model.gguf")
    genome.snapshot()
    # ... external fine-tuning modifies model weights ...
    genome.repair_genes([0, 3, 15])
    genome.save("model_repaired.gguf")
"""

import ctypes
import os
import json
from pathlib import Path
from typing import List, Optional


def _find_library():
    """Locate the compiled molly_llama shared library."""
    # Search paths
    candidates = [
        Path(__file__).parent / "libmolly_llama.so",
        Path(__file__).parent / "molly_llama.dll",
        Path(__file__).parent / "libmolly_llama.dylib",
        Path(__file__).parent.parent.parent / "csrc" / "llama" / "build" / "libmolly_llama.so",
        Path(__file__).parent.parent.parent / "csrc" / "llama" / "build" / "molly_llama.dll",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


class LlamaGenome:
    """
    Gene conversion on GGUF models via llama.cpp integration.

    This is the CPU/edge deployment path. For GPU training with PyTorch,
    use DualGenome instead.
    """

    def __init__(self, model_path: str):
        """
        Load a GGUF model with embedded genome data.

        Args:
            model_path: Path to GGUF file (with or without Molly genome appendix).
        """
        self.model_path = model_path

        lib_path = _find_library()
        if lib_path is None:
            raise RuntimeError(
                "molly_llama library not found. Build it with:\n"
                "  cd csrc/llama && mkdir build && cd build\n"
                "  cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp\n"
                "  make -j$(nproc)"
            )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_bindings()
        self._load(model_path)

    def _setup_bindings(self):
        """Set up ctypes function signatures."""
        # These will be bound once the C API is finalized
        pass

    def _load(self, path: str):
        """Load genome state from GGUF file."""
        self._model_path = path
        # Implementation will call into C++ library

    def snapshot(self):
        """Mark current weights as healthy reference."""
        raise NotImplementedError(
            "Direct llama.cpp snapshot requires model loading. "
            "Use the C++ API directly or export from PyTorch with "
            "LlamaGenome.export_from_pytorch()."
        )

    def repair_genes(self, gene_ids: List[int]):
        """Restore specified genes from complement strand."""
        raise NotImplementedError("Coming in v0.2.0")

    def fix_genes(self, gene_ids: List[int]):
        """Copy current weights to complement for specified genes."""
        raise NotImplementedError("Coming in v0.2.0")

    def save(self, out_path: str):
        """Save model with updated genome data."""
        raise NotImplementedError("Coming in v0.2.0")

    @staticmethod
    def export_from_pytorch(dual_genome, out_path: str):
        """
        Export a PyTorch DualGenome's complement strands to GGUF format.

        This bridges PyTorch training with llama.cpp deployment:
        1. Train + gene convert in PyTorch (GPU)
        2. Export genome state to GGUF
        3. Deploy with llama.cpp (CPU)

        Args:
            dual_genome: A DualGenome instance with trained complement strands.
            out_path: Path for the output genome state file (.json).
        """
        raw = getattr(dual_genome, '_genome', dual_genome)
        state = {
            "n_genes": raw.total_genes,
            "genes": []
        }

        for gene in raw.genes:
            gene_data = {"name": gene.name}

            if hasattr(gene, 'slice_defs'):
                gene_data["slice_defs"] = [
                    {"param_name": pn, "dim": d, "start": s, "end": e}
                    for pn, d, s, e in gene.slice_defs
                ]
                gene_data["complement"] = {}
                gene_data["scales"] = {}
                for pn, d, s, e in gene.slice_defs:
                    key = gene._key(pn, d, s, e)
                    gene_data["complement"][key] = gene.complement[key].tolist()
                    gene_data["scales"][key] = float(gene.scales[key])
            else:
                gene_data["param_names"] = gene.param_names
                gene_data["complement"] = {
                    pn: gene.complement[pn].tolist()
                    for pn in gene.param_names
                }
                gene_data["scales"] = {
                    pn: float(gene.scales[pn])
                    for pn in gene.param_names
                }

            state["genes"].append(gene_data)

        with open(out_path, 'w') as f:
            json.dump(state, f)

        return out_path
