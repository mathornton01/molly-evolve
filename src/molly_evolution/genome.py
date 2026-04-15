"""
DualGenome — high-level API for gene conversion with C++ acceleration.

Wraps TransformerDualGenome with automatic backend dispatch:
  - CUDA: uses compiled kernels for quantize/repair/writeback
  - Python: falls back to existing pure-Python implementation
"""

import logging
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from scipy import stats

# Import existing implementation
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gene_conversion.transformer_genome import (
    TransformerDualGenome, TransformerGene, SlicedTransformerGene,
)

logger = logging.getLogger("molly_evolution")

# Try to load C++ extension
try:
    import molly_evolution._C as _C
    _HAS_CUDA = True
except ImportError:
    _C = None
    _HAS_CUDA = False


class DualGenome:
    """
    Dual-genome wrapper for transformer models.

    Maintains primary (active) and complement (reference) weight strands.
    Supports component-level (75 genes) or head-level (207 genes) granularity.

    Example:
        genome = DualGenome(model, granularity="head")
        genome.snapshot()       # mark current weights as healthy reference
        # ... fine-tune model on new domain ...
        genome.sync_primary()   # update primary strand from model
        # ... score genes and decide which to repair ...
        genome.repair_genes([0, 3, 15, 42])  # restore damaged genes
        genome.snapshot()       # update reference for next cycle
    """

    def __init__(self, model: nn.Module, n_bits: int = 16,
                 granularity: str = "head", backend: str = "auto"):
        """
        Args:
            model: The transformer model to wrap.
            n_bits: Quantization bits for dual-strand encoding (default 16).
            granularity: "component" (75 genes) or "head" (207 genes).
            backend: "auto" (try CUDA, fall back to Python), "cuda", or "python".
        """
        self.model = model
        self.n_bits = n_bits
        self.granularity = granularity

        if backend == "auto":
            self._use_cuda = _HAS_CUDA and torch.cuda.is_available()
        elif backend == "cuda":
            if not _HAS_CUDA:
                raise RuntimeError("CUDA extension not available. "
                                   "Install with: pip install molly-evolution[cuda]")
            self._use_cuda = True
        else:
            self._use_cuda = False

        # Build gene map using existing implementation
        self._genome = TransformerDualGenome(model, n_bits=n_bits,
                                            granularity=granularity)

        if self._use_cuda:
            self._build_cuda_buffers()

    def _build_cuda_buffers(self):
        """Pre-flatten gene data into contiguous GPU buffers for CUDA kernels."""
        device = next(self.model.parameters()).device

        # Build flat index of all gene parameter segments
        self._flat_deltas = []
        self._gene_offsets = [0]
        self._gene_param_map = []  # (param_name, dim, start, end) per segment

        for gene in self._genome.genes:
            n_elements = 0
            if hasattr(gene, 'slice_defs'):
                for pname, dim, start, end in gene.slice_defs:
                    self._gene_param_map.append((pname, dim, start, end))
                    # Count elements in this slice
                    params = dict(self.model.named_parameters())
                    shape = list(params[pname].shape)
                    shape[dim] = end - start
                    n_elements += int(np.prod(shape))
            else:
                for pn in gene.param_names:
                    self._gene_param_map.append((pn, None, None, None))
                    params = dict(self.model.named_parameters())
                    n_elements += params[pn].numel()
            self._gene_offsets.append(self._gene_offsets[-1] + n_elements)

        self._gene_offsets = torch.tensor(self._gene_offsets, dtype=torch.int64,
                                          device=device)

    # ── Delegate to underlying genome ───────────────────────────

    @property
    def genes(self):
        return self._genome.genes

    @property
    def total_genes(self) -> int:
        return self._genome.total_genes

    def snapshot(self):
        """Mark current model weights as healthy reference (complement)."""
        t0 = time.perf_counter()
        if self._use_cuda and _C is not None:
            _C.batch_snapshot(self._genome, self.model)
        else:
            self._genome.snapshot()
        dt = time.perf_counter() - t0
        logger.info(f"snapshot: {dt:.3f}s | {self.total_genes} genes")

    def sync_primary(self):
        """Update primary strands from current model weights."""
        t0 = time.perf_counter()
        if self._use_cuda and _C is not None:
            _C.batch_sync(self._genome, self.model)
        else:
            self._genome.sync_primary()
        dt = time.perf_counter() - t0
        logger.info(f"sync_primary: {dt:.3f}s | {self.total_genes} genes")

    def apply_primary(self):
        """Write primary strands back to model."""
        self._genome.apply_primary()

    def repair_genes(self, gene_ids: List[int]):
        """Purifying selection: restore specified genes from complement."""
        t0 = time.perf_counter()
        if self._use_cuda and _C is not None:
            _C.batch_repair(self._genome, self.model, gene_ids)
        else:
            self._genome.repair_genes(gene_ids)
        dt = time.perf_counter() - t0
        logger.info(f"repair: {dt:.3f}s | {len(gene_ids)}/{self.total_genes} genes")

    def fix_genes(self, gene_ids: List[int]):
        """Adaptive selection: copy primary to complement for specified genes."""
        t0 = time.perf_counter()
        self._genome.fix_genes(gene_ids)
        dt = time.perf_counter() - t0
        logger.info(f"fix: {dt:.3f}s | {len(gene_ids)}/{self.total_genes} genes")

    def gene_summary(self):
        return self._genome.gene_summary()

    # ── Gene conversion decision logic ──────────────────────────

    def select_conversion_genes(self, scores, threshold=0.80, alpha=0.3,
                                max_repair_pct=0.03, max_repair_count=None):
        """
        Select genes for repair/fix based on Bayesian scores.

        Returns (to_repair, to_fix) — sorted lists of gene IDs.
        to_repair is sorted by trade score descending (worst genes first).
        """
        candidates = []
        fixed = []
        for s in scores:
            trade = s["p_del_prev"] - alpha * s["p_ben_curr"]
            if trade > threshold:
                candidates.append((trade, s["gene_id"]))
            elif s["p_del_prev"] < 0.3:
                fixed.append(s["gene_id"])

        # Cap repairs: percentage-based with optional absolute cap
        max_n = max(int(self.total_genes * max_repair_pct), 1)
        if max_repair_count is not None:
            max_n = min(max_n, max_repair_count)

        # Sort by trade score descending, take the most damaging genes first
        candidates.sort(reverse=True)
        to_repair = [gid for _, gid in candidates[:max_n]]

        n_skipped = len(candidates) - len(to_repair)
        if n_skipped > 0:
            logger.info(f"  repair cap: {len(to_repair)}/{len(candidates)} "
                        f"candidates selected (max {max_repair_pct*100:.0f}% "
                        f"= {max_n} genes)")

        return to_repair, fixed

    def apply_conversion(self, scores, threshold=0.80, alpha=0.3,
                         max_repair_pct=0.03, max_repair_count=None):
        """
        Apply gene conversion based on Bayesian scores.

        Returns (n_repaired, n_fixed).
        """
        to_repair, to_fix = self.select_conversion_genes(
            scores, threshold=threshold, alpha=alpha,
            max_repair_pct=max_repair_pct,
            max_repair_count=max_repair_count)
        if to_repair:
            self.repair_genes(to_repair)
        if to_fix:
            self.fix_genes(to_fix)
        return len(to_repair), len(to_fix)

    # ── Save / Load ──────────────────────────────────────────────

    # Filename for the serialized genome state inside a save directory.
    GENOME_FILENAME = "molli_genome.pt"

    def save(self, path: str):
        """
        Save the dual-strand genome state to disk.

        Writes a single torch file containing the quantized complement and
        primary strands plus gene metadata. Can be loaded later with
        ``DualGenome.load(path, model)`` to resume training or redeploy.

        Args:
            path: Either a file path (``*.pt``) or a directory. If a
                directory, the state is written to ``<dir>/molli_genome.pt``.
        """
        # Normalize path: if it's a directory, append the canonical filename.
        if os.path.isdir(path) or path.endswith(os.sep) or not path.endswith(".pt"):
            os.makedirs(path, exist_ok=True)
            out = os.path.join(path, self.GENOME_FILENAME)
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            out = path

        t0 = time.perf_counter()
        gene_states = []
        for gene in self._genome.genes:
            entry = {
                "name": gene.name,
                "n_bits": gene.n_bits,
                "complement": {k: v.cpu() for k, v in gene.complement.items()},
                "primary": {k: v.cpu() for k, v in gene.primary.items()},
                "scales": dict(gene.scales),
                "complement_scales": dict(gene.complement_scales),
            }
            if hasattr(gene, "slice_defs"):
                entry["type"] = "sliced"
                entry["slice_defs"] = list(gene.slice_defs)
            else:
                entry["type"] = "full"
                entry["param_names"] = list(gene.param_names)
            gene_states.append(entry)

        payload = {
            "format_version": 1,
            "n_bits": self.n_bits,
            "granularity": self.granularity,
            "total_genes": self.total_genes,
            "genes": gene_states,
        }
        torch.save(payload, out)
        dt = time.perf_counter() - t0
        logger.info(f"save: {dt:.3f}s | {self.total_genes} genes -> {out}")
        return out

    @classmethod
    def load(cls, path: str, model: nn.Module, backend: str = "auto",
             apply_to_model: bool = True) -> "DualGenome":
        """
        Load a previously-saved dual-strand genome and attach it to ``model``.

        The genome map is rebuilt from the model using the saved
        ``granularity`` setting, then the complement / primary strands and
        scales are overwritten from the file. The caller is responsible for
        ensuring ``model`` has the same architecture as the one that
        produced the saved state.

        Args:
            path: File path or directory containing ``molli_genome.pt``.
            model: A freshly-loaded HuggingFace model to attach the genome to.
            backend: Same semantics as ``DualGenome(..., backend=...)``.
            apply_to_model: If ``True`` (default) write the primary strand
                back to the model, which is what you want when loading a
                standalone genome file. Set to ``False`` when ``model``
                already holds the authoritative (lossless) weights — e.g.
                inside :meth:`MolliTrainer.from_pretrained` — to avoid
                introducing int16 quantization error.
        """
        if os.path.isdir(path):
            file_path = os.path.join(path, cls.GENOME_FILENAME)
        else:
            file_path = path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No genome state found at {file_path}")

        payload = torch.load(file_path, map_location="cpu", weights_only=False)
        granularity = payload.get("granularity", "head")
        n_bits = payload.get("n_bits", 16)

        genome = cls(model, n_bits=n_bits, granularity=granularity,
                     backend=backend)

        saved_genes = payload.get("genes", [])
        if len(saved_genes) != genome.total_genes:
            raise ValueError(
                f"Gene count mismatch: saved state has {len(saved_genes)} "
                f"genes but the model produces {genome.total_genes}. "
                f"Make sure you're loading into the same model architecture.")

        # Overwrite quantized state gene-by-gene.
        for gene, saved in zip(genome._genome.genes, saved_genes):
            gene.complement = {k: v for k, v in saved["complement"].items()}
            gene.primary = {k: v for k, v in saved["primary"].items()}
            gene.scales = dict(saved["scales"])
            gene.complement_scales = dict(saved["complement_scales"])

        if apply_to_model:
            # Apply the primary strand to the model so inference works
            # immediately after loading when no model weights were provided.
            genome._genome.apply_primary()
        logger.info(f"load: {genome.total_genes} genes from {file_path}")
        return genome

    # ── Memory estimation ───────────────────────────────────────

    def memory_footprint(self) -> dict:
        """Estimate memory footprint in MB."""
        n_params = sum(p.numel() for p in self.model.parameters())
        param_dtype = next(self.model.parameters()).dtype
        model_bytes = n_params * (2 if param_dtype == torch.float16 else 4)
        genome_bytes = n_params * 4  # primary + complement (int16 each)

        return {
            "n_params": n_params,
            "n_genes": self.total_genes,
            "model_gpu_mb": model_bytes / 1024 / 1024,
            "genome_cpu_mb": genome_bytes / 1024 / 1024,
            "granularity": self.granularity,
        }
