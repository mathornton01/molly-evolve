"""
DualGenome — high-level API for gene conversion with C++ acceleration.

Wraps TransformerDualGenome with automatic backend dispatch:
  - CUDA: uses compiled kernels for quantize/repair/writeback
  - Python: falls back to existing pure-Python implementation
"""

import logging
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

    def apply_conversion(self, scores, threshold=0.50, alpha=0.3):
        """
        Apply gene conversion based on Bayesian scores.

        Returns (n_repaired, n_fixed).
        """
        repaired, fixed = [], []
        for s in scores:
            trade = s["p_del_prev"] - alpha * s["p_ben_curr"]
            if trade > threshold:
                repaired.append(s["gene_id"])
            elif s["p_del_prev"] < 0.3:
                fixed.append(s["gene_id"])
        if repaired:
            self.repair_genes(repaired)
        if fixed:
            self.fix_genes(fixed)
        return len(repaired), len(fixed)

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
