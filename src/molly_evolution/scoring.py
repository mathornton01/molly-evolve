"""
GeneScorer — vectorized gene scoring with C++ acceleration.

Precomputes contiguous mapping from model parameters to gene scores.
On CUDA: single kernel launch per scoring call.
On CPU: falls back to optimized Python with GPU tensor ops.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from scipy import stats

try:
    import molly_evolution._C as _C
    _HAS_CUDA = True
except ImportError:
    _C = None
    _HAS_CUDA = False


class GeneScorer:
    """
    Fast gene scoring via gradient approximation.

    Uses first-order Taylor expansion to approximate chimeric evaluation:
        delta_loss_g ~ grad_L . (w_complement - w_primary)

    This replaces O(N_genes * N_objectives) forward passes with
    O(N_objectives) forward+backward passes.

    Args:
        genome: DualGenome instance (or TransformerDualGenome).
        model: The transformer model.
        device: torch device.
        use_amp: Whether to use mixed precision for forward/backward.
    """

    def __init__(self, genome, model, device, use_amp=True):
        self.genome = genome
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.n_genes = genome.total_genes

        # Access underlying genome object if wrapped
        self._raw_genome = getattr(genome, '_genome', genome)

        self._use_cuda_kernel = (_HAS_CUDA and device.type == "cuda")
        self._build_repair_map()

    def _build_repair_map(self):
        """Precompute repair directions on GPU for all genes."""
        self.param_gene_map = {}

        for gid, gene in enumerate(self._raw_genome.genes):
            if hasattr(gene, 'slice_defs'):
                for pname, dim, start, end in gene.slice_defs:
                    key = gene._key(pname, dim, start, end)
                    comp = gene._dequantize(gene.complement[key],
                                            gene.scales[key]).cpu()
                    prim = gene._dequantize(gene.primary[key],
                                            gene.scales[key]).cpu()
                    delta = (comp - prim).float().to(self.device)
                    self.param_gene_map.setdefault(pname, []).append(
                        (gid, dim, start, end, delta))
            else:
                for pn in gene.param_names:
                    comp = gene._dequantize(gene.complement[pn],
                                            gene.scales[pn]).cpu()
                    prim = gene._dequantize(gene.primary[pn],
                                            gene.scales[pn]).cpu()
                    delta = (comp - prim).float().to(self.device)
                    self.param_gene_map.setdefault(pn, []).append(
                        (gid, None, None, None, delta))

        if self._use_cuda_kernel:
            self._build_flat_buffers()

    def _build_flat_buffers(self):
        """Flatten all gene data into contiguous GPU buffers for CUDA kernel."""
        segments = []  # list of (gene_id, delta_tensor_flat)
        gene_offsets = [0]
        grad_gather = []  # maps delta position to gradient position

        # Build flattened param order
        param_names = list(dict(self.model.named_parameters()).keys())
        param_offsets = {}
        offset = 0
        for pn in param_names:
            param_offsets[pn] = offset
            offset += dict(self.model.named_parameters())[pn].numel()

        all_deltas = []
        all_grad_indices = []
        current_offset = 0

        for gid in range(self.n_genes):
            gene = self._raw_genome.genes[gid]
            gene_start = current_offset

            if hasattr(gene, 'slice_defs'):
                for pname, dim, start, end in gene.slice_defs:
                    entries = [e for e in self.param_gene_map.get(pname, [])
                               if e[0] == gid and e[1] == dim and
                               e[2] == start and e[3] == end]
                    if entries:
                        delta = entries[0][4]
                        all_deltas.append(delta.flatten())
                        current_offset += delta.numel()
            else:
                for pn in gene.param_names:
                    entries = [e for e in self.param_gene_map.get(pn, [])
                               if e[0] == gid]
                    if entries:
                        delta = entries[0][4]
                        all_deltas.append(delta.flatten())
                        current_offset += delta.numel()

            gene_offsets.append(current_offset)

        if all_deltas:
            self._deltas_flat = torch.cat(all_deltas).to(self.device)
            self._gene_offsets = torch.tensor(gene_offsets, dtype=torch.int64,
                                              device=self.device)
        else:
            self._deltas_flat = None

    def _score_one_eval(self, enc) -> np.ndarray:
        """One forward+backward -> all gene scores."""
        self.model.zero_grad()
        self.model.train()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        # Forward+backward with optional AMP
        if self.use_amp:
            with autocast("cuda", dtype=torch.float16):
                labels = ids.clone()
                labels[mask == 0] = -100
                out = self.model(ids, attention_mask=mask, labels=labels)
            out.loss.backward()
        else:
            labels = ids.clone()
            labels[mask == 0] = -100
            out = self.model(ids, attention_mask=mask, labels=labels)
            out.loss.backward()

        # Compute per-gene dot products
        if self._use_cuda_kernel and self._deltas_flat is not None:
            gene_scores = _C.batched_gene_score(
                self.model, self._deltas_flat, self._gene_offsets,
                self.param_gene_map)
        else:
            gene_scores = self._score_python()

        self.model.zero_grad()
        self.model.eval()
        return gene_scores

    def _score_python(self) -> np.ndarray:
        """Python fallback for gene scoring."""
        gene_scores = torch.zeros(self.n_genes, device=self.device)
        params_dict = dict(self.model.named_parameters())

        for pname, entries in self.param_gene_map.items():
            grad = params_dict[pname].grad
            if grad is None:
                continue
            grad_f = grad.float()
            for gid, dim, start, end, delta in entries:
                if dim is not None:
                    g_slice = grad_f.narrow(dim, start, end - start)
                else:
                    g_slice = grad_f
                gene_scores[gid] += (g_slice * delta).sum()

        return gene_scores.cpu().numpy()

    def score_multi_objective(self, eval_sets, curr_eval,
                              threshold=0.50, alpha=0.3):
        """
        Score all genes against multiple objectives.

        Args:
            eval_sets: list of (name, encodings) for objectives to protect.
            curr_eval: encodings for the current domain being learned.

        Returns:
            List of dicts with gene_id, p_del_prev, p_ben_curr.
        """
        self._raw_genome.sync_primary()
        self._build_repair_map()

        # Score against all previous objectives
        all_prev = {}
        for name, enc in eval_sets:
            all_prev[name] = self._score_one_eval(enc)

        # Score against current domain
        curr_scores = self._score_one_eval(curr_eval)

        # Convert to Bayesian posteriors
        prev_names = [name for name, _ in eval_sets]
        n = self.n_genes

        raw_scores = []
        for gid in range(n):
            deltas_prev = {name: float(-all_prev[name][gid])
                           for name in prev_names}
            delta_curr = float(curr_scores[gid])
            raw_scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                               "delta_curr": delta_curr})

        var_prev = {name: max(np.var([s["deltas_prev"][name]
                                      for s in raw_scores]), 1e-10)
                    for name in prev_names}
        var_curr = max(np.var([s["delta_curr"] for s in raw_scores]), 1e-10)

        scores = []
        for s in raw_scores:
            p_del = {}
            for name in prev_names:
                sig = np.sqrt(var_prev[name] / 2)
                p_del[name] = 1 - stats.norm.cdf(
                    0, loc=s["deltas_prev"][name] / 2, scale=sig)
            p_del_prev = max(p_del.values())
            sig_c = np.sqrt(var_curr / 2)
            p_ben_curr = 1 - stats.norm.cdf(
                0, loc=s["delta_curr"] / 2, scale=sig_c)
            scores.append({"gene_id": s["gene_id"],
                           "p_del_prev": p_del_prev,
                           "p_ben_curr": p_ben_curr})
        return scores
