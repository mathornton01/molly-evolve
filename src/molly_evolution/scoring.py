"""
GeneScorer — vectorized gene scoring with C++ acceleration.

Supports two memory modes:
  - precomputed: all deltas on GPU (fast, O(N_params) GPU memory)
  - streaming: deltas computed on-the-fly (low memory, one param at a time)

Auto-selects streaming for models >500M params.

On CUDA: single kernel launch per scoring call (precomputed mode).
On CPU: falls back to optimized Python with GPU tensor ops.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from scipy import stats

logger = logging.getLogger("molly_evolution")

try:
    import molly_evolution._C as _C
    _HAS_CUDA = True
except ImportError:
    _C = None
    _HAS_CUDA = False


def _gpu_mem_mb():
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


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
        streaming: Memory mode. None=auto, True=streaming, False=precomputed.
    """

    def __init__(self, genome, model, device, use_amp=True, streaming=None):
        self.genome = genome
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.n_genes = genome.total_genes

        # Access underlying genome object if wrapped
        self._raw_genome = getattr(genome, '_genome', genome)

        self._use_cuda_kernel = (_HAS_CUDA and device.type == "cuda")

        # Auto-select memory mode
        n_params = sum(p.numel() for p in model.parameters())
        if streaming is None:
            self.streaming = n_params > 500_000_000
        else:
            self.streaming = streaming

        if self.streaming:
            logger.info("GeneScorer: streaming mode (low memory)")
            self._build_ref_map()
        else:
            logger.info("GeneScorer: precomputed mode (fast)")
            self._build_repair_map()

    # ── Precomputed mode (original) ──────────────────────────────

    def _build_repair_map(self):
        """Precompute repair directions on GPU for all genes."""
        t0 = time.perf_counter()
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

        dt = time.perf_counter() - t0
        logger.info(f"  build_repair_map: {dt:.3f}s | GPU: {_gpu_mem_mb():.0f} MB")

    def _build_flat_buffers(self):
        """Flatten all gene data into contiguous GPU buffers for CUDA kernel."""
        gene_offsets = [0]
        all_deltas = []
        current_offset = 0

        for gid in range(self.n_genes):
            gene = self._raw_genome.genes[gid]

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

    # ── Streaming mode (low memory) ──────────────────────────────

    def _build_ref_map(self):
        """Build lightweight reference map for streaming mode.

        Stores gene object references instead of materialized delta tensors.
        Gene objects are updated in-place by sync_primary, so refs stay valid.
        """
        t0 = time.perf_counter()
        self.param_gene_refs = {}

        for gid, gene in enumerate(self._raw_genome.genes):
            if hasattr(gene, 'slice_defs'):
                for pname, dim, start, end in gene.slice_defs:
                    key = gene._key(pname, dim, start, end)
                    self.param_gene_refs.setdefault(pname, []).append(
                        (gid, dim, start, end, key, gene))
            else:
                for pn in gene.param_names:
                    self.param_gene_refs.setdefault(pn, []).append(
                        (gid, None, None, None, pn, gene))

        dt = time.perf_counter() - t0
        logger.info(f"  build_ref_map: {dt:.3f}s | GPU: {_gpu_mem_mb():.0f} MB")

    # ── Scoring ──────────────────────────────────────────────────

    def _score_one_eval(self, enc) -> np.ndarray:
        """One forward+backward -> all gene scores."""
        t0 = time.perf_counter()
        self.model.zero_grad()
        self.model.train()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        # Forward+backward with optional AMP
        if self.use_amp and self.device.type == "cuda":
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

        t_fwdbwd = time.perf_counter() - t0

        # Compute per-gene dot products
        t1 = time.perf_counter()
        if self._use_cuda_kernel and not self.streaming and self._deltas_flat is not None:
            gene_scores = _C.batched_gene_score(
                self.model, self._deltas_flat, self._gene_offsets,
                self.param_gene_map)
        elif self.streaming:
            gene_scores = self._score_python_streaming()
        else:
            gene_scores = self._score_python()

        t_dot = time.perf_counter() - t1
        logger.debug(f"    fwd+bwd: {t_fwdbwd:.3f}s | dot products: {t_dot:.3f}s | "
                     f"GPU: {_gpu_mem_mb():.0f} MB")

        self.model.zero_grad()
        self.model.eval()
        return gene_scores

    def _score_python(self) -> np.ndarray:
        """Python fallback for gene scoring (precomputed deltas)."""
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

    def _score_python_streaming(self) -> np.ndarray:
        """Streaming scoring: compute deltas on-the-fly, one param at a time.

        Peak GPU memory = model + gradients + ONE delta tensor (not all).
        """
        gene_scores = torch.zeros(self.n_genes, device=self.device)
        params_dict = dict(self.model.named_parameters())

        for pname, entries in self.param_gene_refs.items():
            grad = params_dict[pname].grad
            if grad is None:
                continue
            grad_f = grad.float()

            for gid, dim, start, end, comp_key, gene in entries:
                # Compute delta on-the-fly from int16 strands (CPU -> GPU)
                comp = gene._dequantize(gene.complement[comp_key],
                                        gene.scales[comp_key])
                prim = gene._dequantize(gene.primary[comp_key],
                                        gene.scales[comp_key])
                delta = (comp - prim).float().to(self.device)

                if dim is not None:
                    g_slice = grad_f.narrow(dim, start, end - start)
                else:
                    g_slice = grad_f

                gene_scores[gid] += (g_slice * delta).sum()
                del delta, comp, prim  # free immediately

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
        t0 = time.perf_counter()
        self._raw_genome.sync_primary()

        # Only rebuild precomputed deltas (streaming refs stay valid)
        if not self.streaming:
            self._build_repair_map()

        # Score against all previous objectives
        all_prev = {}
        for name, enc in eval_sets:
            t1 = time.perf_counter()
            all_prev[name] = self._score_one_eval(enc)
            logger.info(f"  scored '{name}': {time.perf_counter()-t1:.3f}s")

        # Score against current domain
        t1 = time.perf_counter()
        curr_scores = self._score_one_eval(curr_eval)
        logger.info(f"  scored 'current': {time.perf_counter()-t1:.3f}s")

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

        dt = time.perf_counter() - t0
        logger.info(f"  total scoring: {dt:.3f}s | {n} genes × "
                    f"{len(eval_sets)+1} objectives")
        return scores

    def memory_estimate(self) -> dict:
        """Estimate memory footprint in MB."""
        n_params = sum(p.numel() for p in self.model.parameters())
        param_dtype = next(self.model.parameters()).dtype
        bytes_per_param = 2 if param_dtype == torch.float16 else 4

        model_mb = n_params * bytes_per_param / 1024 / 1024
        grads_mb = n_params * 4 / 1024 / 1024  # fp32 grads
        genome_cpu_mb = n_params * 4 / 1024 / 1024  # primary + complement int16

        if self.streaming:
            # Only one delta at a time
            max_param = max(p.numel() for p in self.model.parameters())
            delta_mb = max_param * 4 / 1024 / 1024
        else:
            delta_mb = n_params * 4 / 1024 / 1024  # all deltas

        return {
            "model_gpu_mb": model_mb,
            "gradients_gpu_mb": grads_mb,
            "deltas_gpu_mb": delta_mb,
            "genome_cpu_mb": genome_cpu_mb,
            "peak_scoring_gpu_mb": model_mb + grads_mb + delta_mb,
            "n_params": n_params,
            "mode": "streaming" if self.streaming else "precomputed",
        }
