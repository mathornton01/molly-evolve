"""
GeneScorer — Bayesian gene scoring with empirical Bayes shrinkage.

Scoring pipeline:
  1. Taylor approximation: delta_g ~ grad(L) . (w_complement - w_primary)
  2. Split-half noise estimation: score on two data halves independently
  3. Empirical Bayes: shrink gene effects toward population mean

The posterior P(deleterious | data) and P(beneficial | data) are properly
calibrated probabilities that account for observation noise.

Supports two memory modes:
  - precomputed: all deltas on GPU (fast, O(N_params) GPU memory)
  - streaming: deltas computed on-the-fly (low memory, one param at a time)
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
    Bayesian gene scoring via gradient approximation + empirical Bayes.

    Uses first-order Taylor expansion to approximate chimeric evaluation:
        delta_loss_g ~ grad_L . (w_complement - w_primary)

    Then applies empirical Bayes shrinkage to produce calibrated posterior
    probabilities P(deleterious | data) and P(beneficial | data).

    The noise variance is estimated via split-half scoring: the eval data
    is split in two, scored independently, and the disagreement between
    halves estimates observation noise. This separates true signal (tau^2)
    from noise (sigma^2), enabling proper shrinkage.

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
                                            gene.complement_scales[key]).cpu()
                    prim = gene._dequantize(gene.primary[key],
                                            gene.scales[key]).cpu()
                    delta = (comp - prim).float().to(self.device)
                    self.param_gene_map.setdefault(pname, []).append(
                        (gid, dim, start, end, delta))
            else:
                for pn in gene.param_names:
                    comp = gene._dequantize(gene.complement[pn],
                                            gene.complement_scales[pn]).cpu()
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
        """Build lightweight reference map for streaming mode."""
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

    # ── Low-level scoring ────────────────────────────────────────

    def _score_one_eval(self, enc, scoring_batch_size=4) -> np.ndarray:
        """One forward+backward -> all gene scores. Batched to limit memory."""
        t0 = time.perf_counter()
        self.model.zero_grad()
        self.model.train()
        ids_all = enc["input_ids"].to(self.device)
        mask_all = enc["attention_mask"].to(self.device)
        n_samples = ids_all.size(0)

        # Accumulate gradients over mini-batches to limit activation memory
        for b_start in range(0, n_samples, scoring_batch_size):
            b_end = min(b_start + scoring_batch_size, n_samples)
            ids = ids_all[b_start:b_end]
            mask = mask_all[b_start:b_end]

            if self.use_amp and self.device.type == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    labels = ids.clone()
                    labels[mask == 0] = -100
                    out = self.model(ids, attention_mask=mask, labels=labels)
                    # Scale loss to match full-batch average
                    scaled_loss = out.loss * (b_end - b_start) / n_samples
                scaled_loss.backward()
            else:
                labels = ids.clone()
                labels[mask == 0] = -100
                out = self.model(ids, attention_mask=mask, labels=labels)
                scaled_loss = out.loss * (b_end - b_start) / n_samples
                scaled_loss.backward()

            del ids, mask, labels, out, scaled_loss
            torch.cuda.empty_cache()

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
        """Streaming scoring: compute deltas on-the-fly, one param at a time."""
        gene_scores = torch.zeros(self.n_genes, device=self.device)
        params_dict = dict(self.model.named_parameters())

        for pname, entries in self.param_gene_refs.items():
            grad = params_dict[pname].grad
            if grad is None:
                continue
            grad_f = grad.float()

            for gid, dim, start, end, comp_key, gene in entries:
                comp = gene._dequantize(gene.complement[comp_key],
                                        gene.complement_scales[comp_key])
                prim = gene._dequantize(gene.primary[comp_key],
                                        gene.scales[comp_key])
                delta = (comp - prim).float().to(self.device)

                if dim is not None:
                    g_slice = grad_f.narrow(dim, start, end - start)
                else:
                    g_slice = grad_f

                gene_scores[gid] += (g_slice * delta).sum()
                del delta, comp, prim

        return gene_scores.cpu().numpy()

    # ── Split-half noise estimation ──────────────────────────────

    def _score_split_half(self, enc) -> Tuple[np.ndarray, float]:
        """
        Score with split-half noise estimation.

        Splits eval data into two halves, scores each independently.
        The disagreement between halves estimates observation noise.

        Returns:
            (combined_scores, noise_variance)
            combined_scores: average of two half-scores (= full-data estimate)
            noise_variance: estimated variance of the combined score's noise
        """
        n_samples = enc["input_ids"].size(0)

        if n_samples < 4:
            # Too few to split — fall back to single score with heuristic noise
            scores = self._score_one_eval(enc)
            if np.any(np.isnan(scores)):
                logger.warning(f"    NaN in gene scores, replacing with 0")
                scores = np.nan_to_num(scores, nan=0.0)
            noise_var = max(np.var(scores) * 0.1, 1e-10)
            logger.info(f"    (< 4 samples, heuristic noise estimate)")
            return scores, noise_var

        mid = n_samples // 2
        enc_a = {"input_ids": enc["input_ids"][:mid],
                 "attention_mask": enc["attention_mask"][:mid]}
        enc_b = {"input_ids": enc["input_ids"][mid:2*mid],
                 "attention_mask": enc["attention_mask"][mid:2*mid]}

        scores_a = self._score_one_eval(enc_a)
        scores_b = self._score_one_eval(enc_b)

        # Guard against NaN from degenerate forward passes
        if np.any(np.isnan(scores_a)) or np.any(np.isnan(scores_b)):
            logger.warning(f"    NaN in split-half scores, replacing with 0")
            scores_a = np.nan_to_num(scores_a, nan=0.0)
            scores_b = np.nan_to_num(scores_b, nan=0.0)

        # Combined = average of halves (equals full-data estimate for linear ops)
        combined = (scores_a + scores_b) / 2

        # Noise estimation:
        # Each half has noise sigma^2_half. The combined average has sigma^2_half/2.
        # Var(A - B) = 2 * sigma^2_half, so sigma^2_combined = Var(A-B) / 4
        diff = scores_a - scores_b
        noise_var = max(np.var(diff) / 4, 1e-10)

        # Diagnostic: split-half correlation
        std_a, std_b = np.std(scores_a), np.std(scores_b)
        if std_a > 1e-10 and std_b > 1e-10:
            r = np.corrcoef(scores_a, scores_b)[0, 1]
            logger.info(f"    split-half r={r:.3f}")

        return combined, noise_var

    # ── Empirical Bayes ──────────────────────────────────────────

    @staticmethod
    def _empirical_bayes(raw_deltas: np.ndarray, noise_var: float
                         ) -> Tuple[np.ndarray, float, dict]:
        """
        Empirical Bayes shrinkage estimator for gene effects.

        Generative model:
            Prior:      mu_g ~ Normal(mu_0, tau^2)
            Likelihood: d_g | mu_g ~ Normal(mu_g, sigma^2)
            Posterior:  mu_g | d_g ~ Normal(B*d + (1-B)*mu_0, B*sigma^2)

        where B = tau^2 / (tau^2 + sigma^2) is the shrinkage factor.

        Parameters are estimated from data:
            mu_0 = mean(d)              [prior mean]
            tau^2 = Var(d) - sigma^2    [signal variance, method of moments]
            sigma^2 = noise_var         [from split-half estimation]

        Args:
            raw_deltas: observed deltas for all genes, shape (n_genes,)
            noise_var: observation noise variance (from split-half)

        Returns:
            posterior_mean: shrunk estimates, shape (n_genes,)
            posterior_var: scalar posterior variance
            diagnostics: dict with prior params and shrinkage factor
        """
        # Guard against NaN/Inf in inputs
        if np.any(np.isnan(raw_deltas)) or np.any(np.isinf(raw_deltas)):
            logger.warning("    NaN/Inf in raw_deltas, clamping")
            raw_deltas = np.nan_to_num(raw_deltas, nan=0.0, posinf=0.0, neginf=0.0)

        mu_0 = np.mean(raw_deltas)
        total_var = np.var(raw_deltas)

        # Guard against degenerate cases
        if not np.isfinite(mu_0):
            mu_0 = 0.0
        if not np.isfinite(total_var) or total_var < 0:
            total_var = 0.0

        # Signal variance via method of moments: Var(d) = tau^2 + sigma^2
        tau2 = max(total_var - noise_var, 0.0)

        # Shrinkage factor
        denom = tau2 + noise_var
        B = tau2 / denom if denom > 1e-10 else 0.0

        # When SNR is zero but we have nonzero deltas, use a minimum
        # shrinkage to avoid completely ignoring the data
        if B < 1e-6 and total_var > 1e-10:
            B = min(0.1, total_var / (total_var + noise_var + 1e-10))
            logger.info(f"    Applied minimum shrinkage floor: B={B:.4f}")

        # Posterior
        posterior_mean = B * raw_deltas + (1 - B) * mu_0
        posterior_var = max(B * noise_var, 1e-10)

        diagnostics = {
            "prior_mean": float(mu_0),
            "signal_var_tau2": float(tau2),
            "noise_var_sigma2": float(noise_var),
            "total_var": float(total_var),
            "shrinkage_B": float(B),
            "snr": float(tau2 / noise_var) if noise_var > 1e-10 else float('inf'),
        }

        return posterior_mean, posterior_var, diagnostics

    # ── Main scoring API ─────────────────────────────────────────

    def score_multi_objective(self, eval_sets, curr_eval,
                              threshold=0.50, alpha=0.3):
        """
        Score all genes using empirical Bayes with split-half noise estimation.

        For each objective (previous domains + current domain):
          1. Split eval data in half, score each half independently
          2. Estimate noise variance from half-score disagreement
          3. Separate signal (tau^2) from noise (sigma^2)
          4. Apply shrinkage: posterior_mean = B*d + (1-B)*mu_0
          5. Compute calibrated P(deleterious | data) or P(beneficial | data)

        Args:
            eval_sets: list of (name, encodings) for objectives to protect.
            curr_eval: encodings for the current domain being learned.

        Returns:
            List of dicts with gene_id, p_del_prev, p_ben_curr.
        """
        t0 = time.perf_counter()
        self._raw_genome.sync_primary()

        if not self.streaming:
            self._build_repair_map()

        prev_names = [name for name, _ in eval_sets]
        n = self.n_genes

        # Score each previous objective with split-half noise estimation
        all_prev_scores = {}
        all_prev_noise = {}
        for name, enc in eval_sets:
            t1 = time.perf_counter()
            scores, noise_var = self._score_split_half(enc)
            all_prev_scores[name] = scores
            all_prev_noise[name] = noise_var
            logger.info(f"  scored '{name}': {time.perf_counter()-t1:.3f}s")

        # Score current domain
        t1 = time.perf_counter()
        curr_scores, noise_var_curr = self._score_split_half(curr_eval)
        logger.info(f"  scored 'current': {time.perf_counter()-t1:.3f}s")

        # Empirical Bayes for each previous objective -> P(deleterious)
        p_del_all = {}
        for name in prev_names:
            # Negate: positive = repairing helps this domain = gene was deleterious
            raw_deltas = -all_prev_scores[name]
            post_mean, post_var, diag = self._empirical_bayes(
                raw_deltas, all_prev_noise[name])
            post_std = np.sqrt(post_var)

            # P(true effect > 0 | data) = P(gene is deleterious to this domain)
            p_del_all[name] = 1 - stats.norm.cdf(
                0, loc=post_mean, scale=post_std)

            logger.info(f"    '{name}' Bayes: B={diag['shrinkage_B']:.3f}, "
                        f"SNR={diag['snr']:.2f}, "
                        f"tau2={diag['signal_var_tau2']:.2e}, "
                        f"sigma2={diag['noise_var_sigma2']:.2e}")

        # Empirical Bayes for current domain -> P(beneficial)
        # Positive raw score = repair increases loss = gene benefits current domain
        post_mean_c, post_var_c, diag_c = self._empirical_bayes(
            curr_scores, noise_var_curr)
        post_std_c = np.sqrt(post_var_c)

        p_ben = 1 - stats.norm.cdf(0, loc=post_mean_c, scale=post_std_c)

        logger.info(f"    'current' Bayes: B={diag_c['shrinkage_B']:.3f}, "
                    f"SNR={diag_c['snr']:.2f}")

        # Combine: take max P(deleterious) across all protected domains
        scores = []
        for gid in range(n):
            p_del_prev = max(p_del_all[name][gid] for name in prev_names)
            scores.append({
                "gene_id": gid,
                "p_del_prev": float(p_del_prev),
                "p_ben_curr": float(p_ben[gid]),
            })

        dt = time.perf_counter() - t0
        logger.info(f"  total scoring: {dt:.3f}s | {n} genes x "
                    f"{len(eval_sets)+1} objectives (Bayesian)")
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
            max_param = max(p.numel() for p in self.model.parameters())
            delta_mb = max_param * 4 / 1024 / 1024
        else:
            delta_mb = n_params * 4 / 1024 / 1024

        return {
            "model_gpu_mb": model_mb,
            "gradients_gpu_mb": grads_mb,
            "deltas_gpu_mb": delta_mb,
            "genome_cpu_mb": genome_cpu_mb,
            "peak_scoring_gpu_mb": model_mb + grads_mb + delta_mb,
            "n_params": n_params,
            "mode": "streaming" if self.streaming else "precomputed",
        }
