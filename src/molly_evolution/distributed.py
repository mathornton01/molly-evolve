"""
distributed.py — Multi-GPU and FSDP support for Molly Evolution.

Three scaling strategies:
  1. MultiGPUScorer: parallelize objective scoring across GPUs
  2. FSDPGenome: gene conversion on FSDP-sharded models
  3. DeepSpeedGenome: gene conversion with DeepSpeed ZeRO

Memory scaling comparison (7B model):
  Single GPU precomputed: ~70 GB (won't fit)
  Single GPU streaming:   ~21 GB (fits 24GB GPU)
  Multi-GPU streaming:    ~21 GB / GPU (N objectives in parallel)
  FSDP streaming:         ~21/N GB per GPU (N GPUs)
"""

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from molly_evolution.scoring import GeneScorer, _gpu_mem_mb
from molly_evolution.genome import DualGenome

logger = logging.getLogger("molly_evolution")


# ═══════════════════════════════════════════════════════════════════
# 1. Multi-GPU Objective Parallelism
# ═══════════════════════════════════════════════════════════════════

class MultiGPUScorer:
    """
    Score objectives in parallel across multiple GPUs.

    Each GPU gets a model replica + GeneScorer instance. Objectives are
    distributed round-robin across GPUs. CUDA operations on different
    devices run truly in parallel (GIL released during GPU kernel execution).

    Memory per GPU: model weights + gradients + streaming deltas
    Speedup: ~linear in min(n_gpus, n_objectives)

    Example:
        scorer = MultiGPUScorer(genome, model)
        scores = scorer.score_multi_objective(eval_sets, curr_eval)
    """

    def __init__(self, genome, model, use_amp=True):
        self.n_gpus = torch.cuda.device_count()
        self.primary_device = next(model.parameters()).device
        self.use_amp = use_amp

        if self.n_gpus <= 1:
            logger.info("MultiGPUScorer: single GPU, no parallelism")
            self._single_scorer = GeneScorer(
                genome, model, self.primary_device,
                use_amp=use_amp, streaming=True)
            return

        logger.info(f"MultiGPUScorer: {self.n_gpus} GPUs detected")
        self._single_scorer = None

        # Create model replica + scorer on each GPU
        self._scorers = []
        self._genomes = []
        self._models = []

        for i in range(self.n_gpus):
            dev = torch.device(f"cuda:{i}")
            t0 = time.perf_counter()

            if i == 0 and self.primary_device == dev:
                replica = model
            else:
                replica = copy.deepcopy(model).to(dev)

            # Each replica needs its own genome (owns complement/primary state)
            g = DualGenome(replica, granularity=genome.granularity,
                           backend="python")
            # Copy complement strands from original genome
            self._copy_genome_state(genome, g)

            scorer = GeneScorer(g, replica, dev,
                                use_amp=use_amp, streaming=True)

            self._models.append(replica)
            self._genomes.append(g)
            self._scorers.append(scorer)

            dt = time.perf_counter() - t0
            logger.info(f"  GPU {i} ({dev}): replica ready in {dt:.2f}s | "
                        f"mem: {torch.cuda.memory_allocated(dev)/1024/1024:.0f} MB")

    @staticmethod
    def _copy_genome_state(src_genome, dst_genome):
        """Copy complement/primary/scale state between genome instances."""
        src_raw = getattr(src_genome, '_genome', src_genome)
        dst_raw = getattr(dst_genome, '_genome', dst_genome)

        for sg, dg in zip(src_raw.genes, dst_raw.genes):
            for key in sg.complement:
                dg.complement[key] = sg.complement[key].clone()
                dg.primary[key] = sg.primary[key].clone()
                dg.scales[key] = sg.scales[key]

    def sync_weights(self, source_model):
        """Sync all replicas from the source model after training."""
        t0 = time.perf_counter()
        state = source_model.state_dict()
        for i, replica in enumerate(self._models):
            dev = next(replica.parameters()).device
            if replica is not source_model:
                replica.load_state_dict(
                    {k: v.to(dev) for k, v in state.items()})
        dt = time.perf_counter() - t0
        logger.info(f"sync_weights to {self.n_gpus} GPUs: {dt:.3f}s")

    def snapshot_all(self):
        """Snapshot on all genome replicas."""
        for g in self._genomes:
            g.snapshot()

    def score_multi_objective(self, eval_sets, curr_eval,
                              threshold=0.50, alpha=0.3):
        """Score objectives in parallel across GPUs."""
        if self._single_scorer is not None:
            return self._single_scorer.score_multi_objective(
                eval_sets, curr_eval, threshold, alpha)

        t0 = time.perf_counter()

        # Sync primary on all replicas
        for g in self._genomes:
            raw = getattr(g, '_genome', g)
            raw.sync_primary()

        # Rebuild ref maps (streaming mode — very fast)
        for s in self._scorers:
            s._build_ref_map()

        # Distribute objectives across GPUs
        all_objectives = [(name, enc) for name, enc in eval_sets]
        all_objectives.append(("__current__", curr_eval))

        results = {}

        def _score_on_gpu(args):
            gpu_idx, name, enc = args
            scorer = self._scorers[gpu_idx]
            return name, scorer._score_one_eval(enc)

        # Build work items: round-robin assignment to GPUs
        work = []
        for i, (name, enc) in enumerate(all_objectives):
            gpu_idx = i % self.n_gpus
            work.append((gpu_idx, name, enc))

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.n_gpus) as pool:
            for name, scores in pool.map(_score_on_gpu, work):
                results[name] = scores

        # Extract results
        curr_scores = results.pop("__current__")
        all_prev = results

        # Bayesian posteriors (same as GeneScorer.score_multi_objective)
        from scipy import stats as sp_stats
        prev_names = [name for name, _ in eval_sets]
        n = self._scorers[0].n_genes

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
                p_del[name] = 1 - sp_stats.norm.cdf(
                    0, loc=s["deltas_prev"][name] / 2, scale=sig)
            p_del_prev = max(p_del.values())
            sig_c = np.sqrt(var_curr / 2)
            p_ben_curr = 1 - sp_stats.norm.cdf(
                0, loc=s["delta_curr"] / 2, scale=sig_c)
            scores.append({"gene_id": s["gene_id"],
                           "p_del_prev": p_del_prev,
                           "p_ben_curr": p_ben_curr})

        dt = time.perf_counter() - t0
        logger.info(f"MultiGPU scoring: {dt:.3f}s | {n} genes × "
                    f"{len(eval_sets)+1} objectives × {self.n_gpus} GPUs")
        return scores


# ═══════════════════════════════════════════════════════════════════
# 2. FSDP Integration
# ═══════════════════════════════════════════════════════════════════

class FSDPGenome(DualGenome):
    """
    Gene conversion on FSDP-sharded models.

    FSDP shards parameters across ranks. This wrapper uses
    summon_full_params() to materialize full weights for genome
    operations (snapshot, sync, repair), then releases them.

    The genome stores complement/primary on CPU (unsharded), which
    is the same cost as single-GPU. The scaling benefit comes from
    the model itself being sharded during training and scoring.

    Memory per rank:
      Training: model_shard + optimizer_shard + activations
      Genome ops: briefly summons full params (temporary)
      Genome CPU: full complement + primary (int16) — same total

    Usage:
        model = FSDP(model, ...)
        genome = FSDPGenome(model, granularity="head")
        genome.snapshot()
        # ... train with FSDP ...
        scorer = FSDPScorer(genome, model, device)
        scores = scorer.score_multi_objective(eval_sets, curr_eval)
        genome.apply_conversion(scores)
    """

    def __init__(self, fsdp_model: nn.Module, n_bits: int = 16,
                 granularity: str = "head"):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self._fsdp = fsdp_model
        # Initialize genome with full params visible
        with FSDP.summon_full_params(fsdp_model):
            super().__init__(fsdp_model, n_bits=n_bits,
                             granularity=granularity, backend="python")

    def snapshot(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        t0 = time.perf_counter()
        with FSDP.summon_full_params(self._fsdp):
            self._genome.snapshot()
        dt = time.perf_counter() - t0
        logger.info(f"FSDP snapshot: {dt:.3f}s")

    def sync_primary(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        t0 = time.perf_counter()
        with FSDP.summon_full_params(self._fsdp):
            self._genome.sync_primary()
        dt = time.perf_counter() - t0
        logger.info(f"FSDP sync_primary: {dt:.3f}s")

    def repair_genes(self, gene_ids: List[int]):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        t0 = time.perf_counter()
        with FSDP.summon_full_params(self._fsdp, writeback=True):
            self._genome.repair_genes(gene_ids)
        dt = time.perf_counter() - t0
        logger.info(f"FSDP repair: {dt:.3f}s | "
                    f"{len(gene_ids)}/{self.total_genes} genes")

    def fix_genes(self, gene_ids: List[int]):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(self._fsdp):
            self._genome.fix_genes(gene_ids)


class FSDPScorer(GeneScorer):
    """
    Gene scoring on FSDP-sharded models.

    Forward/backward works normally with FSDP. For scoring,
    we summon full params to access gradients and compute dot products.
    Always uses streaming mode (FSDP models are large by definition).
    """

    def __init__(self, genome, fsdp_model, device, use_amp=True):
        self._fsdp = fsdp_model
        super().__init__(genome, fsdp_model, device,
                         use_amp=use_amp, streaming=True)

    def _score_one_eval(self, enc) -> np.ndarray:
        """Forward/backward with FSDP, then score with full params."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.model.zero_grad()
        self.model.train()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        # Forward/backward works normally with FSDP
        if self.use_amp:
            from torch.amp import autocast
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

        # Summon full params+grads to compute dot products
        with FSDP.summon_full_params(self._fsdp):
            gene_scores = self._score_python_streaming()

        self.model.zero_grad()
        self.model.eval()
        return gene_scores


# ═══════════════════════════════════════════════════════════════════
# 3. DeepSpeed ZeRO Integration
# ═══════════════════════════════════════════════════════════════════

class DeepSpeedGenome(DualGenome):
    """
    Gene conversion with DeepSpeed ZeRO-sharded models.

    Uses DeepSpeed's gather/partition APIs to access full parameters.
    Complement/primary stored on CPU (unsharded).

    Usage:
        model, optimizer, _, _ = deepspeed.initialize(...)
        genome = DeepSpeedGenome(model, granularity="head")
        genome.snapshot()
    """

    def __init__(self, ds_engine, n_bits: int = 16,
                 granularity: str = "head"):
        self._ds_engine = ds_engine
        # Access the underlying module
        module = ds_engine.module
        super().__init__(module, n_bits=n_bits,
                         granularity=granularity, backend="python")

    def snapshot(self):
        t0 = time.perf_counter()
        # For ZeRO-3, gather full params before snapshot
        if hasattr(self._ds_engine, 'gather_full_params_and_run'):
            self._ds_engine.gather_full_params_and_run(
                lambda: self._genome.snapshot())
        else:
            # ZeRO-1/2: params already full on each rank
            self._genome.snapshot()
        dt = time.perf_counter() - t0
        logger.info(f"DeepSpeed snapshot: {dt:.3f}s")

    def sync_primary(self):
        t0 = time.perf_counter()
        if hasattr(self._ds_engine, 'gather_full_params_and_run'):
            self._ds_engine.gather_full_params_and_run(
                lambda: self._genome.sync_primary())
        else:
            self._genome.sync_primary()
        dt = time.perf_counter() - t0
        logger.info(f"DeepSpeed sync_primary: {dt:.3f}s")

    def repair_genes(self, gene_ids: List[int]):
        t0 = time.perf_counter()
        if hasattr(self._ds_engine, 'gather_full_params_and_run'):
            self._ds_engine.gather_full_params_and_run(
                lambda: self._genome.repair_genes(gene_ids))
        else:
            self._genome.repair_genes(gene_ids)
        dt = time.perf_counter() - t0
        logger.info(f"DeepSpeed repair: {dt:.3f}s | "
                    f"{len(gene_ids)}/{self.total_genes} genes")


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def estimate_requirements(n_params: int, n_objectives: int = 3,
                          dtype_bytes: int = 2) -> dict:
    """
    Estimate hardware requirements for a given model size.

    Args:
        n_params: Total model parameters.
        n_objectives: Number of objectives to protect.
        dtype_bytes: 2 for fp16, 4 for fp32.

    Returns:
        Dict with memory estimates and recommended configuration.
    """
    mb = 1024 * 1024

    model_mb = n_params * dtype_bytes / mb
    grads_mb = n_params * 4 / mb  # fp32 gradients
    genome_cpu_mb = n_params * 4 / mb  # primary + complement int16

    # Largest single param estimate (scales with model size)
    # LLaMA 7B: ~170MB, 13B: ~340MB, 70B: ~680MB
    max_param_mb = min(n_params * 4 / mb * 0.02, 1000)  # ~2% of total

    # Streaming: model + grads + one delta at a time
    streaming_peak_mb = model_mb + grads_mb + max_param_mb

    # Precomputed: model + grads + all deltas
    precomputed_peak_mb = model_mb + grads_mb + n_params * 4 / mb

    # Training peak (AdamW: 8 bytes/param for states)
    training_peak_mb = model_mb + grads_mb + n_params * 8 / mb

    # Recommendations based on 80GB A100 per GPU
    if streaming_peak_mb < 24_000:
        rec_gpus = 1
        rec_strategy = "single GPU, streaming mode"
    elif streaming_peak_mb < 80_000:
        rec_gpus = max(2, n_objectives)
        rec_strategy = f"{rec_gpus}x GPU, MultiGPUScorer + streaming"
    else:
        rec_gpus = max(4, int(streaming_peak_mb // 70_000) + 1)
        rec_strategy = f"{rec_gpus}x GPU, FSDP + streaming"

    return {
        "n_params_billions": n_params / 1e9,
        "model_gpu_mb": round(model_mb),
        "training_peak_gpu_mb": round(training_peak_mb),
        "scoring_streaming_gpu_mb": round(streaming_peak_mb),
        "scoring_precomputed_gpu_mb": round(precomputed_peak_mb),
        "genome_cpu_mb": round(genome_cpu_mb),
        "recommended_gpus": rec_gpus,
        "recommended_strategy": rec_strategy,
    }
