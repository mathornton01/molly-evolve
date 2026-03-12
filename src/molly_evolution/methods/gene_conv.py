"""
Gene Conversion continual learning method.

Maintains dual-strand genome. After fine-tuning, uses Bayesian scoring
to identify damaged genes and repair them from the complement strand.
"""

import logging
import time
from typing import Dict, List, Tuple

import torch
from torch.amp import GradScaler

from molly_evolution.methods.base import ContinualLearner
from molly_evolution.genome import DualGenome
from molly_evolution.scoring import GeneScorer

logger = logging.getLogger("molly_evolution")


class GeneConvLearner(ContinualLearner):
    """Molly Evolution gene conversion method."""

    name = "gene-conv"

    def __init__(self, model_name: str, device: torch.device,
                 granularity: str = "head", streaming: bool = True,
                 threshold: float = 0.50, alpha: float = 0.3, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.granularity = granularity
        self.streaming = streaming
        self.threshold = threshold
        self.alpha = alpha
        self.genome = None
        self.scorer = None

    def load_model(self):
        from transformers import AutoModelForCausalLM
        logger.info(f"[gene-conv] Loading {self.model_name}...")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16).to(self.device)
        # Enable gradient checkpointing for large models
        n_params = sum(p.numel() for p in self.model.parameters())
        if n_params > 1e9:
            self.model.gradient_checkpointing_enable()
            logger.info(f"[gene-conv] Gradient checkpointing enabled ({n_params/1e9:.1f}B params)")
        dt = time.perf_counter() - t0
        logger.info(f"[gene-conv] Model loaded in {dt:.1f}s")

        self.genome = DualGenome(self.model, granularity=self.granularity,
                                 backend="python")
        logger.info(f"[gene-conv] Genome: {self.genome.total_genes} genes "
                    f"({self.granularity})")

    def snapshot(self):
        t0 = time.perf_counter()
        self.genome.snapshot()
        self.scorer = GeneScorer(self.genome, self.model, self.device,
                                 use_amp=True, streaming=self.streaming)
        logger.info(f"[gene-conv] Snapshot: {time.perf_counter()-t0:.3f}s")

    def _make_optimizer(self, lr):
        """Create optimizer — 8-bit Adam for large models, standard AdamW otherwise."""
        n_params = sum(p.numel() for p in self.model.parameters())
        if n_params > 1e9:
            try:
                import bitsandbytes as bnb
                logger.info(f"[gene-conv] Using 8-bit Adam (model has {n_params/1e9:.1f}B params)")
                return bnb.optim.Adam8bit(self.model.parameters(), lr=lr)
            except ImportError:
                logger.warning("[gene-conv] bitsandbytes not available, using AdamW")
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train_domain(self, train_enc: dict, epochs: int = 3,
                     lr: float = 5e-5, batch_size: int = 1) -> dict:
        t0 = time.perf_counter()
        self.model.train()
        optimizer = self._make_optimizer(lr)

        ids = train_enc["input_ids"].to(self.device)
        mask = train_enc["attention_mask"].to(self.device)

        total_loss = 0
        n_steps = 0
        for epoch in range(epochs):
            for i in range(0, ids.size(0), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_mask = mask[i:i+batch_size]
                loss = self.train_step(batch_ids, batch_mask, optimizer)
                total_loss += loss
                n_steps += 1

        del optimizer
        dt = time.perf_counter() - t0
        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"[gene-conv] Train: {n_steps} steps, "
                    f"loss={avg_loss:.4f}, {dt:.2f}s")
        return {"loss": avg_loss, "time": dt, "steps": n_steps}

    def post_train(self, eval_sets: List[Tuple[str, dict]],
                   curr_eval: dict) -> dict:
        t0 = time.perf_counter()
        scores = self.scorer.score_multi_objective(
            eval_sets, curr_eval,
            threshold=self.threshold, alpha=self.alpha)
        n_rep, n_fix = self.genome.apply_conversion(
            scores, threshold=self.threshold, alpha=self.alpha)
        self.genome.snapshot()
        dt = time.perf_counter() - t0
        logger.info(f"[gene-conv] Post-train: {n_rep} repaired, "
                    f"{n_fix} fixed, {dt:.2f}s")
        return {"repaired": n_rep, "fixed": n_fix, "time": dt}
