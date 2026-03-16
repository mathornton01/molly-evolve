"""
Base class for continual learning methods.

All methods share the same lifecycle:
  1. load_model()    — load pretrained model
  2. snapshot()      — save state before training a new domain
  3. train_domain()  — fine-tune on domain data
  4. post_train()    — method-specific post-processing (repair, merge, etc.)
  5. evaluate()      — compute perplexity on eval data
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

logger = logging.getLogger("molly_evolution")


class ContinualLearner(ABC):
    """Abstract base class for continual learning methods."""

    name: str = "base"

    def __init__(self, model_name: str, device: torch.device, **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._domain_history = []
        self._metrics = {}

    @abstractmethod
    def load_model(self):
        """Load the pretrained model."""
        pass

    @abstractmethod
    def snapshot(self):
        """Save current state before training a new domain."""
        pass

    @abstractmethod
    def train_domain(self, train_enc: dict, epochs: int = 3,
                     lr: float = 5e-5, batch_size: int = 1) -> dict:
        """Fine-tune on domain data. Returns training metrics."""
        pass

    @abstractmethod
    def post_train(self, eval_sets: List[Tuple[str, dict]],
                   curr_eval: dict) -> dict:
        """Post-training processing (repair, merge, etc.)."""
        pass

    def evaluate(self, enc: dict) -> float:
        """Compute perplexity on evaluation data."""
        self.model.eval()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        model_dtype = self._model_dtype()
        with torch.no_grad():
            if self.device.type == "cuda" and model_dtype == torch.bfloat16:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(ids, attention_mask=mask, labels=ids)
            elif self.device.type == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    out = self.model(ids, attention_mask=mask, labels=ids)
            else:
                out = self.model(ids, attention_mask=mask, labels=ids)
        return math.exp(min(out.loss.item(), 20))  # cap at exp(20) to avoid overflow

    def _model_dtype(self):
        """Get the model's parameter dtype."""
        for p in self.model.parameters():
            return p.dtype
        return torch.float32

    def _model_is_fp16(self):
        """Check if model parameters are already in fp16/bf16."""
        return self._model_dtype() in (torch.float16, torch.bfloat16)

    def train_step(self, ids, mask, optimizer, scaler=None):
        """Single training step with optional AMP."""
        optimizer.zero_grad()
        model_dtype = self._model_dtype()
        if self.device.type == "cuda" and scaler is not None and not self._model_is_fp16():
            with autocast("cuda", dtype=torch.float16):
                out = self.model(ids, attention_mask=mask, labels=ids)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif self.device.type == "cuda" and model_dtype == torch.bfloat16:
            # bf16 model — use bf16 autocast (no scaler needed, bf16 doesn't overflow)
            with autocast("cuda", dtype=torch.bfloat16):
                out = self.model(ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            optimizer.step()
        elif self.device.type == "cuda" and model_dtype == torch.float16:
            # fp16 model — use autocast with scaler to avoid NaN
            scaler = scaler or GradScaler()
            with autocast("cuda", dtype=torch.float16):
                out = self.model(ids, attention_mask=mask, labels=ids)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = self.model(ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            optimizer.step()
        return out.loss.item()

    def get_metrics(self) -> dict:
        """Return all recorded metrics."""
        return self._metrics

    def record_eval(self, domain_name: str, eval_sets: List[Tuple[str, dict]]):
        """Evaluate and record perplexity for all eval sets."""
        results = {}
        for name, enc in eval_sets:
            ppl = self.evaluate(enc)
            results[name] = ppl
        self._metrics.setdefault("eval_history", []).append({
            "after_domain": domain_name,
            "perplexities": results,
        })
        return results

    def memory_usage(self) -> dict:
        """Estimate GPU memory usage."""
        if self.model is None:
            return {}
        n_params = sum(p.numel() for p in self.model.parameters())
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        return {
            "n_params": n_params,
            "gpu_allocated_mb": round(gpu_mb),
            "method": self.name,
        }
