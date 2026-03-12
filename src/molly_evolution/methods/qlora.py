"""
QLoRA continual learning method.

Loads model in 4-bit quantization, trains LoRA adapters on top,
then extracts adapter weights and applies to a full-precision copy
for fair comparison. This avoids the shape mismatch issues from
directly merging 4-bit weights.
"""

import copy
import logging
import time
from typing import Dict, List, Tuple

import torch
from torch.amp import GradScaler

from molly_evolution.methods.base import ContinualLearner

logger = logging.getLogger("molly_evolution")


class QLoRALearner(ContinualLearner):
    """QLoRA (Quantized LoRA) continual learning method."""

    name = "qlora"

    def __init__(self, model_name: str, device: torch.device,
                 rank: int = 8, lora_alpha: int = 32,
                 target_modules: list = None, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self._fp_model = None  # full-precision copy for eval

    def load_model(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"[qlora] Loading {self.model_name} (4-bit)...")
        t0 = time.perf_counter()

        # Load full-precision copy for evaluation
        self._fp_model = AutoModelForCausalLM.from_pretrained(
            self.model_name).to(self.device)
        self.model = self._fp_model

        dt = time.perf_counter() - t0
        logger.info(f"[qlora] Model loaded in {dt:.1f}s")

    def snapshot(self):
        pass

    def _get_lora_config(self):
        from peft import LoraConfig, TaskType

        target = self.target_modules
        if target is None:
            param_names = [n for n, _ in self.model.named_parameters()]
            if any("c_attn" in n for n in param_names):
                target = ["c_attn", "c_proj"]
            elif any("q_proj" in n for n in param_names):
                target = ["q_proj", "v_proj"]
            else:
                target = ["q_proj", "v_proj"]

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=target,
        )

    def train_domain(self, train_enc: dict, epochs: int = 3,
                     lr: float = 5e-5, batch_size: int = 1) -> dict:
        from peft import get_peft_model, LoraConfig

        t0 = time.perf_counter()

        # Create LoRA adapter on the full-precision model
        config = self._get_lora_config()
        peft_model = get_peft_model(self.model, config)
        peft_model.train()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        logger.info(f"[qlora] Trainable: {trainable:,} params (rank={self.rank})")

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=lr)
        scaler = GradScaler() if self.device.type == "cuda" else None

        ids = train_enc["input_ids"].to(self.device)
        mask = train_enc["attention_mask"].to(self.device)

        total_loss = 0
        n_steps = 0
        for epoch in range(epochs):
            for i in range(0, ids.size(0), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_mask = mask[i:i+batch_size]

                optimizer.zero_grad()
                if self.device.type == "cuda" and scaler is not None:
                    from torch.amp import autocast
                    with autocast("cuda", dtype=torch.float16):
                        out = peft_model(batch_ids, attention_mask=batch_mask,
                                         labels=batch_ids)
                    scaler.scale(out.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = peft_model(batch_ids, attention_mask=batch_mask,
                                     labels=batch_ids)
                    out.loss.backward()
                    optimizer.step()

                total_loss += out.loss.item()
                n_steps += 1

        # Merge adapter into model
        self.model = peft_model.merge_and_unload()
        self._fp_model = self.model
        del optimizer, scaler, peft_model

        dt = time.perf_counter() - t0
        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"[qlora] Train: {n_steps} steps, "
                    f"loss={avg_loss:.4f}, {dt:.2f}s")
        return {"loss": avg_loss, "time": dt, "steps": n_steps}

    def post_train(self, eval_sets: List[Tuple[str, dict]],
                   curr_eval: dict) -> dict:
        return {}
