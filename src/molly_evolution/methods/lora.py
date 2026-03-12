"""
LoRA continual learning method.

Creates a new LoRA adapter for each domain, trains it, then merges
into the base model. Simple but suffers from O(1/N) dilution as
more domains are added.
"""

import logging
import time
from typing import Dict, List, Tuple

import torch
from torch.amp import GradScaler

from molly_evolution.methods.base import ContinualLearner

logger = logging.getLogger("molly_evolution")


class LoRALearner(ContinualLearner):
    """LoRA (Low-Rank Adaptation) continual learning method."""

    name = "lora"

    def __init__(self, model_name: str, device: torch.device,
                 rank: int = 8, lora_alpha: int = 32,
                 target_modules: list = None, **kwargs):
        super().__init__(model_name, device, **kwargs)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules

    def load_model(self):
        from transformers import AutoModelForCausalLM
        logger.info(f"[lora] Loading {self.model_name}...")
        t0 = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name).to(self.device)
        dt = time.perf_counter() - t0
        logger.info(f"[lora] Model loaded in {dt:.1f}s")

    def snapshot(self):
        # LoRA doesn't need snapshotting — each domain gets a fresh adapter
        pass

    def _get_lora_config(self):
        from peft import LoraConfig, TaskType
        target = self.target_modules
        if target is None:
            # Auto-detect for GPT-2 style models
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
        from peft import get_peft_model

        t0 = time.perf_counter()

        # Create fresh LoRA adapter
        config = self._get_lora_config()
        peft_model = get_peft_model(self.model, config)
        peft_model.train()

        # Only train LoRA parameters
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(f"[lora] Trainable: {trainable:,} / {total:,} "
                    f"({100*trainable/total:.1f}%)")

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

        # Merge adapter into base model
        self.model = peft_model.merge_and_unload()
        del optimizer, scaler, peft_model

        dt = time.perf_counter() - t0
        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"[lora] Train: {n_steps} steps, "
                    f"loss={avg_loss:.4f}, {dt:.2f}s")
        return {"loss": avg_loss, "time": dt, "steps": n_steps,
                "trainable_params": trainable}

    def post_train(self, eval_sets: List[Tuple[str, dict]],
                   curr_eval: dict) -> dict:
        # LoRA has no post-training repair step
        return {}
