"""
MolliTrainer — high-level, LoRA-parity API for MOLLI gene conversion.

This is the "one import, five lines" entry point for users who just want
MOLLI's catastrophic-forgetting protection without orchestrating the
snapshot / train / score / repair / snapshot loop by hand.

Example (mirrors the typical PEFT/LoRA workflow):

    from molly_evolution import MolliTrainer

    trainer = MolliTrainer.from_pretrained("gpt2")
    trainer.fit(
        train_texts=open("my_corpus.txt").read().split("\\n"),
        eval_texts=open("holdout.txt").read().split("\\n"),
        epochs=3,
        lr=5e-5,
    )
    trainer.save_pretrained("./my-molli-model")

    # Later, reload for inference or further training:
    trainer = MolliTrainer.from_pretrained("./my-molli-model")
    print(trainer.generate("Once upon a time"))
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

from molly_evolution.genome import DualGenome
from molly_evolution.scoring import GeneScorer

logger = logging.getLogger("molly_evolution")

TextLike = Union[str, Sequence[str]]


def _ensure_list(texts: TextLike) -> List[str]:
    """Normalize a single string or an iterable of strings into a list."""
    if texts is None:
        return []
    if isinstance(texts, str):
        return [texts]
    return [t for t in texts if t is not None and len(t) > 0]


def _tokenize_block(tokenizer, texts: List[str], max_length: int) -> dict:
    """Tokenize a list of texts into fixed-length blocks suitable for LM training."""
    if not texts:
        return {"input_ids": torch.empty(0, max_length, dtype=torch.long),
                "attention_mask": torch.empty(0, max_length, dtype=torch.long)}

    joined = "\n".join(texts)
    enc = tokenizer(joined, return_tensors="pt", truncation=False, padding=False)
    ids = enc["input_ids"][0]

    # Split into contiguous blocks of max_length tokens.
    n_full = ids.size(0) // max_length
    if n_full == 0:
        # Pad a single block if the corpus is shorter than one window.
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padded = torch.full((max_length,), pad_id, dtype=ids.dtype)
        padded[: ids.size(0)] = ids
        mask = torch.zeros(max_length, dtype=torch.long)
        mask[: ids.size(0)] = 1
        return {"input_ids": padded.unsqueeze(0),
                "attention_mask": mask.unsqueeze(0)}

    blocks = ids[: n_full * max_length].view(n_full, max_length)
    mask = torch.ones_like(blocks)
    return {"input_ids": blocks, "attention_mask": mask}


class MolliTrainer:
    """
    High-level wrapper around :class:`DualGenome` + :class:`GeneScorer`
    that exposes a single ``fit`` / ``save_pretrained`` / ``from_pretrained``
    surface area analogous to HuggingFace PEFT's LoRA helpers.

    Attributes:
        model: the underlying HuggingFace causal-LM model.
        tokenizer: the HuggingFace tokenizer.
        genome: the attached :class:`DualGenome`.
        device: the torch device used for training/inference.
    """

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None,
                 granularity: str = "head", n_bits: int = 16,
                 backend: str = "auto", genome: Optional[DualGenome] = None):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))
        self.device = device
        self.model.to(self.device)

        self.granularity = granularity
        self.n_bits = n_bits

        if genome is None:
            self.genome = DualGenome(self.model, n_bits=n_bits,
                                     granularity=granularity, backend=backend)
            self.genome.snapshot()
            self._snapshot_taken = True
        else:
            self.genome = genome
            self._snapshot_taken = True

    # ── Constructors ─────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_name_or_path: str,
                        device: Optional[torch.device] = None,
                        granularity: str = "head", n_bits: int = 16,
                        backend: str = "auto",
                        torch_dtype: Optional[torch.dtype] = None,
                        **model_kwargs) -> "MolliTrainer":
        """
        Build a trainer from a HuggingFace model name or a saved MOLLI
        checkpoint directory.

        If ``model_name_or_path`` points to a directory containing
        ``molli_genome.pt``, the saved genome state is restored on top of
        the loaded model; otherwise a fresh genome is initialized and
        snapshotted.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Pick a sensible default dtype: bf16 on capable GPUs, else fp32.
        if torch_dtype is None:
            if (torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported()):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Newer transformers uses ``dtype=``; older uses ``torch_dtype=``.
        # Try the new kwarg first and fall back gracefully.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dtype=torch_dtype, **model_kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype, **model_kwargs)

        # Check for a saved genome next to the model and restore it if found.
        genome_file = None
        if os.path.isdir(model_name_or_path):
            candidate = os.path.join(model_name_or_path,
                                     DualGenome.GENOME_FILENAME)
            if os.path.exists(candidate):
                genome_file = candidate

        trainer = cls.__new__(cls)
        trainer.model = model
        trainer.tokenizer = tokenizer
        if trainer.tokenizer.pad_token is None and trainer.tokenizer.eos_token is not None:
            trainer.tokenizer.pad_token = trainer.tokenizer.eos_token
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available()
                      else torch.device("cpu"))
        trainer.device = device
        trainer.model.to(device)
        trainer.granularity = granularity
        trainer.n_bits = n_bits

        if genome_file is not None:
            # The HF model file on disk is the authoritative source of
            # weights — passing apply_to_model=False keeps those lossless
            # floats in place instead of overwriting them with the int16
            # primary strand.
            trainer.genome = DualGenome.load(genome_file, trainer.model,
                                             backend=backend,
                                             apply_to_model=False)
            logger.info(f"[molli] Restored genome from {genome_file}")
        else:
            trainer.genome = DualGenome(trainer.model, n_bits=n_bits,
                                        granularity=granularity, backend=backend)
            trainer.genome.snapshot()
        trainer._snapshot_taken = True
        return trainer

    # ── Training ────────────────────────────────────────────────

    def fit(self,
            train_texts: TextLike,
            eval_texts: Optional[TextLike] = None,
            protect_texts: Optional[Union[TextLike, dict]] = None,
            epochs: int = 3,
            lr: float = 5e-5,
            batch_size: int = 1,
            max_length: int = 256,
            max_repair_pct: float = 0.03,
            threshold: float = 0.50,
            alpha: float = 0.3,
            max_grad_norm: float = 1.0,
            repair: bool = True) -> dict:
        """
        Fine-tune the model on ``train_texts`` and (optionally) run gene
        conversion to preserve the capabilities represented by
        ``protect_texts``.

        Args:
            train_texts: raw text, a list of strings, or a tokenized dict
                ({"input_ids", "attention_mask"}) to fine-tune on.
            eval_texts: held-out text for the *current* domain, used as
                the "beneficial" signal during scoring. If ``None`` the
                last 10% of ``train_texts`` is held out.
            protect_texts: text (or a ``{name: text}`` mapping) whose
                capabilities should be preserved. Genes that hurt these
                objectives will be repaired. ``None`` disables protection
                (repair still runs against ``eval_texts`` only).
            repair: set to ``False`` to skip gene conversion entirely —
                useful for a first-time snapshot on a fresh model.

        Returns:
            A dict with ``loss``, ``train_time``, ``steps``, and (if
            repair ran) ``repaired`` / ``fixed`` counts.
        """
        t0 = time.perf_counter()

        # Snapshot once at the very start if the user skipped init.
        if not self._snapshot_taken:
            self.genome.snapshot()
            self._snapshot_taken = True

        train_enc = self._coerce_encoding(train_texts, max_length)
        if train_enc["input_ids"].numel() == 0:
            raise ValueError("fit() received no training data.")

        if eval_texts is None:
            # Hold out the last ~10% of train blocks for scoring.
            n = train_enc["input_ids"].size(0)
            split = max(n - max(1, n // 10), 1)
            eval_enc = {
                "input_ids": train_enc["input_ids"][split:],
                "attention_mask": train_enc["attention_mask"][split:],
            }
            train_enc = {
                "input_ids": train_enc["input_ids"][:split],
                "attention_mask": train_enc["attention_mask"][:split],
            }
        else:
            eval_enc = self._coerce_encoding(eval_texts, max_length)

        # ── Train ──
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        ids = train_enc["input_ids"].to(self.device)
        mask = train_enc["attention_mask"].to(self.device)

        total_loss = 0.0
        n_steps = 0
        nan_steps = 0
        model_dtype = next(self.model.parameters()).dtype
        use_amp_bf16 = (self.device.type == "cuda"
                        and model_dtype == torch.bfloat16)

        for epoch in range(epochs):
            for i in range(0, ids.size(0), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_mask = mask[i:i + batch_size]
                optimizer.zero_grad()
                if use_amp_bf16:
                    from torch.amp import autocast
                    with autocast("cuda", dtype=torch.bfloat16):
                        out = self.model(batch_ids,
                                         attention_mask=batch_mask,
                                         labels=batch_ids)
                else:
                    out = self.model(batch_ids,
                                     attention_mask=batch_mask,
                                     labels=batch_ids)
                loss_val = out.loss.item()
                if not math.isfinite(loss_val):
                    nan_steps += 1
                    optimizer.zero_grad()
                    continue
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)
                optimizer.step()
                total_loss += loss_val
                n_steps += 1
        del optimizer

        train_time = time.perf_counter() - t0
        avg_loss = total_loss / max(n_steps, 1)
        result = {
            "loss": avg_loss,
            "train_time": train_time,
            "steps": n_steps,
            "nan_steps": nan_steps,
        }

        # ── Gene conversion (optional) ──
        if repair:
            protect_sets = self._build_protect_sets(protect_texts, max_length)
            # Always include the current eval set on protect side; fall back
            # to eval_enc itself if no explicit protect_texts were given.
            if not protect_sets:
                protect_sets = [("current", eval_enc)]

            scorer = GeneScorer(self.genome, self.model, self.device,
                                use_amp=True, streaming=True)
            scores = scorer.score_multi_objective(
                protect_sets, eval_enc,
                threshold=threshold, alpha=alpha)
            n_rep, n_fix = self.genome.apply_conversion(
                scores, threshold=threshold, alpha=alpha,
                max_repair_pct=max_repair_pct)
            self.genome.snapshot()
            result["repaired"] = n_rep
            result["fixed"] = n_fix
            logger.info(f"[molli] fit: loss={avg_loss:.4f} "
                        f"repair={n_rep} fix={n_fix} "
                        f"time={train_time:.1f}s")
        else:
            logger.info(f"[molli] fit (no repair): loss={avg_loss:.4f} "
                        f"time={train_time:.1f}s")

        return result

    # ── Evaluation & inference ──────────────────────────────────

    @torch.no_grad()
    def evaluate(self, texts: TextLike, max_length: int = 256) -> float:
        """Return perplexity on ``texts``."""
        enc = self._coerce_encoding(texts, max_length)
        if enc["input_ids"].numel() == 0:
            return float("nan")
        self.model.eval()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.model(ids, attention_mask=mask, labels=ids)
        return math.exp(min(out.loss.item(), 20))

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64,
                 **generate_kwargs) -> str:
        """One-shot text generation convenience wrapper."""
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, **generate_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # ── Save / Load ─────────────────────────────────────────────

    def save_pretrained(self, path: str):
        """
        Save the model, tokenizer, and genome state to ``path``. The output
        directory is a drop-in HuggingFace checkpoint that can also be
        reloaded with :meth:`MolliTrainer.from_pretrained`.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.genome.save(path)
        logger.info(f"[molli] Saved model + genome to {path}")
        return path

    # ── Internal helpers ────────────────────────────────────────

    def _coerce_encoding(self, data, max_length: int) -> dict:
        """Turn raw text / list / dict into a tokenized encoding."""
        if isinstance(data, dict) and "input_ids" in data:
            ids = data["input_ids"]
            if not isinstance(ids, torch.Tensor):
                ids = torch.as_tensor(ids)
            mask = data.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(ids)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
                mask = mask.unsqueeze(0)
            return {"input_ids": ids, "attention_mask": mask}
        texts = _ensure_list(data)
        return _tokenize_block(self.tokenizer, texts, max_length)

    def _build_protect_sets(self, protect_texts, max_length: int
                            ) -> List[Tuple[str, dict]]:
        """Normalize ``protect_texts`` into a list of (name, encoding) pairs."""
        if protect_texts is None:
            return []
        if isinstance(protect_texts, dict) and "input_ids" not in protect_texts:
            out = []
            for name, t in protect_texts.items():
                out.append((str(name), self._coerce_encoding(t, max_length)))
            return out
        # Single text / list / dict-encoding — treat as one unnamed objective.
        return [("protect", self._coerce_encoding(protect_texts, max_length))]
