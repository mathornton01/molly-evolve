"""
Molly Evolution — Bayesian genetic algorithm for LLM self-evolution.

Inspired by the Amazon Molly fish's gene conversion mechanism.
Maintains dual copies of model weights and selectively repairs damaged
genes after fine-tuning, enabling continual learning without catastrophic
forgetting.

Quick start (LoRA-style, 5 lines):

    from molly_evolution import MolliTrainer

    trainer = MolliTrainer.from_pretrained("gpt2")
    trainer.fit(train_texts=open("corpus.txt").read().split("\\n"))
    trainer.save_pretrained("./my-molli-model")

Low-level API (when you want direct control):

    from molly_evolution import DualGenome, GeneScorer

    genome = DualGenome(model, granularity="head")
    genome.snapshot()
    # ... fine-tune model ...
    scorer = GeneScorer(genome, model, device)
    scores = scorer.score_multi_objective(eval_sets, current_eval)
    genome.apply_conversion(scores)

Scaling:
    from molly_evolution.distributed import (
        MultiGPUScorer, FSDPGenome, estimate_requirements
    )
"""

from molly_evolution._version import __version__
from molly_evolution.genome import DualGenome
from molly_evolution.scoring import GeneScorer
from molly_evolution.trainer import MolliTrainer


def molli_train(model, tokenizer, train_texts, eval_texts=None,
                protect_texts=None, device=None, granularity: str = "head",
                **fit_kwargs):
    """
    One-shot functional helper that applies MOLLI gene conversion to an
    already-loaded HuggingFace model. Returns the :class:`MolliTrainer`
    so callers can save, evaluate, or generate from it.

    This is the minimum-friction entry point for users migrating from a
    LoRA-based training script — you keep the same ``model`` and
    ``tokenizer`` objects you already have, and MOLLI handles the
    snapshot / train / score / repair loop internally.
    """
    trainer = MolliTrainer(model, tokenizer, device=device,
                           granularity=granularity)
    trainer.fit(train_texts=train_texts, eval_texts=eval_texts,
                protect_texts=protect_texts, **fit_kwargs)
    return trainer


def has_cuda_ops() -> bool:
    """Check if compiled CUDA extension is available."""
    try:
        import molly_evolution._C
        return True
    except ImportError:
        return False


__all__ = [
    "DualGenome",
    "GeneScorer",
    "MolliTrainer",
    "molli_train",
    "has_cuda_ops",
    "__version__",
]
