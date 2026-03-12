"""
Molly Evolution — Bayesian genetic algorithm for LLM self-evolution.

Inspired by the Amazon Molly fish's gene conversion mechanism.
Maintains dual copies of model weights and selectively repairs damaged
genes after fine-tuning, enabling continual learning without catastrophic
forgetting.

Usage:
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


def has_cuda_ops() -> bool:
    """Check if compiled CUDA extension is available."""
    try:
        import molly_evolution._C
        return True
    except ImportError:
        return False
