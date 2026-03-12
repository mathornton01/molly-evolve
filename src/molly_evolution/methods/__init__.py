"""
Continual learning methods with a unified interface.

All methods implement ContinualLearner:
  - GeneConvLearner: Molly Evolution (gene conversion)
  - LoRALearner: Low-Rank Adaptation
  - QLoRALearner: Quantized LoRA (4-bit base)

Usage:
    from molly_evolution.methods import get_method

    learner = get_method("gene-conv", model_name="gpt2", device=device)
    learner.load_model()
    learner.snapshot()
    learner.train_domain(train_enc, epochs=3, lr=5e-5)
    learner.post_train(eval_sets, curr_eval)
    ppl = learner.evaluate(test_enc)
"""

from molly_evolution.methods.base import ContinualLearner
from molly_evolution.methods.gene_conv import GeneConvLearner
from molly_evolution.methods.lora import LoRALearner
from molly_evolution.methods.qlora import QLoRALearner

METHODS = {
    "gene-conv": GeneConvLearner,
    "lora": LoRALearner,
    "qlora": QLoRALearner,
}


def get_method(name: str, **kwargs) -> ContinualLearner:
    """Get a continual learning method by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Choose from: {list(METHODS.keys())}")
    return METHODS[name](**kwargs)


def list_methods():
    """List available methods."""
    return list(METHODS.keys())
