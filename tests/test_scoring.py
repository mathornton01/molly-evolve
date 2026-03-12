"""
Tests for GeneScorer — verifies gradient-based scoring matches expectations.

Tests run on CPU with a tiny GPT-2 config to keep execution fast.
"""

import pytest
import torch
import numpy as np

# Skip if transformers not available
transformers = pytest.importorskip("transformers")
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer


@pytest.fixture(scope="module")
def tiny_model():
    """Create a tiny GPT-2 for testing (2 layers, 2 heads, dim 64)."""
    config = GPT2Config(
        vocab_size=1000,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_encoding():
    """Create a sample encoding for scoring."""
    ids = torch.randint(0, 1000, (1, 32))
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask}


class TestGeneScorer:
    def test_import(self):
        from molly_evolution import GeneScorer
        assert GeneScorer is not None

    def test_score_produces_array(self, tiny_model, device, sample_encoding):
        from molly_evolution import DualGenome, GeneScorer

        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Perturb weights to create primary/complement difference
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        genome.sync_primary()

        scorer = GeneScorer(genome, tiny_model, device, use_amp=False)
        scores = scorer._score_one_eval(sample_encoding)

        assert isinstance(scores, np.ndarray)
        assert scores.shape == (genome.total_genes,)
        assert not np.all(scores == 0), "Expected non-zero scores after perturbation"

    def test_score_multi_objective(self, tiny_model, device, sample_encoding):
        from molly_evolution import DualGenome, GeneScorer

        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        genome.sync_primary()

        scorer = GeneScorer(genome, tiny_model, device, use_amp=False)

        eval_sets = [("general", sample_encoding)]
        curr_eval = sample_encoding

        scores = scorer.score_multi_objective(eval_sets, curr_eval)

        assert isinstance(scores, list)
        assert len(scores) == genome.total_genes
        assert all("gene_id" in s for s in scores)
        assert all("p_del_prev" in s for s in scores)
        assert all("p_ben_curr" in s for s in scores)

        # Probabilities should be in [0, 1]
        for s in scores:
            assert 0 <= s["p_del_prev"] <= 1
            assert 0 <= s["p_ben_curr"] <= 1

    def test_head_granularity_more_genes(self, tiny_model, device, sample_encoding):
        from molly_evolution import DualGenome, GeneScorer

        genome_comp = DualGenome(tiny_model, granularity="component", backend="python")
        genome_head = DualGenome(tiny_model, granularity="head", backend="python")

        assert genome_head.total_genes > genome_comp.total_genes

    def test_python_fallback(self, tiny_model, device, sample_encoding):
        """Verify _score_python produces valid results."""
        from molly_evolution import DualGenome, GeneScorer

        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        genome.sync_primary()

        scorer = GeneScorer(genome, tiny_model, device, use_amp=False)
        # Force Python path
        scorer._use_cuda_kernel = False

        scores = scorer._score_one_eval(sample_encoding)
        assert isinstance(scores, np.ndarray)
        assert scores.shape[0] == genome.total_genes
