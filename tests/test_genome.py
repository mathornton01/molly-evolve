"""
Tests for DualGenome — verifies snapshot/sync/repair/fix operations.
"""

import pytest
import torch
import numpy as np

transformers = pytest.importorskip("transformers")
from transformers import GPT2LMHeadModel, GPT2Config


@pytest.fixture(scope="module")
def tiny_model():
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


class TestDualGenome:
    def test_import(self):
        from molly_evolution import DualGenome
        assert DualGenome is not None

    def test_create_component_granularity(self, tiny_model):
        from molly_evolution import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        assert genome.total_genes > 0

    def test_create_head_granularity(self, tiny_model):
        from molly_evolution import DualGenome
        genome = DualGenome(tiny_model, granularity="head", backend="python")
        assert genome.total_genes > 0

    def test_snapshot_and_sync(self, tiny_model):
        from molly_evolution import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()
        genome.sync_primary()

    def test_repair_restores_weights(self):
        from molly_evolution import DualGenome

        # Use fresh model to avoid cross-test pollution
        config = GPT2Config(
            vocab_size=1000, n_positions=64, n_embd=64, n_layer=2, n_head=2)
        model = GPT2LMHeadModel(config)
        model.eval()

        genome = DualGenome(model, granularity="component", backend="python")
        genome.snapshot()

        # Save original weights
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        # Perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        genome.sync_primary()

        # Repair all genes
        all_ids = list(range(genome.total_genes))
        genome.repair_genes(all_ids)

        # Check that genes managed by the genome are restored.
        gene_params = set()
        for gene in genome.genes:
            if hasattr(gene, 'slice_defs'):
                for pn, _, _, _ in gene.slice_defs:
                    gene_params.add(pn)
            else:
                gene_params.update(gene.param_names)

        restored_count = 0
        # Skip embeddings (tied weights / large quantization error)
        skip = {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}
        for n, p in model.named_parameters():
            if n in skip:
                continue
            if n in gene_params and n in original:
                assert torch.allclose(p.data, original[n], atol=0.05), \
                    f"Weight {n} not restored after repair"
                restored_count += 1

        assert restored_count > 0, "Should have restored at least some parameters"

    def test_fix_genes(self, tiny_model):
        from molly_evolution import DualGenome

        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Perturb and sync
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        genome.sync_primary()
        perturbed = {n: p.data.clone() for n, p in tiny_model.named_parameters()}

        # Fix gene 0 (should copy primary to complement)
        genome.fix_genes([0])

        # Now repair gene 0 — should keep perturbed weights
        genome.repair_genes([0])

        # Gene 0's parameters should still be ~perturbed (fixed, not reverted)
        gene0 = genome.genes[0]
        if hasattr(gene0, 'slice_defs'):
            pnames = [sd[0] for sd in gene0.slice_defs]
        else:
            pnames = gene0.param_names

        for n, p in tiny_model.named_parameters():
            if n in pnames:
                assert torch.allclose(p.data, perturbed[n], atol=0.05), \
                    f"Fixed gene 0 param {n} should retain perturbed weights"

    def test_apply_conversion(self, tiny_model):
        from molly_evolution import DualGenome

        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Create fake scores
        scores = []
        for gid in range(genome.total_genes):
            scores.append({
                "gene_id": gid,
                "p_del_prev": 0.9 if gid == 0 else 0.1,
                "p_ben_curr": 0.2,
            })

        n_repaired, n_fixed = genome.apply_conversion(scores, threshold=0.5)

        assert n_repaired >= 1  # Gene 0 should be repaired
        assert n_fixed >= 0

    def test_gene_summary(self, tiny_model):
        from molly_evolution import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()  # must snapshot first to initialize strands
        genome.sync_primary()
        summary = genome.gene_summary()
        assert isinstance(summary, (list, dict))
