"""
Tests for the high-level MolliTrainer API.

These tests exercise the LoRA-parity entry point:
  - MolliTrainer(model, tokenizer) wrapping
  - fit() with and without repair
  - save_pretrained() / from_pretrained() roundtrip preserves weights
  - evaluate() and generate() return sane results
"""

import os
import tempfile

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


@pytest.fixture(scope="module")
def tiny_tokenizer():
    # Tries to load the real gpt2 tokenizer; skip the whole module if the
    # hub is unreachable (offline CI).
    try:
        return GPT2Tokenizer.from_pretrained("gpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gpt2 tokenizer unavailable: {exc}")


@pytest.fixture
def tiny_model(tiny_tokenizer):
    config = GPT2Config(
        vocab_size=tiny_tokenizer.vocab_size,
        n_positions=64,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


@pytest.fixture
def toy_corpus():
    train = ["the quick brown fox jumps over the lazy dog"] * 16
    train += ["pack my box with five dozen liquor jugs"] * 16
    eval_ = ["how vexingly quick daft zebras jump"] * 4
    protect = ["the five boxing wizards jump quickly"] * 4
    return train, eval_, protect


class TestMolliTrainer:
    def test_wrap_existing_model(self, tiny_model, tiny_tokenizer):
        from molly_evolution import MolliTrainer
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"),
                               granularity="head")
        assert trainer.genome.total_genes > 0
        assert trainer.model is tiny_model

    def test_fit_without_repair(self, tiny_model, tiny_tokenizer, toy_corpus):
        from molly_evolution import MolliTrainer
        train, eval_, _ = toy_corpus
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"))
        metrics = trainer.fit(train, eval_texts=eval_, epochs=1, lr=1e-4,
                              max_length=32, repair=False)
        assert metrics["steps"] > 0
        assert "repaired" not in metrics

    def test_fit_with_repair(self, tiny_model, tiny_tokenizer, toy_corpus):
        from molly_evolution import MolliTrainer
        train, eval_, protect = toy_corpus
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"))
        metrics = trainer.fit(
            train, eval_texts=eval_, protect_texts=protect,
            epochs=1, lr=1e-4, max_length=32,
            threshold=0.50, max_repair_pct=0.10)
        assert metrics["steps"] > 0
        assert "repaired" in metrics
        assert "fixed" in metrics

    def test_evaluate_returns_finite_ppl(self, tiny_model, tiny_tokenizer,
                                          toy_corpus):
        from molly_evolution import MolliTrainer
        _, eval_, _ = toy_corpus
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"))
        ppl = trainer.evaluate(eval_, max_length=32)
        assert ppl > 0 and ppl != float("inf")

    def test_generate_runs(self, tiny_model, tiny_tokenizer):
        from molly_evolution import MolliTrainer
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"))
        text = trainer.generate("hello", max_new_tokens=3)
        assert isinstance(text, str) and len(text) >= len("hello")

    def test_save_and_load_roundtrip(self, tiny_model, tiny_tokenizer,
                                      toy_corpus):
        """
        save_pretrained -> from_pretrained should produce a trainer whose
        perplexity on a held-out set exactly matches the original.
        """
        from molly_evolution import MolliTrainer, DualGenome
        train, eval_, protect = toy_corpus
        trainer = MolliTrainer(tiny_model, tiny_tokenizer,
                               device=torch.device("cpu"))
        trainer.fit(train, eval_texts=eval_, protect_texts=protect,
                    epochs=1, lr=1e-4, max_length=32,
                    threshold=0.50, max_repair_pct=0.10)
        ppl_before = trainer.evaluate(eval_, max_length=32)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt")
            trainer.save_pretrained(path)
            assert os.path.exists(os.path.join(path, DualGenome.GENOME_FILENAME))
            assert os.path.exists(os.path.join(path, "config.json"))

            reloaded = MolliTrainer.from_pretrained(
                path, device=torch.device("cpu"), granularity="head")
            assert reloaded.genome.total_genes == trainer.genome.total_genes
            ppl_after = reloaded.evaluate(eval_, max_length=32)

        # Weights should match exactly, so perplexities should agree closely.
        assert abs(ppl_before - ppl_after) < 1e-3, (
            f"ppl drift after save/load: {ppl_before} vs {ppl_after}")

    def test_genome_save_load_unit(self, tiny_model):
        """Direct DualGenome.save/load unit test without the trainer."""
        from molly_evolution import DualGenome
        genome = DualGenome(tiny_model, granularity="component",
                            backend="python")
        genome.snapshot()

        # Mutate weights in-place and sync
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        genome.sync_primary()

        with tempfile.TemporaryDirectory() as tmp:
            out = genome.save(tmp)
            assert os.path.exists(out)

            # Build a fresh model of the same shape and load the genome.
            config = tiny_model.config
            fresh = GPT2LMHeadModel(config)
            reloaded = DualGenome.load(tmp, fresh, backend="python")
            assert reloaded.total_genes == genome.total_genes
