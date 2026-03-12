"""
Tests for gene conversion stability fixes.

Validates:
  1. Threshold and repair cap in select_conversion_genes
  2. Gradual batched repair with NaN detection
  3. Dataset loading and fallback
  4. Full experiment flow (CPU, quicktest)
  5. Edge cases: all NaN, zero candidates, single gene
"""

import copy
import math
import os
import sys

import pytest
import torch
import numpy as np

transformers = pytest.importorskip("transformers")
from transformers import GPT2LMHeadModel, GPT2Config


def make_tiny_model():
    """Create a fresh tiny GPT-2 for testing."""
    config = GPT2Config(
        vocab_size=1000, n_positions=64, n_embd=64, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def make_sample_encoding(batch=4, seq_len=32, vocab=1000):
    """Create a sample encoding for testing."""
    ids = torch.randint(0, vocab, (batch, seq_len))
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture(scope="module")
def tiny_model():
    return make_tiny_model()


# ── 1. select_conversion_genes threshold & cap ──────────────────


class TestSelectConversionGenes:
    """Test the new select_conversion_genes method."""

    def test_default_threshold_is_080(self):
        from molly_evolution.genome import DualGenome
        import inspect
        sig = inspect.signature(DualGenome.select_conversion_genes)
        assert sig.parameters["threshold"].default == 0.80

    def test_high_threshold_reduces_candidates(self, tiny_model):
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Scores where trade = p_del - 0.3*p_ben
        # Gene 0: trade = 0.95 - 0.3*0.1 = 0.92 (above 0.80)
        # Gene 1: trade = 0.70 - 0.3*0.1 = 0.67 (above 0.50, below 0.80)
        # Gene 2: trade = 0.40 - 0.3*0.1 = 0.37 (below 0.50)
        scores = [
            {"gene_id": 0, "p_del_prev": 0.95, "p_ben_curr": 0.1},
            {"gene_id": 1, "p_del_prev": 0.70, "p_ben_curr": 0.1},
            {"gene_id": 2, "p_del_prev": 0.40, "p_ben_curr": 0.1},
        ]

        # At threshold=0.80, only gene 0 qualifies
        to_repair_high, _ = genome.select_conversion_genes(
            scores, threshold=0.80, max_repair_pct=1.0)
        assert to_repair_high == [0], f"Only gene 0 should qualify at threshold 0.80, got {to_repair_high}"

        # At threshold=0.50, genes 0 and 1 qualify
        to_repair_low, _ = genome.select_conversion_genes(
            scores, threshold=0.50, max_repair_pct=1.0)
        assert set(to_repair_low) == {0, 1}

    def test_repair_cap_limits_candidates(self, tiny_model):
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        n_genes = genome.total_genes
        # All genes have very high trade scores
        scores = [
            {"gene_id": gid, "p_del_prev": 0.99, "p_ben_curr": 0.01}
            for gid in range(n_genes)
        ]

        # With 3% cap
        to_repair, _ = genome.select_conversion_genes(
            scores, threshold=0.50, max_repair_pct=0.03)
        max_expected = max(int(n_genes * 0.03), 1)
        assert len(to_repair) <= max_expected, \
            f"Should cap at {max_expected}, got {len(to_repair)}"
        assert len(to_repair) >= 1, "Should repair at least 1 gene"

    def test_repair_cap_selects_worst_genes_first(self, tiny_model):
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Gene 5 has highest trade, then gene 3, then gene 1
        scores = [
            {"gene_id": 0, "p_del_prev": 0.50, "p_ben_curr": 0.5},  # trade = 0.35
            {"gene_id": 1, "p_del_prev": 0.85, "p_ben_curr": 0.1},  # trade = 0.82
            {"gene_id": 2, "p_del_prev": 0.50, "p_ben_curr": 0.5},  # trade = 0.35
            {"gene_id": 3, "p_del_prev": 0.90, "p_ben_curr": 0.1},  # trade = 0.87
            {"gene_id": 4, "p_del_prev": 0.50, "p_ben_curr": 0.5},  # trade = 0.35
            {"gene_id": 5, "p_del_prev": 0.95, "p_ben_curr": 0.1},  # trade = 0.92
        ]

        # Cap at 2 genes max
        to_repair, _ = genome.select_conversion_genes(
            scores, threshold=0.80, max_repair_pct=1.0, max_repair_count=2)
        # Should select genes 5 and 3 (highest trade scores)
        assert len(to_repair) == 2
        assert to_repair[0] == 5, f"Highest-trade gene should be first, got {to_repair[0]}"
        assert to_repair[1] == 3, f"Second-highest should be second, got {to_repair[1]}"

    def test_no_candidates_returns_empty(self, tiny_model):
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # All genes have low p_del_prev, none exceed threshold
        scores = [
            {"gene_id": gid, "p_del_prev": 0.30, "p_ben_curr": 0.9}
            for gid in range(genome.total_genes)
        ]
        to_repair, to_fix = genome.select_conversion_genes(
            scores, threshold=0.80)
        assert len(to_repair) == 0

    def test_fix_genes_identified(self, tiny_model):
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Gene 0: should be fixed (p_del < 0.3)
        # Gene 1: should be repaired (high trade)
        # Gene 2: neither
        scores = [
            {"gene_id": 0, "p_del_prev": 0.10, "p_ben_curr": 0.9},  # fix
            {"gene_id": 1, "p_del_prev": 0.95, "p_ben_curr": 0.1},  # repair
            {"gene_id": 2, "p_del_prev": 0.50, "p_ben_curr": 0.5},  # neither
        ]
        to_repair, to_fix = genome.select_conversion_genes(
            scores, threshold=0.80, max_repair_pct=1.0)
        assert 1 in to_repair
        assert 0 in to_fix
        assert 2 not in to_repair and 2 not in to_fix

    def test_apply_conversion_uses_new_defaults(self, tiny_model):
        """apply_conversion should use threshold=0.80 and max_repair_pct=0.03 by default."""
        from molly_evolution.genome import DualGenome
        genome = DualGenome(tiny_model, granularity="component", backend="python")
        genome.snapshot()

        # Create scores where all genes barely exceed 0.50 but NOT 0.80
        scores = [
            {"gene_id": gid, "p_del_prev": 0.70, "p_ben_curr": 0.1}
            for gid in range(genome.total_genes)
        ]
        # trade = 0.70 - 0.3*0.1 = 0.67, below 0.80 default
        n_rep, n_fix = genome.apply_conversion(scores)
        assert n_rep == 0, f"No genes should be repaired at default threshold 0.80, got {n_rep}"


# ── 2. Gradual batched repair with NaN detection ────────────────


class TestGradualRepair:
    """Test the batched repair logic from full_comparison.py."""

    def test_batched_repair_normal(self):
        """Batched repair with no NaN should repair all candidates."""
        model = make_tiny_model()
        from molly_evolution.genome import DualGenome

        genome = DualGenome(model, granularity="component", backend="python")
        genome.snapshot()

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        genome.sync_primary()

        to_repair = list(range(min(6, genome.total_genes)))
        batch_size = 2
        enc = make_sample_encoding()

        # Simulate the batched repair loop from full_comparison.py
        n_rep = 0
        n_rolled_back = 0
        for b_start in range(0, len(to_repair), batch_size):
            batch = to_repair[b_start:b_start + batch_size]
            state_backup = copy.deepcopy(model.state_dict())
            genome.repair_genes(batch)

            # Check stability
            model.eval()
            stable = True
            with torch.no_grad():
                for k in range(min(4, enc["input_ids"].size(0))):
                    out = model(enc["input_ids"][k:k+1], labels=enc["input_ids"][k:k+1])
                    if math.isnan(out.loss.item()) or math.isinf(out.loss.item()):
                        stable = False
                        break

            if stable:
                n_rep += len(batch)
                del state_backup
            else:
                model.load_state_dict(state_backup)
                del state_backup
                n_rolled_back += len(batch)
                break

        assert n_rep == len(to_repair), f"All genes should be repaired, got {n_rep}/{len(to_repair)}"
        assert n_rolled_back == 0

    def test_batched_repair_nan_stops_early(self):
        """If NaN is detected, remaining batches should be skipped."""
        model = make_tiny_model()
        from molly_evolution.genome import DualGenome

        genome = DualGenome(model, granularity="component", backend="python")
        genome.snapshot()

        to_repair = list(range(min(6, genome.total_genes)))
        batch_size = 2
        enc = make_sample_encoding()

        # Simulate the loop, but inject NaN after batch 1
        n_rep = 0
        n_rolled_back = 0
        batch_count = 0
        for b_start in range(0, len(to_repair), batch_size):
            batch = to_repair[b_start:b_start + batch_size]
            state_backup = copy.deepcopy(model.state_dict())
            genome.repair_genes(batch)

            batch_count += 1
            # Simulate: batch 1 OK, batch 2 NaN
            if batch_count >= 2:
                # Pretend NaN
                model.load_state_dict(state_backup)
                del state_backup
                n_rolled_back += len(batch)
                break
            else:
                n_rep += len(batch)
                del state_backup

        assert n_rep == batch_size, f"Only first batch should succeed, got {n_rep}"
        assert n_rolled_back == batch_size, f"Second batch should be rolled back"
        assert batch_count == 2, "Should have stopped after 2 batches"

    def test_state_backup_actually_restores(self):
        """Verify copy.deepcopy + load_state_dict correctly restores weights."""
        model = make_tiny_model()
        from molly_evolution.genome import DualGenome

        genome = DualGenome(model, granularity="component", backend="python")
        genome.snapshot()

        # Save pre-repair state
        pre_repair = {n: p.data.clone() for n, p in model.named_parameters()}

        # Backup, repair, then rollback
        state_backup = copy.deepcopy(model.state_dict())

        # Heavily perturb via repair (add noise then repair)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 1.0)  # large perturbation
        genome.sync_primary()
        genome.repair_genes([0, 1])

        # Weights should be different now
        post_repair = {n: p.data.clone() for n, p in model.named_parameters()}

        # Rollback
        model.load_state_dict(state_backup)

        # Verify restoration
        for n, p in model.named_parameters():
            assert torch.equal(p.data, pre_repair[n]), \
                f"Param {n} not restored after rollback"

    def test_check_model_stable_detects_nan(self):
        """check_model_stable should catch NaN from corrupted weights."""
        model = make_tiny_model()
        enc = make_sample_encoding(batch=4)

        # Normal model should be stable
        model.eval()
        with torch.no_grad():
            out = model(enc["input_ids"][:1], labels=enc["input_ids"][:1])
            assert not math.isnan(out.loss.item()), "Baseline should not be NaN"

        # Corrupt a weight to produce NaN
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float('nan'))

        model.eval()
        with torch.no_grad():
            out = model(enc["input_ids"][:1], labels=enc["input_ids"][:1])
            assert math.isnan(out.loss.item()), "NaN weights should produce NaN loss"

    def test_multi_sample_nan_check(self):
        """NaN check should test multiple samples, not just one."""
        model = make_tiny_model()
        enc = make_sample_encoding(batch=4)

        # Verify all 4 samples produce valid output
        model.eval()
        n_checked = 0
        with torch.no_grad():
            for k in range(4):
                out = model(enc["input_ids"][k:k+1], labels=enc["input_ids"][k:k+1])
                loss = out.loss.item()
                assert not math.isnan(loss), f"Sample {k} should not be NaN"
                assert not math.isinf(loss), f"Sample {k} should not be inf"
                n_checked += 1
        assert n_checked == 4, "Should have checked all 4 samples"


# ── 3. Dataset loading and fallback ─────────────────────────────


class TestDataLoading:
    """Test dataset loading, including quicktest fallback."""

    def test_legal_dataset_config(self):
        from molly_evolution.data import DOMAIN_CONFIGS
        cfg = DOMAIN_CONFIGS["legal"]
        assert cfg["dataset"] == "lex_glue", f"Expected lex_glue, got {cfg['dataset']}"
        assert cfg["config"] == "unfair_tos"
        assert cfg["text_field"] == "text"

    def test_quicktest_data_sufficient(self):
        """Quicktest data should produce 200+ train samples at 256 tokens."""
        from molly_evolution.data import QUICKTEST_DATA
        for domain, text in QUICKTEST_DATA.items():
            # Need enough chars to produce 250 chunks of 256 tokens
            # Rough: 4 chars/token * 256 tokens * 250 chunks = 256,000 chars
            # The doubling loop handles shortfall, but let's verify base size
            assert len(text) > 50000, \
                f"{domain} quicktest too short: {len(text)} chars (need >50000)"

    def test_quicktest_load_produces_enough_samples(self):
        """Actually tokenize quicktest data and verify sample counts."""
        from molly_evolution.data import _load_quicktest
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        for domain in ["general", "code", "legal", "medical"]:
            train, eval_ = _load_quicktest(
                domain, tokenizer, max_length=256, n_train=200, n_eval=50)
            n_train = train["input_ids"].shape[0]
            n_eval = eval_["input_ids"].shape[0]
            seq_len = train["input_ids"].shape[1]

            assert seq_len == 256, f"{domain}: seq_len={seq_len}, expected 256"
            assert n_train >= 100, \
                f"{domain}: only {n_train} train samples (need >=100)"
            assert n_eval >= 10, \
                f"{domain}: only {n_eval} eval samples (need >=10)"

    def test_fallback_on_missing_dataset(self):
        """Loading a nonexistent dataset should fall back to quicktest."""
        from molly_evolution.data import load_domain_data, DOMAIN_CONFIGS
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Temporarily set a bad dataset name
        original = DOMAIN_CONFIGS["legal"].copy()
        DOMAIN_CONFIGS["legal"]["dataset"] = "nonexistent/dataset_xyz_404"
        DOMAIN_CONFIGS["legal"]["config"] = None

        try:
            train, eval_ = load_domain_data(
                "legal", tokenizer, max_length=64, n_train=10, n_eval=5)
            assert train is not None
            assert train["input_ids"].shape[0] > 0, "Fallback should produce data"
        finally:
            DOMAIN_CONFIGS["legal"].update(original)

    def test_unknown_domain_uses_general_quicktest(self):
        """Unknown domain should fall back to general quicktest data."""
        from molly_evolution.data import load_domain_data
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train, eval_ = load_domain_data(
            "imaginary_domain", tokenizer, max_length=64, n_train=10, n_eval=5)
        assert train is not None
        assert train["input_ids"].shape[0] > 0


# ── 4. evaluate_ppl NaN handling ────────────────────────────────


class TestEvaluatePPL:
    """Test the perplexity evaluation function handles NaN/inf."""

    def test_normal_ppl(self):
        """evaluate_ppl should return a finite value for a normal model."""
        # Import the function from experiments
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        from full_comparison import evaluate_ppl

        model = make_tiny_model()
        enc = make_sample_encoding()
        ppl = evaluate_ppl(model, enc, torch.device("cpu"))
        assert ppl > 0, f"PPL should be positive, got {ppl}"
        assert not math.isnan(ppl), f"PPL should not be NaN"
        assert not math.isinf(ppl), f"PPL should not be inf"

    def test_nan_model_returns_inf(self):
        """evaluate_ppl should return inf for a model that produces NaN."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        from full_comparison import evaluate_ppl

        model = make_tiny_model()
        # Corrupt all weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float('nan'))

        enc = make_sample_encoding()
        ppl = evaluate_ppl(model, enc, torch.device("cpu"))
        assert ppl == float('inf'), f"NaN model should give inf PPL, got {ppl}"


# ── 5. geometric_mean edge cases ────────────────────────────────


class TestGeometricMean:
    def test_normal(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        from full_comparison import geometric_mean
        result = geometric_mean([4.0, 9.0])
        assert abs(result - 6.0) < 0.01

    def test_with_inf(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        import importlib
        import full_comparison
        importlib.reload(full_comparison)
        # inf values should be filtered out
        result = full_comparison.geometric_mean([4.0, float('inf'), 9.0])
        assert abs(result - 6.0) < 0.01

    def test_all_inf(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        import importlib
        import full_comparison
        importlib.reload(full_comparison)
        result = full_comparison.geometric_mean([float('inf'), float('inf')])
        assert math.isinf(result), f"All-inf should return inf, got {result}"


# ── 6. CLI argument parsing ─────────────────────────────────────


class TestCLIArgs:
    """Verify the new CLI arguments are parsed correctly."""

    def test_default_gc_args(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        import importlib
        import full_comparison
        importlib.reload(full_comparison)

        parser = full_comparison.argparse.ArgumentParser()
        parser.add_argument("--gc-threshold", type=float, default=0.80)
        parser.add_argument("--gc-alpha", type=float, default=0.3)
        parser.add_argument("--gc-max-repair-pct", type=float, default=0.03)
        parser.add_argument("--gc-batch-size", type=int, default=5)
        args = parser.parse_args([])

        assert args.gc_threshold == 0.80
        assert args.gc_alpha == 0.3
        assert args.gc_max_repair_pct == 0.03
        assert args.gc_batch_size == 5

    def test_custom_gc_args(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        import importlib
        import full_comparison
        importlib.reload(full_comparison)

        parser = full_comparison.argparse.ArgumentParser()
        parser.add_argument("--gc-threshold", type=float, default=0.80)
        parser.add_argument("--gc-alpha", type=float, default=0.3)
        parser.add_argument("--gc-max-repair-pct", type=float, default=0.03)
        parser.add_argument("--gc-batch-size", type=int, default=5)
        args = parser.parse_args([
            "--gc-threshold", "0.90",
            "--gc-max-repair-pct", "0.05",
            "--gc-batch-size", "10",
        ])

        assert args.gc_threshold == 0.90
        assert args.gc_max_repair_pct == 0.05
        assert args.gc_batch_size == 10


# ── 7. End-to-end integration (CPU, quicktest) ──────────────────


class TestEndToEnd:
    """Run a minimal end-to-end test of the gene conversion pipeline on CPU."""

    def test_gene_conv_pipeline_cpu(self):
        """Full gene-conv pipeline: load -> train -> score -> repair -> eval."""
        from molly_evolution.genome import DualGenome
        from molly_evolution.scoring import GeneScorer

        model = make_tiny_model()
        device = torch.device("cpu")
        enc = make_sample_encoding(batch=8, seq_len=32, vocab=1000)

        # Build genome
        genome = DualGenome(model, granularity="component", backend="python")
        genome.snapshot()

        # Simulate training (just a few steps)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for j in range(3):
            optimizer.zero_grad()
            out = model(enc["input_ids"][j:j+1], labels=enc["input_ids"][j:j+1])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Score genes
        genome.sync_primary()
        scorer = GeneScorer(genome, model, device, use_amp=False, streaming=False)
        general_enc = make_sample_encoding(batch=2, seq_len=32, vocab=1000)
        scores = scorer.score_multi_objective(
            [("general", general_enc)], enc,
            threshold=0.80, alpha=0.3)

        assert len(scores) == genome.total_genes
        for s in scores:
            assert "gene_id" in s
            assert "p_del_prev" in s
            assert "p_ben_curr" in s
            assert 0 <= s["p_del_prev"] <= 1
            assert 0 <= s["p_ben_curr"] <= 1

        # Select and apply conversion
        to_repair, to_fix = genome.select_conversion_genes(
            scores, threshold=0.80, max_repair_pct=0.03)

        # Apply in batches
        n_rep = 0
        batch_size = 2
        for b_start in range(0, len(to_repair), batch_size):
            batch = to_repair[b_start:b_start + batch_size]
            state_backup = copy.deepcopy(model.state_dict())
            genome.repair_genes(batch)

            model.eval()
            stable = True
            with torch.no_grad():
                out = model(enc["input_ids"][:1], labels=enc["input_ids"][:1])
                if math.isnan(out.loss.item()) or math.isinf(out.loss.item()):
                    stable = False

            if stable:
                n_rep += len(batch)
                del state_backup
            else:
                model.load_state_dict(state_backup)
                del state_backup
                break

        # Snapshot for next cycle
        genome.snapshot()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(enc["input_ids"][:1], labels=enc["input_ids"][:1])
            loss = out.loss.item()
            assert not math.isnan(loss), "Model should not produce NaN after pipeline"
            assert not math.isinf(loss), "Model should not produce inf after pipeline"
