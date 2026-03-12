"""
Tests for C++/CUDA extension ops.

These tests verify that the compiled C++ ops match the Python reference
implementation. Tests are skipped if the extension is not built.
"""

import pytest
import torch
import numpy as np


def has_cpp_ext():
    try:
        import molly_evolution._C
        return True
    except ImportError:
        return False


def has_cuda():
    return has_cpp_ext() and torch.cuda.is_available()


@pytest.mark.skipif(not has_cpp_ext(), reason="C++ extension not built")
class TestQuantize:
    def test_quantize_roundtrip(self):
        import molly_evolution._C as _C
        t = torch.randn(100)
        q, scale = _C.quantize_tensor(t, 16)
        restored = _C.dequantize_tensor(q, scale)
        assert torch.allclose(t, restored, atol=0.01)

    def test_quantize_zeros(self):
        import molly_evolution._C as _C
        t = torch.zeros(50)
        q, scale = _C.quantize_tensor(t, 16)
        assert scale == 0.0
        assert torch.all(q == 0)

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_quantize_cuda(self):
        import molly_evolution._C as _C
        t = torch.randn(1000, device="cuda")
        q, scale = _C.quantize_tensor(t, 16)
        restored = _C.dequantize_tensor(q, scale)
        assert torch.allclose(t, restored, atol=0.01)


@pytest.mark.skipif(not has_cpp_ext(), reason="C++ extension not built")
class TestGeneScoring:
    def test_batched_score_simple(self):
        import molly_evolution._C as _C

        # 3 genes, each with 10 elements
        n_genes = 3
        n_each = 10
        grad = torch.randn(n_genes * n_each)
        deltas = torch.randn(n_genes * n_each)
        offsets = torch.tensor([0, 10, 20, 30], dtype=torch.int64)

        scores = _C.batched_gene_score(grad, deltas, offsets)

        assert scores.shape == (3,)

        # Verify against manual computation
        for g in range(n_genes):
            expected = (grad[g*10:(g+1)*10] * deltas[g*10:(g+1)*10]).sum()
            assert torch.allclose(scores[g], expected, atol=1e-5)

    def test_batched_score_unequal_genes(self):
        import molly_evolution._C as _C

        grad = torch.randn(25)
        deltas = torch.randn(25)
        offsets = torch.tensor([0, 5, 15, 25], dtype=torch.int64)

        scores = _C.batched_gene_score(grad, deltas, offsets)

        expected_0 = (grad[0:5] * deltas[0:5]).sum()
        expected_1 = (grad[5:15] * deltas[5:15]).sum()
        expected_2 = (grad[15:25] * deltas[15:25]).sum()

        assert torch.allclose(scores[0], expected_0, atol=1e-5)
        assert torch.allclose(scores[1], expected_1, atol=1e-5)
        assert torch.allclose(scores[2], expected_2, atol=1e-5)

    @pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
    def test_batched_score_cuda(self):
        import molly_evolution._C as _C

        n_genes = 50
        n_each = 1000
        grad = torch.randn(n_genes * n_each, device="cuda")
        deltas = torch.randn(n_genes * n_each, device="cuda")
        offsets = torch.arange(0, (n_genes + 1) * n_each, n_each,
                               dtype=torch.int64, device="cuda")

        scores = _C.batched_gene_score(grad, deltas, offsets)

        # Verify against CPU reference
        scores_cpu = _C.batched_gene_score(
            grad.cpu(), deltas.cpu(), offsets.cpu())

        assert torch.allclose(scores.cpu(), scores_cpu, atol=1e-4)


@pytest.mark.skipif(not has_cpp_ext(), reason="C++ extension not built")
class TestRepair:
    def test_repair_full(self):
        import molly_evolution._C as _C

        param = torch.randn(10, 20)
        complement = torch.randint(-32767, 32767, (10, 20), dtype=torch.int16)
        scale = 0.001

        original = param.clone()
        _C.repair_gene(param, complement, scale)

        expected = complement.float() * scale
        assert torch.allclose(param, expected, atol=1e-6)

    def test_repair_slice(self):
        import molly_evolution._C as _C

        param = torch.randn(10, 20)
        original = param.clone()

        # Repair columns 5:10
        complement = torch.randint(-32767, 32767, (10, 5), dtype=torch.int16)
        scale = 0.001

        _C.repair_gene(param, complement, scale, dim=1, start=5, end=10)

        # Only columns 5:10 should have changed
        expected_slice = complement.float() * scale
        assert torch.allclose(param[:, 5:10], expected_slice, atol=1e-6)
        # Other columns should be unchanged
        assert torch.allclose(param[:, :5], original[:, :5])
        assert torch.allclose(param[:, 10:], original[:, 10:])

    def test_has_cuda(self):
        import molly_evolution._C as _C
        result = _C.has_cuda()
        assert isinstance(result, bool)
