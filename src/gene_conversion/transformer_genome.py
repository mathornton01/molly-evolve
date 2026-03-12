"""
Transformer-scale Dual Genome Module

Maps GPT-2 (and similar transformer) parameters onto biologically meaningful
gene boundaries. Each gene corresponds to a functional component:

    - Attention Q/K/V projection (per layer)
    - Attention output projection (per layer)
    - Attention layer norm (per layer)
    - MLP up-projection (per layer)
    - MLP down-projection (per layer)
    - MLP layer norm (per layer)
    - Token embeddings
    - Position embeddings
    - Final layer norm

Gene conversion operates at this component level: repairing or fixing
entire functional units rather than individual weights.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class TransformerGene:
    """
    A single gene: a group of related parameter tensors forming
    one functional unit of the transformer.

    Stores dual-strand quantized snapshots for gene conversion.
    """

    def __init__(self, name: str, param_names: List[str], n_bits: int = 16):
        self.name = name
        self.param_names = param_names
        self.n_bits = n_bits

        # These get populated by snapshot/sync
        self.complement: Dict[str, torch.Tensor] = {}  # known-good reference
        self.primary: Dict[str, torch.Tensor] = {}     # current (possibly mutated)
        self.scales: Dict[str, float] = {}

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Symmetric quantization to n-bit integer."""
        max_val = tensor.abs().max().item()
        scale = max_val / (2 ** (self.n_bits - 1) - 1) if max_val > 0 else 1.0
        qmin = -(2 ** (self.n_bits - 1))
        qmax = 2 ** (self.n_bits - 1) - 1
        quantized = torch.clamp(
            torch.round(tensor / scale), qmin, qmax
        ).to(torch.int16)
        return quantized, scale

    def _dequantize(self, quantized: torch.Tensor, scale: float) -> torch.Tensor:
        return quantized.float() * scale

    def snapshot_from_model(self, model: nn.Module):
        """Capture current model weights as the complement (reference) strand."""
        params = dict(model.named_parameters())
        for pname in self.param_names:
            q, s = self._quantize(params[pname].data)
            self.complement[pname] = q
            self.primary[pname] = q.clone()
            self.scales[pname] = s

    def sync_primary_from_model(self, model: nn.Module):
        """Update primary strand from current model weights."""
        params = dict(model.named_parameters())
        for pname in self.param_names:
            q, s = self._quantize(params[pname].data)
            self.primary[pname] = q
            self.scales[pname] = s

    def apply_primary_to_model(self, model: nn.Module):
        """Write primary strand back to model."""
        params = dict(model.named_parameters())
        for pname in self.param_names:
            params[pname].data = self._dequantize(
                self.primary[pname], self.scales[pname]
            ).to(params[pname].device)

    def apply_complement_to_model(self, model: nn.Module):
        """Write complement strand back to model (full repair)."""
        params = dict(model.named_parameters())
        for pname in self.param_names:
            # Use complement's own scale (from snapshot time)
            params[pname].data = self._dequantize(
                self.complement[pname], self.scales[pname]
            ).to(params[pname].device)

    def repair(self):
        """Purifying selection: copy complement -> primary."""
        for pname in self.param_names:
            self.primary[pname] = self.complement[pname].clone()

    def fix(self):
        """Adaptive selection: copy primary -> complement."""
        for pname in self.param_names:
            self.complement[pname] = self.primary[pname].clone()

    def divergence(self) -> float:
        """Relative L2 divergence between primary and complement."""
        total_div = 0.0
        total_norm = 0.0
        for pname in self.param_names:
            p = self.primary[pname].float()
            c = self.complement[pname].float()
            total_div += (p - c).norm().item() ** 2
            total_norm += c.norm().item() ** 2
        if total_norm == 0:
            return 0.0
        return math.sqrt(total_div / total_norm)

    @property
    def n_params(self) -> int:
        """Total number of parameters in this gene."""
        return sum(t.numel() for t in self.primary.values())


class TransformerDualGenome:
    """
    Dual-genome wrapper for GPT-2 style transformers.

    Maps model parameters to biologically meaningful genes and provides
    gene conversion operations (repair, fix) at the component level.

    Supports HuggingFace GPT2LMHeadModel out of the box.
    """

    def __init__(self, model: nn.Module, n_bits: int = 16):
        self.model = model
        self.n_bits = n_bits
        self.genes: List[TransformerGene] = []
        self._build_gene_map()

    def _build_gene_map(self):
        """
        Auto-detect transformer structure and create gene boundaries.
        Supports HuggingFace GPT-2 naming conventions.
        """
        all_params = set(dict(self.model.named_parameters()).keys())
        mapped_params = set()

        # Detect model structure
        param_list = list(all_params)

        # Find transformer layers by looking for repeated patterns
        # GPT-2 HF: transformer.h.{i}.attn.c_attn.weight, etc.
        layer_indices = set()
        for pname in param_list:
            parts = pname.split(".")
            for j, part in enumerate(parts):
                if part == "h" and j + 1 < len(parts) and parts[j + 1].isdigit():
                    layer_indices.add(int(parts[j + 1]))

        layer_indices = sorted(layer_indices)

        # Try to detect the prefix (e.g., "transformer.h" or "model.layers")
        prefix = self._detect_layer_prefix(param_list)

        for layer_idx in layer_indices:
            layer_params = [p for p in param_list if f"{prefix}.{layer_idx}." in p]

            # Group by component
            groups = self._group_layer_params(layer_params, prefix, layer_idx)
            for group_name, group_params in groups.items():
                gene = TransformerGene(
                    name=f"L{layer_idx}_{group_name}",
                    param_names=group_params,
                    n_bits=self.n_bits,
                )
                self.genes.append(gene)
                mapped_params.update(group_params)

        # Non-layer params (embeddings, final LN, LM head)
        remaining = all_params - mapped_params
        non_layer_groups = self._group_non_layer_params(sorted(remaining))
        for group_name, group_params in non_layer_groups.items():
            gene = TransformerGene(
                name=group_name,
                param_names=group_params,
                n_bits=self.n_bits,
            )
            self.genes.append(gene)
            mapped_params.update(group_params)

        # Safety: any unmapped params go into a catch-all gene
        still_remaining = all_params - mapped_params
        if still_remaining:
            gene = TransformerGene(
                name="other",
                param_names=sorted(still_remaining),
                n_bits=self.n_bits,
            )
            self.genes.append(gene)

    def _detect_layer_prefix(self, param_list: list) -> str:
        """Detect the prefix path to transformer layers."""
        for pname in param_list:
            # HuggingFace GPT-2: transformer.h.0.attn...
            if "transformer.h." in pname:
                return "transformer.h"
            # Other common patterns
            if "model.layers." in pname:
                return "model.layers"
            if "encoder.layer." in pname:
                return "encoder.layer"
        # Fallback: look for .h. pattern
        for pname in param_list:
            if ".h." in pname:
                parts = pname.split(".h.")
                return parts[0] + ".h"
        return "h"

    def _group_layer_params(
        self, params: List[str], prefix: str, layer_idx: int
    ) -> Dict[str, List[str]]:
        """Group parameters within a transformer layer into functional genes."""
        groups: Dict[str, List[str]] = {}

        for pname in params:
            # Determine component
            suffix = pname.split(f"{prefix}.{layer_idx}.")[-1]

            if "attn" in suffix and ("c_attn" in suffix or "q_proj" in suffix
                                     or "k_proj" in suffix or "v_proj" in suffix
                                     or "qkv" in suffix):
                key = "attn_qkv"
            elif "attn" in suffix and ("c_proj" in suffix or "o_proj" in suffix
                                       or "out_proj" in suffix):
                key = "attn_out"
            elif "ln_1" in suffix or "input_layernorm" in suffix:
                key = "ln_attn"
            elif "ln_2" in suffix or "post_attention_layernorm" in suffix:
                key = "ln_mlp"
            elif "mlp" in suffix and ("c_fc" in suffix or "up_proj" in suffix
                                      or "fc1" in suffix or "gate" in suffix):
                key = "mlp_up"
            elif "mlp" in suffix and ("c_proj" in suffix or "down_proj" in suffix
                                      or "fc2" in suffix):
                key = "mlp_down"
            elif "attn" in suffix:
                key = "attn_other"
            elif "mlp" in suffix:
                key = "mlp_other"
            else:
                key = "other"

            groups.setdefault(key, []).append(pname)

        return groups

    def _group_non_layer_params(
        self, params: List[str]
    ) -> Dict[str, List[str]]:
        """Group non-layer parameters (embeddings, final LN, etc.)."""
        groups: Dict[str, List[str]] = {}

        for pname in params:
            if "wte" in pname or "embed_tokens" in pname or "token_emb" in pname:
                key = "token_embed"
            elif "wpe" in pname or "position_embed" in pname:
                key = "pos_embed"
            elif "ln_f" in pname or "final_layer_norm" in pname or "norm" in pname:
                key = "final_ln"
            elif "lm_head" in pname:
                key = "lm_head"
            else:
                key = "other_global"

            groups.setdefault(key, []).append(pname)

        return groups

    # ── Genome Operations ───────────────────────────────────────

    def snapshot(self):
        """Mark current model weights as the healthy reference (complement)."""
        for gene in self.genes:
            gene.snapshot_from_model(self.model)

    def sync_primary(self):
        """Update primary strands from current model weights."""
        for gene in self.genes:
            gene.sync_primary_from_model(self.model)

    def apply_primary(self):
        """Write primary strands back to model."""
        for gene in self.genes:
            gene.apply_primary_to_model(self.model)

    def apply_complement(self):
        """Write complement strands back to model (full restore)."""
        for gene in self.genes:
            gene.apply_complement_to_model(self.model)

    def repair_genes(self, gene_ids: List[int]):
        """Purifying selection: repair specified genes then update model."""
        for gid in gene_ids:
            self.genes[gid].repair()
            self.genes[gid].apply_primary_to_model(self.model)

    def fix_genes(self, gene_ids: List[int]):
        """Adaptive selection: fix specified genes in complement."""
        for gid in gene_ids:
            self.genes[gid].fix()

    # ── Diagnostics ─────────────────────────────────────────────

    @property
    def total_genes(self) -> int:
        return len(self.genes)

    def gene_summary(self) -> List[Dict]:
        """Summary of all genes: name, param count, divergence."""
        results = []
        for i, gene in enumerate(self.genes):
            results.append({
                "id": i,
                "name": gene.name,
                "n_params": gene.n_params,
                "divergence": gene.divergence(),
            })
        return results

    def print_gene_map(self):
        """Print the full gene map."""
        total_params = 0
        for i, gene in enumerate(self.genes):
            np_ = gene.n_params
            total_params += np_
            print(f"  Gene {i:3d}: {gene.name:25s}  {np_:>10,d} params")
        print(f"  {'TOTAL':>34s}  {total_params:>10,d} params")
