"""
Genetic Weight Encoding — Core Module

Encodes neural network weights as dual-strand "genetic chromosomes" inspired by
the Amazon Molly's genome structure. Each weight tensor becomes a chromosome with:

  - Primary strand:     quantized weights used for forward computation
  - Complementary strand: redundant copy enabling selective repair

Gene conversion copies contiguous blocks ("genes") between strands:
  - Repair:   complement → primary  (purifying selection — revert harmful mutations)
  - Fixation: primary → complement  (adaptive selection — preserve beneficial mutations)

The dual-strand structure is analogous to the Amazon Molly's two ancestral
haploid genomes (from P. mexicana and P. latipinna), which serve as mutual
references for gene conversion.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GeneticChromosome:
    """
    A weight tensor encoded as a dual-strand genetic chromosome.

    Each scalar weight is a "codon" — a quantized integer value.
    Codons are grouped into "genes" — contiguous blocks that are the
    unit of gene conversion (copied together, like biological gene conversion
    operates on contiguous DNA stretches).

    Two strands maintain the same information:
      - primary:     the "active" genome (subject to mutation via training)
      - complement:  the "reference" genome (snapshot of a known-good state)

    When strands disagree, a mutation has occurred. Gene conversion selectively
    resolves disagreements by copying between strands.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        n_bits: int = 16,
        gene_size: Optional[int] = None,
    ):
        """
        Encode a weight tensor as a genetic chromosome.

        Args:
            weights:   float weight tensor to encode
            n_bits:    quantization depth (bits per codon). 16-bit gives
                       ~1/32768 relative precision — negligible quantization error
            gene_size: number of codons per gene. Default: sqrt(n_codons)
        """
        self.shape = weights.shape
        self.n_bits = n_bits
        self.n_codons = weights.numel()

        # Symmetric quantization parameters
        max_val = weights.abs().max().item()
        self.scale = max_val / (2 ** (n_bits - 1) - 1) if max_val > 0 else 1.0

        # Primary strand: quantized from current weights
        self.primary = self._quantize(weights)

        # Complementary strand: initially identical (no mutations yet)
        self.complement = self.primary.clone()

        # Partition into genes
        if gene_size is None:
            gene_size = max(1, int(math.sqrt(self.n_codons)))
        boundaries = list(range(0, self.n_codons, gene_size))
        if boundaries[-1] != self.n_codons:
            boundaries.append(self.n_codons)
        self.gene_boundaries = boundaries

    def _quantize(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize float weights to n-bit integers (symmetric, zero-centered)."""
        qmin = -(2 ** (self.n_bits - 1))
        qmax = 2 ** (self.n_bits - 1) - 1
        return torch.clamp(
            torch.round(weights.flatten() / self.scale), qmin, qmax
        ).to(torch.int32)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def n_genes(self) -> int:
        return len(self.gene_boundaries) - 1

    def gene_slice(self, gene_idx: int) -> slice:
        return slice(self.gene_boundaries[gene_idx], self.gene_boundaries[gene_idx + 1])

    # ── Encode / Decode ─────────────────────────────────────────────

    def decode(self, strand: str = "primary") -> torch.Tensor:
        """Dequantize a strand back to float weights in original shape."""
        source = self.primary if strand == "primary" else self.complement
        return (source.float() * self.scale).reshape(self.shape)

    def update_primary(self, new_weights: torch.Tensor):
        """Re-quantize new float weights into the primary strand."""
        self.primary = self._quantize(new_weights)

    def snapshot(self):
        """Copy primary → complement. Marks current state as the reference."""
        self.complement = self.primary.clone()

    # ── Mutation Detection ──────────────────────────────────────────

    def detect_divergence(self) -> List[float]:
        """
        Per-gene relative L2 divergence between primary and complement.
        A divergence of 0 means no mutation; higher = more change.
        """
        divergences = []
        for i in range(self.n_genes):
            s = self.gene_slice(i)
            p = self.primary[s].float()
            c = self.complement[s].float()
            denom = c.norm().item()
            if denom == 0:
                denom = 1.0
            divergences.append((p - c).norm().item() / denom)
        return divergences

    # ── Gene Conversion ─────────────────────────────────────────────

    def repair_gene(self, gene_idx: int):
        """
        Purifying selection: copy complement → primary for one gene.
        Reverts the mutation at this locus to the known-good reference.
        """
        s = self.gene_slice(gene_idx)
        self.primary[s] = self.complement[s].clone()

    def fix_gene(self, gene_idx: int):
        """
        Adaptive selection: copy primary → complement for one gene.
        Accepts the mutation as the new reference (beneficial fixation).
        """
        s = self.gene_slice(gene_idx)
        self.complement[s] = self.primary[s].clone()


class DualGenomeModule(nn.Module):
    """
    Wraps any nn.Module with dual-genome genetic encoding.

    The wrapped module's parameters are encoded as GeneticChromosomes.
    Training modifies the module's weights normally (introducing "mutations").
    After training, gene conversion can selectively repair or fix mutations
    based on Bayesian fitness scoring.

    Usage:
        model = SmallMLP()
        train(model, task_a_data)            # Train normally

        genome = DualGenomeModule(model)     # Wrap with genetic encoding
        genome.snapshot()                    # Mark current state as healthy

        train(model, task_b_data)            # More training (mutations)
        genome.sync_primary()               # Update chromosomes from weights

        # Now: primary = mutated weights, complement = healthy reference
        # Gene conversion can selectively repair harmful mutations
    """

    def __init__(
        self,
        module: nn.Module,
        n_bits: int = 16,
        gene_size: Optional[int] = None,
    ):
        super().__init__()
        self.module = module
        self.n_bits = n_bits
        self.gene_size = gene_size
        self.chromosomes: Dict[str, GeneticChromosome] = {}
        self._encode()

    def _encode(self):
        """Encode all module parameters as genetic chromosomes."""
        for name, param in self.module.named_parameters():
            self.chromosomes[name] = GeneticChromosome(
                param.data, self.n_bits, self.gene_size
            )

    def forward(self, x):
        return self.module(x)

    # ── Genome Management ───────────────────────────────────────────

    def snapshot(self):
        """
        Mark current module weights as the "healthy" reference genome.
        Updates both primary and complement strands from current weights.
        """
        for name, param in self.module.named_parameters():
            self.chromosomes[name].update_primary(param.data)
            self.chromosomes[name].snapshot()

    def sync_primary(self):
        """
        Update primary strands from current module weights.
        Call after training to reflect accumulated mutations in the encoding.
        Complement strands are NOT modified.
        """
        for name, param in self.module.named_parameters():
            self.chromosomes[name].update_primary(param.data)

    def apply_primary(self):
        """Write primary strands back to module weights (dequantized)."""
        for name, param in self.module.named_parameters():
            param.data = self.chromosomes[name].decode("primary")

    def apply_complement(self):
        """Write complement strands back to module weights (full restore)."""
        for name, param in self.module.named_parameters():
            param.data = self.chromosomes[name].decode("complement")

    # ── Gene Indexing ───────────────────────────────────────────────

    @property
    def total_genes(self) -> int:
        return sum(c.n_genes for c in self.chromosomes.values())

    def gene_info(self) -> List[Tuple[str, int]]:
        """
        Flat list of (chromosome_name, local_gene_idx) for every gene.
        Index into this list with flat gene IDs.
        """
        info = []
        for name, chrom in self.chromosomes.items():
            for i in range(chrom.n_genes):
                info.append((name, i))
        return info

    def _resolve_gene(self, flat_gene_id: int) -> Tuple[str, int]:
        """Convert a flat gene ID to (chromosome_name, local_gene_idx)."""
        offset = 0
        for name, chrom in self.chromosomes.items():
            if flat_gene_id < offset + chrom.n_genes:
                return name, flat_gene_id - offset
            offset += chrom.n_genes
        raise IndexError(f"Gene ID {flat_gene_id} out of range (total: {self.total_genes})")

    # ── Gene Conversion ─────────────────────────────────────────────

    def repair_genes(self, gene_ids: List[int]):
        """
        Purifying selection: for each gene ID, copy complement → primary.
        Then write the repaired weights back to the module.
        """
        for gid in gene_ids:
            name, local_idx = self._resolve_gene(gid)
            self.chromosomes[name].repair_gene(local_idx)
        self.apply_primary()

    def fix_genes(self, gene_ids: List[int]):
        """
        Adaptive selection: for each gene ID, copy primary → complement.
        Accepts these mutations as the new reference.
        """
        for gid in gene_ids:
            name, local_idx = self._resolve_gene(gid)
            self.chromosomes[name].fix_gene(local_idx)

    # ── Diagnostics ─────────────────────────────────────────────────

    def detect_all_divergence(self) -> List[Tuple[str, int, float]]:
        """Per-gene divergence between primary and complement, across all chromosomes."""
        results = []
        for name, chrom in self.chromosomes.items():
            for i, d in enumerate(chrom.detect_divergence()):
                results.append((name, i, d))
        return results

    def divergence_by_layer(self) -> Dict[str, float]:
        """Mean divergence per chromosome (i.e., per parameter tensor)."""
        result = {}
        for name, chrom in self.chromosomes.items():
            divs = chrom.detect_divergence()
            result[name] = sum(divs) / len(divs) if divs else 0.0
        return result
