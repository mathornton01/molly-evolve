#!/usr/bin/env python3
"""
Prototype: Self-Repairing Weight Encoding via Gene Conversion

Demonstrates that encoding neural network weights as dual-strand genetic
chromosomes enables selective repair of catastrophic forgetting while
preserving new capabilities.

Experiment:
  - 4D input space with two tasks that share SOME but not all features
  - Task A uses features [x1, x2, x3], Task B uses features [x2, x3, x4]
  - Features x2, x3 are shared; x1 is A-only; x4 is B-only
  - Training on B after A causes forgetting of x1-related weights
  - Gene conversion selectively repairs x1-related genes while keeping
    x4-related genes and shared features intact

Expected: gene conversion achieves better COMBINED accuracy than either
"no repair" (forgets A) or "full repair" (forgets B).

Usage:
    cd molly-evolve
    python experiments/prototype_encoding.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.encoding import DualGenomeModule


# ===================================================================
#  Synthetic Data — 4D with partial feature overlap
# ===================================================================

def make_data(n: int, task: str, seed: int) -> tuple:
    """
    Generate 4D binary classification data.

    Task A: class = sign( sin(x1) + x2 + x3 )    -- uses features 1,2,3
    Task B: class = sign( x2 + x3 + tanh(x4) )    -- uses features 2,3,4

    Shared features: x2, x3 (linear contribution to both)
    A-only feature:  x1 (nonlinear -- sin)
    B-only feature:  x4 (nonlinear -- tanh)

    This structure means:
      - Hidden neurons for x2, x3 should transfer between tasks
      - Hidden neurons for x1 will degrade when training on B (not reinforced)
      - Hidden neurons for x4 are new in B
      - Gene conversion should repair x1 damage and keep x4 learning
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4).astype(np.float32)

    if task == "A":
        score = np.sin(X[:, 0]) + X[:, 1] + X[:, 2]
    elif task == "B":
        score = X[:, 1] + X[:, 2] + np.tanh(X[:, 3])
    else:
        raise ValueError(f"Unknown task: {task}")

    y = (score > 0).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)


# ===================================================================
#  Model
# ===================================================================

class SmallMLP(nn.Module):
    def __init__(self, input_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
#  Training & Evaluation
# ===================================================================

def train_model(model, X, y, epochs=300, lr=0.01):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0.5) == y).float().mean().item()


def bce_loss(model, X, y):
    model.eval()
    with torch.no_grad():
        return nn.BCELoss()(model(X), y).item()


# ===================================================================
#  Bayesian Gene Scoring
# ===================================================================

def bayesian_gene_scoring(genome, X_a, y_a, X_b, y_b):
    """
    Score each gene via chimeric evaluation.

    For each gene g:
      - Repair ONLY gene g (complement -> primary)
      - Measure effect on Task A loss and Task B loss
      - Restore original state
      - Compute Bayesian posterior on mutation effect

    Returns per-gene scores with P(deleterious for A) and P(beneficial for B).
    """
    genome.sync_primary()

    saved = {}
    for name, chrom in genome.chromosomes.items():
        saved[name] = chrom.primary.clone()

    baseline_loss_a = bce_loss(genome, X_a, y_a)
    baseline_loss_b = bce_loss(genome, X_b, y_b)

    raw_scores = []
    for gene_id in range(genome.total_genes):
        genome.repair_genes([gene_id])

        chi_loss_a = bce_loss(genome, X_a, y_a)
        chi_loss_b = bce_loss(genome, X_b, y_b)

        for name, chrom in genome.chromosomes.items():
            chrom.primary = saved[name].clone()
        genome.apply_primary()

        # delta_a > 0 means repairing this gene REDUCED Task A loss (gene was harmful)
        # delta_b > 0 means repairing this gene INCREASED Task B loss (gene was helpful)
        raw_scores.append({
            "gene_id": gene_id,
            "delta_a": baseline_loss_a - chi_loss_a,
            "delta_b": chi_loss_b - baseline_loss_b,
        })

    # Empirical Bayes: set prior variance from observed deltas
    var_a = max(np.var([s["delta_a"] for s in raw_scores]), 1e-10)
    var_b = max(np.var([s["delta_b"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        # Gaussian conjugate posterior: prior N(0, var), noise var = var
        # Posterior: N(delta/2, var/2)
        sig_a = np.sqrt(var_a / 2)
        sig_b = np.sqrt(var_b / 2)

        scores.append({
            **s,
            "p_deleterious_a": 1 - stats.norm.cdf(0, loc=s["delta_a"] / 2, scale=sig_a),
            "p_beneficial_b": 1 - stats.norm.cdf(0, loc=s["delta_b"] / 2, scale=sig_b),
        })

    return scores


# ===================================================================
#  Gene Conversion with Weighted Trade-off
# ===================================================================

def gene_conversion(genome, scores, repair_threshold=0.50, alpha=0.3,
                    conversion_rate=0.6, seed=0):
    """
    Apply gene conversion using a weighted trade-off score.

    For each gene, compute:
        trade_off = P(deleterious_A) - alpha * P(beneficial_B)

    If trade_off > repair_threshold:
        REPAIR (complement -> primary): revert this mutation

    If P(deleterious_A) < 0.3:
        FIX (primary -> complement): accept this mutation permanently

    Else:
        SKIP

    alpha controls the balance:
        alpha=0: pure Task A repair (ignores Task B impact)
        alpha=1: never repair if gene helps B at all
        alpha=0.3: repair if A-damage is ~3x more than B-benefit

    In biology, gene conversion does not have perfect information about
    consequences -- it operates locally with some stochasticity. The
    conversion_rate parameter models this: only a fraction of genes
    undergo conversion at each cycle.
    """
    rng = np.random.RandomState(seed)
    repaired, fixed, skipped = [], [], []

    for s in scores:
        gid = s["gene_id"]

        if rng.random() > conversion_rate:
            skipped.append(gid)
            continue

        trade_off = s["p_deleterious_a"] - alpha * s["p_beneficial_b"]

        if trade_off > repair_threshold:
            repaired.append(gid)
        elif s["p_deleterious_a"] < 0.3:
            fixed.append(gid)
        else:
            skipped.append(gid)

    if repaired:
        genome.repair_genes(repaired)
    if fixed:
        genome.fix_genes(fixed)

    return {"repaired": repaired, "fixed": fixed, "skipped": skipped}


# ===================================================================
#  Helpers
# ===================================================================

def layer_name(param_name):
    """Readable layer name from 'net.0.weight' style names."""
    parts = param_name.split(".")
    idx = parts[1] if len(parts) > 1 else "?"
    kind = parts[2] if len(parts) > 2 else "?"
    names = {"0": "input", "2": "hidden", "4": "output"}
    return f"{names.get(idx, 'L'+idx)}_{kind}"


def print_layer_breakdown(genome, gene_ids, label):
    """Show which layers the affected genes belong to."""
    if not gene_ids:
        return
    info = genome.gene_info()
    counts = {}
    for gid in gene_ids:
        name, _ = info[gid]
        ln = layer_name(name)
        counts[ln] = counts.get(ln, 0) + 1
    parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
    print(f"    {label}: {', '.join(parts)}")


# ===================================================================
#  Main Experiment
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    GENE_SIZE = 16

    print("=" * 65)
    print("  MOLLY-EVOLVE: Self-Repairing Weight Encoding Prototype")
    print("  Inspired by Amazon Molly (P. formosa) gene conversion")
    print("=" * 65)
    print()
    print("  Tasks (4D input with partial feature overlap):")
    print("    A: sign( sin(x1) + x2 + x3 )  -- features 1,2,3")
    print("    B: sign( x2 + x3 + tanh(x4) )  -- features 2,3,4")
    print("    Shared: x2, x3 | A-only: x1 | B-only: x4")

    # -- Data --
    X_a_tr, y_a_tr = make_data(3000, "A", seed=42)
    X_a_te, y_a_te = make_data(1000, "A", seed=99)
    X_b_tr, y_b_tr = make_data(3000, "B", seed=42)
    X_b_te, y_b_te = make_data(1000, "B", seed=99)

    # ---- Phase 1: Train on Task A ----
    print("\n[Phase 1] Train on Task A")
    model = SmallMLP(input_dim=4, hidden=64)
    train_model(model, X_a_tr, y_a_tr, epochs=500, lr=0.005)

    acc_a0 = accuracy(model, X_a_te, y_a_te)
    acc_b0 = accuracy(model, X_b_te, y_b_te)
    print(f"  Task A: {acc_a0:.1%}    Task B: {acc_b0:.1%}")

    # ---- Phase 2: Encode ----
    print("\n[Phase 2] Encode weights as genetic chromosomes")
    genome = DualGenomeModule(model, n_bits=16, gene_size=GENE_SIZE)
    genome.snapshot()

    print(f"  Chromosomes: {len(genome.chromosomes)}   "
          f"Total genes: {genome.total_genes}")
    for name, chrom in genome.chromosomes.items():
        print(f"    {layer_name(name):20s}  {chrom.n_codons:5d} codons  "
              f"{chrom.n_genes:3d} genes")

    # ---- Phase 3: Fine-tune on Task B ----
    print("\n[Phase 3] Fine-tune on Task B (causes forgetting of A)")
    train_model(model, X_b_tr, y_b_tr, epochs=500, lr=0.005)

    acc_a1 = accuracy(model, X_a_te, y_a_te)
    acc_b1 = accuracy(model, X_b_te, y_b_te)
    forgetting = acc_a0 - acc_a1
    print(f"  Task A: {acc_a1:.1%}  (was {acc_a0:.1%}, "
          f"forgetting: {forgetting:+.1%})")
    print(f"  Task B: {acc_b1:.1%}  (was {acc_b0:.1%})")

    genome.sync_primary()

    # Mutation heatmap
    div = genome.divergence_by_layer()
    print("\n  Mutation divergence per layer:")
    for name, d in div.items():
        bar = "#" * min(int(d * 40), 60)
        print(f"    {layer_name(name):20s}  {d:.4f}  {bar}")

    # Save mutated state
    mut_primaries = {}
    for name, chrom in genome.chromosomes.items():
        mut_primaries[name] = chrom.primary.clone()

    # ---- Phase 4: Bayesian gene scoring ----
    print(f"\n[Phase 4] Bayesian gene scoring "
          f"({genome.total_genes} chimeric evaluations)")
    scores = bayesian_gene_scoring(genome, X_a_te, y_a_te, X_b_te, y_b_te)

    n_del = sum(1 for s in scores if s["p_deleterious_a"] > 0.65)
    n_ben = sum(1 for s in scores if s["p_beneficial_b"] > 0.65)
    print(f"  Deleterious for A (P > 0.65):  {n_del:3d} genes")
    print(f"  Beneficial for B (P > 0.65):   {n_ben:3d} genes")

    # Top deleterious genes
    top = sorted(scores, key=lambda s: s["p_deleterious_a"], reverse=True)[:5]
    info = genome.gene_info()
    print("\n  Top 5 genes most harmful to Task A:")
    for s in top:
        nm, loc = info[s["gene_id"]]
        print(f"    gene {s['gene_id']:3d} ({layer_name(nm)} #{loc})  "
              f"P(del_A)={s['p_deleterious_a']:.3f}  "
              f"P(ben_B)={s['p_beneficial_b']:.3f}  "
              f"d_a={s['delta_a']:+.4f}")

    # ---- Phase 5: Compare strategies ----
    #
    # Metrics:
    #   Mean     = (A + B) / 2                        (rewards total accuracy)
    #   Worst    = min(A, B)                           (rewards balance)
    #   Harmonic = 2*A*B / (A+B)                       (penalizes imbalance)
    #
    # The "Worst" metric is the most relevant for gene conversion:
    # in biology, fitness = survival of the organism across ALL selective
    # pressures, not just the average. A fish that handles 90% of both
    # pressures beats one that handles 100% of one and 50% of the other.

    def hmean(a, b):
        return 2 * a * b / (a + b) if (a + b) > 0 else 0

    print("\n[Phase 5] Comparing strategies")
    print("-" * 75)
    hdr = (f"  {'Strategy':32s} {'Task A':>7s} {'Task B':>7s}"
           f" {'Mean':>7s} {'Worst':>7s} {'Harmon':>7s}")
    print(hdr)
    print("-" * 75)

    def print_row(label, a, b):
        print(f"  {label:32s} {a:>7.1%} {b:>7.1%}"
              f" {(a+b)/2:>7.1%} {min(a,b):>7.1%} {hmean(a,b):>7.1%}")

    # 1. After Task A only
    print_row("After Task A only", acc_a0, acc_b0)

    # 2. No repair (post-B forgetting)
    print_row("After Task B (no repair)", acc_a1, acc_b1)

    # 3. Full repair (restore everything to Task A)
    genome.apply_complement()
    acc_a_full = accuracy(genome, X_a_te, y_a_te)
    acc_b_full = accuracy(genome, X_b_te, y_b_te)
    print_row("Full repair (all genes)", acc_a_full, acc_b_full)

    # 4. Restore mutated state, then apply gene conversion
    for name, chrom in genome.chromosomes.items():
        chrom.primary = mut_primaries[name].clone()
    genome.apply_primary()

    result = gene_conversion(genome, scores,
                             repair_threshold=0.50,
                             alpha=0.3,
                             conversion_rate=0.6)

    acc_a_gc = accuracy(genome, X_a_te, y_a_te)
    acc_b_gc = accuracy(genome, X_b_te, y_b_te)
    print_row("Gene conversion (selective)", acc_a_gc, acc_b_gc)

    # 5. Oracle: train on BOTH tasks jointly
    model_oracle = SmallMLP(input_dim=4, hidden=64)
    torch.manual_seed(42)
    model_oracle.train()
    opt = optim.Adam(model_oracle.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    for _ in range(500):
        opt.zero_grad()
        loss_a = criterion(model_oracle(X_a_tr), y_a_tr)
        loss_b = criterion(model_oracle(X_b_tr), y_b_tr)
        (loss_a + loss_b).backward()
        opt.step()

    acc_a_or = accuracy(model_oracle, X_a_te, y_a_te)
    acc_b_or = accuracy(model_oracle, X_b_te, y_b_te)
    print_row("Oracle (joint training)", acc_a_or, acc_b_or)

    print("-" * 75)

    # ---- Phase 6: Analysis ----
    print(f"\n[Phase 6] Gene conversion details")
    print(f"  Genes repaired (complement -> primary):  {len(result['repaired'])}")
    print(f"  Genes fixed (primary -> complement):     {len(result['fixed'])}")
    print(f"  Genes skipped:                           {len(result['skipped'])}")

    print_layer_breakdown(genome, result["repaired"], "Repaired")
    print_layer_breakdown(genome, result["fixed"], "Fixed")

    # Verdict — use "worst task" as primary metric
    worst_gc = min(acc_a_gc, acc_b_gc)
    worst_none = min(acc_a1, acc_b1)
    worst_full = min(acc_a_full, acc_b_full)
    worst_oracle = min(acc_a_or, acc_b_or)
    best_baseline_worst = max(worst_none, worst_full)

    print("\n" + "=" * 75)
    print("  Worst-task accuracy (the metric that matters for robustness):")
    print(f"    No repair:       {worst_none:.1%}")
    print(f"    Full repair:     {worst_full:.1%}")
    print(f"    Gene conversion: {worst_gc:.1%}")
    print(f"    Oracle:          {worst_oracle:.1%}")

    if worst_gc > best_baseline_worst:
        gain = worst_gc - best_baseline_worst
        print(f"\n  Gene conversion OUTPERFORMS both baselines by +{gain:.1%}")
        if worst_oracle > 0:
            closeness = worst_gc / worst_oracle * 100
            print(f"  Reaches {closeness:.0f}% of oracle worst-task performance")
    else:
        print("\n  Gene conversion did not outperform on worst-task metric.")
        print("  (Try adjusting thresholds or conversion_rate)")
    print("=" * 75)


if __name__ == "__main__":
    main()
