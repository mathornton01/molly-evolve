#!/usr/bin/env python3
"""
GPT-2 Scale Gene Conversion Experiment

Demonstrates gene conversion on a real language model:
  1. Load pretrained GPT-2-small (124M params)
  2. Evaluate baseline perplexity on WikiText-2 (general language = Task A)
  3. Snapshot weights as healthy genome
  4. Fine-tune on a specific domain (Task B)
  5. Measure catastrophic forgetting on Task A
  6. Apply Bayesian gene conversion
  7. Compare: no repair, full repair, gene conversion, oracle

Gene boundaries map to transformer components (75 genes total):
  - Per layer: attn_qkv, attn_out, ln_attn, mlp_up, mlp_down, ln_mlp
  - Global: token_embed, pos_embed, final_ln

Usage:
    cd molly-evolve
    python experiments/gpt2_gene_conversion.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.transformer_genome import TransformerDualGenome

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
#  Data Loading
# ===================================================================

def load_wikitext2(tokenizer, max_length=512, max_samples=500):
    """Load WikiText-2 test set as tokenized sequences."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50]

    encodings = tokenizer(
        texts[:max_samples],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return encodings


def load_domain_data(tokenizer, max_length=512, max_samples=1000):
    """
    Load domain-specific data for Task B fine-tuning.

    Tries multiple code datasets in order of preference, falling back
    to a synthetic code corpus if nothing else works.
    """
    from datasets import load_dataset

    texts = []

    # Strategy 1: try bigcode/the-stack-smol (Parquet, no scripts)
    try:
        print("    Trying bigcode/the-stack-smol...")
        dataset = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for i, item in enumerate(dataset):
            if len(texts) >= max_samples:
                break
            content = item.get("content", "")
            if 50 < len(content) < 2000:
                texts.append(content)
    except Exception as e:
        print(f"    stack-smol failed: {e}")

    # Strategy 2: try code_search_net
    if len(texts) < 50:
        try:
            print("    Trying code_search_net...")
            dataset = load_dataset(
                "code_search_net", "python", split="train",
                trust_remote_code=True,
            )
            for item in dataset:
                if len(texts) >= max_samples:
                    break
                code = item.get("whole_func_string", item.get("func_code_string", ""))
                if 50 < len(code) < 2000:
                    texts.append(code)
        except Exception as e:
            print(f"    code_search_net failed: {e}")

    # Strategy 3: generate synthetic Python-like code
    if len(texts) < 50:
        print("    Using synthetic code corpus (fallback)...")
        import random
        random.seed(42)
        templates = [
            "def {f}({a}):\n    result = {a} * 2\n    return result\n",
            "class {f}:\n    def __init__(self, {a}):\n        self.{a} = {a}\n    def get(self):\n        return self.{a}\n",
            "import os\nimport sys\n\ndef {f}(path):\n    if os.path.exists(path):\n        with open(path) as f:\n            data = f.read()\n        return data\n    return None\n",
            "for i in range(100):\n    if i % 3 == 0:\n        print('{f}', i)\n    elif i % 5 == 0:\n        print('{a}', i)\n",
            "try:\n    result = {f}()\n    print(result)\nexcept Exception as e:\n    print(f'Error: {{e}}')\n    raise\n",
            "import numpy as np\n\ndef {f}(arr):\n    mean = np.mean(arr)\n    std = np.std(arr)\n    return (arr - mean) / std\n",
            "from typing import List, Dict\n\ndef {f}(items: List[Dict]) -> List:\n    return [item['{a}'] for item in items if '{a}' in item]\n",
        ]
        funcs = ["process", "compute", "transform", "analyze", "validate",
                 "convert", "extract", "generate", "optimize", "evaluate",
                 "calculate", "normalize", "aggregate", "serialize", "parse"]
        args = ["data", "value", "items", "config", "params", "input",
                "output", "result", "count", "index", "name", "path"]
        for _ in range(max_samples):
            tmpl = random.choice(templates)
            code = tmpl.format(f=random.choice(funcs), a=random.choice(args))
            texts.append(code)

    print(f"    Loaded {len(texts)} code samples")

    encodings = tokenizer(
        texts[:max_samples],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return encodings


# ===================================================================
#  Evaluation
# ===================================================================

@torch.no_grad()
def evaluate_perplexity(model, encodings, device, batch_size=8):
    """
    Compute perplexity on tokenized data.
    Lower perplexity = better language modeling.

    Properly masks padding tokens by setting their labels to -100,
    so they don't contribute to the loss.
    """
    model.eval()
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    total_loss = 0.0
    total_tokens = 0
    n_batches = (len(input_ids) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(input_ids))
        batch_ids = input_ids[start:end]
        batch_mask = attention_mask[start:end]

        # Mask padding in labels so loss ignores them
        labels = batch_ids.clone()
        labels[batch_mask == 0] = -100

        outputs = model(batch_ids, attention_mask=batch_mask, labels=labels)

        # HF computes mean loss over non-ignored tokens internally
        # Count actual (non-padding) tokens for proper weighting
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = min(np.exp(avg_loss), 10000.0)  # cap at 10k
    return perplexity


# ===================================================================
#  Fine-tuning
# ===================================================================

def fine_tune(model, encodings, device, epochs=3, batch_size=4, lr=5e-5,
              max_steps=None):
    """Fine-tune GPT-2 on domain data."""
    from torch.optim import AdamW

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    n_batches = (len(input_ids) + batch_size - 1) // batch_size
    step = 0

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(input_ids))
        input_ids = input_ids[perm]
        attention_mask = attention_mask[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(input_ids))
            batch_ids = input_ids[start:end]
            batch_mask = attention_mask[start:end]

            labels = batch_ids.clone()
            labels[batch_mask == 0] = -100
            outputs = model(batch_ids, attention_mask=batch_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if max_steps and step >= max_steps:
                avg = epoch_loss / (i + 1)
                print(f"    Epoch {epoch+1} (stopped at step {step}): "
                      f"avg loss = {avg:.4f}")
                return

        avg = epoch_loss / n_batches
        print(f"    Epoch {epoch+1}/{epochs}: avg loss = {avg:.4f}")


# ===================================================================
#  Bayesian Gene Scoring
# ===================================================================

def bayesian_gene_scoring(genome, model, eval_a, eval_b, device, batch_size=8):
    """
    Score each gene using chimeric evaluation.

    For each gene:
      1. Repair just that gene (complement -> primary)
      2. Evaluate perplexity on Task A and Task B
      3. Restore original state
      4. Compute Bayesian posterior on mutation effect

    Lower perplexity = better, so:
      delta_a = ppl_mutated(A) - ppl_chimeric(A)
        positive = repairing this gene REDUCED A perplexity = gene was harmful for A
      delta_b = ppl_chimeric(B) - ppl_mutated(B)
        positive = repairing this gene INCREASED B perplexity = gene was helpful for B
    """
    genome.sync_primary()

    # Save state
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    # Baseline perplexities
    ppl_a_base = evaluate_perplexity(model, eval_a, device, batch_size)
    ppl_b_base = evaluate_perplexity(model, eval_b, device, batch_size)
    print(f"    Baseline: ppl_A={ppl_a_base:.2f}  ppl_B={ppl_b_base:.2f}")

    raw_scores = []
    n_genes = genome.total_genes

    for gene_id in range(n_genes):
        # Repair just this gene
        genome.repair_genes([gene_id])

        ppl_a_chi = evaluate_perplexity(model, eval_a, device, batch_size)
        ppl_b_chi = evaluate_perplexity(model, eval_b, device, batch_size)

        # Restore ALL genes
        for i, gene in enumerate(genome.genes):
            for pn in gene.param_names:
                gene.primary[pn] = saved[i][pn].clone()
        genome.apply_primary()

        raw_scores.append({
            "gene_id": gene_id,
            "gene_name": genome.genes[gene_id].name,
            "delta_a": ppl_a_base - ppl_a_chi,   # positive = repairing helps A
            "delta_b": ppl_b_chi - ppl_b_base,    # positive = repairing hurts B
            "ppl_a_chi": ppl_a_chi,
            "ppl_b_chi": ppl_b_chi,
        })

        if (gene_id + 1) % 10 == 0 or gene_id == n_genes - 1:
            print(f"    Scored {gene_id+1}/{n_genes} genes...", end="\r")

    print()

    # Empirical Bayes posteriors
    var_a = max(np.var([s["delta_a"] for s in raw_scores]), 1e-10)
    var_b = max(np.var([s["delta_b"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        sig_a = np.sqrt(var_a / 2)
        sig_b = np.sqrt(var_b / 2)

        scores.append({
            **s,
            "p_deleterious_a": 1 - stats.norm.cdf(0, loc=s["delta_a"]/2, scale=sig_a),
            "p_beneficial_b": 1 - stats.norm.cdf(0, loc=s["delta_b"]/2, scale=sig_b),
        })

    return scores, ppl_a_base, ppl_b_base


# ===================================================================
#  Gene Conversion
# ===================================================================

def gene_conversion(genome, scores, repair_threshold=0.50, alpha=0.3,
                    conversion_rate=0.6, seed=0):
    """
    Apply gene conversion with weighted trade-off scoring.
    Same logic as the prototype, scaled to transformer genes.
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
#  Main Experiment
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  MOLLY-EVOLVE: GPT-2 Scale Gene Conversion")
    print(f"  Device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print("=" * 70)

    # ---- Load model and tokenizer ----
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n[Setup] Loading GPT-2-small (124M params)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Load data ----
    print("\n[Setup] Loading datasets...")
    print("  Task A: WikiText-2 (general language modeling)")
    eval_a = load_wikitext2(tokenizer, max_length=256, max_samples=200)

    print("  Task B: Python code (domain specialization)")
    train_b = load_domain_data(tokenizer, max_length=256, max_samples=500)
    eval_b_data = load_domain_data(tokenizer, max_length=256, max_samples=100)
    print(f"  Task A eval: {len(eval_a['input_ids'])} sequences")
    print(f"  Task B train: {len(train_b['input_ids'])} sequences")
    print(f"  Task B eval: {len(eval_b_data['input_ids'])} sequences")

    # ---- Phase 1: Baseline evaluation ----
    print("\n[Phase 1] Baseline GPT-2 perplexity")
    ppl_a_orig = evaluate_perplexity(model, eval_a, device)
    ppl_b_orig = evaluate_perplexity(model, eval_b_data, device)
    print(f"  WikiText-2 (A): {ppl_a_orig:.2f}")
    print(f"  Code (B):       {ppl_b_orig:.2f}")

    # ---- Phase 2: Encode as dual genome ----
    print("\n[Phase 2] Encoding as transformer dual genome")
    genome = TransformerDualGenome(model, n_bits=16)
    genome.snapshot()

    print(f"  Total genes: {genome.total_genes}")
    genome.print_gene_map()

    # ---- Phase 3: Fine-tune on code (Task B) ----
    print("\n[Phase 3] Fine-tuning on Python code (aggressive LR for forgetting)")
    t0 = time.time()
    fine_tune(model, train_b, device, epochs=5, batch_size=4, lr=3e-4)
    ft_time = time.time() - t0
    print(f"  Fine-tuning took {ft_time:.1f}s")

    ppl_a_ft = evaluate_perplexity(model, eval_a, device)
    ppl_b_ft = evaluate_perplexity(model, eval_b_data, device)
    forgetting = ppl_a_ft - ppl_a_orig
    print(f"\n  WikiText-2 (A): {ppl_a_ft:.2f}  "
          f"(was {ppl_a_orig:.2f}, degradation: +{forgetting:.2f})")
    print(f"  Code (B):       {ppl_b_ft:.2f}  "
          f"(was {ppl_b_orig:.2f}, improvement: {ppl_b_orig - ppl_b_ft:.2f})")

    genome.sync_primary()

    # Mutation heatmap
    summary = genome.gene_summary()
    mutated = [(s["name"], s["divergence"]) for s in summary if s["divergence"] > 0.001]
    print(f"\n  Mutated genes: {len(mutated)}/{genome.total_genes}")
    top_mut = sorted(mutated, key=lambda x: x[1], reverse=True)[:10]
    print("  Most mutated:")
    for name, div in top_mut:
        bar = "#" * min(int(div * 100), 50)
        print(f"    {name:25s}  {div:.4f}  {bar}")

    # Save mutated state
    mut_state = {}
    for i, gene in enumerate(genome.genes):
        mut_state[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    # ---- Phase 4: Bayesian gene scoring ----
    print(f"\n[Phase 4] Bayesian gene scoring ({genome.total_genes} chimeric evaluations)")
    t0 = time.time()
    scores, _, _ = bayesian_gene_scoring(
        genome, model, eval_a, eval_b_data, device, batch_size=8
    )
    score_time = time.time() - t0
    print(f"  Scoring took {score_time:.1f}s")

    n_del = sum(1 for s in scores if s["p_deleterious_a"] > 0.65)
    n_ben = sum(1 for s in scores if s["p_beneficial_b"] > 0.65)
    print(f"  Deleterious for A (P > 0.65): {n_del} genes")
    print(f"  Beneficial for B (P > 0.65):  {n_ben} genes")

    top = sorted(scores, key=lambda s: s["p_deleterious_a"], reverse=True)[:5]
    print("\n  Top 5 genes most harmful to WikiText-2:")
    for s in top:
        print(f"    {s['gene_name']:25s}  P(del_A)={s['p_deleterious_a']:.3f}  "
              f"P(ben_B)={s['p_beneficial_b']:.3f}  "
              f"d_a={s['delta_a']:+.2f}")

    # ---- Phase 5: Compare strategies ----
    def hmean(a, b):
        """Harmonic mean of inverse perplexities (higher = better)."""
        inv_a = 1.0 / a if a > 0 else 0
        inv_b = 1.0 / b if b > 0 else 0
        return 2.0 / (a + b) * a * b if (a + b) > 0 else 0

    print("\n[Phase 5] Comparing strategies")
    print("-" * 75)
    print(f"  {'Strategy':35s} {'PPL(A)':>8s} {'PPL(B)':>8s}"
          f" {'Worst':>8s} {'GeoMean':>8s}")
    print("-" * 75)

    def geo_mean(a, b):
        """Geometric mean — standard metric for cross-domain perplexity."""
        return np.sqrt(a * b)

    def print_row(label, ppl_a, ppl_b):
        print(f"  {label:35s} {ppl_a:>8.2f} {ppl_b:>8.2f}"
              f" {max(ppl_a, ppl_b):>8.2f} {geo_mean(ppl_a, ppl_b):>8.2f}")

    # 1. Original pretrained
    print_row("Pretrained GPT-2", ppl_a_orig, ppl_b_orig)

    # 2. No repair (post fine-tuning)
    print_row("After code fine-tune (no repair)", ppl_a_ft, ppl_b_ft)

    # 3. Full repair
    genome.apply_complement()
    ppl_a_full = evaluate_perplexity(model, eval_a, device)
    ppl_b_full = evaluate_perplexity(model, eval_b_data, device)
    print_row("Full repair (all genes)", ppl_a_full, ppl_b_full)

    # 4. Restore mutated state, apply gene conversion
    for i, gene in enumerate(genome.genes):
        for pn in gene.param_names:
            gene.primary[pn] = mut_state[i][pn].clone()
    genome.apply_primary()

    result = gene_conversion(genome, scores,
                             repair_threshold=0.50,
                             alpha=0.3,
                             conversion_rate=1.0)  # no random skip for now

    ppl_a_gc = evaluate_perplexity(model, eval_a, device)
    ppl_b_gc = evaluate_perplexity(model, eval_b_data, device)
    print_row("Gene conversion (selective)", ppl_a_gc, ppl_b_gc)

    print("-" * 75)

    # ---- Phase 6: Analysis ----
    print(f"\n[Phase 6] Gene conversion details")
    print(f"  Genes repaired: {len(result['repaired'])}")
    print(f"  Genes fixed:    {len(result['fixed'])}")
    print(f"  Genes skipped:  {len(result['skipped'])}")

    if result["repaired"]:
        print("\n  Repaired genes:")
        for gid in result["repaired"]:
            g = genome.genes[gid]
            print(f"    {g.name:25s}  ({g.n_params:,d} params)")

    if result["fixed"]:
        print("\n  Fixed genes (accepted as beneficial):")
        for gid in result["fixed"]:
            g = genome.genes[gid]
            print(f"    {g.name:25s}  ({g.n_params:,d} params)")

    # Verdict — geometric mean is the standard cross-domain perplexity metric
    # (equivalent to average cross-entropy loss across domains)
    gm_none = geo_mean(ppl_a_ft, ppl_b_ft)
    gm_full = geo_mean(ppl_a_full, ppl_b_full)
    gm_gc = geo_mean(ppl_a_gc, ppl_b_gc)
    gm_orig = geo_mean(ppl_a_orig, ppl_b_orig)

    print("\n" + "=" * 75)
    print("  Geometric mean perplexity (lower = better, standard cross-domain metric):")
    print(f"    Pretrained:      {gm_orig:.2f}")
    print(f"    No repair:       {gm_none:.2f}")
    print(f"    Full repair:     {gm_full:.2f}")
    print(f"    Gene conversion: {gm_gc:.2f}")

    best_baseline = min(gm_none, gm_full)
    if gm_gc < best_baseline:
        pct = (best_baseline - gm_gc) / best_baseline * 100
        print(f"\n  Gene conversion OUTPERFORMS best baseline by {pct:.1f}%!")
        print(f"  Recovered {(1 - (ppl_a_gc - ppl_a_orig)/(ppl_a_ft - ppl_a_orig))*100:.0f}%"
              f" of WikiText capability")
        print(f"  Retained {(1 - (ppl_b_gc - ppl_b_ft)/(ppl_b_orig - ppl_b_ft))*100:.0f}%"
              f" of code improvement")
    else:
        print("\n  Gene conversion did not outperform on geometric mean metric.")
    print("=" * 75)


if __name__ == "__main__":
    main()
