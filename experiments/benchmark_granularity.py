#!/usr/bin/env python3
"""
Fast A/B test: Component-level (75 genes) vs Head-level (207 genes)

Uses minimal data to run quickly. Tests whether finer gene granularity
improves gene conversion quality.

Target runtime: ~8-10 minutes on RTX 4090.
"""

import copy
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.transformer_genome import TransformerDualGenome

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
#  Minimal data loading (speed-optimized)
# ===================================================================

def load_fast_data(tokenizer, max_length=128):
    """3 domains with minimal samples for fast iteration."""
    from datasets import load_dataset
    import random
    rng = random.Random(42)

    domains = {}

    # Code
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [item.get("whole_func_string", "") for item in ds
                 if 80 < len(item.get("whole_func_string", "")) < 800][:120]
    except Exception:
        texts = ["def f(x):\n    return x * 2\n"] * 120
    domains["Code"] = (texts[:80], texts[80:120])

    # Legal
    legal = [
        "WHEREAS, the parties hereto desire to enter into this Agreement; "
        "NOW, THEREFORE, in consideration of the mutual covenants herein.",
        "The Defendant hereby moves this Court for summary judgment pursuant to "
        "Rule 56 of the Federal Rules of Civil Procedure.",
        "ARTICLE III. REPRESENTATIONS AND WARRANTIES. Each party represents and "
        "warrants that it has full power and authority to enter into this Agreement.",
        "The arbitration shall be conducted in accordance with the Commercial "
        "Arbitration Rules of the American Arbitration Association.",
        "CONFIDENTIALITY. Neither party shall disclose Confidential Information "
        "of the other party to any third party without prior written consent.",
    ]
    lt = [" ".join(rng.choices(legal, k=rng.randint(2, 4))) for _ in range(120)]
    domains["Legal"] = (lt[:80], lt[80:120])

    # Medical
    med = [
        "The patient presented with acute onset of dyspnea and chest pain. "
        "Physical examination revealed bilateral crackles on auscultation.",
        "A randomized controlled trial evaluated the efficacy of metformin "
        "versus placebo in patients with type 2 diabetes mellitus.",
        "Histopathological examination revealed moderately differentiated "
        "adenocarcinoma with lymphovascular invasion.",
        "The CRISPR-Cas9 system was employed to generate a knockout model of "
        "the BRCA1 gene in murine mammary epithelial cells.",
        "Meta-analysis of 12 RCTs demonstrated significant reduction in "
        "all-cause mortality (RR 0.78, 95% CI 0.65-0.93, p=0.006).",
    ]
    mt = [" ".join(rng.choices(med, k=rng.randint(2, 4))) for _ in range(120)]
    domains["Medical"] = (mt[:80], mt[80:120])

    # Tokenize
    encoded = {}
    for name, (train, evl) in domains.items():
        encoded[name] = {
            "train": tokenizer(train, truncation=True, max_length=max_length,
                               padding=True, return_tensors="pt"),
            "eval": tokenizer(evl, truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt"),
        }

    # General eval (small)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    gen_texts = [t for t in ds["text"] if len(t.strip()) > 50][:40]
    gen_eval = tokenizer(gen_texts, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")

    return encoded, gen_eval


# ===================================================================
#  Core functions
# ===================================================================

@torch.no_grad()
def perplexity(model, enc, device, batch_size=20):
    model.eval()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(ids), batch_size):
        b_ids = ids[i:i+batch_size]
        b_mask = mask[i:i+batch_size]
        labels = b_ids.clone()
        labels[b_mask == 0] = -100
        out = model(b_ids, attention_mask=b_mask, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tok += n
    return min(np.exp(total_loss / total_tok), 10000.0)


def fine_tune(model, enc, device, epochs=2, batch_size=8, lr=2e-4):
    from torch.optim import AdamW
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    for epoch in range(epochs):
        perm = torch.randperm(len(ids))
        ids, mask = ids[perm], mask[perm]
        nb = (len(ids) + batch_size - 1) // batch_size
        for i in range(nb):
            s, e = i * batch_size, min((i+1) * batch_size, len(ids))
            labels = ids[s:e].clone()
            labels[mask[s:e] == 0] = -100
            out = model(ids[s:e], attention_mask=mask[s:e], labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()


def score_genes_multi(genome, model, eval_sets, curr_eval, device):
    """Multi-objective gene scoring."""
    genome.sync_primary()
    saved = {}
    for i, gene in enumerate(genome.genes):
        if hasattr(gene, 'slice_defs'):
            saved[i] = {k: v.clone() for k, v in gene.primary.items()}
        else:
            saved[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    base_prev = {name: perplexity(model, enc, device)
                 for name, enc in eval_sets}
    base_curr = perplexity(model, curr_eval, device)

    scores = []
    n = genome.total_genes
    for gid in range(n):
        genome.repair_genes([gid])

        deltas_prev = {}
        for name, enc in eval_sets:
            ppl = perplexity(model, enc, device)
            deltas_prev[name] = base_prev[name] - ppl

        ppl_curr = perplexity(model, curr_eval, device)
        delta_curr = ppl_curr - base_curr

        # Restore
        gene = genome.genes[gid]
        if hasattr(gene, 'slice_defs'):
            for k, v in saved[gid].items():
                gene.primary[k] = v.clone()
        else:
            for pn in gene.param_names:
                gene.primary[pn] = saved[gid][pn].clone()

        # Restore ALL genes (in case repair touched shared params)
        for i, g in enumerate(genome.genes):
            if i == gid:
                continue
            if hasattr(g, 'slice_defs'):
                for k, v in saved[i].items():
                    g.primary[k] = v.clone()
            else:
                for pn in g.param_names:
                    g.primary[pn] = saved[i][pn].clone()
        genome.apply_primary()

        scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                        "delta_curr": delta_curr})

        if (gid + 1) % 25 == 0 or gid == n - 1:
            print(f"      {gid+1}/{n}", end=" ", flush=True)
    print()

    # Bayesian posteriors
    prev_names = [name for name, _ in eval_sets]
    var_prev = {name: max(np.var([s["deltas_prev"][name] for s in scores]), 1e-10)
                for name in prev_names}
    var_curr = max(np.var([s["delta_curr"] for s in scores]), 1e-10)

    for s in scores:
        p_del_per_obj = {}
        for name in prev_names:
            sig = np.sqrt(var_prev[name] / 2)
            p_del_per_obj[name] = 1 - stats.norm.cdf(
                0, loc=s["deltas_prev"][name] / 2, scale=sig)
        s["p_del_prev"] = max(p_del_per_obj.values())
        sig_curr = np.sqrt(var_curr / 2)
        s["p_ben_curr"] = 1 - stats.norm.cdf(
            0, loc=s["delta_curr"] / 2, scale=sig_curr)

    return scores


def gene_conversion(genome, scores, threshold=0.50, alpha=0.3):
    repaired, fixed = [], []
    for s in scores:
        trade = s["p_del_prev"] - alpha * s["p_ben_curr"]
        if trade > threshold:
            repaired.append(s["gene_id"])
        elif s["p_del_prev"] < 0.3:
            fixed.append(s["gene_id"])
    if repaired:
        genome.repair_genes(repaired)
    if fixed:
        genome.fix_genes(fixed)
    return len(repaired), len(fixed)


def eval_all(model, gen_eval, domain_data, device):
    r = {"General": perplexity(model, gen_eval, device)}
    for name, data in domain_data.items():
        r[name] = perplexity(model, data["eval"], device)
    return r


def geo_mean(ppls):
    return np.exp(np.mean(np.log(list(ppls.values()))))


# ===================================================================
#  Run one gene conversion trial
# ===================================================================

def run_gc_trial(base_model, granularity, domain_data, gen_eval, device):
    """Run gene conversion with given granularity. Returns per-step results."""
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16, granularity=granularity)
    genome.snapshot()

    label = f"{granularity} ({genome.total_genes} genes)"
    print(f"\n  [{label}]")

    domain_names = list(domain_data.keys())
    results = []

    for step_idx, dname in enumerate(domain_names):
        data = domain_data[dname]
        print(f"    {dname}: training...", end=" ", flush=True)
        t0 = time.time()
        fine_tune(model, data["train"], device)
        t_train = time.time() - t0

        # Build eval sets: general + all previous domains
        eval_sets = [("General", gen_eval)]
        for prev in domain_names[:step_idx]:
            eval_sets.append((prev, domain_data[prev]["eval"]))

        print(f"scoring {genome.total_genes} genes vs {len(eval_sets)} obj...")
        t0 = time.time()
        genome.sync_primary()
        scores = score_genes_multi(genome, model, eval_sets, data["eval"], device)
        t_score = time.time() - t0

        n_rep, n_fix = gene_conversion(genome, scores)
        print(f"      repaired {n_rep}, fixed {n_fix} "
              f"(train {t_train:.0f}s, score {t_score:.0f}s)")

        genome.snapshot()
        r = eval_all(model, gen_eval, domain_data, device)
        results.append(r)

    del model
    torch.cuda.empty_cache()
    return results


# ===================================================================
#  Main
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    print("=" * 70)
    print("  Gene Granularity A/B Test: Component (75) vs Head (207)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("  GPT-2 loaded")

    print("  Loading minimal data...")
    domain_data, gen_eval = load_fast_data(tokenizer)
    domain_names = list(domain_data.keys())

    # Baseline
    base_on_gpu = copy.deepcopy(base_model).to(device)
    baseline = eval_all(base_on_gpu, gen_eval, domain_data, device)
    del base_on_gpu
    torch.cuda.empty_cache()
    print(f"  Baseline GM: {geo_mean(baseline):.2f}")

    total_start = time.time()

    # Run both granularities
    comp_results = run_gc_trial(base_model, "component", domain_data, gen_eval, device)
    head_results = run_gc_trial(base_model, "head", domain_data, gen_eval, device)

    total_time = time.time() - total_start

    # Results
    print(f"\n{'='*70}")
    print("  RESULTS: Component (75 genes) vs Head (207 genes)")
    print(f"{'='*70}\n")

    header = f"  {'Step':>12s}"
    for dn in ["General"] + domain_names:
        header += f"  {dn[:6]:>6s}"
    header += f"  {'GM':>7s}"
    print(header)
    print(f"  {'-'*12}  {'------  ' * (1 + len(domain_names))}-------")

    # Baseline
    row = f"  {'Pretrained':>12s}"
    for dn in ["General"] + domain_names:
        row += f"  {baseline[dn]:>6.1f}"
    row += f"  {geo_mean(baseline):>7.2f}"
    print(row)
    print()

    for step_idx, dname in enumerate(domain_names):
        # Component
        r = comp_results[step_idx]
        row = f"  {'C+' + dname:>12s}"
        for dn in ["General"] + domain_names:
            row += f"  {r[dn]:>6.1f}"
        gm_c = geo_mean(r)
        row += f"  {gm_c:>7.2f}"
        print(row)

        # Head
        r = head_results[step_idx]
        row = f"  {'H+' + dname:>12s}"
        for dn in ["General"] + domain_names:
            row += f"  {r[dn]:>6.1f}"
        gm_h = geo_mean(r)
        row += f"  {gm_h:>7.2f}"
        diff = ((gm_c - gm_h) / gm_c) * 100
        marker = "<-- head wins" if gm_h < gm_c else ""
        row += f"  {marker}"
        print(row)
        print()

    # Summary
    final_c = geo_mean(comp_results[-1])
    final_h = geo_mean(head_results[-1])
    print(f"  Final GM:  component={final_c:.2f}  head={final_h:.2f}  "
          f"diff={((final_c - final_h)/final_c)*100:+.1f}%")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
