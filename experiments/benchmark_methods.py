#!/usr/bin/env python3
"""
Benchmark: Gene Conversion vs Existing Continual Learning Methods

Compares gene conversion against four established approaches to
catastrophic forgetting, all on the same 3-domain sequential task:

  1. No repair        — naive sequential fine-tuning (lower bound)
  2. L2 regularization — penalize weight divergence from previous checkpoint
  3. EWC              — Elastic Weight Consolidation (Fisher-weighted penalty)
  4. LoRA             — Low-Rank Adapters (freeze base, train adapters per domain)
  5. Weight averaging — fine-tune each domain separately, average all models
  6. Gene conversion  — our method (Bayesian selective repair)

All methods use the same domains, data, eval, and base model (GPT-2-small).

Usage:
    cd molly-evolve
    python experiments/benchmark_methods.py
"""

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.transformer_genome import TransformerDualGenome

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
#  Data — reuse from iterative_evolution
# ===================================================================

@dataclass
class Domain:
    name: str
    train_texts: List[str] = field(default_factory=list)
    eval_texts: List[str] = field(default_factory=list)
    train_enc: dict = field(default=None, repr=False)
    eval_enc: dict = field(default=None, repr=False)


def load_domains(tokenizer, max_length=256):
    from datasets import load_dataset
    import random
    rng = random.Random(42)
    domains = []

    # Code
    print("    Code...", end="", flush=True)
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [item.get("whole_func_string", "") for item in ds
                 if 80 < len(item.get("whole_func_string", "")) < 1500][:400]
    except Exception:
        texts = ["def f(x): return x * 2\n"] * 400
    d = Domain("Code", texts[:300], texts[300:400])
    domains.append(d)
    print(" done")

    # Legal
    print("    Legal...", end="", flush=True)
    legal = [
        "WHEREAS, the parties hereto desire to enter into this Agreement; "
        "NOW, THEREFORE, in consideration of the mutual covenants and agreements "
        "set forth herein, the parties agree as follows:",
        "Section 1. Definitions. For purposes of this Agreement, the following "
        "terms shall have the meanings set forth below:",
        "The Defendant hereby moves this Court for summary judgment pursuant to "
        "Rule 56 of the Federal Rules of Civil Procedure on the grounds that "
        "there is no genuine dispute as to any material fact.",
        "ARTICLE III. REPRESENTATIONS AND WARRANTIES. Each party represents and "
        "warrants to the other party that: (a) it has full power and authority "
        "to enter into this Agreement.",
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
        "date first written above.",
        "The Court finds that the plaintiff has established a prima facie case "
        "of negligence. The defendant owed a duty of care to the plaintiff.",
        "Pursuant to 28 U.S.C. 1332, this Court has diversity jurisdiction over "
        "this matter. The amount in controversy exceeds $75,000.",
        "INDEMNIFICATION. The Indemnifying Party shall defend, indemnify, and "
        "hold harmless the Indemnified Party from and against any claims.",
        "The arbitration shall be conducted in accordance with the Commercial "
        "Arbitration Rules of the American Arbitration Association.",
        "CONFIDENTIALITY. Neither party shall disclose any Confidential "
        "Information of the other party to any third party without prior consent.",
    ]
    lt = [" ".join(rng.choices(legal, k=rng.randint(2, 5))) for _ in range(400)]
    domains.append(Domain("Legal", lt[:300], lt[300:400]))
    print(" done")

    # Medical
    print("    Medical...", end="", flush=True)
    med = [
        "The patient presented with acute onset of dyspnea and chest pain. "
        "Physical examination revealed bilateral crackles on auscultation.",
        "A randomized controlled trial was conducted to evaluate the efficacy "
        "of metformin versus placebo in patients with type 2 diabetes mellitus.",
        "Histopathological examination of the biopsy specimen revealed "
        "moderately differentiated adenocarcinoma with lymphovascular invasion.",
        "The mechanism of action involves inhibition of the enzyme "
        "cyclooxygenase-2, reducing prostaglandin synthesis.",
        "MRI of the brain with gadolinium contrast demonstrated a 2.3 cm "
        "enhancing lesion in the right temporal lobe with surrounding edema.",
        "Pharmacokinetic analysis revealed a mean half-life of 6.2 hours with "
        "peak plasma concentration achieved at 2.1 hours post-administration.",
        "The CRISPR-Cas9 system was employed to generate a knockout model of "
        "the BRCA1 gene in murine mammary epithelial cells.",
        "Complete blood count revealed leukocytosis with WBC 18,500/uL. "
        "C-reactive protein was elevated at 142 mg/L.",
        "Meta-analysis of 12 RCTs demonstrated a significant reduction in "
        "all-cause mortality (RR 0.78, 95% CI 0.65-0.93, p=0.006).",
        "PCR assay targeting 16S rRNA confirmed Staphylococcus aureus. "
        "Susceptibility testing revealed methicillin resistance (MRSA).",
    ]
    mt = [" ".join(rng.choices(med, k=rng.randint(2, 5))) for _ in range(400)]
    domains.append(Domain("Medical", mt[:300], mt[300:400]))
    print(" done")

    for d in domains:
        d.train_enc = tokenizer(d.train_texts, truncation=True,
                                max_length=max_length, padding=True,
                                return_tensors="pt")
        d.eval_enc = tokenizer(d.eval_texts, truncation=True,
                               max_length=max_length, padding=True,
                               return_tensors="pt")

    return domains


def load_general_eval(tokenizer, max_length=256, max_samples=150):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]
    return tokenizer(texts, truncation=True, max_length=max_length,
                     padding=True, return_tensors="pt")


# ===================================================================
#  Evaluation
# ===================================================================

@torch.no_grad()
def perplexity(model, enc, device, batch_size=16):
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


def eval_all(model, gen_eval, domains, device):
    r = {"General": perplexity(model, gen_eval, device)}
    for d in domains:
        r[d.name] = perplexity(model, d.eval_enc, device)
    return r


def geo_mean(ppls):
    vals = list(ppls.values())
    return np.exp(np.mean(np.log(vals)))


# ===================================================================
#  Training helpers
# ===================================================================

def fine_tune(model, enc, device, epochs=4, batch_size=4, lr=2e-4,
              regularizer=None):
    """Fine-tune with optional regularization callback."""
    from torch.optim import AdamW
    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    for epoch in range(epochs):
        perm = torch.randperm(len(ids))
        ids, mask = ids[perm], mask[perm]
        eloss = 0.0
        nb = (len(ids) + batch_size - 1) // batch_size

        for i in range(nb):
            s, e = i * batch_size, min((i+1) * batch_size, len(ids))
            labels = ids[s:e].clone()
            labels[mask[s:e] == 0] = -100

            out = model(ids[s:e], attention_mask=mask[s:e], labels=labels)
            loss = out.loss

            if regularizer:
                loss = loss + regularizer(model)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            eloss += out.loss.item()  # log task loss only

        print(f"      Ep {epoch+1}/{epochs}: {eloss/nb:.4f}")


# ===================================================================
#  Method 1: No repair (naive sequential)
# ===================================================================

def run_no_repair(base_model, domains, gen_eval, device):
    print("\n  [1/6] No Repair (naive sequential fine-tuning)")
    model = copy.deepcopy(base_model).to(device)
    for d in domains:
        print(f"    Training on {d.name}...")
        fine_tune(model, d.train_enc, device)
    return eval_all(model, gen_eval, domains, device)


# ===================================================================
#  Method 2: L2 Regularization
# ===================================================================

def run_l2_reg(base_model, domains, gen_eval, device, lam=100.0):
    print(f"\n  [2/6] L2 Regularization (lambda={lam})")
    model = copy.deepcopy(base_model).to(device)

    for d in domains:
        # Save reference weights
        ref = {n: p.data.clone() for n, p in model.named_parameters()}

        def l2_reg(m, ref=ref, lam=lam):
            loss = 0.0
            for n, p in m.named_parameters():
                loss += ((p - ref[n]) ** 2).sum()
            return lam * loss

        print(f"    Training on {d.name}...")
        fine_tune(model, d.train_enc, device, regularizer=l2_reg)

    return eval_all(model, gen_eval, domains, device)


# ===================================================================
#  Method 3: EWC (Elastic Weight Consolidation)
# ===================================================================

def compute_fisher(model, enc, device, n_samples=200):
    """Compute diagonal Fisher information matrix approximation."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    ids = enc["input_ids"].to(device)[:n_samples]
    mask = enc["attention_mask"].to(device)[:n_samples]

    for i in range(len(ids)):
        model.zero_grad()
        labels = ids[i:i+1].clone()
        labels[mask[i:i+1] == 0] = -100
        out = model(ids[i:i+1], attention_mask=mask[i:i+1], labels=labels)
        out.loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2

    for n in fisher:
        fisher[n] /= len(ids)

    return fisher


def run_ewc(base_model, domains, gen_eval, device, lam=500.0):
    print(f"\n  [3/6] EWC (lambda={lam})")
    model = copy.deepcopy(base_model).to(device)

    # Accumulate Fisher + reference weights from all previous tasks
    all_fisher = []

    for d_idx, d in enumerate(domains):
        if all_fisher:
            def ewc_reg(m, all_f=all_fisher, lam=lam):
                loss = 0.0
                for (ref, fisher) in all_f:
                    for n, p in m.named_parameters():
                        loss += (fisher[n] * (p - ref[n]) ** 2).sum()
                return lam * loss
        else:
            ewc_reg = None

        print(f"    Training on {d.name}...")
        fine_tune(model, d.train_enc, device, regularizer=ewc_reg)

        # Compute Fisher for this task
        print(f"      Computing Fisher information...")
        fisher = compute_fisher(model, d.train_enc, device)
        ref = {n: p.data.clone() for n, p in model.named_parameters()}
        all_fisher.append((ref, fisher))

    return eval_all(model, gen_eval, domains, device)


# ===================================================================
#  Method 4: LoRA (per-domain adapters, merged)
# ===================================================================

def run_lora(base_model, domains, gen_eval, device, rank=8):
    print(f"\n  [4/6] LoRA (rank={rank}, per-domain adapters merged)")
    from peft import LoraConfig, get_peft_model, TaskType

    # Train a separate LoRA adapter on each domain from the same base
    adapters = []

    for d in domains:
        print(f"    Training LoRA adapter for {d.name}...")
        model = copy.deepcopy(base_model).to(device)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
        )
        peft_model = get_peft_model(model, config)

        # Train only LoRA params
        from torch.optim import AdamW
        peft_model.train()
        opt = AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=2e-4, weight_decay=0.01,
        )
        ids = d.train_enc["input_ids"].to(device)
        mask = d.train_enc["attention_mask"].to(device)

        for epoch in range(4):
            perm = torch.randperm(len(ids))
            ids, mask = ids[perm], mask[perm]
            eloss = 0.0
            nb = (len(ids) + 3) // 4
            for i in range(nb):
                s, e = i * 4, min((i+1) * 4, len(ids))
                labels = ids[s:e].clone()
                labels[mask[s:e] == 0] = -100
                out = peft_model(ids[s:e], attention_mask=mask[s:e], labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                eloss += out.loss.item()
            print(f"      Ep {epoch+1}/4: {eloss/nb:.4f}")

        # Merge adapter into base
        merged = peft_model.merge_and_unload()
        # Save the delta from base
        base_state = base_model.state_dict()
        delta = {}
        for n, p in merged.named_parameters():
            if n in base_state:
                delta[n] = (p.data.cpu() - base_state[n].cpu())
        adapters.append(delta)

        del peft_model, merged, model
        torch.cuda.empty_cache()

    # Average all deltas and apply to base
    print("    Merging all adapters (averaged delta)...")
    final = copy.deepcopy(base_model).to(device)
    for n, p in final.named_parameters():
        avg_delta = sum(a[n] for a in adapters if n in a) / len(adapters)
        p.data += avg_delta.to(device)

    return eval_all(final, gen_eval, domains, device)


# ===================================================================
#  Method 5: Weight Averaging (model soup)
# ===================================================================

def run_weight_avg(base_model, domains, gen_eval, device):
    print("\n  [5/6] Weight Averaging (model soup)")
    models = []

    for d in domains:
        print(f"    Fine-tuning separate model for {d.name}...")
        m = copy.deepcopy(base_model).to(device)
        fine_tune(m, d.train_enc, device)
        models.append({n: p.data.cpu().clone() for n, p in m.named_parameters()})
        del m
        torch.cuda.empty_cache()

    # Average all models
    print("    Averaging weights...")
    avg_model = copy.deepcopy(base_model).to(device)
    n_models = len(models)
    for n, p in avg_model.named_parameters():
        p.data = sum(m[n] for m in models).to(device) / n_models

    return eval_all(avg_model, gen_eval, domains, device)


# ===================================================================
#  Method 6: Gene Conversion
# ===================================================================

def score_genes(genome, model, gen_eval, domain_eval, device):
    genome.sync_primary()
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    ppl_a = perplexity(model, gen_eval, device)
    ppl_b = perplexity(model, domain_eval, device)

    raw = []
    for gid in range(genome.total_genes):
        genome.repair_genes([gid])
        pa = perplexity(model, gen_eval, device)
        pb = perplexity(model, domain_eval, device)
        for i, gene in enumerate(genome.genes):
            for pn in gene.param_names:
                gene.primary[pn] = saved[i][pn].clone()
        genome.apply_primary()
        raw.append({"gene_id": gid, "delta_a": ppl_a - pa, "delta_b": pb - ppl_b})
        if (gid + 1) % 15 == 0 or gid == genome.total_genes - 1:
            print(f"      {gid+1}/{genome.total_genes}", end="\r")
    print()

    va = max(np.var([s["delta_a"] for s in raw]), 1e-10)
    vb = max(np.var([s["delta_b"] for s in raw]), 1e-10)
    scores = []
    for s in raw:
        sa, sb = np.sqrt(va/2), np.sqrt(vb/2)
        scores.append({
            **s,
            "p_del_a": 1 - stats.norm.cdf(0, loc=s["delta_a"]/2, scale=sa),
            "p_ben_b": 1 - stats.norm.cdf(0, loc=s["delta_b"]/2, scale=sb),
        })
    return scores


def gene_conversion(genome, scores):
    repaired, fixed, skipped = [], [], []
    for s in scores:
        trade = s["p_del_a"] - 0.3 * s["p_ben_b"]
        if trade > 0.50:
            repaired.append(s["gene_id"])
        elif s["p_del_a"] < 0.3:
            fixed.append(s["gene_id"])
        else:
            skipped.append(s["gene_id"])
    if repaired:
        genome.repair_genes(repaired)
    if fixed:
        genome.fix_genes(fixed)
    return len(repaired), len(fixed)


def run_gene_conversion(base_model, domains, gen_eval, device):
    print("\n  [6/6] Gene Conversion (our method)")
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16)
    genome.snapshot()

    for d in domains:
        print(f"    Training on {d.name}...")
        fine_tune(model, d.train_enc, device)

        print(f"    Scoring genes...")
        genome.sync_primary()
        scores = score_genes(genome, model, gen_eval, d.eval_enc, device)

        n_rep, n_fix = gene_conversion(genome, scores)
        print(f"    Repaired: {n_rep}, Fixed: {n_fix}")

        genome.snapshot()

    return eval_all(model, gen_eval, domains, device)


# ===================================================================
#  Main
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU required!"

    print("=" * 75)
    print("  MOLLY-EVOLVE: Benchmark — Gene Conversion vs Alternatives")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("  Domains: Code -> Legal -> Medical (sequential)")
    print("=" * 75)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("\n[Setup]")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")  # keep on CPU
    print("  GPT-2-small loaded")

    print("  Loading domains...")
    domains = load_domains(tokenizer)
    gen_eval = load_general_eval(tokenizer)

    # Baseline
    print("\n[Baseline]")
    base_on_gpu = copy.deepcopy(base_model).to(device)
    baseline = eval_all(base_on_gpu, gen_eval, domains, device)
    del base_on_gpu
    torch.cuda.empty_cache()
    gm_base = geo_mean(baseline)
    parts = [f"{k}: {v:.1f}" for k, v in baseline.items()]
    print(f"  Pretrained: {' | '.join(parts)}  (GM: {gm_base:.2f})")

    # Run all methods
    results = {"Pretrained": baseline}
    timings = {}
    total_start = time.time()

    for name, fn in [
        ("No Repair", lambda: run_no_repair(base_model, domains, gen_eval, device)),
        ("L2 Reg", lambda: run_l2_reg(base_model, domains, gen_eval, device)),
        ("EWC", lambda: run_ewc(base_model, domains, gen_eval, device)),
        ("LoRA", lambda: run_lora(base_model, domains, gen_eval, device)),
        ("Wt Avg", lambda: run_weight_avg(base_model, domains, gen_eval, device)),
        ("Gene Conv", lambda: run_gene_conversion(base_model, domains, gen_eval, device)),
    ]:
        t0 = time.time()
        try:
            results[name] = fn()
        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = None
        timings[name] = time.time() - t0
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Final comparison
    print(f"\n{'='*75}")
    print("  FINAL RESULTS")
    print(f"{'='*75}")
    print()

    # Header
    domain_names = ["General"] + [d.name for d in domains]
    hdr = f"  {'Method':15s}"
    for dn in domain_names:
        hdr += f" {dn:>8s}"
    hdr += f" {'GeoMean':>8s} {'Time':>6s}"
    print(hdr)
    print("  " + "-" * (15 + 9 * len(domain_names) + 9 + 7))

    ranked = []
    for method, ppls in results.items():
        if ppls is None:
            print(f"  {method:15s}  FAILED")
            continue
        gm = geo_mean(ppls)
        ranked.append((method, ppls, gm))
        row = f"  {method:15s}"
        for dn in domain_names:
            row += f" {ppls[dn]:>8.1f}"
        t = timings.get(method, 0)
        row += f" {gm:>8.2f} {t:>5.0f}s"
        print(row)

    # Rank by geometric mean
    ranked.sort(key=lambda x: x[2])
    print()
    print("  Rankings (by geometric mean perplexity, lower = better):")
    for i, (method, _, gm) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"    {i+1}. {method:15s}  {gm:.2f}{marker}")

    winner = ranked[0][0]
    print(f"\n  Total benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 75)


if __name__ == "__main__":
    main()
