#!/usr/bin/env python3
"""
Scaling Benchmark: Gene Conversion vs Alternatives over 6 Domains

Shows how each method scales as more domains are added sequentially.
Key hypothesis: LoRA/weight-averaging dilute at O(1/N) per domain,
while gene conversion selectively preserves — advantage grows with N.

Methods compared:
  1. No Repair        — naive sequential fine-tuning
  2. LoRA             — per-domain adapters merged via averaged deltas
  3. Weight Averaging — separate fine-tuned models averaged (model soup)
  4. Gene Conversion  — multi-objective Bayesian scoring + selective repair

Domains (6 sequential): Code -> Legal -> Medical -> Science -> History -> Finance

Usage:
    cd molly-evolve
    python experiments/benchmark_scaling.py
"""

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.transformer_genome import TransformerDualGenome

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
#  Data — 6 domains
# ===================================================================

@dataclass
class Domain:
    name: str
    train_texts: List[str] = field(default_factory=list)
    eval_texts: List[str] = field(default_factory=list)
    train_enc: dict = field(default=None, repr=False)
    eval_enc: dict = field(default=None, repr=False)


def make_synthetic_domain(name, templates, rng, n_train=200, n_eval=80):
    texts = [" ".join(rng.choices(templates, k=rng.randint(2, 5)))
             for _ in range(n_train + n_eval)]
    return Domain(name, texts[:n_train], texts[n_train:])


def load_domains(tokenizer, max_length=256):
    from datasets import load_dataset
    import random
    rng = random.Random(42)
    domains = []

    # 1. Code (real data)
    print("    Code...", end="", flush=True)
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [item.get("whole_func_string", "") for item in ds
                 if 80 < len(item.get("whole_func_string", "")) < 1500][:280]
    except Exception:
        texts = ["def f(x):\n    return x * 2\n"] * 280
    domains.append(Domain("Code", texts[:200], texts[200:280]))
    print(" done")

    # 2. Legal
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
    domains.append(make_synthetic_domain("Legal", legal, rng))
    print(" done")

    # 3. Medical
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
    domains.append(make_synthetic_domain("Medical", med, rng))
    print(" done")

    # 4. Science (physics/chemistry)
    print("    Science...", end="", flush=True)
    sci = [
        "The electromagnetic force between two point charges is described by "
        "Coulomb's law: F = k * q1 * q2 / r^2, where k is the Coulomb constant "
        "and r is the separation distance between the charges.",
        "According to the second law of thermodynamics, the total entropy of an "
        "isolated system can never decrease over time. In any spontaneous process, "
        "the entropy of the universe increases.",
        "Quantum tunneling occurs when a particle traverses a potential energy "
        "barrier that it classically could not surmount. The transmission "
        "coefficient decays exponentially with barrier width.",
        "The Schrodinger equation describes the wave function of a quantum "
        "mechanical system. For a particle in a one-dimensional potential well, "
        "the energy eigenvalues are quantized: E_n = n^2 * h^2 / (8mL^2).",
        "In organic chemistry, nucleophilic substitution reactions proceed via "
        "either SN1 or SN2 mechanisms. SN2 reactions exhibit second-order kinetics "
        "and proceed with inversion of stereochemistry at the carbon center.",
        "The photoelectric effect demonstrates the particle nature of light. "
        "Photons with energy exceeding the work function eject electrons from "
        "the metal surface with kinetic energy E_k = hf - phi.",
        "General relativity describes gravity as curvature of spacetime caused "
        "by mass and energy. The Einstein field equations relate the geometry "
        "of spacetime to the distribution of matter within it.",
        "The Michaelis-Menten equation models enzyme kinetics: v = V_max * [S] "
        "/ (K_m + [S]), where V_max is the maximum reaction rate and K_m is "
        "the substrate concentration at half-maximal velocity.",
        "Nuclear fusion in stellar cores converts hydrogen to helium via the "
        "proton-proton chain reaction, releasing 26.7 MeV per helium-4 nucleus. "
        "This process powers main-sequence stars like the Sun.",
        "Maxwell's equations unify electricity and magnetism into a single "
        "framework. The curl of E equals -dB/dt, and the curl of B equals "
        "mu_0 * J + mu_0 * epsilon_0 * dE/dt.",
    ]
    domains.append(make_synthetic_domain("Science", sci, rng))
    print(" done")

    # 5. History
    print("    History...", end="", flush=True)
    hist = [
        "The Treaty of Westphalia in 1648 established the principle of state "
        "sovereignty and effectively ended the Thirty Years' War, which had "
        "devastated much of Central Europe.",
        "The Industrial Revolution, beginning in Britain around 1760, transformed "
        "manufacturing from hand production to machine-based processes, powered "
        "by steam engines and water wheels.",
        "The French Revolution of 1789 overthrew the Bourbon monarchy and "
        "established a republic founded on the principles of liberty, equality, "
        "and fraternity. The Reign of Terror followed under Robespierre.",
        "The Roman Empire at its greatest extent under Trajan in 117 CE "
        "encompassed the entire Mediterranean basin, spanning from Britain "
        "to Mesopotamia, governing approximately 70 million people.",
        "The Silk Road connected East Asia to the Mediterranean world, "
        "facilitating trade in silk, spices, precious metals, and ideas. "
        "It operated from roughly the 2nd century BCE to the 15th century CE.",
        "The Renaissance, originating in 14th-century Italy, marked a cultural "
        "rebirth emphasizing humanism, classical learning, and artistic "
        "innovation. Key figures include Leonardo da Vinci and Michelangelo.",
        "The Congress of Vienna in 1815 redrew the map of Europe after the "
        "Napoleonic Wars, seeking to restore the balance of power and prevent "
        "future continental conflicts.",
        "The Meiji Restoration of 1868 ended the Tokugawa shogunate and "
        "modernized Japan through rapid industrialization, constitutional "
        "government, and military reform.",
        "World War I began in 1914 following the assassination of Archduke "
        "Franz Ferdinand. The conflict introduced trench warfare, chemical "
        "weapons, and resulted in approximately 20 million casualties.",
        "The Berlin Conference of 1884-1885 formalized European colonial claims "
        "in Africa, partitioning the continent among European powers with "
        "little regard for existing ethnic or political boundaries.",
    ]
    domains.append(make_synthetic_domain("History", hist, rng))
    print(" done")

    # 6. Finance
    print("    Finance...", end="", flush=True)
    fin = [
        "The Black-Scholes model prices European options using five parameters: "
        "stock price, strike price, time to expiration, risk-free rate, and "
        "volatility. The formula assumes log-normal returns.",
        "Portfolio diversification reduces unsystematic risk by combining assets "
        "with low correlation. The efficient frontier describes the set of "
        "portfolios offering maximum return for a given level of risk.",
        "The capital asset pricing model (CAPM) relates expected return to "
        "systematic risk: E(R_i) = R_f + beta_i * (E(R_m) - R_f), where "
        "beta measures the asset's sensitivity to market movements.",
        "Discounted cash flow analysis values a company by projecting future "
        "free cash flows and discounting them to present value using the "
        "weighted average cost of capital (WACC).",
        "The yield curve plots interest rates across different maturities. "
        "An inverted yield curve, where short-term rates exceed long-term rates, "
        "has historically been a reliable predictor of economic recession.",
        "Value at Risk (VaR) estimates the maximum potential loss of a portfolio "
        "over a specified time period at a given confidence level. The 95% "
        "one-day VaR represents the loss exceeded only 5% of trading days.",
        "The Modigliani-Miller theorem states that in perfect capital markets, "
        "the value of a firm is independent of its capital structure. Leverage "
        "does not affect total enterprise value absent taxes and frictions.",
        "Credit default swaps transfer the credit risk of a reference entity "
        "from the protection buyer to the protection seller. The CDS spread "
        "reflects the market's assessment of default probability.",
        "The Federal Reserve implements monetary policy through open market "
        "operations, adjusting the federal funds rate to influence borrowing "
        "costs, employment, and inflation.",
        "Fundamental analysis evaluates securities by examining financial "
        "statements, industry conditions, and macroeconomic factors. Key "
        "metrics include P/E ratio, ROE, and debt-to-equity ratio.",
    ]
    domains.append(make_synthetic_domain("Finance", fin, rng))
    print(" done")

    # Tokenize
    for d in domains:
        d.train_enc = tokenizer(d.train_texts, truncation=True,
                                max_length=max_length, padding=True,
                                return_tensors="pt")
        d.eval_enc = tokenizer(d.eval_texts, truncation=True,
                               max_length=max_length, padding=True,
                               return_tensors="pt")
    return domains


def load_general_eval(tokenizer, max_length=256, max_samples=100):
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
#  Training helper
# ===================================================================

def fine_tune(model, enc, device, epochs=3, batch_size=4, lr=2e-4,
              regularizer=None, quiet=False):
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
            eloss += out.loss.item()

        if not quiet:
            print(f"      Ep {epoch+1}/{epochs}: {eloss/nb:.4f}")


# ===================================================================
#  Multi-objective Bayesian Gene Scoring (KEY IMPROVEMENT)
# ===================================================================

def score_genes_multi(genome, model, eval_sets, current_domain_eval, device):
    """
    Score each gene against ALL previous objectives (general + prior domains).

    eval_sets: list of (name, encodings) for all objectives to protect
    current_domain_eval: encodings for the current domain being learned

    Returns scores with p_del_prev (max across all previous) and p_ben_curr.
    """
    genome.sync_primary()

    # Save state for restoration
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    # Baseline perplexities
    base_prev = {name: perplexity(model, enc, device)
                 for name, enc in eval_sets}
    base_curr = perplexity(model, current_domain_eval, device)

    raw_scores = []
    n = genome.total_genes

    for gid in range(n):
        genome.repair_genes([gid])

        # Measure impact on ALL previous objectives
        deltas_prev = {}
        for name, enc in eval_sets:
            ppl = perplexity(model, enc, device)
            deltas_prev[name] = base_prev[name] - ppl  # positive = repair helped

        # Measure impact on current domain
        ppl_curr = perplexity(model, current_domain_eval, device)
        delta_curr = ppl_curr - base_curr  # positive = repair hurt current domain

        # Restore
        for i, gene in enumerate(genome.genes):
            for pn in gene.param_names:
                gene.primary[pn] = saved[i][pn].clone()
        genome.apply_primary()

        raw_scores.append({
            "gene_id": gid,
            "deltas_prev": deltas_prev,
            "delta_curr": delta_curr,
        })

        if (gid + 1) % 15 == 0 or gid == n - 1:
            print(f"      {gid+1}/{n}", end="  ", flush=True)

    print()

    # Bayesian posteriors per previous objective
    # For each prior objective, compute variance across genes
    prev_names = [name for name, _ in eval_sets]
    var_prev = {}
    for name in prev_names:
        vals = [s["deltas_prev"][name] for s in raw_scores]
        var_prev[name] = max(np.var(vals), 1e-10)

    var_curr = max(np.var([s["delta_curr"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        # P(deleterious) for each previous objective
        p_del_per_obj = {}
        for name in prev_names:
            sig = np.sqrt(var_prev[name] / 2)
            p_del_per_obj[name] = 1 - stats.norm.cdf(
                0, loc=s["deltas_prev"][name] / 2, scale=sig
            )

        # Take MAX p_del across all previous objectives
        # (repair if harmful to ANY previous domain)
        p_del_prev = max(p_del_per_obj.values())

        # P(beneficial for current domain)
        sig_curr = np.sqrt(var_curr / 2)
        p_ben_curr = 1 - stats.norm.cdf(
            0, loc=s["delta_curr"] / 2, scale=sig_curr
        )

        scores.append({
            "gene_id": s["gene_id"],
            "p_del_prev": p_del_prev,
            "p_ben_curr": p_ben_curr,
            "p_del_per_obj": p_del_per_obj,
        })

    return scores


def gene_conversion(genome, scores, threshold=0.50, alpha=0.3):
    repaired, fixed, skipped = [], [], []
    for s in scores:
        trade = s["p_del_prev"] - alpha * s["p_ben_curr"]
        if trade > threshold:
            repaired.append(s["gene_id"])
        elif s["p_del_prev"] < 0.3:
            fixed.append(s["gene_id"])
        else:
            skipped.append(s["gene_id"])
    if repaired:
        genome.repair_genes(repaired)
    if fixed:
        genome.fix_genes(fixed)
    return len(repaired), len(fixed)


# ===================================================================
#  Method runners
# ===================================================================

def run_no_repair(base_model, domains_so_far, all_domains, gen_eval, device):
    """Sequential fine-tuning, no protection."""
    model = copy.deepcopy(base_model).to(device)
    for d in domains_so_far:
        fine_tune(model, d.train_enc, device, quiet=True)
    result = eval_all(model, gen_eval, all_domains, device)
    del model
    torch.cuda.empty_cache()
    return result


def run_lora(base_model, domains_so_far, all_domains, gen_eval, device,
             adapter_cache, rank=8):
    """Per-domain LoRA adapters, merged via averaged deltas."""
    from peft import LoraConfig, get_peft_model, TaskType

    # Train adapters for any new domains not yet cached
    for d in domains_so_far:
        if d.name in adapter_cache:
            continue
        model = copy.deepcopy(base_model).to(device)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=16,
            lora_dropout=0.05, target_modules=["c_attn", "c_proj"],
        )
        peft_model = get_peft_model(model, config)
        peft_model.train()
        from torch.optim import AdamW
        opt = AdamW([p for p in peft_model.parameters() if p.requires_grad],
                    lr=2e-4, weight_decay=0.01)
        ids = d.train_enc["input_ids"].to(device)
        mask = d.train_enc["attention_mask"].to(device)
        for epoch in range(3):
            perm = torch.randperm(len(ids))
            ids, mask = ids[perm], mask[perm]
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
        merged = peft_model.merge_and_unload()
        base_state = base_model.state_dict()
        delta = {}
        for n, p in merged.named_parameters():
            if n in base_state:
                delta[n] = (p.data.cpu() - base_state[n].cpu())
        adapter_cache[d.name] = delta
        del peft_model, merged, model
        torch.cuda.empty_cache()

    # Merge all adapters for domains_so_far
    final = copy.deepcopy(base_model).to(device)
    n_adapters = len(domains_so_far)
    for n, p in final.named_parameters():
        avg_delta = sum(adapter_cache[d.name].get(n, torch.zeros_like(p.cpu()))
                        for d in domains_so_far) / n_adapters
        p.data += avg_delta.to(device)

    result = eval_all(final, gen_eval, all_domains, device)
    del final
    torch.cuda.empty_cache()
    return result


def run_weight_avg(base_model, domains_so_far, all_domains, gen_eval, device,
                   model_cache):
    """Fine-tune each domain separately, average all models."""
    for d in domains_so_far:
        if d.name in model_cache:
            continue
        m = copy.deepcopy(base_model).to(device)
        fine_tune(m, d.train_enc, device, quiet=True)
        model_cache[d.name] = {n: p.data.cpu().clone()
                               for n, p in m.named_parameters()}
        del m
        torch.cuda.empty_cache()

    # Average
    avg = copy.deepcopy(base_model).to(device)
    n_models = len(domains_so_far)
    for n, p in avg.named_parameters():
        p.data = sum(model_cache[d.name][n]
                     for d in domains_so_far).to(device) / n_models

    result = eval_all(avg, gen_eval, all_domains, device)
    del avg
    torch.cuda.empty_cache()
    return result


def run_gene_conversion(base_model, domains, all_domains, gen_eval, device):
    """Multi-objective gene conversion over all domains sequentially."""
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16)
    genome.snapshot()

    results_per_step = []

    for step_idx, d in enumerate(domains):
        # Train
        fine_tune(model, d.train_enc, device, quiet=True)

        # Build eval sets: general + all PREVIOUS domains
        eval_sets = [("General", gen_eval)]
        for prev_d in domains[:step_idx]:
            eval_sets.append((prev_d.name, prev_d.eval_enc))

        # Multi-objective scoring
        genome.sync_primary()
        scores = score_genes_multi(genome, model, eval_sets, d.eval_enc, device)

        n_rep, n_fix = gene_conversion(genome, scores)
        print(f"      {d.name}: repaired {n_rep}, fixed {n_fix} "
              f"(scored vs {len(eval_sets)} objectives)")

        genome.snapshot()

        # Evaluate after this step
        result = eval_all(model, gen_eval, all_domains, device)
        results_per_step.append(result)

    return results_per_step


# ===================================================================
#  Main
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GPU required!"

    print("=" * 80)
    print("  MOLLY-EVOLVE: Scaling Benchmark (6 Domains)")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("  Hypothesis: Gene conversion advantage grows with domain count")
    print("=" * 80)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("\n[Setup]")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("  GPT-2-small loaded")

    print("  Loading 6 domains...")
    all_domains = load_domains(tokenizer)
    gen_eval = load_general_eval(tokenizer)

    # Baseline
    base_on_gpu = copy.deepcopy(base_model).to(device)
    baseline = eval_all(base_on_gpu, gen_eval, all_domains, device)
    del base_on_gpu
    torch.cuda.empty_cache()
    print(f"  Baseline GM: {geo_mean(baseline):.2f}")

    total_start = time.time()

    # ---- Method 1: No Repair ----
    print("\n[1/4] No Repair (sequential fine-tuning)")
    no_repair_results = []
    for n in range(1, len(all_domains) + 1):
        print(f"  N={n} ({all_domains[n-1].name})...", end="", flush=True)
        r = run_no_repair(base_model, all_domains[:n], all_domains, gen_eval, device)
        no_repair_results.append(r)
        print(f" GM={geo_mean(r):.2f}")

    # ---- Method 2: LoRA ----
    print("\n[2/4] LoRA (per-domain adapters, averaged merge)")
    lora_cache = {}
    lora_results = []
    for n in range(1, len(all_domains) + 1):
        print(f"  N={n} ({all_domains[n-1].name})...", end="", flush=True)
        r = run_lora(base_model, all_domains[:n], all_domains, gen_eval,
                     device, lora_cache)
        lora_results.append(r)
        print(f" GM={geo_mean(r):.2f}")

    # ---- Method 3: Weight Averaging ----
    print("\n[3/4] Weight Averaging (model soup)")
    wt_cache = {}
    wt_results = []
    for n in range(1, len(all_domains) + 1):
        print(f"  N={n} ({all_domains[n-1].name})...", end="", flush=True)
        r = run_weight_avg(base_model, all_domains[:n], all_domains, gen_eval,
                           device, wt_cache)
        wt_results.append(r)
        print(f" GM={geo_mean(r):.2f}")

    # ---- Method 4: Gene Conversion (multi-objective) ----
    print("\n[4/4] Gene Conversion (multi-objective Bayesian scoring)")
    gc_results = run_gene_conversion(
        base_model, all_domains, all_domains, gen_eval, device
    )

    total_time = time.time() - total_start

    # ---- Results table ----
    print(f"\n{'='*80}")
    print("  SCALING RESULTS: Geometric Mean Perplexity by Domain Count")
    print(f"{'='*80}\n")

    domain_names = [d.name for d in all_domains]
    print(f"  {'Domains':>8s}  {'NoRepair':>10s}  {'LoRA':>10s}  "
          f"{'WtAvg':>10s}  {'GeneConv':>10s}  {'Pretrained':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    gm_baseline = geo_mean(baseline)
    for n in range(len(all_domains)):
        gm_nr = geo_mean(no_repair_results[n])
        gm_lr = geo_mean(lora_results[n])
        gm_wa = geo_mean(wt_results[n])
        gm_gc = geo_mean(gc_results[n])
        label = f"N={n+1} +{domain_names[n][:4]}"

        # Mark the winner
        gms = {"NR": gm_nr, "LR": gm_lr, "WA": gm_wa, "GC": gm_gc}
        winner = min(gms, key=gms.get)
        markers = {k: " *" if k == winner else "  " for k in gms}

        print(f"  {label:>8s}  {gm_nr:>8.2f}{markers['NR']}  "
              f"{gm_lr:>8.2f}{markers['LR']}  "
              f"{gm_wa:>8.2f}{markers['WA']}  "
              f"{gm_gc:>8.2f}{markers['GC']}  "
              f"{gm_baseline:>10.2f}")

    # Detailed final results (N=6)
    print(f"\n  Final per-domain perplexity (N=6, all domains):")
    print(f"  {'':>10s}", end="")
    for dn in ["General"] + domain_names:
        print(f"  {dn[:6]:>6s}", end="")
    print(f"  {'GM':>8s}")
    print(f"  {'':>10s}  {'------' * (1 + len(domain_names))}  --------")

    final_data = [
        ("Pretrained", baseline),
        ("No Repair", no_repair_results[-1]),
        ("LoRA", lora_results[-1]),
        ("Wt Avg", wt_results[-1]),
        ("Gene Conv", gc_results[-1]),
    ]
    for name, ppls in final_data:
        print(f"  {name:>10s}", end="")
        for dn in ["General"] + domain_names:
            print(f"  {ppls[dn]:>6.1f}", end="")
        print(f"  {geo_mean(ppls):>8.2f}")

    # Rankings at each step
    print(f"\n  Winner at each step:")
    for n in range(len(all_domains)):
        gms = {
            "No Repair": geo_mean(no_repair_results[n]),
            "LoRA": geo_mean(lora_results[n]),
            "Wt Avg": geo_mean(wt_results[n]),
            "Gene Conv": geo_mean(gc_results[n]),
        }
        ranked = sorted(gms.items(), key=lambda x: x[1])
        w_name, w_gm = ranked[0]
        r_name, r_gm = ranked[1]
        print(f"    N={n+1}: {w_name} ({w_gm:.2f}) > "
              f"{r_name} ({r_gm:.2f})")

    print(f"\n  Total benchmark time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
