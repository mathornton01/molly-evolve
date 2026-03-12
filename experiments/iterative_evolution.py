#!/usr/bin/env python3
"""
Iterative Multi-Domain Evolution via Gene Conversion

The core experiment: GPT-2 acquires new domains sequentially while gene
conversion protects previously learned capabilities. Over multiple cycles,
the model accumulates skills without catastrophic forgetting.

Evolution loop:
  Cycle 1: Pretrained -> fine-tune on Code    -> gene conversion -> model v1
  Cycle 2: model v1   -> fine-tune on Legal   -> gene conversion -> model v2
  Cycle 3: model v2   -> fine-tune on Medical -> gene conversion -> model v3

After each cycle, we evaluate on ALL previous domains to track retention.

Usage:
    cd molly-evolve
    python experiments/iterative_evolution.py
"""

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
#  Domain definitions
# ===================================================================

@dataclass
class Domain:
    name: str
    train_texts: List[str] = field(default_factory=list)
    eval_texts: List[str] = field(default_factory=list)
    train_enc: dict = field(default=None, repr=False)
    eval_enc: dict = field(default=None, repr=False)


def load_domains(tokenizer, max_length=256) -> List[Domain]:
    """
    Load multiple domains for iterative evolution.
    Uses datasets that are reliably available.
    """
    from datasets import load_dataset

    domains = []

    # Domain 1: Python code (code_search_net)
    print("  Loading Code domain...")
    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = []
        for item in ds:
            code = item.get("whole_func_string", "")
            if 80 < len(code) < 1500:
                texts.append(code)
            if len(texts) >= 400:
                break
        d = Domain(name="Code")
        d.train_texts = texts[:300]
        d.eval_texts = texts[300:400]
        domains.append(d)
    except Exception as e:
        print(f"    Failed: {e}")

    # Domain 2: Legal text (from pile-of-law or synthetic)
    print("  Loading Legal domain...")
    legal_templates = [
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
        "to enter into this Agreement; (b) this Agreement constitutes a valid "
        "and binding obligation.",
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
        "date first written above. By signing below, each party acknowledges "
        "that it has read, understands, and agrees to be bound by the terms.",
        "The Court finds that the plaintiff has established a prima facie case "
        "of negligence. The defendant owed a duty of care to the plaintiff, "
        "breached that duty, and the breach was the proximate cause of damages.",
        "Pursuant to 28 U.S.C. 1332, this Court has diversity jurisdiction over "
        "this matter. The amount in controversy exceeds $75,000, exclusive of "
        "interest and costs, and complete diversity exists between the parties.",
        "INDEMNIFICATION. The Indemnifying Party shall defend, indemnify, and "
        "hold harmless the Indemnified Party from and against any and all "
        "claims, damages, losses, costs, and expenses arising out of or "
        "relating to any breach of this Agreement.",
        "The arbitration shall be conducted in accordance with the Commercial "
        "Arbitration Rules of the American Arbitration Association. The "
        "arbitrator's decision shall be final and binding upon the parties.",
        "CONFIDENTIALITY. During the term of this Agreement and for a period of "
        "five (5) years thereafter, neither party shall disclose any Confidential "
        "Information of the other party to any third party without prior written "
        "consent, except as required by law or regulation.",
    ]
    import random
    rng = random.Random(42)
    legal_texts = []
    for _ in range(400):
        n_paragraphs = rng.randint(2, 5)
        text = " ".join(rng.choices(legal_templates, k=n_paragraphs))
        legal_texts.append(text)
    d = Domain(name="Legal")
    d.train_texts = legal_texts[:300]
    d.eval_texts = legal_texts[300:400]
    domains.append(d)

    # Domain 3: Medical/scientific text
    print("  Loading Medical domain...")
    medical_templates = [
        "The patient presented with acute onset of dyspnea and chest pain. "
        "Physical examination revealed bilateral crackles on auscultation. "
        "Chest X-ray showed bilateral infiltrates consistent with pneumonia.",
        "A randomized controlled trial was conducted to evaluate the efficacy "
        "of metformin versus placebo in patients with type 2 diabetes mellitus. "
        "Primary endpoint was change in HbA1c from baseline at 12 weeks.",
        "Histopathological examination of the biopsy specimen revealed "
        "moderately differentiated adenocarcinoma with lymphovascular invasion. "
        "Immunohistochemistry was positive for CK7 and negative for CK20.",
        "The mechanism of action involves inhibition of the enzyme "
        "cyclooxygenase-2 (COX-2), thereby reducing prostaglandin synthesis "
        "and attenuating the inflammatory response at the site of injury.",
        "Magnetic resonance imaging (MRI) of the brain with gadolinium contrast "
        "demonstrated a 2.3 cm enhancing lesion in the right temporal lobe "
        "with surrounding vasogenic edema and mass effect on adjacent structures.",
        "Pharmacokinetic analysis revealed a mean half-life of 6.2 hours with "
        "peak plasma concentration achieved at 2.1 hours post-administration. "
        "Bioavailability was approximately 85% following oral administration.",
        "The CRISPR-Cas9 system was employed to generate a knockout model of "
        "the BRCA1 gene in murine mammary epithelial cells. Transfection "
        "efficiency was confirmed by Sanger sequencing of the target locus.",
        "Complete blood count revealed leukocytosis with a white blood cell "
        "count of 18,500/uL, neutrophilic predominance, and a left shift. "
        "C-reactive protein was elevated at 142 mg/L.",
        "Meta-analysis of 12 randomized controlled trials (n=4,521) demonstrated "
        "a statistically significant reduction in all-cause mortality (RR 0.78, "
        "95% CI 0.65-0.93, p=0.006) with the intervention compared to control.",
        "The polymerase chain reaction (PCR) assay targeting the 16S rRNA gene "
        "confirmed the presence of Staphylococcus aureus. Antibiotic "
        "susceptibility testing revealed methicillin resistance (MRSA).",
    ]
    medical_texts = []
    for _ in range(400):
        n_paragraphs = rng.randint(2, 5)
        text = " ".join(rng.choices(medical_templates, k=n_paragraphs))
        medical_texts.append(text)
    d = Domain(name="Medical")
    d.train_texts = medical_texts[:300]
    d.eval_texts = medical_texts[300:400]
    domains.append(d)

    # Tokenize all domains
    for d in domains:
        d.train_enc = tokenizer(
            d.train_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        d.eval_enc = tokenizer(
            d.eval_texts, truncation=True, max_length=max_length,
            padding=True, return_tensors="pt",
        )
        print(f"    {d.name}: {len(d.train_texts)} train, "
              f"{len(d.eval_texts)} eval sequences")

    return domains


def load_general_eval(tokenizer, max_length=256, max_samples=200):
    """Load WikiText-2 as the general capability eval."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]
    return tokenizer(
        texts, truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    )


# ===================================================================
#  Evaluation (GPU-optimized)
# ===================================================================

@torch.no_grad()
def evaluate_perplexity(model, encodings, device, batch_size=16):
    """Compute perplexity with proper padding masking. Uses larger batches for speed."""
    model.eval()
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    total_loss = 0.0
    total_tokens = 0
    n_batches = (len(input_ids) + batch_size - 1) // batch_size

    for i in range(n_batches):
        s, e = i * batch_size, min((i + 1) * batch_size, len(input_ids))
        ids = input_ids[s:e]
        mask = attention_mask[s:e]
        labels = ids.clone()
        labels[mask == 0] = -100

        outputs = model(ids, attention_mask=mask, labels=labels)
        n_tok = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tok
        total_tokens += n_tok

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return min(np.exp(avg_loss), 10000.0)


def evaluate_all_domains(model, general_eval, domains, device, label=""):
    """Evaluate on general + all domain evals. Returns dict of perplexities."""
    results = {}
    results["General"] = evaluate_perplexity(model, general_eval, device)
    for d in domains:
        results[d.name] = evaluate_perplexity(model, d.eval_enc, device)

    if label:
        parts = [f"{k}: {v:.1f}" for k, v in results.items()]
        print(f"  {label:30s}  {' | '.join(parts)}")

    return results


# ===================================================================
#  Fine-tuning
# ===================================================================

def fine_tune(model, encodings, device, epochs=3, batch_size=4, lr=2e-4):
    """Fine-tune with padding-masked labels."""
    from torch.optim import AdamW

    model.train()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    n_batches = (len(input_ids) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(len(input_ids))
        input_ids = input_ids[perm]
        attention_mask = attention_mask[perm]
        epoch_loss = 0.0

        for i in range(n_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, len(input_ids))
            ids = input_ids[s:e]
            mask = attention_mask[s:e]
            labels = ids.clone()
            labels[mask == 0] = -100

            outputs = model(ids, attention_mask=mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            epoch_loss += outputs.loss.item()

        print(f"      Epoch {epoch+1}/{epochs}: loss = {epoch_loss/n_batches:.4f}")


# ===================================================================
#  Bayesian Gene Scoring (optimized)
# ===================================================================

def score_genes(genome, model, general_eval, domain_eval, device):
    """
    Score each gene via chimeric evaluation.
    Optimized: uses larger batch size and only evaluates on general + current domain.
    """
    genome.sync_primary()

    # Save state
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {pn: gene.primary[pn].clone() for pn in gene.param_names}

    ppl_a_base = evaluate_perplexity(model, general_eval, device)
    ppl_b_base = evaluate_perplexity(model, domain_eval, device)

    raw_scores = []
    n = genome.total_genes

    for gid in range(n):
        genome.repair_genes([gid])

        ppl_a = evaluate_perplexity(model, general_eval, device)
        ppl_b = evaluate_perplexity(model, domain_eval, device)

        # Restore
        for i, gene in enumerate(genome.genes):
            for pn in gene.param_names:
                gene.primary[pn] = saved[i][pn].clone()
        genome.apply_primary()

        raw_scores.append({
            "gene_id": gid,
            "gene_name": genome.genes[gid].name,
            "delta_a": ppl_a_base - ppl_a,
            "delta_b": ppl_b - ppl_b_base,
        })

        if (gid + 1) % 15 == 0 or gid == n - 1:
            print(f"      Scored {gid+1}/{n} genes", end="\r")

    print()

    # Bayesian posteriors
    var_a = max(np.var([s["delta_a"] for s in raw_scores]), 1e-10)
    var_b = max(np.var([s["delta_b"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        sig_a = np.sqrt(var_a / 2)
        sig_b = np.sqrt(var_b / 2)
        scores.append({
            **s,
            "p_del_a": 1 - stats.norm.cdf(0, loc=s["delta_a"]/2, scale=sig_a),
            "p_ben_b": 1 - stats.norm.cdf(0, loc=s["delta_b"]/2, scale=sig_b),
        })

    return scores, ppl_a_base, ppl_b_base


# ===================================================================
#  Gene Conversion
# ===================================================================

def gene_conversion(genome, scores, repair_threshold=0.50, alpha=0.3):
    """Apply gene conversion with weighted trade-off. No random skipping."""
    repaired, fixed, skipped = [], [], []

    for s in scores:
        gid = s["gene_id"]
        trade_off = s["p_del_a"] - alpha * s["p_ben_b"]

        if trade_off > repair_threshold:
            repaired.append(gid)
        elif s["p_del_a"] < 0.3:
            fixed.append(gid)
        else:
            skipped.append(gid)

    if repaired:
        genome.repair_genes(repaired)
    if fixed:
        genome.fix_genes(fixed)

    return {"repaired": repaired, "fixed": fixed, "skipped": skipped}


# ===================================================================
#  Main: Iterative Evolution Loop
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This experiment requires a GPU!"

    print("=" * 75)
    print("  MOLLY-EVOLVE: Iterative Multi-Domain Evolution")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 75)

    # ---- Setup ----
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n[Setup] Loading GPT-2-small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    print("[Setup] Loading domains...")
    domains = load_domains(tokenizer, max_length=256)
    general_eval = load_general_eval(tokenizer, max_length=256, max_samples=150)

    # ---- Baseline ----
    print("\n[Baseline] Pretrained GPT-2 perplexity across all domains:")
    baseline = evaluate_all_domains(model, general_eval, domains, device,
                                    "Pretrained")

    # ---- Encode as dual genome ----
    genome = TransformerDualGenome(model, n_bits=16)
    genome.snapshot()
    print(f"\n  Genome: {genome.total_genes} genes, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # ---- Evolution history ----
    history = [{"cycle": 0, "domain": "pretrained", "perplexities": baseline}]

    # Also track a "no conversion" baseline: train sequentially without repair
    model_no_repair = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # ---- Evolution loop ----
    total_start = time.time()

    for cycle_idx, domain in enumerate(domains):
        cycle_num = cycle_idx + 1
        print(f"\n{'='*75}")
        print(f"  EVOLUTION CYCLE {cycle_num}: Acquiring '{domain.name}' domain")
        print(f"{'='*75}")

        # Phase A: Fine-tune on new domain
        print(f"\n  [A] Fine-tuning on {domain.name}...")
        t0 = time.time()
        fine_tune(model, domain.train_enc, device, epochs=4, batch_size=4, lr=2e-4)
        print(f"      Fine-tuning took {time.time()-t0:.1f}s")

        # Also fine-tune the no-repair baseline
        fine_tune(model_no_repair, domain.train_enc, device,
                  epochs=4, batch_size=4, lr=2e-4)

        # Phase B: Evaluate post-training (before conversion)
        print(f"\n  [B] Post-training evaluation:")
        pre_gc = evaluate_all_domains(model, general_eval, domains, device,
                                      f"After {domain.name} (pre-GC)")

        # Phase C: Bayesian gene scoring
        print(f"\n  [C] Bayesian gene scoring...")
        t0 = time.time()
        genome.sync_primary()
        scores, _, _ = score_genes(
            genome, model, general_eval, domain.eval_enc, device
        )
        print(f"      Scoring took {time.time()-t0:.1f}s")

        n_del = sum(1 for s in scores if s["p_del_a"] > 0.65)
        print(f"      Deleterious for general: {n_del}/{genome.total_genes} genes")

        # Phase D: Gene conversion
        print(f"\n  [D] Gene conversion...")
        result = gene_conversion(genome, scores)
        print(f"      Repaired: {len(result['repaired'])} genes")
        print(f"      Fixed:    {len(result['fixed'])} genes")
        print(f"      Skipped:  {len(result['skipped'])} genes")

        if result["repaired"]:
            repaired_names = [genome.genes[g].name for g in result["repaired"]]
            # Show summary by layer type
            by_type = {}
            for name in repaired_names:
                # Extract component type (e.g., "mlp_down", "attn_qkv")
                parts = name.split("_", 1)
                comp = parts[1] if len(parts) > 1 else name
                by_type[comp] = by_type.get(comp, 0) + 1
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_type.items()))
            print(f"      Repair breakdown: {summary}")

        # Phase E: Post-conversion evaluation
        print(f"\n  [E] Post-conversion evaluation:")
        post_gc = evaluate_all_domains(model, general_eval, domains, device,
                                       f"After {domain.name} (post-GC)")

        # Update complement strand for next cycle
        genome.snapshot()

        history.append({
            "cycle": cycle_num,
            "domain": domain.name,
            "pre_gc": pre_gc,
            "post_gc": post_gc,
            "repaired": len(result["repaired"]),
            "fixed": len(result["fixed"]),
        })

    total_time = time.time() - total_start

    # ---- Final comparison ----
    print(f"\n{'='*75}")
    print("  FINAL RESULTS: Evolution History")
    print(f"{'='*75}")

    # Evaluate no-repair baseline on everything
    print("\n  No-repair baseline (sequential fine-tuning, no gene conversion):")
    no_repair_final = evaluate_all_domains(
        model_no_repair, general_eval, domains, device, "No repair (final)"
    )

    print("\n  Gene conversion model (after all evolution cycles):")
    gc_final = evaluate_all_domains(
        model, general_eval, domains, device, "Gene conversion (final)"
    )

    # Compute geometric means
    def geo_mean_all(ppls):
        vals = list(ppls.values())
        return np.exp(np.mean(np.log(vals)))

    gm_baseline = geo_mean_all(baseline)
    gm_no_repair = geo_mean_all(no_repair_final)
    gm_gc = geo_mean_all(gc_final)

    print(f"\n  Geometric mean perplexity across ALL domains (lower = better):")
    print(f"    Pretrained:      {gm_baseline:.2f}")
    print(f"    No repair:       {gm_no_repair:.2f}")
    print(f"    Gene conversion: {gm_gc:.2f}")

    if gm_gc < gm_no_repair:
        pct = (gm_no_repair - gm_gc) / gm_no_repair * 100
        print(f"\n  Gene conversion outperforms no-repair by {pct:.1f}%!")

    # Show evolution trajectory
    print(f"\n  Evolution trajectory (General domain perplexity):")
    print(f"    Pretrained:  {baseline['General']:.1f}")
    for h in history[1:]:
        pre = h["pre_gc"]["General"]
        post = h["post_gc"]["General"]
        recovery = ((pre - post) / (pre - baseline["General"])) * 100 if pre > baseline["General"] else 0
        print(f"    After {h['domain']:8s}:  {pre:.1f} -> {post:.1f} "
              f"(recovered {recovery:.0f}% of damage, "
              f"repaired {h['repaired']} genes)")

    print(f"\n  Total experiment time: {total_time:.0f}s "
          f"({total_time/60:.1f} min)")
    print("=" * 75)


if __name__ == "__main__":
    main()
