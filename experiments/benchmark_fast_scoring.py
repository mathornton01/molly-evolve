#!/usr/bin/env python3
"""
Fast gene scoring via gradient approximation.

Instead of O(N_genes * N_objectives) forward passes for chimeric evaluation,
use first-order Taylor expansion: delta_loss_g ~ grad_L . (w_complement - w_primary)

This requires only O(N_objectives) forward+backward passes total.

Compares:
  1. Gene Conversion (chimeric scoring) — accurate but slow
  2. Gene Conversion (gradient scoring) — fast approximation
  3. LoRA — speed reference
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
#  Data (same minimal set)
# ===================================================================

def load_fast_data(tokenizer, max_length=128):
    from datasets import load_dataset
    import random
    rng = random.Random(42)
    domains = {}

    try:
        ds = load_dataset("code_search_net", "python", split="train")
        texts = [item.get("whole_func_string", "") for item in ds
                 if 80 < len(item.get("whole_func_string", "")) < 800][:120]
    except Exception:
        texts = ["def f(x):\n    return x * 2\n"] * 120
    domains["Code"] = (texts[:80], texts[80:120])

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

    encoded = {}
    for name, (train, evl) in domains.items():
        encoded[name] = {
            "train": tokenizer(train, truncation=True, max_length=max_length,
                               padding=True, return_tensors="pt"),
            "eval": tokenizer(evl, truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt"),
        }

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    gen_texts = [t for t in ds["text"] if len(t.strip()) > 50][:40]
    gen_eval = tokenizer(gen_texts, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")
    return encoded, gen_eval


# ===================================================================
#  Core utilities
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


def eval_all(model, gen_eval, domain_data, device):
    r = {"General": perplexity(model, gen_eval, device)}
    for name, data in domain_data.items():
        r[name] = perplexity(model, data["eval"], device)
    return r


def geo_mean(ppls):
    return np.exp(np.mean(np.log(list(ppls.values()))))


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


# ===================================================================
#  GRADIENT-BASED GENE SCORING (the key innovation for speed)
# ===================================================================

def score_genes_gradient(genome, model, eval_sets, curr_eval, device):
    """
    O(N_objectives) scoring via gradient approximation.

    For each eval set, one forward+backward pass gives us dL/dw.
    Per-gene impact of repair: delta_loss_g ~ sum(dL/dw_g * (w_complement - w_primary))

    ~100x faster than chimeric evaluation for 207 genes.
    """
    genome.sync_primary()

    # Precompute repair direction (complement - primary) for each gene
    gene_repairs = []
    for gene in genome.genes:
        repairs = {}
        if hasattr(gene, 'slice_defs'):
            for pname, dim, start, end in gene.slice_defs:
                key = gene._key(pname, dim, start, end)
                comp = gene._dequantize(gene.complement[key], gene.scales[key]).cpu()
                prim = gene._dequantize(gene.primary[key], gene.scales[key]).cpu()
                repairs[(pname, dim, start, end)] = (comp - prim).float()
        else:
            for pn in gene.param_names:
                comp = gene._dequantize(gene.complement[pn], gene.scales[pn]).cpu()
                prim = gene._dequantize(gene.primary[pn], gene.scales[pn]).cpu()
                repairs[(pn, None, None, None)] = (comp - prim).float()
        gene_repairs.append(repairs)

    def grad_scores_for_eval(enc):
        """One forward+backward -> per-gene approximate loss change."""
        model.zero_grad()
        model.train()
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        # Accumulate gradients across batches
        bs = 20
        for i in range(0, len(ids), bs):
            b_ids = ids[i:i+bs]
            b_mask = mask[i:i+bs]
            labels = b_ids.clone()
            labels[b_mask == 0] = -100
            out = model(b_ids, attention_mask=b_mask, labels=labels)
            out.loss.backward()

        # Compute per-gene dot product: grad . repair_direction
        params_dict = dict(model.named_parameters())
        gene_scores = []
        for repairs in gene_repairs:
            dot = 0.0
            for (pname, dim, start, end), delta_w in repairs.items():
                grad = params_dict[pname].grad
                if grad is None:
                    continue
                if dim is not None:
                    grad = grad.narrow(dim, start, end - start)
                dot += (grad.float() * delta_w.to(grad.device)).sum().item()
            gene_scores.append(dot)

        model.zero_grad()
        model.eval()
        return gene_scores

    # Score against all previous objectives
    all_prev = {}
    for name, enc in eval_sets:
        all_prev[name] = grad_scores_for_eval(enc)

    # Score against current domain
    curr_scores = grad_scores_for_eval(curr_eval)

    # Convert to Bayesian posteriors
    # Negative grad score = repair reduces loss = helps objective (deleterious mutation)
    # Positive grad score = repair increases loss = hurts objective
    n = genome.total_genes
    prev_names = [name for name, _ in eval_sets]

    raw_scores = []
    for gid in range(n):
        deltas_prev = {name: -all_prev[name][gid] for name in prev_names}
        delta_curr = curr_scores[gid]
        raw_scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                           "delta_curr": delta_curr})

    var_prev = {name: max(np.var([s["deltas_prev"][name] for s in raw_scores]), 1e-10)
                for name in prev_names}
    var_curr = max(np.var([s["delta_curr"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        p_del = {}
        for name in prev_names:
            sig = np.sqrt(var_prev[name] / 2)
            p_del[name] = 1 - stats.norm.cdf(
                0, loc=s["deltas_prev"][name] / 2, scale=sig)
        p_del_prev = max(p_del.values())
        sig_c = np.sqrt(var_curr / 2)
        p_ben_curr = 1 - stats.norm.cdf(
            0, loc=s["delta_curr"] / 2, scale=sig_c)
        scores.append({"gene_id": s["gene_id"], "p_del_prev": p_del_prev,
                        "p_ben_curr": p_ben_curr})
    return scores


# ===================================================================
#  CHIMERIC SCORING (original, for quality comparison)
# ===================================================================

def score_genes_chimeric(genome, model, eval_sets, curr_eval, device):
    """Original O(N_genes * N_objectives) chimeric evaluation."""
    genome.sync_primary()
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {k: v.clone() for k, v in gene.primary.items()}

    base_prev = {name: perplexity(model, enc, device) for name, enc in eval_sets}
    base_curr = perplexity(model, curr_eval, device)

    raw_scores = []
    n = genome.total_genes
    for gid in range(n):
        genome.repair_genes([gid])
        deltas_prev = {}
        for name, enc in eval_sets:
            ppl = perplexity(model, enc, device)
            deltas_prev[name] = base_prev[name] - ppl
        ppl_curr = perplexity(model, curr_eval, device)
        delta_curr = ppl_curr - base_curr

        for i, g in enumerate(genome.genes):
            for k, v in saved[i].items():
                g.primary[k] = v.clone()
        genome.apply_primary()

        raw_scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                           "delta_curr": delta_curr})
        if (gid + 1) % 30 == 0 or gid == n - 1:
            print(f"      {gid+1}/{n}", end=" ", flush=True)
    print()

    prev_names = [name for name, _ in eval_sets]
    var_prev = {name: max(np.var([s["deltas_prev"][name] for s in raw_scores]), 1e-10)
                for name in prev_names}
    var_curr = max(np.var([s["delta_curr"] for s in raw_scores]), 1e-10)

    scores = []
    for s in raw_scores:
        p_del = {}
        for name in prev_names:
            sig = np.sqrt(var_prev[name] / 2)
            p_del[name] = 1 - stats.norm.cdf(
                0, loc=s["deltas_prev"][name] / 2, scale=sig)
        p_del_prev = max(p_del.values())
        sig_c = np.sqrt(var_curr / 2)
        p_ben_curr = 1 - stats.norm.cdf(
            0, loc=s["delta_curr"] / 2, scale=sig_c)
        scores.append({"gene_id": s["gene_id"], "p_del_prev": p_del_prev,
                        "p_ben_curr": p_ben_curr})
    return scores


# ===================================================================
#  Runners
# ===================================================================

def run_gc(base_model, domain_data, gen_eval, device, scoring="gradient"):
    label = f"Gene Conv ({scoring})"
    print(f"\n  [{label}]")
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16, granularity="head")
    genome.snapshot()

    domain_names = list(domain_data.keys())
    t_train_total, t_score_total = 0.0, 0.0

    for step_idx, dname in enumerate(domain_names):
        print(f"    {dname}:", end=" ", flush=True)

        t0 = time.time()
        fine_tune(model, domain_data[dname]["train"], device)
        t_train = time.time() - t0
        t_train_total += t_train

        eval_sets = [("General", gen_eval)]
        for prev in domain_names[:step_idx]:
            eval_sets.append((prev, domain_data[prev]["eval"]))

        t0 = time.time()
        genome.sync_primary()
        if scoring == "gradient":
            scores = score_genes_gradient(genome, model, eval_sets,
                                          domain_data[dname]["eval"], device)
        else:
            scores = score_genes_chimeric(genome, model, eval_sets,
                                          domain_data[dname]["eval"], device)
        t_score = time.time() - t0
        t_score_total += t_score

        n_r, n_f = gene_conversion(genome, scores)
        print(f"rep={n_r} fix={n_f} (train {t_train:.1f}s, score {t_score:.1f}s)")
        genome.snapshot()

    result = eval_all(model, gen_eval, domain_data, device)
    del model
    torch.cuda.empty_cache()
    return result, t_train_total, t_score_total


def run_lora(base_model, domain_data, gen_eval, device, rank=8):
    from peft import LoraConfig, get_peft_model, TaskType
    print("\n  [LoRA]")
    domain_names = list(domain_data.keys())
    adapters = []

    for dname in domain_names:
        print(f"    {dname}:", end=" ", flush=True)
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
        ids = domain_data[dname]["train"]["input_ids"].to(device)
        mask = domain_data[dname]["train"]["attention_mask"].to(device)
        for epoch in range(2):
            perm = torch.randperm(len(ids))
            ids, mask = ids[perm], mask[perm]
            nb = (len(ids) + 7) // 8
            for i in range(nb):
                s, e = i * 8, min((i+1) * 8, len(ids))
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
        adapters.append(delta)
        del peft_model, merged, model
        torch.cuda.empty_cache()
        print("done")

    final = copy.deepcopy(base_model).to(device)
    for n, p in final.named_parameters():
        avg = sum(a.get(n, torch.zeros_like(p.cpu())) for a in adapters) / len(adapters)
        p.data += avg.to(device)
    result = eval_all(final, gen_eval, domain_data, device)
    del final
    torch.cuda.empty_cache()
    return result


# ===================================================================
#  Main
# ===================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda"

    print("=" * 70)
    print("  Speed Benchmark: Gradient Scoring vs Chimeric vs LoRA")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")

    domain_data, gen_eval = load_fast_data(tokenizer)
    domain_names = list(domain_data.keys())

    base_on_gpu = copy.deepcopy(base_model).to(device)
    baseline = eval_all(base_on_gpu, gen_eval, domain_data, device)
    del base_on_gpu
    torch.cuda.empty_cache()
    gm_base = geo_mean(baseline)
    print(f"  Baseline GM: {gm_base:.2f}")

    # 1. LoRA (speed reference)
    t0 = time.time()
    lora_result = run_lora(base_model, domain_data, gen_eval, device)
    t_lora = time.time() - t0

    # 2. Gene Conversion — gradient scoring (FAST)
    t0 = time.time()
    gc_grad_result, gc_grad_train, gc_grad_score = run_gc(
        base_model, domain_data, gen_eval, device, scoring="gradient")
    t_gc_grad = time.time() - t0

    # 3. Gene Conversion — chimeric scoring (SLOW, quality reference)
    t0 = time.time()
    gc_chim_result, gc_chim_train, gc_chim_score = run_gc(
        base_model, domain_data, gen_eval, device, scoring="chimeric")
    t_gc_chim = time.time() - t0

    # Results
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}\n")

    header = f"  {'Method':>20s}"
    for dn in ["General"] + domain_names:
        header += f"  {dn[:6]:>7s}"
    header += f"  {'GM':>7s}  {'Total':>6s}  {'Score':>6s}"
    print(header)
    print(f"  {'-'*20}  {'-------  ' * 4}-------  ------  ------")

    data = [
        ("Pretrained", baseline, 0, 0),
        ("LoRA", lora_result, t_lora, 0),
        ("GC (gradient)", gc_grad_result, t_gc_grad, gc_grad_score),
        ("GC (chimeric)", gc_chim_result, t_gc_chim, gc_chim_score),
    ]

    for name, ppls, t_total, t_score in data:
        gm = geo_mean(ppls)
        row = f"  {name:>20s}"
        for dn in ["General"] + domain_names:
            row += f"  {ppls[dn]:>7.1f}"
        row += f"  {gm:>7.2f}  {t_total:>5.1f}s"
        if t_score > 0:
            row += f"  {t_score:>5.1f}s"
        else:
            row += f"  {'--':>6s}"
        print(row)

    print(f"\n  Scoring speedup: {gc_chim_score/max(gc_grad_score, 0.1):.0f}x "
          f"({gc_chim_score:.1f}s -> {gc_grad_score:.1f}s)")
    print(f"  Overall speedup vs chimeric: {t_gc_chim/max(t_gc_grad, 0.1):.1f}x")
    print(f"  vs LoRA speed: {t_gc_grad/max(t_lora, 0.1):.1f}x")

    # Quality comparison
    gm_grad = geo_mean(gc_grad_result)
    gm_chim = geo_mean(gc_chim_result)
    gm_lora = geo_mean(lora_result)
    print(f"\n  Quality: gradient GM={gm_grad:.2f}, chimeric GM={gm_chim:.2f} "
          f"(diff={((gm_grad-gm_chim)/gm_chim)*100:+.1f}%)")
    if gm_grad < gm_lora:
        print(f"  Gradient scoring BEATS LoRA ({gm_grad:.2f} vs {gm_lora:.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
