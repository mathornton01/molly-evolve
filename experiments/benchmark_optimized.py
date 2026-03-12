#!/usr/bin/env python3
"""
Fully optimized gene conversion: mixed precision + vectorized scoring + torch.compile

Optimizations applied:
  1. torch.amp.autocast (fp16) — ~2x faster forward/backward
  2. Precompute repair vectors on GPU — no CPU<->GPU transfers during scoring
  3. Vectorized dot products per parameter — no Python loops over genes
  4. torch.compile — JIT fuses ops for additional speedup
  5. Larger batch sizes where possible
"""

import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
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
#  Core utilities (with mixed precision)
# ===================================================================

@torch.no_grad()
def perplexity(model, enc, device, batch_size=40):
    model.eval()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(ids), batch_size):
        b_ids = ids[i:i+batch_size]
        b_mask = mask[i:i+batch_size]
        labels = b_ids.clone()
        labels[b_mask == 0] = -100
        with autocast("cuda", dtype=torch.float16):
            out = model(b_ids, attention_mask=b_mask, labels=labels)
        n = (labels != -100).sum().item()
        total_loss += out.loss.float().item() * n
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
    scaler = torch.amp.GradScaler("cuda")
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
            with autocast("cuda", dtype=torch.float16):
                out = model(ids[s:e], attention_mask=mask[s:e], labels=labels)
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
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
#  VECTORIZED GRADIENT SCORING (fully optimized)
# ===================================================================

class FastGeneScorer:
    """
    Precomputes a vectorized mapping from model parameters to gene scores.
    Scoring is a single batched operation per eval set — no Python loops.
    """

    def __init__(self, genome, model, device):
        self.genome = genome
        self.model = model
        self.device = device
        self.n_genes = genome.total_genes

        # Build repair direction vectors and gene-to-param mapping
        # For each param, store list of (gene_id, slice_or_full, repair_delta)
        self._build_repair_map()

    def _build_repair_map(self):
        """Precompute repair directions on GPU for all genes."""
        self.param_gene_map = {}  # {param_name: list of (gene_id, dim, start, end, delta_gpu)}

        for gid, gene in enumerate(self.genome.genes):
            if hasattr(gene, 'slice_defs'):
                for pname, dim, start, end in gene.slice_defs:
                    key = gene._key(pname, dim, start, end)
                    comp = gene._dequantize(gene.complement[key], gene.scales[key])
                    prim = gene._dequantize(gene.primary[key], gene.scales[key])
                    delta = (comp - prim).float().to(self.device)
                    self.param_gene_map.setdefault(pname, []).append(
                        (gid, dim, start, end, delta))
            else:
                for pn in gene.param_names:
                    comp = gene._dequantize(gene.complement[pn], gene.scales[pn])
                    prim = gene._dequantize(gene.primary[pn], gene.scales[pn])
                    delta = (comp - prim).float().to(self.device)
                    self.param_gene_map.setdefault(pn, []).append(
                        (gid, None, None, None, delta))

    def score(self, enc):
        """One forward+backward -> all gene scores via vectorized dot products."""
        self.model.zero_grad()
        self.model.train()
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        # Single forward+backward with mixed precision
        with autocast("cuda", dtype=torch.float16):
            labels = ids.clone()
            labels[mask == 0] = -100
            out = self.model(ids, attention_mask=mask, labels=labels)
        out.loss.backward()

        # Vectorized dot products — accumulate into GPU tensor
        gene_scores = torch.zeros(self.n_genes, device=self.device)
        params_dict = dict(self.model.named_parameters())

        for pname, entries in self.param_gene_map.items():
            grad = params_dict[pname].grad
            if grad is None:
                continue
            grad_f = grad.float()

            for gid, dim, start, end, delta in entries:
                if dim is not None:
                    g_slice = grad_f.narrow(dim, start, end - start)
                else:
                    g_slice = grad_f
                gene_scores[gid] += (g_slice * delta).sum()

        self.model.zero_grad()
        self.model.eval()
        return gene_scores.cpu().numpy()


def score_genes_fast(genome, model, eval_sets, curr_eval, device):
    """Fully optimized multi-objective gradient scoring."""
    genome.sync_primary()
    scorer = FastGeneScorer(genome, model, device)

    # Score against all previous objectives
    all_prev = {}
    for name, enc in eval_sets:
        all_prev[name] = scorer.score(enc)

    # Score against current domain
    curr_scores = scorer.score(curr_eval)

    # Convert to Bayesian posteriors
    n = genome.total_genes
    prev_names = [name for name, _ in eval_sets]

    raw_scores = []
    for gid in range(n):
        deltas_prev = {name: -all_prev[name][gid] for name in prev_names}
        delta_curr = curr_scores[gid]
        raw_scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                           "delta_curr": float(delta_curr)})

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

def run_gc_optimized(base_model, domain_data, gen_eval, device, use_compile=False):
    label = "GC optimized" + (" +compile" if use_compile else "")
    print(f"\n  [{label}]")
    model = copy.deepcopy(base_model).to(device)

    if use_compile:
        model = torch.compile(model)
        # Warmup compile with a dummy forward pass
        dummy = torch.randint(0, 1000, (1, 32), device=device)
        with autocast("cuda", dtype=torch.float16):
            model(dummy)
        print("    (compiled)", end=" ")

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
        scores = score_genes_fast(genome, model, eval_sets,
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


def run_gc_baseline(base_model, domain_data, gen_eval, device):
    """Previous gradient scoring without optimizations (for comparison)."""
    print("\n  [GC gradient (no amp)]")
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16, granularity="head")
    genome.snapshot()

    domain_names = list(domain_data.keys())
    t_train_total, t_score_total = 0.0, 0.0

    for step_idx, dname in enumerate(domain_names):
        print(f"    {dname}:", end=" ", flush=True)

        # Train without AMP
        t0 = time.time()
        from torch.optim import AdamW
        model.train()
        opt = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
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
                out = model(ids[s:e], attention_mask=mask[s:e], labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        t_train = time.time() - t0
        t_train_total += t_train

        # Score without AMP
        eval_sets = [("General", gen_eval)]
        for prev in domain_names[:step_idx]:
            eval_sets.append((prev, domain_data[prev]["eval"]))

        t0 = time.time()
        genome.sync_primary()

        # Precompute repairs
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

        all_prev = {}
        for name, enc in eval_sets:
            model.zero_grad()
            model.train()
            _ids = enc["input_ids"].to(device)
            _mask = enc["attention_mask"].to(device)
            for i in range(0, len(_ids), 20):
                b_ids = _ids[i:i+20]
                b_mask = _mask[i:i+20]
                lbls = b_ids.clone()
                lbls[b_mask == 0] = -100
                out = model(b_ids, attention_mask=b_mask, labels=lbls)
                out.loss.backward()

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
            all_prev[name] = np.array(gene_scores)
            model.zero_grad()
            model.eval()

        # Current domain
        model.zero_grad()
        model.train()
        _ids = domain_data[dname]["eval"]["input_ids"].to(device)
        _mask = domain_data[dname]["eval"]["attention_mask"].to(device)
        for i in range(0, len(_ids), 20):
            b_ids = _ids[i:i+20]
            b_mask = _mask[i:i+20]
            lbls = b_ids.clone()
            lbls[b_mask == 0] = -100
            out = model(b_ids, attention_mask=b_mask, labels=lbls)
            out.loss.backward()
        params_dict = dict(model.named_parameters())
        curr_scores = []
        for repairs in gene_repairs:
            dot = 0.0
            for (pname, dim, start, end), delta_w in repairs.items():
                grad = params_dict[pname].grad
                if grad is None:
                    continue
                if dim is not None:
                    grad = grad.narrow(dim, start, end - start)
                dot += (grad.float() * delta_w.to(grad.device)).sum().item()
            curr_scores.append(dot)
        curr_scores = np.array(curr_scores)
        model.zero_grad()
        model.eval()

        t_score = time.time() - t0
        t_score_total += t_score

        # Bayesian posteriors
        prev_names = [name for name, _ in eval_sets]
        n = genome.total_genes
        raw = []
        for gid in range(n):
            dp = {name: -all_prev[name][gid] for name in prev_names}
            raw.append({"gene_id": gid, "deltas_prev": dp,
                        "delta_curr": float(curr_scores[gid])})
        vp = {name: max(np.var([s["deltas_prev"][name] for s in raw]), 1e-10)
              for name in prev_names}
        vc = max(np.var([s["delta_curr"] for s in raw]), 1e-10)
        sc = []
        for s in raw:
            pd = {}
            for name in prev_names:
                sig = np.sqrt(vp[name] / 2)
                pd[name] = 1 - stats.norm.cdf(0, loc=s["deltas_prev"][name]/2, scale=sig)
            pdp = max(pd.values())
            sigc = np.sqrt(vc / 2)
            pbc = 1 - stats.norm.cdf(0, loc=s["delta_curr"]/2, scale=sigc)
            sc.append({"gene_id": s["gene_id"], "p_del_prev": pdp, "p_ben_curr": pbc})

        n_r, n_f = gene_conversion(genome, sc)
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
    print("  Optimization Benchmark: AMP + Vectorized + Compile")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")

    domain_data, gen_eval = load_fast_data(tokenizer)
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

    # 2. GC baseline (gradient scoring, no AMP)
    t0 = time.time()
    gc_base_result, gc_base_train, gc_base_score = run_gc_baseline(
        base_model, domain_data, gen_eval, device)
    t_gc_base = time.time() - t0

    # 3. GC optimized (AMP + vectorized scoring, no compile)
    t0 = time.time()
    gc_opt_result, gc_opt_train, gc_opt_score = run_gc_optimized(
        base_model, domain_data, gen_eval, device, use_compile=False)
    t_gc_opt = time.time() - t0

    # 4. GC optimized + torch.compile (skip on Windows — no Triton)
    gc_comp_result, gc_comp_train, gc_comp_score, t_gc_comp = None, 0, 0, 0
    try:
        import triton
        t0 = time.time()
        gc_comp_result, gc_comp_train, gc_comp_score = run_gc_optimized(
            base_model, domain_data, gen_eval, device, use_compile=True)
        t_gc_comp = time.time() - t0
    except (ImportError, RuntimeError):
        print("\n  [torch.compile skipped — Triton not available on Windows]")

    # Results
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}\n")

    domain_names = list(domain_data.keys())
    header = f"  {'Method':>22s}"
    for dn in ["General"] + domain_names:
        header += f"  {dn[:6]:>6s}"
    header += f"  {'GM':>6s}  {'Total':>6s}  {'Train':>5s}  {'Score':>5s}"
    print(header)
    print(f"  {'-'*22}  {'------  ' * 4}------  ------  -----  -----")

    data = [
        ("Pretrained", baseline, 0, 0, 0),
        ("LoRA", lora_result, t_lora, t_lora, 0),
        ("GC grad (no amp)", gc_base_result, t_gc_base, gc_base_train, gc_base_score),
        ("GC + AMP + vectorized", gc_opt_result, t_gc_opt, gc_opt_train, gc_opt_score),
    ]
    if gc_comp_result is not None:
        data.append(("GC + AMP + compile", gc_comp_result, t_gc_comp,
                      gc_comp_train, gc_comp_score))

    for name, ppls, t_total, t_train, t_score in data:
        gm = geo_mean(ppls)
        row = f"  {name:>22s}"
        for dn in ["General"] + domain_names:
            row += f"  {ppls[dn]:>6.1f}"
        row += f"  {gm:>6.2f}  {t_total:>5.1f}s"
        row += f"  {t_train:>4.1f}s" if t_train > 0 else f"  {'--':>5s}"
        row += f"  {t_score:>4.1f}s" if t_score > 0 else f"  {'--':>5s}"
        print(row)

    print(f"\n  Speedups vs baseline gradient scoring ({t_gc_base:.1f}s):")
    print(f"    AMP + vectorized: {t_gc_base/max(t_gc_opt,0.1):.1f}x "
          f"({t_gc_opt:.1f}s)")
    if t_gc_comp > 0:
        print(f"    AMP + compile:    {t_gc_base/max(t_gc_comp,0.1):.1f}x "
              f"({t_gc_comp:.1f}s)")
    print(f"    LoRA reference:   {t_lora:.1f}s")
    print(f"    GC vs LoRA:       {t_gc_opt/max(t_lora,0.1):.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
