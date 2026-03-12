#!/usr/bin/env python3
"""
Quick comparison: Gene Conversion (head-level) vs LoRA vs QLoRA

Uses the same minimal data as benchmark_granularity.py for speed.
Gene conversion result from that test: GM=8.73
"""

import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.gene_conversion.transformer_genome import TransformerDualGenome
from scipy import stats

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===================================================================
#  Minimal data (same as benchmark_granularity.py)
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


# ===================================================================
#  Gene Conversion (head-level, multi-objective)
# ===================================================================

def score_genes_multi(genome, model, eval_sets, curr_eval, device):
    genome.sync_primary()
    saved = {}
    for i, gene in enumerate(genome.genes):
        saved[i] = {k: v.clone() for k, v in gene.primary.items()}

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

        # Restore all genes
        for i, g in enumerate(genome.genes):
            for k, v in saved[i].items():
                g.primary[k] = v.clone()
        genome.apply_primary()

        scores.append({"gene_id": gid, "deltas_prev": deltas_prev,
                        "delta_curr": delta_curr})
        if (gid + 1) % 30 == 0 or gid == n - 1:
            print(f"      {gid+1}/{n}", end=" ", flush=True)
    print()

    prev_names = [name for name, _ in eval_sets]
    var_prev = {name: max(np.var([s["deltas_prev"][name] for s in scores]), 1e-10)
                for name in prev_names}
    var_curr = max(np.var([s["delta_curr"] for s in scores]), 1e-10)
    for s in scores:
        p_del = {}
        for name in prev_names:
            sig = np.sqrt(var_prev[name] / 2)
            p_del[name] = 1 - stats.norm.cdf(0, loc=s["deltas_prev"][name]/2, scale=sig)
        s["p_del_prev"] = max(p_del.values())
        sig_c = np.sqrt(var_curr / 2)
        s["p_ben_curr"] = 1 - stats.norm.cdf(0, loc=s["delta_curr"]/2, scale=sig_c)
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


def run_gene_conversion(base_model, domain_data, gen_eval, device):
    print("\n  [Gene Conversion - head-level, 207 genes]")
    model = copy.deepcopy(base_model).to(device)
    genome = TransformerDualGenome(model, n_bits=16, granularity="head")
    genome.snapshot()

    domain_names = list(domain_data.keys())
    for step_idx, dname in enumerate(domain_names):
        print(f"    {dname}: train...", end=" ", flush=True)
        fine_tune(model, domain_data[dname]["train"], device)
        eval_sets = [("General", gen_eval)]
        for prev in domain_names[:step_idx]:
            eval_sets.append((prev, domain_data[prev]["eval"]))
        print(f"score vs {len(eval_sets)} obj...")
        genome.sync_primary()
        scores = score_genes_multi(genome, model, eval_sets,
                                   domain_data[dname]["eval"], device)
        n_r, n_f = gene_conversion(genome, scores)
        print(f"      repaired {n_r}, fixed {n_f}")
        genome.snapshot()

    result = eval_all(model, gen_eval, domain_data, device)
    del model
    torch.cuda.empty_cache()
    return result


# ===================================================================
#  LoRA / QLoRA runners
# ===================================================================

def run_lora_variant(base_model, domain_data, gen_eval, device,
                     use_qlora=False, rank=8):
    from peft import LoraConfig, get_peft_model, TaskType

    label = "QLoRA" if use_qlora else "LoRA"
    print(f"\n  [{label} - rank={rank}, per-domain adapters merged]")

    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    domain_names = list(domain_data.keys())
    adapters = []

    for dname in domain_names:
        print(f"    {dname}: training adapter...", end=" ", flush=True)

        if use_qlora:
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(
                "gpt2", quantization_config=bnb_config,
                device_map={"": device}
            )
        else:
            model = copy.deepcopy(base_model).to(device)

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=16,
            lora_dropout=0.05, target_modules=["c_attn", "c_proj"],
        )
        peft_model = get_peft_model(model, config)
        peft_model.train()

        from torch.optim import AdamW
        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        opt = AdamW(trainable, lr=2e-4, weight_decay=0.01)
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

        # For QLoRA, extract LoRA weights and apply to full-precision base
        if use_qlora:
            # Save just the LoRA adapter matrices
            lora_state = {k: v.cpu().clone()
                          for k, v in peft_model.state_dict().items()
                          if "lora_" in k}
            del peft_model, model
            torch.cuda.empty_cache()

            # Apply adapter to full-precision base for clean delta
            fp_model = copy.deepcopy(base_model).to(device)
            fp_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=16,
                lora_dropout=0.05, target_modules=["c_attn", "c_proj"],
            )
            fp_peft = get_peft_model(fp_model, fp_config)
            # Load trained adapter weights
            missing, unexpected = fp_peft.load_state_dict(
                lora_state, strict=False)
            merged = fp_peft.merge_and_unload()
            base_state = base_model.state_dict()
            delta = {}
            for n, p in merged.named_parameters():
                if n in base_state:
                    delta[n] = (p.data.cpu() - base_state[n].cpu())
            adapters.append(delta)
            del fp_model, fp_peft, merged
            model = None  # already deleted above
        else:
            merged = peft_model.merge_and_unload()
            base_state = base_model.state_dict()
            delta = {}
            for n, p in merged.named_parameters():
                if n in base_state:
                    delta[n] = (p.data.cpu() - base_state[n].cpu())
            adapters.append(delta)
            del merged

        if not use_qlora:
            del peft_model, model
        torch.cuda.empty_cache()
        print("done")

    # Average all deltas and apply
    print(f"    Merging {len(adapters)} adapters...")
    final = copy.deepcopy(base_model).to(device)
    n_adapters = len(adapters)
    for n, p in final.named_parameters():
        avg_delta = sum(a.get(n, torch.zeros_like(p.cpu()))
                        for a in adapters) / n_adapters
        p.data += avg_delta.to(device)

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
    print("  Gene Conversion vs LoRA vs QLoRA")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("  3 domains, minimal data, fast iteration")
    print("=" * 70)

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("  GPT-2 loaded")

    print("  Loading data...")
    domain_data, gen_eval = load_fast_data(tokenizer)
    domain_names = list(domain_data.keys())

    # Baseline
    base_on_gpu = copy.deepcopy(base_model).to(device)
    baseline = eval_all(base_on_gpu, gen_eval, domain_data, device)
    del base_on_gpu
    torch.cuda.empty_cache()
    gm_base = geo_mean(baseline)

    total_start = time.time()
    results = {"Pretrained": baseline}

    # Run LoRA
    t0 = time.time()
    results["LoRA"] = run_lora_variant(base_model, domain_data, gen_eval,
                                        device, use_qlora=False)
    t_lora = time.time() - t0

    # Run QLoRA
    t0 = time.time()
    results["QLoRA"] = run_lora_variant(base_model, domain_data, gen_eval,
                                         device, use_qlora=True)
    t_qlora = time.time() - t0

    # Run Gene Conversion (head-level)
    t0 = time.time()
    results["Gene Conv"] = run_gene_conversion(base_model, domain_data,
                                                gen_eval, device)
    t_gc = time.time() - t0

    total_time = time.time() - total_start

    # Results
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}\n")

    header = f"  {'Method':>12s}"
    for dn in ["General"] + domain_names:
        header += f"  {dn[:6]:>7s}"
    header += f"  {'GM':>7s}  {'Time':>5s}"
    print(header)
    print(f"  {'-'*12}  {'-------  ' * (1 + len(domain_names))}-------  -----")

    timings = {"Pretrained": 0, "LoRA": t_lora, "QLoRA": t_qlora,
               "Gene Conv": t_gc}

    ranked = []
    for method, ppls in results.items():
        gm = geo_mean(ppls)
        ranked.append((method, ppls, gm))
        row = f"  {method:>12s}"
        for dn in ["General"] + domain_names:
            row += f"  {ppls[dn]:>7.1f}"
        t = timings[method]
        row += f"  {gm:>7.2f}  {t:>4.0f}s"
        print(row)

    ranked.sort(key=lambda x: x[2])
    print(f"\n  Rankings (GM, lower = better):")
    for i, (method, _, gm) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"    {i+1}. {method:>12s}  {gm:.2f}{marker}")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
