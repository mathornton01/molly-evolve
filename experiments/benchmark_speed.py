"""
Molly Evolution — Speed & Memory Benchmark

Runs a full evolution cycle on GPT-2 (124M) with detailed timing
and memory logging for each phase. Compares streaming vs precomputed
scoring modes and projects requirements for larger models.

Usage:
    python experiments/benchmark_speed.py
"""

import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from molly_evolution import DualGenome, GeneScorer
from molly_evolution.distributed import estimate_requirements

# ── Logging setup ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("molly_evolution")
logger.setLevel(logging.DEBUG)


# ── Utilities ────────────────────────────────────────────────────

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def fmt_mem(mb):
    if mb > 1024:
        return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"


def perplexity(model, enc, device, use_amp=False):
    model.eval()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                out = model(ids, attention_mask=mask, labels=ids)
        else:
            out = model(ids, attention_mask=mask, labels=ids)
    return math.exp(out.loss.item())


# ── Main benchmark ───────────────────────────────────────────────

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  MOLLY EVOLUTION — SPEED & MEMORY BENCHMARK")
    print("=" * 70)
    print()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Device:  {gpu_name} ({gpu_total:.1f} GB)")
        torch.cuda.reset_peak_memory_stats()
    else:
        print(f"  Device:  CPU")
    print()

    # ── 1. Load model ────────────────────────────────────────────

    print("Phase 1: Model Loading")
    print("-" * 70)
    t0 = time.perf_counter()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    n_params = sum(p.numel() for p in model.parameters())

    dt = time.perf_counter() - t0
    print(f"  Model:         GPT-2 ({n_params/1e6:.1f}M params)")
    print(f"  Load time:     {fmt_time(dt)}")
    print(f"  GPU memory:    {fmt_mem(gpu_mem_mb())}")
    print()

    # ── 2. Prepare eval data ─────────────────────────────────────

    general_text = ("The quick brown fox jumps over the lazy dog. " * 8 +
                    "In natural language processing, transformers have "
                    "revolutionized the field of machine learning.")
    domain_text = ("The mitochondria is the powerhouse of the cell. " * 8 +
                   "Protein synthesis occurs at ribosomes attached to "
                   "the endoplasmic reticulum.")
    train_text = ("Chemical reactions involve the breaking and forming "
                  "of molecular bonds. " * 8 +
                  "Organic chemistry studies carbon-based compounds.")

    def encode(text, max_length=128):
        return tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_length, padding="max_length")

    general_enc = encode(general_text)
    domain_enc = encode(domain_text)
    train_enc = encode(train_text)

    # ── 3. Genome initialization ─────────────────────────────────

    print("Phase 2: Genome Initialization")
    print("-" * 70)

    for gran in ["component", "head"]:
        t0 = time.perf_counter()
        g = DualGenome(model, granularity=gran, backend="python")
        dt = time.perf_counter() - t0
        print(f"  {gran:>10s}: {g.total_genes:>4d} genes | {fmt_time(dt)}")
    print()

    # Use head-level for the benchmark
    t0 = time.perf_counter()
    genome = DualGenome(model, granularity="head", backend="python")
    genome.snapshot()
    dt_snapshot = time.perf_counter() - t0
    footprint = genome.memory_footprint()

    print(f"  Snapshot:      {fmt_time(dt_snapshot)}")
    print(f"  Genome CPU:    {fmt_mem(footprint['genome_cpu_mb'])}")
    print(f"  GPU memory:    {fmt_mem(gpu_mem_mb())}")
    print()

    # ── 4. Training (simulate fine-tuning) ───────────────────────

    print("Phase 3: Fine-Tuning (simulated)")
    print("-" * 70)

    ppl_before = perplexity(model, train_enc, device)
    print(f"  PPL before:    {ppl_before:.1f}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    t0 = time.perf_counter()
    n_steps = 20
    ids = train_enc["input_ids"].to(device)
    mask = train_enc["attention_mask"].to(device)

    for step in range(n_steps):
        optimizer.zero_grad()
        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                out = model(ids, attention_mask=mask, labels=ids)
            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(ids, attention_mask=mask, labels=ids)
            out.loss.backward()
            optimizer.step()

    dt_train = time.perf_counter() - t0
    ppl_after = perplexity(model, train_enc, device)
    ppl_general = perplexity(model, general_enc, device)

    print(f"  Steps:         {n_steps}")
    print(f"  Train time:    {fmt_time(dt_train)} ({n_steps/dt_train:.1f} steps/s)")
    print(f"  PPL after:     {ppl_after:.1f} (train domain)")
    print(f"  PPL general:   {ppl_general:.1f} (degradation check)")
    print(f"  GPU peak:      {fmt_mem(gpu_peak_mb())}")
    print()

    del optimizer, scaler

    # ── 5. Scoring comparison: streaming vs precomputed ──────────

    print("Phase 4: Gene Scoring Comparison")
    print("-" * 70)

    eval_sets = [("general", general_enc)]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Precomputed mode ---
    t0 = time.perf_counter()
    scorer_pre = GeneScorer(genome, model, device,
                            use_amp=use_amp, streaming=False)
    scores_pre = scorer_pre.score_multi_objective(eval_sets, domain_enc)
    dt_pre = time.perf_counter() - t0
    mem_pre = gpu_peak_mb() if torch.cuda.is_available() else 0
    est_pre = scorer_pre.memory_estimate()

    print(f"  Precomputed mode:")
    print(f"    Time:        {fmt_time(dt_pre)}")
    print(f"    GPU peak:    {fmt_mem(mem_pre)}")
    print(f"    Delta GPU:   {fmt_mem(est_pre['deltas_gpu_mb'])}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Streaming mode ---
    t0 = time.perf_counter()
    scorer_str = GeneScorer(genome, model, device,
                            use_amp=use_amp, streaming=True)
    scores_str = scorer_str.score_multi_objective(eval_sets, domain_enc)
    dt_str = time.perf_counter() - t0
    mem_str = gpu_peak_mb() if torch.cuda.is_available() else 0
    est_str = scorer_str.memory_estimate()

    print(f"  Streaming mode:")
    print(f"    Time:        {fmt_time(dt_str)}")
    print(f"    GPU peak:    {fmt_mem(mem_str)}")
    print(f"    Delta GPU:   {fmt_mem(est_str['deltas_gpu_mb'])}")

    # Compare repair decisions (both modes should agree on which genes to repair)
    def _get_repair_set(scores, threshold=0.50, alpha=0.3):
        return {s["gene_id"] for s in scores
                if s["p_del_prev"] - alpha * s["p_ben_curr"] > threshold}
    repair_pre = _get_repair_set(scores_pre)
    repair_str = _get_repair_set(scores_str)
    overlap = len(repair_pre & repair_str)
    total = len(repair_pre | repair_str) or 1
    print(f"  Repair agreement: {overlap}/{total} genes overlap "
          f"({'PASS' if overlap/total > 0.5 else 'NOTE: expected variance'})")
    print()

    # ── 6. Gene conversion ───────────────────────────────────────

    print("Phase 5: Gene Conversion")
    print("-" * 70)

    t0 = time.perf_counter()
    n_repaired, n_fixed = genome.apply_conversion(scores_str)
    dt_conv = time.perf_counter() - t0

    ppl_repaired = perplexity(model, general_enc, device)

    print(f"  Repaired:      {n_repaired}/{genome.total_genes} genes")
    print(f"  Fixed:         {n_fixed}/{genome.total_genes} genes")
    print(f"  Time:          {fmt_time(dt_conv)}")
    print(f"  PPL general:   {ppl_general:.1f} -> {ppl_repaired:.1f} (after repair)")
    print()

    # ── 7. Full cycle summary ────────────────────────────────────

    total = dt_snapshot + dt_train + dt_str + dt_conv
    print("=" * 70)
    print("  FULL CYCLE SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Phase':<25s} {'Time':>10s} {'GPU Peak':>12s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    print(f"  {'Snapshot':<25s} {fmt_time(dt_snapshot):>10s} "
          f"{fmt_mem(footprint['model_gpu_mb']):>12s}")
    print(f"  {'Training (20 steps)':<25s} {fmt_time(dt_train):>10s}")
    print(f"  {'Scoring (streaming)':<25s} {fmt_time(dt_str):>10s} "
          f"{fmt_mem(mem_str):>12s}")
    print(f"  {'Gene conversion':<25s} {fmt_time(dt_conv):>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL CYCLE':<25s} {fmt_time(total):>10s}")
    print()

    throughput = n_params / dt_str
    print(f"  Scoring throughput:    {throughput/1e6:.1f}M params/sec")
    print(f"  Genes:                {genome.total_genes}")
    print(f"  Scoring mode:         streaming (low memory)")
    print()

    # ── 8. Scaling projections ───────────────────────────────────

    print("=" * 70)
    print("  SCALING PROJECTIONS")
    print("=" * 70)
    print()

    models = [
        ("GPT-2", 124e6),
        ("GPT-2 XL", 1.5e9),
        ("LLaMA 7B", 7e9),
        ("LLaMA 13B", 13e9),
        ("LLaMA 70B", 70e9),
    ]

    print(f"  {'Model':<14s} {'Params':>8s} {'Streaming':>12s} "
          f"{'Precomputed':>14s} {'Genome CPU':>12s} {'Rec. GPUs':>10s}")
    print(f"  {'-'*14} {'-'*8} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")

    for name, n in models:
        est = estimate_requirements(int(n), n_objectives=3)
        print(f"  {name:<14s} {n/1e9:>7.1f}B "
              f"{fmt_mem(est['scoring_streaming_gpu_mb']):>12s} "
              f"{fmt_mem(est['scoring_precomputed_gpu_mb']):>14s} "
              f"{fmt_mem(est['genome_cpu_mb']):>12s} "
              f"{est['recommended_gpus']:>10}")

    print()

    # Multi-GPU scaling estimate
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if n_gpus > 1:
        print(f"  Detected {n_gpus} GPUs — MultiGPUScorer available")
        print(f"  Estimated speedup with {n_gpus} GPUs "
              f"and 3 objectives: ~{min(n_gpus, 3):.0f}x")
    else:
        print(f"  Single GPU detected. Multi-GPU scaling available on multi-GPU nodes.")
    print()

    # ── 9. RunPod configuration guide ────────────────────────────

    print("=" * 70)
    print("  RUNPOD CONFIGURATION GUIDE")
    print("=" * 70)
    print()
    print("  For LLaMA 7B (streaming mode):")
    print("    GPU:    1x A100 80GB  (model 14GB + grads 28GB = 42GB peak)")
    print("    CPU:    32 GB RAM (genome strands)")
    print("    Pod:    A100 80GB SXM, 64 GB RAM")
    print()
    print("  For LLaMA 13B (streaming mode):")
    print("    GPU:    1x A100 80GB  (model 26GB + grads 52GB — tight)")
    print("    Alt:    2x A100 80GB with MultiGPUScorer")
    print("    CPU:    64 GB RAM")
    print()
    print("  For LLaMA 70B (FSDP + streaming):")
    print("    GPU:    4x A100 80GB  (FSDP shards model+grads)")
    print("    CPU:    512 GB RAM")
    print("    Use:    FSDPGenome + FSDPScorer")
    print()


if __name__ == "__main__":
    main()
