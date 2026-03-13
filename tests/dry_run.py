"""
Targeted end-to-end dry run that exercises ALL code paths.
Tests gene-conv, scoring, repair, LoRA, and edge cases.
"""
import sys
import os
import math
import copy
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
from molly_evolution.genome import DualGenome
from molly_evolution.scoring import GeneScorer
from full_comparison import evaluate_ppl, geometric_mean

print("=== TARGETED END-TO-END DRY RUN ===")
print()

# Tiny model for speed
config = GPT2Config(vocab_size=1000, n_positions=64, n_embd=64, n_layer=2, n_head=2)
model = GPT2LMHeadModel(config)
device = torch.device("cpu")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params")

# Create data
enc_train = {"input_ids": torch.randint(0, 1000, (10, 32)),
             "attention_mask": torch.ones(10, 32, dtype=torch.long)}
enc_eval = {"input_ids": torch.randint(0, 1000, (8, 32)),
            "attention_mask": torch.ones(8, 32, dtype=torch.long)}
enc_general = {"input_ids": torch.randint(0, 1000, (8, 32)),
               "attention_mask": torch.ones(8, 32, dtype=torch.long)}

# ========== GENE CONV PATH ==========
print()
print("--- Gene Conversion ---")
genome = DualGenome(model, granularity="head", backend="python")
genome.snapshot()
print(f"  Genes: {genome.total_genes}")

# Train
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
ids = enc_train["input_ids"]
mask = enc_train["attention_mask"]
step_losses = []
for epoch in range(2):
    for j in range(ids.size(0)):
        optimizer.zero_grad()
        out = model(ids[j:j+1], attention_mask=mask[j:j+1], labels=ids[j:j+1])
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_losses.append(out.loss.item())
    print(f"  Epoch {epoch+1}: loss={step_losses[-1]:.4f}")
del optimizer

# Post-training stability check
model.eval()
with torch.no_grad():
    out = model(ids[:1], attention_mask=mask[:1], labels=ids[:1])
    stable = not (math.isnan(out.loss.item()) or math.isinf(out.loss.item()))
print(f"  Post-training stable: {stable}")
assert stable, "Model unstable after training"

# Scoring (Bayesian)
t0 = time.perf_counter()
genome.sync_primary()
scorer = GeneScorer(genome, model, device, use_amp=False, streaming=True)

prev_evals = [("general", enc_general)]
scores = scorer.score_multi_objective(prev_evals, enc_eval, threshold=0.80, alpha=0.3)
print(f"  Scoring: {time.perf_counter()-t0:.1f}s, {len(scores)} genes")

# Verify score structure
for s in scores:
    assert "gene_id" in s and "p_del_prev" in s and "p_ben_curr" in s
    assert 0 <= s["p_del_prev"] <= 1, f"p_del={s['p_del_prev']}"
    assert 0 <= s["p_ben_curr"] <= 1, f"p_ben={s['p_ben_curr']}"
print("  All probabilities in [0,1]")

# Gene selection
to_repair, to_fix = genome.select_conversion_genes(
    scores, threshold=0.80, alpha=0.3, max_repair_pct=0.03)
print(f"  Selected: {len(to_repair)} repair, {len(to_fix)} fix")

# Batched repair with CPU state backup
batch_size = 5
n_rep = 0
n_rolled_back = 0
if to_repair:
    for b_start in range(0, len(to_repair), batch_size):
        batch = to_repair[b_start:b_start + batch_size]
        state_backup = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        genome.repair_genes(batch)

        model.eval()
        check_stable = True
        with torch.no_grad():
            for k in range(min(4, ids.size(0))):
                out = model(ids[k:k+1], attention_mask=mask[k:k+1], labels=ids[k:k+1])
                if math.isnan(out.loss.item()) or math.isinf(out.loss.item()):
                    check_stable = False
                    break
        if check_stable:
            n_rep += len(batch)
            del state_backup
        else:
            model.load_state_dict(state_backup)
            del state_backup
            n_rolled_back += len(batch)
            break
    print(f"  Repaired: {n_rep}, rolled back: {n_rolled_back}")
else:
    print("  No genes above threshold -- skipping repair")

# Fix genes
if to_fix:
    genome.fix_genes(to_fix)
    print(f"  Fixed: {len(to_fix)} genes")

# Snapshot for next cycle
genome.snapshot()

# Evaluate
model.eval()
ppls = {}
for name, enc in [("general", enc_general), ("code", enc_eval)]:
    ppl = evaluate_ppl(model, enc, device)
    ppls[name] = ppl
gm = geometric_mean(list(ppls.values()))
ppl_str = " | ".join(f"{n}:{v:.1f}" for n, v in ppls.items())
print(f"  PPL: {ppl_str} | GM: {gm:.1f}")

# ========== SECOND DOMAIN (multi-objective) ==========
print()
print("--- Domain 2 (multi-objective scoring) ---")
enc_legal = {"input_ids": torch.randint(0, 1000, (8, 32)),
             "attention_mask": torch.ones(8, 32, dtype=torch.long)}

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for j in range(5):
    optimizer.zero_grad()
    out = model(enc_legal["input_ids"][j:j+1], labels=enc_legal["input_ids"][j:j+1])
    out.loss.backward()
    optimizer.step()
print(f"  Trained 5 steps, final loss={out.loss.item():.4f}")
del optimizer

genome.sync_primary()
scorer = GeneScorer(genome, model, device, use_amp=False, streaming=True)
prev_evals_2 = [("general", enc_general), ("code", enc_eval)]
scores_2 = scorer.score_multi_objective(prev_evals_2, enc_legal, threshold=0.80, alpha=0.3)
print(f"  Scored {len(scores_2)} genes with {len(prev_evals_2)} previous objectives")
for s in scores_2:
    assert 0 <= s["p_del_prev"] <= 1 and 0 <= s["p_ben_curr"] <= 1
print("  All probabilities in [0,1]")

# ========== LoRA PATH ==========
print()
print("--- LoRA ---")
from peft import get_peft_model, LoraConfig, TaskType

model2 = GPT2LMHeadModel(config)
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
                         lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
peft_model = get_peft_model(model2, lora_config)
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

peft_model.train()
optimizer = torch.optim.AdamW(
    [p for p in peft_model.parameters() if p.requires_grad], lr=5e-5)
for j in range(5):
    optimizer.zero_grad()
    out = peft_model(ids[j:j+1], labels=ids[j:j+1])
    out.loss.backward()
    optimizer.step()
print(f"  Trained 5 steps, final loss={out.loss.item():.4f}")

# Merge and re-wrap (tests PEFT cleanup)
model2 = peft_model.merge_and_unload()
del optimizer, peft_model
print(f"  Merged. Model type: {type(model2).__name__}")

# Re-wrap with fresh LoRA (simulates domain 2+)
peft_model2 = get_peft_model(model2, lora_config)
peft_model2.train()
opt2 = torch.optim.AdamW([p for p in peft_model2.parameters() if p.requires_grad], lr=5e-5)
opt2.zero_grad()
out = peft_model2(ids[0:1], labels=ids[0:1])
out.loss.backward()
opt2.step()
model2 = peft_model2.merge_and_unload()
del opt2, peft_model2
print("  Re-wrapped, trained 1 step, merged again: OK")

model2.eval()
with torch.no_grad():
    out = model2(ids[:1], labels=ids[:1])
    print(f"  Final eval loss: {out.loss.item():.4f}")
    assert not math.isnan(out.loss.item())

# ========== EDGE CASES ==========
print()
print("--- Edge Cases ---")

# All scores below threshold
all_low_scores = [{"gene_id": i, "p_del_prev": 0.3, "p_ben_curr": 0.5}
                  for i in range(genome.total_genes)]
to_rep, to_fx = genome.select_conversion_genes(all_low_scores, threshold=0.80)
assert len(to_rep) == 0
print(f"  All-low-threshold: 0 repair, {len(to_fx)} fix")

# All above threshold (capped)
all_high_scores = [{"gene_id": i, "p_del_prev": 0.99, "p_ben_curr": 0.01}
                   for i in range(genome.total_genes)]
to_rep, to_fx = genome.select_conversion_genes(
    all_high_scores, threshold=0.80, max_repair_pct=0.03)
max_n = max(int(genome.total_genes * 0.03), 1)
assert len(to_rep) == max_n
print(f"  All-high-threshold: {len(to_rep)} repair (capped at {max_n})")

# 2-sample scoring (heuristic noise path)
enc_tiny = {"input_ids": torch.randint(0, 1000, (2, 32)),
            "attention_mask": torch.ones(2, 32, dtype=torch.long)}
scores_tiny, noise_tiny = scorer._score_split_half(enc_tiny)
assert len(scores_tiny) == genome.total_genes
assert noise_tiny > 0
print(f"  2-sample scoring: noise_var={noise_tiny:.2e} (heuristic)")

# geometric_mean edge cases
assert abs(geometric_mean([4.0, 9.0]) - 6.0) < 0.01
assert geometric_mean([float("inf"), float("inf")]) == float("inf")
assert abs(geometric_mean([4.0, float("inf"), 9.0]) - 6.0) < 0.01
print("  geometric_mean edge cases: OK")

# JSON with NaN
import json
data = {"final_loss": float("nan"), "ppls": {"gen": 5.2}}
json_str = json.dumps(data, default=str)
print(f"  JSON with NaN: serializes OK")

print()
print("=== ALL CHECKS PASSED ===")
