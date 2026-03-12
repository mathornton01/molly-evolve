"""
Full comparison experiment: Gene Conversion vs LoRA on LLaMA-2-7B.

Runs on real HuggingFace datasets with detailed logging, checkpointing,
and model saving for Ollama deployment.

Usage:
    python experiments/full_comparison.py \
        --model meta-llama/Llama-2-7b-hf \
        --output /workspace/results \
        --epochs 3 --n-train 200 --n-eval 50 --max-length 256
"""

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import torch
import numpy as np

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("experiment")


def gpu_stats():
    """Get current GPU memory stats."""
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        "gpu_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def geometric_mean(values):
    values = [v for v in values if v > 0 and not math.isnan(v) and not math.isinf(v)]
    if not values:
        return float('inf')
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved: {path}")


def save_model_hf(model, tokenizer, path):
    """Save model in HuggingFace format."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    logger.info(f"Model saved: {path}")


def create_modelfile(model_path, output_path):
    """Create Ollama Modelfile for importing."""
    modelfile = f"""FROM {model_path}

TEMPLATE \"\"\"{{{{- if .System }}}}
<s>[INST] <<SYS>>
{{{{ .System }}}}
<</SYS>>
{{{{- end }}}}

{{{{ .Prompt }}}} [/INST] {{{{ .Response }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
"""
    path = os.path.join(output_path, "Modelfile")
    with open(path, "w") as f:
        f.write(modelfile)
    logger.info(f"Ollama Modelfile: {path}")


def evaluate_ppl(model, enc, device):
    """Compute perplexity, handling overflow and NaN gracefully."""
    model.eval()
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    total_loss = 0
    n_batches = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, ids.size(0), 4):  # batch of 4 for eval
            batch_ids = ids[i:i+4]
            batch_mask = mask[i:i+4]
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(batch_ids, attention_mask=batch_mask, labels=batch_ids)
            else:
                out = model(batch_ids, attention_mask=batch_mask, labels=batch_ids)
            loss_val = out.loss.item()
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
                n_batches += 1
    if n_batches == 0:
        return float('inf')
    avg_loss = total_loss / n_batches
    ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    return ppl


def run_gene_conv(model_name, tokenizer, domain_data, all_eval_sets, domains,
                  output_dir, device, args):
    """Run gene conversion method with full logging."""
    from molly_evolution.genome import DualGenome
    from molly_evolution.scoring import GeneScorer
    from transformers import AutoModelForCausalLM

    method_dir = os.path.join(output_dir, "gene-conv")
    os.makedirs(method_dir, exist_ok=True)

    results = {
        "method": "gene-conv",
        "model": model_name,
        "config": {"epochs": args.epochs, "lr": args.lr, "max_length": args.max_length,
                    "n_train": args.n_train, "n_eval": args.n_eval,
                    "threshold": args.gc_threshold, "alpha": args.gc_alpha,
                    "max_repair_pct": args.gc_max_repair_pct,
                    "repair_batch_size": args.gc_batch_size},
        "domains": [],
        "timing": {},
        "gpu_memory": {},
    }

    # Load model
    logger.info("=" * 60)
    logger.info("  GENE CONVERSION")
    logger.info("=" * 60)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1e9:
        model.gradient_checkpointing_enable()
    results["timing"]["load_model"] = time.perf_counter() - t0
    results["gpu_memory"]["after_load"] = gpu_stats()
    logger.info(f"Model loaded in {results['timing']['load_model']:.1f}s")
    logger.info(f"GPU: {gpu_stats()}")

    # Build genome
    t0 = time.perf_counter()
    genome = DualGenome(model, granularity="head", backend="python")
    results["n_genes"] = genome.total_genes
    results["n_params"] = n_params
    logger.info(f"Genome: {genome.total_genes} genes, {n_params/1e9:.1f}B params")

    # Snapshot
    genome.snapshot()
    scorer = GeneScorer(genome, model, device, use_amp=True, streaming=True)
    results["timing"]["snapshot"] = time.perf_counter() - t0
    results["gpu_memory"]["after_snapshot"] = gpu_stats()
    logger.info(f"Snapshot: {results['timing']['snapshot']:.2f}s")

    # Train each domain
    for i, domain in enumerate(domains):
        logger.info(f"\n{'-' * 60}")
        logger.info(f"  Domain {i+1}/{len(domains)}: {domain}")
        logger.info(f"{'-' * 60}")

        domain_result = {
            "domain": domain,
            "step": i + 1,
            "timing": {},
            "train": {},
            "scoring": {},
            "conversion": {},
            "perplexities": {},
        }

        train_enc = domain_data[domain][0]
        eval_enc = domain_data[domain][1]

        # Training
        t0 = time.perf_counter()
        model.train()

        if device.type == "cuda":
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)
                logger.info(f"  Using 8-bit Adam")
            except ImportError:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        ids = train_enc["input_ids"].to(device)
        mask = train_enc["attention_mask"].to(device)

        step_losses = []
        nan_streak = 0
        training_aborted = False
        use_amp_train = device.type == "cuda"
        for epoch in range(args.epochs):
            epoch_loss = 0
            n_steps = 0
            for j in range(0, ids.size(0)):
                optimizer.zero_grad()
                if use_amp_train:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(ids[j:j+1], attention_mask=mask[j:j+1],
                                    labels=ids[j:j+1])
                else:
                    out = model(ids[j:j+1], attention_mask=mask[j:j+1],
                                labels=ids[j:j+1])
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_val = out.loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    nan_streak += 1
                    if nan_streak >= 20:
                        logger.error(f"  ABORT: {nan_streak} consecutive NaN/inf losses")
                        training_aborted = True
                        break
                else:
                    nan_streak = 0
                    epoch_loss += loss_val
                    n_steps += 1
                step_losses.append(loss_val)

            if n_steps == 0:
                logger.warning(f"  Epoch {epoch+1}/{args.epochs}: no valid steps (all NaN/inf)")
            else:
                avg = epoch_loss / n_steps
                logger.info(f"  Epoch {epoch+1}/{args.epochs}: loss={avg:.4f}")
            if training_aborted:
                break

        del optimizer
        torch.cuda.empty_cache()

        train_time = time.perf_counter() - t0
        domain_result["timing"]["train"] = train_time
        domain_result["train"]["step_losses"] = [l for l in step_losses if not math.isnan(l)]
        domain_result["train"]["final_loss"] = step_losses[-1] if step_losses else 0
        domain_result["train"]["n_steps"] = len(step_losses)
        domain_result["train"]["aborted"] = training_aborted
        logger.info(f"  Training: {train_time:.1f}s, {len(step_losses)} steps")
        logger.info(f"  GPU after train: {gpu_stats()}")

        # Post-training stability check
        model.eval()
        post_train_stable = True
        use_amp_check = device.type == "cuda"
        with torch.no_grad():
            for k in range(min(2, ids.size(0))):
                if use_amp_check:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(ids[k:k+1], attention_mask=mask[k:k+1],
                                    labels=ids[k:k+1])
                else:
                    out = model(ids[k:k+1], attention_mask=mask[k:k+1],
                                labels=ids[k:k+1])
                if math.isnan(out.loss.item()) or math.isinf(out.loss.item()):
                    post_train_stable = False
                    break

        if not post_train_stable or training_aborted:
            logger.error(f"  MODEL UNSTABLE after training domain '{domain}' — "
                         f"skipping scoring/repair for this domain")
            domain_result["conversion"] = {
                "n_repaired": 0, "n_fixed": 0,
                "n_candidates": 0, "n_rolled_back": 0,
                "skipped": True, "reason": "model_unstable_after_training",
            }
            # Still evaluate PPL to record the state
            t0 = time.perf_counter()
            ppls = {}
            for name, enc in all_eval_sets:
                ppl = evaluate_ppl(model, enc, device)
                ppls[name] = ppl
            domain_result["timing"]["eval"] = time.perf_counter() - t0
            domain_result["perplexities"] = ppls
            domain_result["gm_ppl"] = geometric_mean(list(ppls.values()))
            domain_result["gpu_memory"] = gpu_stats()
            results["domains"].append(domain_result)
            save_json(results, os.path.join(method_dir, "results.json"))
            continue

        # Scoring
        t0 = time.perf_counter()
        prev_evals = [(n, e) for n, e in all_eval_sets
                      if n != domain and n in ["general"] + domains[:i]]
        if not prev_evals:
            prev_evals = [all_eval_sets[0]]

        scores = scorer.score_multi_objective(
            prev_evals, eval_enc, threshold=args.gc_threshold,
            alpha=args.gc_alpha)

        score_time = time.perf_counter() - t0
        domain_result["timing"]["score"] = score_time
        domain_result["scoring"]["n_objectives"] = len(prev_evals) + 1
        # Save gene scores for analysis (scores is a list of dicts)
        domain_result["scoring"]["gene_scores"] = {
            "p_del_max": [s["p_del_prev"] for s in scores],
            "p_ben": [s["p_ben_curr"] for s in scores],
        }
        logger.info(f"  Scoring: {score_time:.1f}s")

        # Gradual batched conversion with NaN stability checks
        t0 = time.perf_counter()

        to_repair, to_fix = genome.select_conversion_genes(
            scores, threshold=args.gc_threshold, alpha=args.gc_alpha,
            max_repair_pct=args.gc_max_repair_pct)

        # Prepare multi-sample NaN test inputs (use up to 4 diverse samples)
        n_test = min(4, train_enc["input_ids"].size(0))
        test_ids_all = train_enc["input_ids"][:n_test].to(device)
        test_mask_all = train_enc["attention_mask"][:n_test].to(device)

        use_amp = device.type == "cuda"

        def check_model_stable():
            """Test model stability on multiple samples."""
            model.eval()
            with torch.no_grad():
                for k in range(n_test):
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            out = model(test_ids_all[k:k+1],
                                        attention_mask=test_mask_all[k:k+1],
                                        labels=test_ids_all[k:k+1])
                    else:
                        out = model(test_ids_all[k:k+1],
                                    attention_mask=test_mask_all[k:k+1],
                                    labels=test_ids_all[k:k+1])
                    if math.isnan(out.loss.item()) or math.isinf(out.loss.item()):
                        return False
            return True

        # Apply repairs in small batches with NaN check after each
        batch_size = args.gc_batch_size
        n_rep = 0
        n_rep_rolled_back = 0
        if to_repair:
            logger.info(f"  Repairing {len(to_repair)} genes in batches of {batch_size}...")
            for b_start in range(0, len(to_repair), batch_size):
                batch = to_repair[b_start:b_start + batch_size]
                state_backup = copy.deepcopy(model.state_dict())
                genome.repair_genes(batch)

                if check_model_stable():
                    n_rep += len(batch)
                    del state_backup
                    logger.info(f"    batch {b_start//batch_size + 1}: "
                                f"repaired {len(batch)} genes OK "
                                f"(total {n_rep})")
                else:
                    model.load_state_dict(state_backup)
                    del state_backup
                    n_rep_rolled_back += len(batch)
                    logger.warning(f"    batch {b_start//batch_size + 1}: "
                                   f"NaN detected, rolled back {len(batch)} genes")
                    # Stop further repairs — model is at its stability limit
                    remaining = len(to_repair) - b_start - len(batch)
                    if remaining > 0:
                        logger.warning(f"    stopping early, {remaining} genes skipped")
                    break

        # Apply fixes (these are low-risk: just updating reference)
        n_fix = 0
        if to_fix:
            genome.fix_genes(to_fix)
            n_fix = len(to_fix)

        genome.snapshot()
        scorer = GeneScorer(genome, model, device, use_amp=True, streaming=True)
        convert_time = time.perf_counter() - t0
        domain_result["timing"]["convert"] = convert_time
        domain_result["conversion"]["n_repaired"] = n_rep
        domain_result["conversion"]["n_fixed"] = n_fix
        domain_result["conversion"]["n_candidates"] = len(to_repair)
        domain_result["conversion"]["n_rolled_back"] = n_rep_rolled_back
        logger.info(f"  Conversion: {n_rep} repaired, {n_fix} fixed, "
                    f"{n_rep_rolled_back} rolled back, {convert_time:.1f}s")

        # Evaluate all domains
        t0 = time.perf_counter()
        ppls = {}
        for name, enc in all_eval_sets:
            ppl = evaluate_ppl(model, enc, device)
            ppls[name] = ppl
        eval_time = time.perf_counter() - t0
        domain_result["timing"]["eval"] = eval_time
        domain_result["perplexities"] = ppls

        gm = geometric_mean(list(ppls.values()))
        domain_result["gm_ppl"] = gm

        ppl_str = " | ".join(f"{n}:{v:.1f}" for n, v in ppls.items())
        logger.info(f"  PPL: {ppl_str}")
        logger.info(f"  GM: {gm:.1f}")
        logger.info(f"  Domain total: {sum(domain_result['timing'].values()):.1f}s")

        domain_result["gpu_memory"] = gpu_stats()
        results["domains"].append(domain_result)

        # Checkpoint after each domain
        save_json(results, os.path.join(method_dir, "results.json"))

    results["timing"]["total"] = time.perf_counter() - t_total
    results["final_gm"] = results["domains"][-1]["gm_ppl"] if results["domains"] else 0

    # Save final model
    logger.info("\nSaving gene-conv model...")
    model_save_path = os.path.join(method_dir, "model")
    save_model_hf(model, tokenizer, model_save_path)
    create_modelfile(model_save_path, method_dir)

    # Save final results
    save_json(results, os.path.join(method_dir, "results.json"))

    logger.info(f"\nGene-conv complete: GM={results['final_gm']:.1f}, "
                f"total={results['timing']['total']:.1f}s")

    # Cleanup
    del model, genome, scorer
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_lora(model_name, tokenizer, domain_data, all_eval_sets, domains,
             output_dir, device, args):
    """Run LoRA method with full logging."""
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    method_dir = os.path.join(output_dir, "lora")
    os.makedirs(method_dir, exist_ok=True)

    results = {
        "method": "lora",
        "model": model_name,
        "config": {"epochs": args.epochs, "lr": args.lr, "max_length": args.max_length,
                    "n_train": args.n_train, "n_eval": args.n_eval,
                    "rank": 8, "lora_alpha": 32},
        "domains": [],
        "timing": {},
        "gpu_memory": {},
    }

    logger.info("\n" + "=" * 60)
    logger.info("  LoRA")
    logger.info("=" * 60)
    t_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1e9:
        model.gradient_checkpointing_enable()
    results["timing"]["load_model"] = time.perf_counter() - t0
    results["n_params"] = n_params
    results["gpu_memory"]["after_load"] = gpu_stats()
    logger.info(f"Model loaded in {results['timing']['load_model']:.1f}s")

    # Detect target modules
    param_names = [n for n, _ in model.named_parameters()]
    if any("q_proj" in n for n in param_names):
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["c_attn", "c_proj"]

    for i, domain in enumerate(domains):
        logger.info(f"\n{'-' * 60}")
        logger.info(f"  Domain {i+1}/{len(domains)}: {domain}")
        logger.info(f"{'-' * 60}")

        domain_result = {
            "domain": domain,
            "step": i + 1,
            "timing": {},
            "train": {},
            "perplexities": {},
        }

        train_enc = domain_data[domain][0]

        # Create fresh LoRA adapter
        t0 = time.perf_counter()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8, lora_alpha=32, lora_dropout=0.05,
            target_modules=target_modules,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.train()

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        domain_result["train"]["trainable_params"] = trainable
        domain_result["train"]["trainable_pct"] = round(100 * trainable / total, 2)
        logger.info(f"  LoRA trainable: {trainable:,} / {total:,} "
                    f"({100*trainable/total:.2f}%)")

        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad], lr=args.lr)

        ids = train_enc["input_ids"].to(device)
        mask = train_enc["attention_mask"].to(device)

        step_losses = []
        nan_streak = 0
        lora_aborted = False
        use_amp_lora = device.type == "cuda"
        for epoch in range(args.epochs):
            epoch_loss = 0
            n_steps = 0
            for j in range(0, ids.size(0)):
                optimizer.zero_grad()
                if use_amp_lora:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = peft_model(ids[j:j+1], attention_mask=mask[j:j+1],
                                         labels=ids[j:j+1])
                else:
                    out = peft_model(ids[j:j+1], attention_mask=mask[j:j+1],
                                     labels=ids[j:j+1])
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                optimizer.step()
                loss_val = out.loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    nan_streak += 1
                    if nan_streak >= 20:
                        logger.error(f"  ABORT LoRA: {nan_streak} consecutive NaN/inf losses")
                        lora_aborted = True
                        break
                else:
                    nan_streak = 0
                    epoch_loss += loss_val
                    n_steps += 1
                step_losses.append(loss_val)

            if n_steps == 0:
                logger.warning(f"  Epoch {epoch+1}/{args.epochs}: no valid steps (all NaN/inf)")
            else:
                avg = epoch_loss / n_steps
                logger.info(f"  Epoch {epoch+1}/{args.epochs}: loss={avg:.4f}")
            if lora_aborted:
                break

        # Merge adapter
        model = peft_model.merge_and_unload()
        if n_params > 1e9:
            model.gradient_checkpointing_enable()
        del optimizer, peft_model
        torch.cuda.empty_cache()

        train_time = time.perf_counter() - t0
        domain_result["timing"]["train"] = train_time
        domain_result["train"]["step_losses"] = [l for l in step_losses
                                                     if not math.isnan(l) and not math.isinf(l)]
        domain_result["train"]["final_loss"] = step_losses[-1] if step_losses else 0
        domain_result["train"]["n_steps"] = len(step_losses)
        domain_result["train"]["aborted"] = lora_aborted
        logger.info(f"  Training + merge: {train_time:.1f}s")

        # Evaluate all domains
        t0 = time.perf_counter()
        ppls = {}
        for name, enc in all_eval_sets:
            ppl = evaluate_ppl(model, enc, device)
            ppls[name] = ppl
        eval_time = time.perf_counter() - t0
        domain_result["timing"]["eval"] = eval_time
        domain_result["perplexities"] = ppls

        gm = geometric_mean(list(ppls.values()))
        domain_result["gm_ppl"] = gm

        ppl_str = " | ".join(f"{n}:{v:.1f}" for n, v in ppls.items())
        logger.info(f"  PPL: {ppl_str}")
        logger.info(f"  GM: {gm:.1f}")

        domain_result["gpu_memory"] = gpu_stats()
        results["domains"].append(domain_result)

        # Checkpoint
        save_json(results, os.path.join(method_dir, "results.json"))

    results["timing"]["total"] = time.perf_counter() - t_total
    results["final_gm"] = results["domains"][-1]["gm_ppl"] if results["domains"] else 0

    # Save final model
    logger.info("\nSaving LoRA model...")
    model_save_path = os.path.join(method_dir, "model")
    save_model_hf(model, tokenizer, model_save_path)
    create_modelfile(model_save_path, method_dir)

    save_json(results, os.path.join(method_dir, "results.json"))

    logger.info(f"\nLoRA complete: GM={results['final_gm']:.1f}, "
                f"total={results['timing']['total']:.1f}s")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Full comparison experiment")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output", default="/workspace/results")
    parser.add_argument("--domains", default="code,legal,medical")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quicktest", action="store_true",
                        help="Use built-in data (no download)")
    parser.add_argument("--gc-threshold", type=float, default=0.80,
                        help="Gene conversion repair threshold (default 0.80)")
    parser.add_argument("--gc-alpha", type=float, default=0.3,
                        help="Gene conversion alpha weighting (default 0.3)")
    parser.add_argument("--gc-max-repair-pct", type=float, default=0.03,
                        help="Max fraction of genes to repair per domain (default 0.03)")
    parser.add_argument("--gc-batch-size", type=int, default=5,
                        help="Genes to repair per batch before NaN check (default 5)")
    args = parser.parse_args()

    device = torch.device(args.device)
    domains = [d.strip() for d in args.domains.split(",")]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Add file logging
    fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(fh)
    logging.getLogger("molly_evolution").addHandler(fh)

    logger.info("=" * 60)
    logger.info("  Full Comparison Experiment")
    logger.info("=" * 60)
    logger.info(f"  Model:   {args.model}")
    logger.info(f"  Domains: {' -> '.join(domains)}")
    logger.info(f"  Epochs:  {args.epochs}")
    logger.info(f"  N-train: {args.n_train}")
    logger.info(f"  N-eval:  {args.n_eval}")
    logger.info(f"  Max-len: {args.max_length}")
    logger.info(f"  Output:  {output_dir}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:     {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        logger.info(f"  GPU:     None (CPU mode)")
        logger.info(f"  VRAM:    N/A")
    logger.info(f"  GC:      threshold={args.gc_threshold}, alpha={args.gc_alpha}, "
                f"max_repair={args.gc_max_repair_pct*100:.0f}%, "
                f"batch={args.gc_batch_size}")
    logger.info("")

    # Save experiment config
    save_json(vars(args), os.path.join(output_dir, "config.json"))

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    from molly_evolution.data import load_domain_data
    logger.info("Loading data...")
    domain_data = {}
    for d in ["general"] + domains:
        train_enc, eval_enc = load_domain_data(
            d, tokenizer, max_length=args.max_length,
            n_train=args.n_train, n_eval=args.n_eval,
            quicktest=args.quicktest)
        domain_data[d] = (train_enc, eval_enc)

    all_eval_sets = [(d, domain_data[d][1]) for d in ["general"] + domains
                     if domain_data[d][1] is not None]

    # Save data stats
    data_stats = {}
    for d in ["general"] + domains:
        train, eval_ = domain_data[d]
        data_stats[d] = {
            "train_samples": train["input_ids"].shape[0] if train is not None else 0,
            "eval_samples": eval_["input_ids"].shape[0] if eval_ is not None else 0,
            "seq_length": train["input_ids"].shape[1] if train is not None else 0,
        }
    save_json(data_stats, os.path.join(output_dir, "data_stats.json"))

    # Run gene conversion
    gc_results = run_gene_conv(
        args.model, tokenizer, domain_data, all_eval_sets, domains,
        output_dir, device, args)

    # Run LoRA
    lora_results = run_lora(
        args.model, tokenizer, domain_data, all_eval_sets, domains,
        output_dir, device, args)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"  {'Method':<12s} {'Final GM':>10s} {'Time':>10s}")
    logger.info(f"  {'-'*12} {'-'*10} {'-'*10}")
    for r in [gc_results, lora_results]:
        logger.info(f"  {r['method']:<12s} {r['final_gm']:>10.1f} "
                    f"{r['timing']['total']:>9.1f}s")

    winner = gc_results if gc_results["final_gm"] < lora_results["final_gm"] else lora_results
    logger.info(f"\n  Winner: {winner['method']} (lowest GM perplexity)")

    # Full per-domain comparison
    if gc_results["domains"] and lora_results["domains"]:
        logger.info(f"\n  Per-domain perplexity after final domain:")
        all_domains = ["general"] + domains
        header = f"  {'Method':<12s}" + "".join(f" {d:>10s}" for d in all_domains) + f" {'GM':>10s}"
        logger.info(header)
        logger.info("  " + "-" * len(header))
        for r in [gc_results, lora_results]:
            ppls = r["domains"][-1]["perplexities"]
            row = f"  {r['method']:<12s}"
            for d in all_domains:
                v = ppls.get(d, float('nan'))
                row += f" {v:>10.1f}"
            row += f" {r['final_gm']:>10.1f}"
            logger.info(row)

    # Save combined summary
    gc_final_ppls = gc_results["domains"][-1]["perplexities"] if gc_results["domains"] else {}
    lora_final_ppls = lora_results["domains"][-1]["perplexities"] if lora_results["domains"] else {}
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "domains": domains,
        "gene_conv": {"final_gm": gc_results["final_gm"],
                       "total_time": gc_results["timing"]["total"],
                       "final_ppls": gc_final_ppls,
                       "config": gc_results["config"]},
        "lora": {"final_gm": lora_results["final_gm"],
                  "total_time": lora_results["timing"]["total"],
                  "final_ppls": lora_final_ppls},
        "winner": winner["method"],
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
        },
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))

    logger.info(f"\n  Results saved to: {output_dir}")
    logger.info(f"  Models saved to:")
    logger.info(f"    gene-conv: {output_dir}/gene-conv/model/")
    logger.info(f"    lora:      {output_dir}/lora/model/")
    logger.info(f"\n  To import to Ollama:")
    logger.info(f"    ollama create molly-gc -f {output_dir}/gene-conv/Modelfile")
    logger.info(f"    ollama create molly-lora -f {output_dir}/lora/Modelfile")
    logger.info(f"\n  Or convert to GGUF first:")
    logger.info(f"    python llama.cpp/convert_hf_to_gguf.py {output_dir}/gene-conv/model/")
    logger.info(f"    python llama.cpp/convert_hf_to_gguf.py {output_dir}/lora/model/")


if __name__ == "__main__":
    main()
