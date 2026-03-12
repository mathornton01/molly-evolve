"""
molly — CLI for Molly Evolution continual learning.

Commands:
  molly evolve    Run continual learning on sequential domains
  molly compare   Compare methods (gene-conv, lora, qlora) side-by-side
  molly benchmark Speed and memory benchmark
  molly deploy    Generate deployment files for RunPod / Docker
  molly info      Show scaling estimates for a model size
"""

import argparse
import logging
import math
import os
import sys
import time

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    logging.getLogger("molly_evolution").setLevel(level)
    # Quiet noisy libraries
    for lib in ["transformers", "datasets", "urllib3", "httpx"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def geometric_mean(values):
    """Geometric mean of positive values."""
    values = [v for v in values if v > 0]
    if not values:
        return 0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


# =================================================================
# evolve command
# =================================================================

def cmd_evolve(args):
    """Run continual learning with a single method."""
    import torch
    from molly_evolution.methods import get_method
    from molly_evolution.data import load_domain_data
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    domains = [d.strip() for d in args.domains.split(",")]

    print(f"\n{'='*60}")
    print(f"  Molly Evolution — Continual Learning")
    print(f"{'='*60}")
    print(f"  Model:   {args.model}")
    print(f"  Method:  {args.method}")
    print(f"  Domains: {' -> '.join(domains)}")
    print(f"  Device:  {device}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load domain data
    print("Loading data...")
    domain_data = {}
    for d in domains:
        train_enc, eval_enc = load_domain_data(
            d, tokenizer, max_length=args.max_length,
            n_train=args.n_train, n_eval=args.n_eval,
            quicktest=args.quicktest)
        domain_data[d] = (train_enc, eval_enc)

    # Also load general eval if not in domains
    if "general" not in domains:
        _, gen_eval = load_domain_data(
            "general", tokenizer, max_length=args.max_length,
            n_train=1, n_eval=args.n_eval, quicktest=args.quicktest)
        domain_data["general"] = (None, gen_eval)

    # Initialize method
    learner = get_method(args.method, model_name=args.model, device=device)
    learner.load_model()

    # Build eval sets (always include general)
    all_eval_sets = []
    for d in ["general"] + domains:
        if d in domain_data and domain_data[d][1] is not None:
            all_eval_sets.append((d, domain_data[d][1]))

    # Initial snapshot
    learner.snapshot()

    # Train on each domain sequentially
    for i, domain in enumerate(domains):
        print(f"\n{'-'*60}")
        print(f"  Domain {i+1}/{len(domains)}: {domain}")
        print(f"{'-'*60}")

        train_enc, eval_enc = domain_data[domain]

        # Train
        train_metrics = learner.train_domain(
            train_enc, epochs=args.epochs, lr=args.lr)

        # Post-train (gene conversion, etc.)
        prev_evals = [(n, e) for n, e in all_eval_sets
                      if n != domain and n in ["general"] + domains[:i]]
        if not prev_evals:
            prev_evals = [all_eval_sets[0]]  # at least general

        post_metrics = learner.post_train(prev_evals, eval_enc)

        # Evaluate all domains
        print(f"\n  Perplexity after training on '{domain}':")
        ppls = learner.record_eval(domain, all_eval_sets)
        for name, ppl in ppls.items():
            marker = " <--" if name == domain else ""
            print(f"    {name:<15s} {ppl:>8.1f}{marker}")
        gm = geometric_mean(list(ppls.values()))
        print(f"    {'GM':<15s} {gm:>8.1f}")

    print(f"\n{'='*60}")
    print(f"  DONE — Final geometric mean PPL: {gm:.1f}")
    print(f"{'='*60}\n")


# =================================================================
# compare command
# =================================================================

def cmd_compare(args):
    """Compare multiple methods side-by-side."""
    import torch
    from molly_evolution.methods import get_method
    from molly_evolution.data import load_domain_data
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    domains = [d.strip() for d in args.domains.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    print(f"\n{'='*60}")
    print(f"  Molly Evolution — Method Comparison")
    print(f"{'='*60}")
    print(f"  Model:   {args.model}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Domains: {' -> '.join(domains)}")
    print(f"  Device:  {device}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load all data once
    print("Loading data...")
    domain_data = {}
    for d in ["general"] + domains:
        train_enc, eval_enc = load_domain_data(
            d, tokenizer, max_length=args.max_length,
            n_train=args.n_train, n_eval=args.n_eval,
            quicktest=args.quicktest)
        domain_data[d] = (train_enc, eval_enc)

    all_eval_sets = [(d, domain_data[d][1]) for d in ["general"] + domains
                     if domain_data[d][1] is not None]

    # Run each method
    results = {}
    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"  Running: {method_name}")
        print(f"{'='*60}")

        learner = get_method(method_name, model_name=args.model, device=device)
        learner.load_model()
        learner.snapshot()

        method_results = []
        total_time = 0

        for i, domain in enumerate(domains):
            print(f"\n  Domain {i+1}/{len(domains)}: {domain}")

            train_enc = domain_data[domain][0]
            eval_enc = domain_data[domain][1]

            t0 = time.perf_counter()
            learner.train_domain(train_enc, epochs=args.epochs, lr=args.lr)

            prev_evals = [(n, e) for n, e in all_eval_sets
                          if n != domain and n in ["general"] + domains[:i]]
            if not prev_evals:
                prev_evals = [all_eval_sets[0]]
            learner.post_train(prev_evals, eval_enc)
            dt = time.perf_counter() - t0
            total_time += dt

            ppls = learner.record_eval(domain, all_eval_sets)
            gm = geometric_mean(list(ppls.values()))
            method_results.append({"domain": domain, "ppls": ppls, "gm": gm})

            ppl_str = " | ".join(f"{n}:{v:.1f}" for n, v in ppls.items())
            print(f"    PPL: {ppl_str} | GM: {gm:.1f}")

        results[method_name] = {
            "steps": method_results,
            "final_gm": method_results[-1]["gm"],
            "total_time": total_time,
        }

        # Clean up GPU memory
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    print(f"  {'Method':<12s} {'Final GM':>10s} {'Time':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for method_name in methods:
        r = results[method_name]
        print(f"  {method_name:<12s} {r['final_gm']:>10.1f} "
              f"{r['total_time']:>9.1f}s")

    # Find winner
    winner = min(methods, key=lambda m: results[m]["final_gm"])
    print(f"\n  Winner: {winner} (lowest GM perplexity)")
    print()


# =================================================================
# benchmark command
# =================================================================

def cmd_benchmark(args):
    """Run speed/memory benchmark."""
    # Delegate to existing benchmark script
    benchmark_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                  "experiments", "benchmark_speed.py")
    if os.path.exists(benchmark_path):
        exec(open(benchmark_path).read())
    else:
        print("Benchmark script not found. Running inline benchmark...")
        import torch
        from molly_evolution.distributed import estimate_requirements
        n_params = int(args.params.replace("B", "e9").replace("M", "e6"))
        est = estimate_requirements(n_params)
        print(f"\nScaling estimates for {args.params} model:")
        for k, v in est.items():
            print(f"  {k}: {v}")


# =================================================================
# deploy command
# =================================================================

def cmd_deploy(args):
    """Generate deployment files."""
    if args.target == "runpod":
        _deploy_runpod(args)
    elif args.target == "docker":
        _deploy_docker(args)
    else:
        print(f"Unknown target: {args.target}. Use 'runpod' or 'docker'.")


def _deploy_runpod(args):
    """Generate RunPod deployment script."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "..",
                               "scripts", "deploy_runpod.py")
    if os.path.exists(script_path):
        print(f"RunPod deploy script: {script_path}")
        print(f"\nUsage:")
        print(f"  python {script_path} --model {args.model} --gpu {args.gpu}")
    else:
        print("Deploy script not found.")


def _deploy_docker(args):
    """Show Docker build instructions."""
    dockerfile = os.path.join(os.path.dirname(__file__), "..", "..", "Dockerfile")
    if os.path.exists(dockerfile):
        print(f"Dockerfile: {dockerfile}")
        print(f"\nBuild and run:")
        print(f"  docker build -t molly-evolution .")
        print(f"  docker run --gpus all -it molly-evolution \\")
        print(f"    molly evolve --model {args.model} --domains general,code,legal")
    else:
        print("Dockerfile not found.")


# =================================================================
# info command
# =================================================================

def cmd_info(args):
    """Show scaling estimates."""
    from molly_evolution.distributed import estimate_requirements

    sizes = {
        "gpt2": 124e6, "gpt2-medium": 345e6, "gpt2-large": 774e6,
        "gpt2-xl": 1.5e9, "llama-7b": 7e9, "llama-13b": 13e9,
        "llama-70b": 70e9,
    }

    model = args.model.lower()
    if model in sizes:
        n = int(sizes[model])
    else:
        try:
            n = int(float(model.replace("b", "e9").replace("m", "e6")))
        except ValueError:
            print(f"Unknown model: {args.model}")
            print(f"Known models: {', '.join(sizes.keys())}")
            print(f"Or specify param count: 7b, 13b, 1.5b, 345m")
            return

    est = estimate_requirements(n, n_objectives=args.objectives)

    print(f"\n{'='*50}")
    print(f"  Scaling Estimate: {est['n_params_billions']:.1f}B params")
    print(f"{'='*50}")
    print(f"  Model GPU (fp16):     {est['model_gpu_mb']:>8,} MB")
    print(f"  Training peak:        {est['training_peak_gpu_mb']:>8,} MB")
    print(f"  Scoring (streaming):  {est['scoring_streaming_gpu_mb']:>8,} MB")
    print(f"  Scoring (precomputed):{est['scoring_precomputed_gpu_mb']:>8,} MB")
    print(f"  Genome CPU:           {est['genome_cpu_mb']:>8,} MB")
    print(f"  Recommended GPUs:     {est['recommended_gpus']}")
    print(f"  Strategy:             {est['recommended_strategy']}")
    print()


# =================================================================
# Main parser
# =================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="molly",
        description="Molly Evolution — Continual learning for LLMs",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # evolve
    p_evolve = sub.add_parser("evolve", help="Run continual learning")
    p_evolve.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    p_evolve.add_argument("--method", default="gene-conv",
                          choices=["gene-conv", "lora", "qlora"])
    p_evolve.add_argument("--domains", default="code,legal,medical",
                          help="Comma-separated domain list")
    p_evolve.add_argument("--epochs", type=int, default=3)
    p_evolve.add_argument("--lr", type=float, default=5e-5)
    p_evolve.add_argument("--max-length", type=int, default=256)
    p_evolve.add_argument("--n-train", type=int, default=100)
    p_evolve.add_argument("--n-eval", type=int, default=50)
    p_evolve.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    p_evolve.add_argument("--quicktest", action="store_true",
                          help="Use built-in data (no download)")
    p_evolve.set_defaults(func=cmd_evolve)

    # compare
    p_compare = sub.add_parser("compare", help="Compare methods")
    p_compare.add_argument("--model", default="gpt2")
    p_compare.add_argument("--methods", default="gene-conv,lora",
                           help="Comma-separated methods")
    p_compare.add_argument("--domains", default="code,legal,medical")
    p_compare.add_argument("--epochs", type=int, default=3)
    p_compare.add_argument("--lr", type=float, default=5e-5)
    p_compare.add_argument("--max-length", type=int, default=256)
    p_compare.add_argument("--n-train", type=int, default=100)
    p_compare.add_argument("--n-eval", type=int, default=50)
    p_compare.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    p_compare.add_argument("--quicktest", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Speed/memory benchmark")
    p_bench.add_argument("--params", default="124M",
                         help="Model size (e.g., 7B, 13B, 124M)")
    p_bench.set_defaults(func=cmd_benchmark)

    # deploy
    p_deploy = sub.add_parser("deploy", help="Deploy to RunPod / Docker")
    p_deploy.add_argument("target", choices=["runpod", "docker"],
                          help="Deployment target")
    p_deploy.add_argument("--model", default="gpt2")
    p_deploy.add_argument("--gpu", default="a100-80gb")
    p_deploy.set_defaults(func=cmd_deploy)

    # info
    p_info = sub.add_parser("info", help="Scaling estimates for a model")
    p_info.add_argument("model", help="Model name or param count (e.g., 7b)")
    p_info.add_argument("--objectives", type=int, default=3)
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
