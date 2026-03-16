#!/usr/bin/env python3
"""
Extended molly-evolve comparison runner with:
  - Checkpointing (skip completed method+domain combos on restart)
  - Rich structured logging to JSON + text
  - Live status file for the dashboard
  - 5-domain and 6-domain experiments back-to-back
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/home/mathornton/micah/molly-evolve")
STATUS_FILE = BASE / "experiment_status.json"
VENV_PYTHON = str(BASE / "venv" / "bin" / "python")
EXPERIMENT_SCRIPT = str(BASE / "experiments" / "full_comparison.py")

COMMON_ARGS = [
    "--model", "Qwen/Qwen2.5-7B",
    "--quicktest",
    "--gc-max-repair-pct", "0.15",
    "--n-train", "100",
    "--n-eval", "49",
    "--max-length", "256",
]

EXPERIMENTS = [
    {
        "name": "5-domain",
        "domains": "code,legal,medical,science,finance",
        "output": str(BASE / "results_5domain"),
        "log": str(BASE / "5domain_results.txt"),
    },
    {
        "name": "6-domain",
        "domains": "code,legal,medical,science,finance,general",
        "output": str(BASE / "results_6domain"),
        "log": str(BASE / "6domain_results.txt"),
    },
]


def update_status(status: dict):
    """Atomically update the status file for the dashboard."""
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = str(STATUS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2, default=str)
    os.replace(tmp, str(STATUS_FILE))


def parse_results_json(output_dir: str) -> dict:
    """Try to read intermediate results from the experiment output."""
    results = {}
    for method in ["gene-conv", "lora", "qlora"]:
        rpath = os.path.join(output_dir, method, "results.json")
        if os.path.exists(rpath):
            try:
                with open(rpath) as f:
                    results[method] = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                results["_summary"] = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return results


def check_experiment_done(output_dir: str) -> bool:
    """Check if an experiment already completed (summary.json exists with winner)."""
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                s = json.load(f)
            return "winner" in s
        except (json.JSONDecodeError, IOError):
            pass
    return False


def tail_log(logfile: str, n: int = 5) -> list:
    """Get last n lines of a log file."""
    try:
        with open(logfile, "r") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except (IOError, FileNotFoundError):
        return []


def run_experiment(exp: dict, status: dict):
    """Run a single experiment with live status updates."""
    name = exp["name"]
    output = exp["output"]
    logfile = exp["log"]

    # Skip if already done
    if check_experiment_done(output):
        print(f"\n>>> {name} already complete, skipping.")
        status["experiments"][name] = {
            "status": "complete (cached)",
            "results": parse_results_json(output),
        }
        update_status(status)
        return

    print(f"\n{'='*60}")
    print(f"  {name.upper()} RUN: {exp['domains']}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    status["experiments"][name] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "domains": exp["domains"],
        "results": {},
    }
    status["current_experiment"] = name
    update_status(status)

    cmd = [
        VENV_PYTHON, EXPERIMENT_SCRIPT,
        *COMMON_ARGS,
        "--domains", exp["domains"],
        "--output", output,
    ]

    os.makedirs(output, exist_ok=True)

    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE),
        )

        last_status_update = 0
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

            # Update status every 10 seconds
            now = time.time()
            if now - last_status_update > 10:
                last_status_update = now
                status["experiments"][name]["results"] = parse_results_json(output)
                status["experiments"][name]["last_log"] = tail_log(logfile, 8)
                update_status(status)

        proc.wait()

    # Final update
    final_results = parse_results_json(output)
    status["experiments"][name]["status"] = "complete" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
    status["experiments"][name]["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["experiments"][name]["results"] = final_results
    status["experiments"][name]["last_log"] = tail_log(logfile, 8)
    update_status(status)


def main():
    status = {
        "title": "Molly Evolution — Extended Comparison",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "current_experiment": None,
        "experiments": {},
        "model": "Qwen/Qwen2.5-7B",
        "config": {
            "gc_max_repair_pct": 0.15,
            "n_train": 100,
            "n_eval": 49,
            "max_length": 256,
        },
    }
    update_status(status)

    print("=" * 60)
    print("  MOLLY EVOLUTION — EXTENDED COMPARISON")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(f"  Status file: {STATUS_FILE}")
    print(f"  Dashboard:   python {BASE}/dashboard.py")
    print("=" * 60)

    for exp in EXPERIMENTS:
        run_experiment(exp, status)

    status["current_experiment"] = None
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    update_status(status)

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
