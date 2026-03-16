#!/usr/bin/env python3
"""
Live terminal dashboard for molly-evolve experiments.
Reads experiment_status.json and displays progress.

Usage:
    python dashboard.py              # auto-refresh every 5s
    python dashboard.py --once       # print once and exit
    watch -n5 python dashboard.py --once   # alternative
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.live import Live
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

BASE = Path("/home/mathornton/micah/molly-evolve")
STATUS_FILE = BASE / "experiment_status.json"


def load_status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def format_duration(start_str, end_str=None):
    try:
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str) if end_str else datetime.now(timezone.utc)
        delta = end - start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except (ValueError, TypeError):
        return "—"


def extract_domain_ppls(results_dict):
    """Extract per-method, per-domain PPL progression."""
    rows = []
    for method in ["gene-conv", "lora", "qlora"]:
        if method not in results_dict:
            continue
        r = results_dict[method]
        domains = r.get("domains", [])
        if not domains:
            continue
        last = domains[-1]
        ppls = last.get("perplexities", {})
        gm = last.get("gm_ppl", None)
        total_time = r.get("timing", {}).get("total", None)
        n_done = len(domains)
        n_total = r.get("n_domains", n_done)
        rows.append({
            "method": method,
            "ppls": ppls,
            "gm": gm,
            "time": total_time,
            "progress": f"{n_done}/{n_total}" if n_total else str(n_done),
            "status": "done" if r.get("final_gm") else "running",
        })
    return rows


def render_rich(status):
    console = Console()
    console.clear()

    # Header
    title = status.get("title", "Molly Evolution Experiment")
    model = status.get("model", "?")
    started = status.get("started_at", "?")
    elapsed = format_duration(started, status.get("finished_at"))
    current = status.get("current_experiment")
    is_done = status.get("finished_at") is not None

    state_str = "[bold green]✓ COMPLETE[/]" if is_done else f"[bold yellow]⟳ RUNNING: {current}[/]"

    header = Text()
    header.append(f"  {title}\n", style="bold white")
    header.append(f"  Model: {model}  |  Elapsed: {elapsed}  |  ", style="dim")
    console.print(Panel(header, title=state_str, border_style="blue" if not is_done else "green"))

    # Per-experiment panels
    for exp_name, exp in status.get("experiments", {}).items():
        exp_status = exp.get("status", "?")
        domains = exp.get("domains", "?")
        started_at = exp.get("started_at")
        finished_at = exp.get("finished_at")

        if exp_status == "running":
            style = "yellow"
            icon = "⟳"
            elapsed_exp = format_duration(started_at) if started_at else "—"
        elif "complete" in exp_status:
            style = "green"
            icon = "✓"
            elapsed_exp = format_duration(started_at, finished_at) if started_at else "—"
        else:
            style = "red"
            icon = "✗"
            elapsed_exp = "—"

        # Results table
        results = exp.get("results", {})
        method_rows = extract_domain_ppls(results)

        if method_rows:
            table = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
            table.add_column("Method", style="bold", width=12)
            table.add_column("Progress", justify="center", width=10)

            # Collect all domain names
            all_domains = set()
            for row in method_rows:
                all_domains.update(row["ppls"].keys())
            domain_order = ["general", "code", "legal", "medical", "science", "finance"]
            domain_cols = [d for d in domain_order if d in all_domains]
            for d in sorted(all_domains):
                if d not in domain_cols:
                    domain_cols.append(d)

            for d in domain_cols:
                table.add_column(d[:6], justify="right", width=7)
            table.add_column("GM", justify="right", width=7, style="bold")
            table.add_column("Time", justify="right", width=9)

            # Find best GM for highlighting
            gms = [r["gm"] for r in method_rows if r["gm"] is not None]
            best_gm = min(gms) if gms else None

            for row in method_rows:
                vals = [row["method"], row["progress"]]
                for d in domain_cols:
                    v = row["ppls"].get(d)
                    vals.append(f"{v:.1f}" if v is not None else "—")
                gm_str = f"{row['gm']:.2f}" if row["gm"] is not None else "—"
                if row["gm"] == best_gm and best_gm is not None:
                    gm_str = f"[bold green]{gm_str}[/]"
                vals.append(gm_str)
                t = row.get("time")
                vals.append(f"{t:.0f}s" if t else "—")
                table.add_row(*vals)

            console.print(Panel(
                table,
                title=f"{icon} {exp_name} ({domains})  [{elapsed_exp}]",
                border_style=style,
            ))
        else:
            # No results yet — show log tail
            last_log = exp.get("last_log", [])
            log_text = "\n".join(last_log) if last_log else "  Waiting for results..."
            console.print(Panel(
                log_text,
                title=f"{icon} {exp_name} ({domains})  [{elapsed_exp}]",
                border_style=style,
            ))

        # Summary if available
        summary = results.get("_summary")
        if summary and "winner" in summary:
            winner = summary["winner"]
            console.print(f"    [bold green]Winner: {winner}[/]")

    # Log tail from current experiment
    if current and not is_done:
        exp = status.get("experiments", {}).get(current, {})
        last_log = exp.get("last_log", [])
        if last_log:
            console.print(Panel(
                "\n".join(last_log),
                title="Recent Log",
                border_style="dim",
            ))

    updated = status.get("updated_at", "?")
    console.print(f"\n  [dim]Last updated: {updated}  |  Refresh: 5s  |  Ctrl+C to exit[/]")


def render_plain(status):
    """Fallback for no-rich environments."""
    print("=" * 60)
    print(f"  {status.get('title', 'Experiment')}")
    print(f"  Status: {'COMPLETE' if status.get('finished_at') else 'RUNNING'}")
    print(f"  Elapsed: {format_duration(status.get('started_at', ''), status.get('finished_at'))}")
    print("=" * 60)

    for exp_name, exp in status.get("experiments", {}).items():
        print(f"\n--- {exp_name} ({exp.get('status', '?')}) ---")
        results = exp.get("results", {})
        for method in ["gene-conv", "lora", "qlora"]:
            if method in results:
                r = results[method]
                domains = r.get("domains", [])
                if domains:
                    last = domains[-1]
                    ppls = last.get("perplexities", {})
                    gm = last.get("gm_ppl", "?")
                    ppl_str = " | ".join(f"{k}:{v:.1f}" for k, v in ppls.items())
                    print(f"  {method:<12s}  {ppl_str}  GM={gm:.2f}" if isinstance(gm, float) else f"  {method}: {ppl_str}")

        last_log = exp.get("last_log", [])
        if last_log:
            print("  Log:")
            for l in last_log:
                print(f"    {l}")

    print(f"\nUpdated: {status.get('updated_at', '?')}")


def main():
    once = "--once" in sys.argv

    if once:
        status = load_status()
        if not status:
            print("No experiment status found. Run run_extended.py first.")
            return
        if HAS_RICH:
            render_rich(status)
        else:
            render_plain(status)
        return

    # Live refresh mode
    print("Watching experiment_status.json... (Ctrl+C to exit)")
    try:
        while True:
            status = load_status()
            if status:
                if HAS_RICH:
                    render_rich(status)
                else:
                    render_plain(status)
            else:
                print("Waiting for experiment to start...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
