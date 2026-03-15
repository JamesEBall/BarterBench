#!/usr/bin/env python3
"""Backfill process metrics into existing results.json entries.

Run once to add TER, ISS, OER, TRR to all existing runs.
Safe to re-run — skips entries that already have process_metrics unless --force.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scoring import (
    compute_trade_efficiency_ratio,
    compute_information_security_score,
    compute_offer_execution_rate,
    compute_trade_relevance_rate,
)

RESULTS_FILES = [
    Path(__file__).parent / "results.json",
    Path(__file__).parent / "benchmark_results.json",
    Path(__file__).parent / "arena" / "results.json",
]

METRIC_NAMES = {
    "trade_efficiency_ratio": "TER",
    "information_security_score": "ISS",
    "offer_execution_rate": "OER",
    "trade_relevance_rate": "TRR",
}
# ISS_active uses per_model_active instead of per_model — shown alongside ISS
ISS_ACTIVE_KEY = "information_security_score"


def compute_process_metrics(entry):
    process_metrics = {}
    ter = compute_trade_efficiency_ratio(entry)
    if ter:
        process_metrics["trade_efficiency_ratio"] = ter
    iss = compute_information_security_score(entry)
    if iss:
        process_metrics["information_security_score"] = iss
    oer = compute_offer_execution_rate(entry)
    if oer:
        process_metrics["offer_execution_rate"] = oer
    trr = compute_trade_relevance_rate(entry)
    if trr:
        process_metrics["trade_relevance_rate"] = trr
    return process_metrics


def backfill_file(path, force=False):
    if not path.exists():
        return 0, 0

    with open(path) as f:
        data = json.load(f)

    updated = 0
    skipped = 0
    for entry in data:
        if entry.get("error"):
            continue
        if entry.get("process_metrics") and not force:
            skipped += 1
            continue
        # Skip obviously failed runs (no trades, no meaningful activity)
        if entry.get("num_trades", 0) == 0 and entry.get("avg_goal_completion", 0) == 0:
            skipped += 1
            continue

        pm = compute_process_metrics(entry)
        if pm:
            entry["process_metrics"] = pm
            updated += 1

    if updated:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return updated, skipped


def print_summary(path):
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)

    entries_with_pm = [d for d in data if d.get("process_metrics")]
    if not entries_with_pm:
        return

    print(f"\n{'─'*60}")
    print(f"  {path.name}  ({len(entries_with_pm)} entries with process metrics)")
    print(f"{'─'*60}")

    # Aggregate per-model across all entries
    model_sums = {k: {} for k in METRIC_NAMES}
    model_cnts = {k: {} for k in METRIC_NAMES}

    for entry in entries_with_pm:
        pm = entry["process_metrics"]
        for key in METRIC_NAMES:
            if key not in pm:
                continue
            per_model = pm[key].get("per_model", {})
            for model, val in per_model.items():
                if val is None:
                    continue
                model_sums[key].setdefault(model, 0)
                model_sums[key][model] += val
                model_cnts[key].setdefault(model, 0)
                model_cnts[key][model] += 1

    models = sorted(set(
        m for k in METRIC_NAMES for m in model_sums[k]
    ))

    # Also aggregate ISS_active (per_model_active field)
    iss_active_sums = {}
    iss_active_cnts = {}
    for entry in entries_with_pm:
        pm = entry["process_metrics"]
        iss = pm.get(ISS_ACTIVE_KEY, {})
        for model, val in iss.get("per_model_active", {}).items():
            if val is None:
                continue
            iss_active_sums.setdefault(model, 0)
            iss_active_sums[model] += val
            iss_active_cnts.setdefault(model, 0)
            iss_active_cnts[model] += 1

    col_labels = list(METRIC_NAMES.values()) + ["ISS_act"]
    header = f"  {'Model':<20s}"
    for label in col_labels:
        header += f" {label:>8s}"
    print(header)
    print(f"  {'─'*20}" + "─" * (9 * len(col_labels)))

    for model in models:
        row = f"  {model:<20s}"
        for key, label in METRIC_NAMES.items():
            cnt = model_cnts[key].get(model, 0)
            if cnt > 0:
                avg = model_sums[key][model] / cnt
                if label in ("TER",):
                    row += f" {avg:>8.3f}"
                else:
                    row += f" {avg*100:>7.1f}%"
            else:
                row += f" {'N/A':>8s}"
        # ISS_active column
        cnt_a = iss_active_cnts.get(model, 0)
        if cnt_a > 0:
            row += f" {iss_active_sums[model] / cnt_a * 100:>7.1f}%"
        else:
            row += f" {'N/A':>8s}"
        print(row)


if __name__ == "__main__":
    force = "--force" in sys.argv
    total_updated = 0

    for results_file in RESULTS_FILES:
        updated, skipped = backfill_file(results_file, force=force)
        if updated or skipped:
            print(f"{results_file.name}: updated={updated}, skipped={skipped}")
        total_updated += updated

    print(f"\nTotal entries updated: {total_updated}")

    for results_file in RESULTS_FILES:
        print_summary(results_file)
