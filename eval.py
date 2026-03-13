"""CLI entry point: run marketplace eval across scenarios."""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent import MarketAgent, RandomAgent
from engine import MarketEngine
from elo import record_match, print_elo_leaderboard, reset_ratings, load_ratings, file_lock
from bradley_terry import compute_bt_ratings, print_bt_leaderboard, reset_bt
from scoring import compute_metrics, compute_cost_efficiency
from model_registry import get_model_info, compute_dollar_cost, get_size_tier

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_FILE = Path(__file__).parent / "results.json"
RUNS_DIR = Path(__file__).parent / "runs"
BENCHMARK_RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"
EXPERIMENTS_FILE = Path(__file__).parent / "experiments.json"


# ---- File locking for parallel runs ----

def _locked_append_result(entry, results_file):
    """Thread-safe append of a result entry to a JSON array file."""
    with file_lock(results_file):
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
        else:
            results = []
        results.append(entry)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    return results


# ---- Scenario helpers ----

def load_scenario(name):
    """Load a single scenario JSON file."""
    f = SCENARIOS_DIR / f"{name}.json"
    if not f.exists():
        print(f"Error: scenario '{name}' not found at {f}")
        sys.exit(1)
    with open(f) as fh:
        return json.load(fh)


def list_scenarios():
    """List all available scenarios."""
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    scenarios = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            if data["agents"][0].get("target") is not None:
                scenarios.append(data["name"])
    return scenarios


# ---- Model config helpers ----

def parse_model_config(config_str):
    """Parse 'haiku:3,opus:3' into [('haiku', 3), ('opus', 3)]."""
    parts = config_str.split(",")
    config = []
    for p in parts:
        if ":" in p:
            model, count = p.split(":")
            config.append((model.strip(), int(count.strip())))
        else:
            config.append((p.strip(), 1))
    return config


def auto_model_config(scenario):
    """Split agents 50/50 for a 2-way matchup."""
    n = len(scenario["agents"])
    half = n // 2
    return half, n - half


def assign_models(model_config, num_agents):
    """Assign models to agent slots with stratified pairing.

    For 2-model matchups, paired role slots (0&1, 2&3, etc.) get one of each model
    so neither model monopolises structurally advantaged positions.
    Falls back to random shuffle for >2 models or odd slot counts.
    """
    assignments = []
    for model, count in model_config:
        assignments.extend([model] * count)

    total = sum(c for _, c in model_config)
    if total != num_agents:
        print(f"Error: model config specifies {total} agents but scenario has {num_agents}")
        sys.exit(1)

    models = list(set(assignments))
    if len(models) == 2 and num_agents % 2 == 0:
        # Stratified: assign one of each model to each role pair
        result = [None] * num_agents
        for pair_start in range(0, num_agents, 2):
            pair = list(models)
            random.shuffle(pair)
            result[pair_start] = pair[0]
            result[pair_start + 1] = pair[1]
        return result
    else:
        random.shuffle(assignments)
        return assignments


# ---- Results persistence ----

def load_existing_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def save_run_file(entry, scenario_name):
    RUNS_DIR.mkdir(exist_ok=True)
    run_file = RUNS_DIR / f"run_{entry['run_id']:04d}_{scenario_name}.json"
    with open(run_file, "w") as f:
        json.dump(entry, f, indent=2)


# ---- Leaderboard ----

def print_leaderboard(results):
    """Print aggregate model performance across runs."""
    if not results:
        return

    model_stats = {}
    for r in results:
        if "error" in r:
            continue
        for model, score in r.get("model_goal_completion", {}).items():
            if model not in model_stats:
                model_stats[model] = {"scores": [], "trades": [], "invalids": []}
            model_stats[model]["scores"].append(score)
            model_stats[model]["trades"].append(r.get("num_trades", 0))
            model_stats[model]["invalids"].append(r.get("invalid_rate", 0))

    elo_ratings = load_ratings()

    print("\n" + "=" * 85)
    print("  LEADERBOARD")
    print("=" * 85)
    print(f"  {'Model':<12} {'Avg Goal%':>10} {'± 95% CI':>9} {'StdDev':>8} {'Invalid%':>9} {'Runs':>6}")
    print("-" * 85)

    rows = []
    for model, stats in model_stats.items():
        n = len(stats["scores"])
        scores = stats["scores"]
        mean = sum(scores) / n * 100
        if n > 1:
            variance = sum((s * 100 - mean) ** 2 for s in scores) / (n - 1)
            std = math.sqrt(variance)
            ci95 = 1.96 * std / math.sqrt(n)
        else:
            std = 0.0
            ci95 = 0.0
        rows.append((
            model,
            elo_ratings.get(model, 0),
            mean,
            std,
            ci95,
            sum(stats["invalids"]) / n * 100,
            n,
        ))
    rows.sort(key=lambda r: (r[1] if r[1] else 0, r[2]), reverse=True)

    for model, elo, score, std, ci95, invalids, n in rows:
        elo_str = f" ELO {elo:.0f}" if elo else ""
        ci_str = f"±{ci95:.1f}%" if n > 1 else "  n/a"
        print(f"  {model:<12} {score:>9.1f}% {ci_str:>9} {std:>7.1f}% {invalids:>8.1f}% {n:>6}{elo_str}")
    print("=" * 85)


# ---- Benchmark mode ----

def build_benchmark_config(anchor, test_models, num_agents):
    """Build model config for benchmark: half anchor, rest split across test models."""
    anchor_count = num_agents // 2
    remaining = num_agents - anchor_count

    config = [(anchor, anchor_count)]
    if test_models:
        base = remaining // len(test_models)
        extra = remaining % len(test_models)
        for i, model in enumerate(test_models):
            count = base + (1 if i < extra else 0)
            if count > 0:
                config.append((model, count))

    config_str = ",".join(f"{m}:{c}" for m, c in config)
    return config, config_str


def print_benchmark_leaderboard(run_entries, anchor):
    """Print benchmark leaderboard: model performance vs anchor."""
    if not run_entries:
        return

    model_stats = {}
    for entry in run_entries:
        if "error" in entry:
            continue
        mgc = entry.get("model_goal_completion", {})
        anchor_score = mgc.get(anchor, 0)

        for model, score in mgc.items():
            if model not in model_stats:
                model_stats[model] = {"scores": [], "vs_anchor": []}
            model_stats[model]["scores"].append(score)
            if model != anchor:
                if score - anchor_score >= 0.02:
                    model_stats[model]["vs_anchor"].append("win")
                elif anchor_score - score >= 0.02:
                    model_stats[model]["vs_anchor"].append("loss")
                else:
                    model_stats[model]["vs_anchor"].append("draw")

    # Aggregate scarce item capture
    scarce_totals = {}
    scarce_counts = {}
    for entry in run_entries:
        if "error" in entry:
            continue
        for item, captures in entry.get("scarce_capture", {}).items():
            if item not in scarce_totals:
                scarce_totals[item] = {}
                scarce_counts[item] = 0
            scarce_counts[item] += 1
            for model, qty in captures.items():
                scarce_totals[item][model] = scarce_totals[item].get(model, 0) + qty

    print("\n" + "=" * 90)
    print("  BENCHMARK LEADERBOARD")
    print("=" * 90)
    print(f"  {'Model':<14} {'Avg Goal%':>10} {'± 95% CI':>9} {'StdDev':>8} {'vs Anchor':>12} {'W-L-D':>10}")
    print("-" * 90)

    rows = []
    for model, stats in model_stats.items():
        n = len(stats["scores"])
        scores = stats["scores"]
        avg = sum(scores) / n * 100
        if n > 1:
            variance = sum((s * 100 - avg) ** 2 for s in scores) / (n - 1)
            std = math.sqrt(variance)
            ci95 = 1.96 * std / math.sqrt(n)
        else:
            std = 0.0
            ci95 = 0.0
        wins = stats["vs_anchor"].count("win")
        losses = stats["vs_anchor"].count("loss")
        draws = stats["vs_anchor"].count("draw")
        if model == anchor:
            vs_str = "baseline"
            wld = "-"
        else:
            total = wins + losses + draws
            vs_str = f"{wins / total * 100:.0f}% wins" if total > 0 else "-"
            wld = f"{wins}-{losses}-{draws}"
        rows.append((model, avg, std, ci95, vs_str, wld, model == anchor, n))

    rows.sort(key=lambda r: r[1], reverse=True)

    for model, avg, std, ci95, vs_str, wld, is_anchor, n in rows:
        marker = " *" if is_anchor else "  "
        ci_str = f"±{ci95:.1f}%" if n > 1 else "  n/a"
        print(f"{marker}{model:<12} {avg:>9.1f}% {ci_str:>9} {std:>7.1f}% {vs_str:>12} {wld:>10}")

    if scarce_totals:
        print()
        print(f"  Scarce Item Capture (avg per run):")
        for item in sorted(scarce_totals.keys()):
            parts = []
            for model in sorted(scarce_totals[item].keys()):
                avg_qty = scarce_totals[item][model] / scarce_counts[item]
                parts.append(f"{model}={avg_qty:.1f}")
            print(f"    {item}: {', '.join(parts)}")

    print()
    print("  * = anchor (baseline)")
    print("=" * 80)


def run_benchmark(scenario_name, anchor, test_models, runs, verbose,
                  simultaneous=False, parallel=1, temperature=1.0, history_rounds=3):
    """Run benchmark mode: test models against anchor in a single scenario."""
    procedural = scenario_name == "procedural"

    if procedural:
        # Default: 8 agents for procedural benchmark
        default_agents = 8
        scenario = _generate_scenario_for_eval(default_agents)
        scenario_name = scenario["name"]
        num_agents = default_agents
        print(f"Procedural scenario: {scenario_name} ({num_agents} agents, seed={scenario['_seed']})")
    else:
        scenario = load_scenario(scenario_name)
        num_agents = len(scenario["agents"])

    config, config_str = build_benchmark_config(anchor, test_models, num_agents)

    print(f"Benchmark Mode: {scenario_name}")
    print(f"  Anchor: {anchor} ({config[0][1]} agents)")
    test_info = ", ".join(f"{m}({c})" for m, c in config[1:])
    print(f"  Test models: {test_info}")
    flags = []
    if simultaneous:
        flags.append("simultaneous")
    if parallel > 1:
        flags.append(f"parallel={parallel}")
    flag_str = f" | {', '.join(flags)}" if flags else ""
    print(f"  Agents: {num_agents} | Rounds: {scenario.get('max_rounds', 10)} | Runs: {runs}{flag_str}")
    print()

    if BENCHMARK_RESULTS_FILE.exists():
        with open(BENCHMARK_RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = []

    run_offset = len(all_results)
    run_entries = []

    def _run_one(run_num):
        run_id = run_offset + run_num + 1
        # Fresh scenario each run for procedural mode
        if procedural and run_num > 0:
            run_scenario = _generate_scenario_for_eval(num_agents)
            run_name = run_scenario["name"]
        else:
            run_scenario = scenario
            run_name = scenario_name
        print(f"  Run {run_num + 1}/{runs}")
        entry = run_single(run_name, run_scenario, config, config_str, run_id, verbose,
                           simultaneous=simultaneous, live_updates=True, temperature=temperature,
                           history_rounds=history_rounds)
        entry["mode"] = "benchmark"
        entry["anchor_model"] = anchor
        _locked_append_result(entry, BENCHMARK_RESULTS_FILE)
        save_run_file(entry, run_name)
        return entry

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(_run_one, i) for i in range(runs)]
            run_entries = [f.result() for f in futures]
    else:
        for run_num in range(runs):
            entry = _run_one(run_num)
            run_entries.append(entry)

    print_benchmark_leaderboard(run_entries, anchor)

    # Statistical summary with confidence intervals (if multi-run)
    if len(run_entries) > 1:
        from scoring import compute_aggregate_statistics
        stats = compute_aggregate_statistics(run_entries)
        if stats:
            print(f"\n  Statistical Summary ({stats['total_runs']} runs):")
            for model, s in sorted(stats["per_model"].items(), key=lambda x: -x[1]["mean"]):
                print(f"    {model:<16s} {s['mean']*100:.1f}% ± {s['std']*100:.1f}% "
                      f"[{s['ci_lower']*100:.1f}-{s['ci_upper']*100:.1f}%] (95% CI)")

    print(f"\nResults saved to {BENCHMARK_RESULTS_FILE}")


# ---- Cross-provider matrix (all pairs) ----

def run_matrix(models, runs_per_pair, verbose, simultaneous=False, parallel=1,
               temperature=1.0, history_rounds=3):
    """Run round-robin pairwise comparisons across all model pairs.

    For N models, runs N*(N-1)/2 pairwise matchups on procedural scenarios.
    """
    from scoring import compute_aggregate_statistics
    from itertools import combinations

    pairs = list(combinations(models, 2))
    total_matches = len(pairs) * runs_per_pair

    print(f"\n{'='*70}")
    print(f"CROSS-PROVIDER MODEL MATRIX")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")
    print(f"Pairs: {len(pairs)} | Runs per pair: {runs_per_pair} | Total matches: {total_matches}")
    print()

    all_entries = []
    for pair_idx, (model_a, model_b) in enumerate(pairs):
        print(f"\n--- Pair {pair_idx + 1}/{len(pairs)}: {model_a} vs {model_b} ---")
        pair_entries = []

        for run_num in range(runs_per_pair):
            scenario = _generate_scenario_for_eval(
                num_agents=6, num_items=4, num_scarce=1,
                extra_seed=hash((model_a, model_b, run_num)),
            )
            scenario_name = f"matrix_{model_a}_vs_{model_b}"
            config = [(model_a, 3), (model_b, 3)]
            config_str = f"{model_a}:3,{model_b}:3"
            run_id = hash((model_a, model_b, run_num)) % 100000

            entry = run_single(scenario_name, scenario, config, config_str, run_id, verbose,
                               simultaneous=simultaneous, temperature=temperature,
                               history_rounds=history_rounds)
            entry["mode"] = "matrix"
            _locked_append_result(entry, RESULTS_FILE)
            pair_entries.append(entry)

        # Print pair summary
        stats = compute_aggregate_statistics(pair_entries)
        if stats:
            for model, s in stats["per_model"].items():
                print(f"    {model}: {s['mean']*100:.1f}% ± {s['std']*100:.1f}% ({s['n_runs']} runs)")

        all_entries.extend(pair_entries)

    # Print overall matrix
    print(f"\n{'='*70}")
    print(f"MATRIX RESULTS ({len(all_entries)} total matches)")
    print(f"{'='*70}")

    # Aggregate all results
    stats = compute_aggregate_statistics(all_entries)
    if stats:
        print(f"\n{'Model':<20s} {'Mean':>8s} {'±Std':>8s} {'95% CI':>16s} {'N':>4s}")
        print("-" * 60)
        sorted_models = sorted(stats["per_model"].items(), key=lambda x: -x[1]["mean"])
        for model, s in sorted_models:
            print(f"  {model:<18s} {s['mean']*100:>7.1f}% {s['std']*100:>7.1f}% "
                  f"[{s['ci_lower']*100:.1f}-{s['ci_upper']*100:.1f}%] {s['n_runs']:>4d}")

        # Pairwise significance
        if stats.get("pairwise"):
            print(f"\nPairwise Comparisons:")
            for key, comp in stats["pairwise"].items():
                sig = "*" if comp["significant"] else ""
                print(f"  {key}: diff={comp['mean_diff']*100:+.1f}pp "
                      f"CI=[{comp['ci_lower']*100:+.1f}, {comp['ci_upper']*100:+.1f}] "
                      f"p={comp['p_value']:.3f} {sig}")

    print_elo_leaderboard()
    compute_bt_ratings()
    print_bt_leaderboard()


# ---- Run a single match (model vs model) ----

def run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose,
               simultaneous=False, live_updates=True, temperature=1.0, history_rounds=3):
    """Run a single marketplace match. Returns the result entry."""
    # Seed RNG for reproducibility — each run gets a unique but deterministic seed
    seed = hash((scenario_name, run_id)) % (2**32)
    random.seed(seed)
    num_agents = len(scenario["agents"])
    model_assignments = assign_models(model_config, num_agents)
    agents = []
    for i in range(num_agents):
        if model_assignments[i] == "random":
            agents.append(RandomAgent(agent_idx=i, seed=seed + i))
        else:
            agents.append(MarketAgent(model_name=model_assignments[i], agent_idx=i,
                                      temperature=temperature, history_rounds=history_rounds))

    mode_str = " [simultaneous]" if simultaneous else ""
    print(f"  [{scenario_name}]{mode_str} models: {' '.join(model_assignments)}")
    print(f"  Running marketplace...", end="", flush=True)

    try:
        engine = MarketEngine(scenario, agents, simultaneous=simultaneous,
                              run_id=run_id, live_updates=live_updates)
        result = engine.run()
        result["scenario_data"] = scenario
        metrics = compute_metrics(result)

        clean_history = list(result.get("history", []))

        entry = {
            "run_id": run_id,
            "seed": seed,
            "temperature": temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "model_config": model_config_str,
            "model_assignments": model_assignments,
            **metrics,
            "elapsed_seconds": result["elapsed_seconds"],
            "backend": agents[0].backend,
            "agent_tokens": [
                {"model": agents[i].model_name,
                 "agent_idx": i,
                 "tokens": agents[i].total_input_tokens + agents[i].total_output_tokens,
                 "input_tokens": agents[i].total_input_tokens,
                 "output_tokens": agents[i].total_output_tokens,
                 "cost_usd": compute_dollar_cost(
                     agents[i].model_name,
                     agents[i].total_input_tokens,
                     agents[i].total_output_tokens),
                 }
                for i in range(num_agents)
            ],
            "initial_inventories": result["initial_inventories"],
            "targets": [dict(a.get("target", {})) for a in scenario["agents"]],
            "agent_results": result["agent_results"],
            "trades": result["trades"],
            "history": clean_history,
        }

        # Model metadata (dimensions for analysis: size, family, provider, cost tier)
        entry["model_metadata"] = {
            model: get_model_info(model) for model in set(model_assignments)
        }

        # Per-agent latency statistics
        entry["agent_latencies"] = [
            {
                "model": agents[i].model_name,
                "agent_idx": i,
                "mean_latency": (sum(agents[i].turn_latencies) / len(agents[i].turn_latencies)
                                 if agents[i].turn_latencies else 0),
                "max_latency": max(agents[i].turn_latencies) if agents[i].turn_latencies else 0,
                "min_latency": min(agents[i].turn_latencies) if agents[i].turn_latencies else 0,
                "total_latency": sum(agents[i].turn_latencies),
                "num_turns": len(agents[i].turn_latencies),
            }
            for i in range(num_agents)
        ]

        # Reproducibility metadata
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).parent)
            ).decode().strip()[:12]
        except Exception:
            git_sha = "unknown"
        entry["reproducibility"] = {
            "python_version": sys.version.split()[0],
            "barterbench_version": "1.0.0",
            "git_sha": git_sha,
            "seed": seed,
            "temperature": temperature,
            "history_rounds": history_rounds,
            "simultaneous": simultaneous,
        }

        # Cost-adjusted performance
        cost_eff = compute_cost_efficiency(entry)
        if cost_eff:
            entry["cost_efficiency"] = cost_eff

        status = f"goal={metrics['avg_goal_completion']*100:.0f}% trades={metrics['num_trades']} invalid={metrics['invalid_rate']*100:.0f}%"
        print(f" done ({status})")

        for model, score in metrics["model_goal_completion"].items():
            tokens_info = ""
            if cost_eff and model in cost_eff["per_model"]:
                gc_per_k = cost_eff["per_model"][model]["goal_completion_per_1k_tokens"]
                tokens_info = f" ({gc_per_k:.3f} gc/1k tok)"
            print(f"    {model}: {score*100:.1f}% goal completion{tokens_info}")

        match = record_match(entry)
        if match:
            w = match["winner"]
            elo_a = match["elo_after"][match["model_a"]]
            elo_b = match["elo_after"][match["model_b"]]
            if w == "draw":
                print(f"    ELO: draw — {match['model_a']}={elo_a:.0f} {match['model_b']}={elo_b:.0f}")
            else:
                print(f"    ELO: {w} wins — {match['model_a']}={elo_a:.0f} {match['model_b']}={elo_b:.0f}")

        if verbose:
            print()
            for h in result["history"]:
                tag = f"    [R{h['round']}] Trader {h['agent']} ({h['model']})"
                if h["action"] == "post_offer":
                    inv = " [INVALID]" if h.get("invalid") else ""
                    print(f"{tag} POST: give {h.get('give',{})} want {h.get('want',{})}{inv} — {h.get('message','')}")
                elif h["action"] == "accept_offer":
                    inv = " [INVALID]" if h.get("invalid") else ""
                    print(f"{tag} ACCEPT offer #{h.get('offer_id','?')}{inv} — {h.get('message','')}")
                elif h["action"] == "pass_turn":
                    print(f"{tag} PASS — {h.get('message','')}")
            print()

        return entry

    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "model_config": model_config_str,
            "error": str(e),
        }


def resume_run(checkpoint_path, verbose=False):
    """Resume a match from a checkpoint file."""
    from pathlib import Path
    cp_path = Path(checkpoint_path) if Path(checkpoint_path).is_absolute() else Path(__file__).parent / checkpoint_path
    if not cp_path.exists():
        print(f"Error: checkpoint file not found: {cp_path}")
        sys.exit(1)

    engine, start_round, initial_inventories, elapsed_offset = MarketEngine.from_checkpoint(str(cp_path))
    scenario_name = engine.scenario.get("name", "unknown")
    model_assignments = [a.model_name for a in engine.agents]

    print(f"Resuming: {scenario_name}")
    print(f"  Round {start_round}/{engine.max_rounds} | Agents: {engine.num_agents} | Models: {' '.join(model_assignments)}")
    print(f"  Prior: {len(engine.trades)} trades, {len(engine.history)} actions, {elapsed_offset:.0f}s elapsed")
    print(f"  Running marketplace...", end="", flush=True)

    try:
        result = engine.run(start_round=start_round, initial_inventories=initial_inventories,
                            elapsed_offset=elapsed_offset)
        result["scenario_data"] = engine.scenario
        metrics = compute_metrics(result)

        model_config_str = ",".join(f"{m}:{model_assignments.count(m)}" for m in dict.fromkeys(model_assignments))
        entry = {
            "run_id": engine.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "model_config": model_config_str,
            "model_assignments": model_assignments,
            "resumed_from_round": start_round,
            **metrics,
            "elapsed_seconds": result["elapsed_seconds"],
            "backend": engine.agents[0].backend,
            "agent_tokens": [
                {"model": engine.agents[i].model_name,
                 "tokens": engine.agents[i].total_input_tokens + engine.agents[i].total_output_tokens}
                for i in range(engine.num_agents)
            ],
            "initial_inventories": result["initial_inventories"],
            "targets": [dict(a.get("target", {})) for a in engine.scenario["agents"]],
            "agent_results": result["agent_results"],
            "trades": result["trades"],
            "history": result["history"],
        }

        cost_eff = compute_cost_efficiency(entry)
        if cost_eff:
            entry["cost_efficiency"] = cost_eff

        status = f"goal={metrics['avg_goal_completion']*100:.0f}% trades={metrics['num_trades']} invalid={metrics['invalid_rate']*100:.0f}%"
        print(f" done ({status})")

        for model, score in metrics["model_goal_completion"].items():
            print(f"    {model}: {score*100:.1f}% goal completion")

        _locked_append_result(entry, RESULTS_FILE)
        save_run_file(entry, scenario_name)

        match = record_match(entry)
        if match:
            w = match["winner"]
            elo_a = match["elo_after"][match["model_a"]]
            elo_b = match["elo_after"][match["model_b"]]
            if w == "draw":
                print(f"    ELO: draw — {match['model_a']}={elo_a:.0f} {match['model_b']}={elo_b:.0f}")
            else:
                print(f"    ELO: {w} wins — {match['model_a']}={elo_a:.0f} {match['model_b']}={elo_b:.0f}")

        if verbose:
            print()
            for h in result["history"]:
                tag = f"    [R{h['round']}] Trader {h['agent']} ({h['model']})"
                if h["action"] == "post_offer":
                    inv = " [INVALID]" if h.get("invalid") else ""
                    print(f"{tag} POST: give {h.get('give',{})} want {h.get('want',{})}{inv} — {h.get('message','')}")
                elif h["action"] == "accept_offer":
                    inv = " [INVALID]" if h.get("invalid") else ""
                    print(f"{tag} ACCEPT offer #{h.get('offer_id','?')}{inv} — {h.get('message','')}")
                elif h["action"] == "pass_turn":
                    print(f"{tag} PASS — {h.get('message','')}")
            print()

        print_leaderboard([entry])
        print(f"\nResults saved to {RESULTS_FILE}")

    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()


# ---- Orchestration modes ----

def _generate_scenario_for_eval(num_agents, run_seed=None):
    """Generate a fresh procedural scenario for contamination-free eval."""
    from scenario_gen import generate_scenario
    seed = run_seed if run_seed is not None else random.randint(0, 2**32 - 1)
    num_items = max(3, num_agents // 2)
    num_scarce = max(1, num_items // 3)
    scenario = generate_scenario(num_agents=num_agents, num_items=num_items,
                                 num_scarce=num_scarce, seed=seed)
    scenario["_generated"] = True
    scenario["_seed"] = seed
    return scenario


def run_eval(scenario_name, model_config_str, runs, verbose,
             simultaneous=False, parallel=1, temperature=1.0, history_rounds=3):
    """Run eval for a single scenario. Use scenario_name='procedural' for fresh generated scenarios."""
    procedural = scenario_name == "procedural"
    model_config = parse_model_config(model_config_str)
    total_assigned = sum(c for _, c in model_config)

    if procedural:
        # Generate a scenario matching the model config agent count
        scenario = _generate_scenario_for_eval(total_assigned)
        scenario_name = scenario["name"]
        num_agents = total_assigned
        print(f"Procedural scenario: {scenario_name} ({num_agents} agents, seed={scenario['_seed']})")
    else:
        scenario = load_scenario(scenario_name)
        num_agents = len(scenario["agents"])
        if total_assigned != num_agents:
            print(f"Error: scenario '{scenario_name}' has {num_agents} agents but model config specifies {total_assigned}")
            print(f"  Use --models with correct counts, e.g. --models haiku:{num_agents // 2},opus:{num_agents // 2}")
            sys.exit(1)

    results = load_existing_results()
    run_offset = len(results)

    flags = []
    if simultaneous:
        flags.append("simultaneous")
    if parallel > 1:
        flags.append(f"parallel={parallel}")
    flag_str = f" | {', '.join(flags)}" if flags else ""
    print(f"Marketplace Eval: {scenario_name}")
    print(f"  Agents: {num_agents} | Models: {model_config_str} | Rounds: {scenario.get('max_rounds', 10)} | Runs: {runs}{flag_str}")
    print()

    def _run_one(run_num):
        run_id = run_offset + run_num + 1
        # For procedural mode, generate a fresh scenario each run (different seed)
        if procedural and run_num > 0:
            run_scenario = _generate_scenario_for_eval(total_assigned)
            run_scenario_name = run_scenario["name"]
        else:
            run_scenario = scenario
            run_scenario_name = scenario_name
        entry = run_single(run_scenario_name, run_scenario, model_config, model_config_str, run_id, verbose,
                           simultaneous=simultaneous, live_updates=True, temperature=temperature,
                           history_rounds=history_rounds)
        _locked_append_result(entry, RESULTS_FILE)
        save_run_file(entry, run_scenario_name)
        return entry

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(_run_one, i) for i in range(runs)]
            run_entries = [f.result() for f in futures]
    else:
        run_entries = []
        for run_num in range(runs):
            entry = _run_one(run_num)
            run_entries.append(entry)

    results = load_existing_results()
    print_leaderboard(results)
    print_elo_leaderboard()
    compute_bt_ratings()
    print_bt_leaderboard()
    print(f"\nResults saved to {RESULTS_FILE}")


def run_tournament(model_a, model_b, runs_per_scenario, verbose,
                   simultaneous=False, parallel=1, temperature=1.0, history_rounds=3):
    """Run a full tournament: model_a vs model_b across all scenarios."""
    scenarios = list_scenarios()
    if not scenarios:
        print("Error: no scenarios found")
        sys.exit(1)

    results = load_existing_results()
    run_offset = len(results)

    total_matches = len(scenarios) * runs_per_scenario
    flags = []
    if simultaneous:
        flags.append("simultaneous")
    if parallel > 1:
        flags.append(f"parallel={parallel}")
    flag_str = f" | {', '.join(flags)}" if flags else ""
    print(f"Tournament: {model_a} vs {model_b}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Runs per scenario: {runs_per_scenario}")
    print(f"  Total matches: {total_matches}{flag_str}")
    print()

    match_num = 0
    for scenario_name in scenarios:
        scenario = load_scenario(scenario_name)
        num_agents = len(scenario["agents"])
        half_a, half_b = auto_model_config(scenario)
        model_config = [(model_a, half_a), (model_b, half_b)]
        model_config_str = f"{model_a}:{half_a},{model_b}:{half_b}"

        print(f"--- {scenario_name} ({num_agents} agents, {scenario.get('max_rounds', 10)} rounds) ---")

        _match_lock = threading.Lock()

        def _run_one(run_num):
            nonlocal match_num
            with _match_lock:
                match_num += 1
                current_match = match_num
            run_id = run_offset + current_match
            print(f"\n  Match {current_match}/{total_matches}")
            entry = run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose,
                               simultaneous=simultaneous, live_updates=True, temperature=temperature,
                               history_rounds=history_rounds)
            _locked_append_result(entry, RESULTS_FILE)
            save_run_file(entry, scenario_name)
            return entry

        if parallel > 1:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(_run_one, i) for i in range(runs_per_scenario)]
                [f.result() for f in futures]
        else:
            for run_num in range(runs_per_scenario):
                _run_one(run_num)

        print()

    results = load_existing_results()
    print_leaderboard(results)
    print_elo_leaderboard()
    compute_bt_ratings()
    print_bt_leaderboard()
    print(f"\nResults saved to {RESULTS_FILE}")


# ---- Experiments management ----

def _load_experiments():
    """Load experiments list from experiments.json."""
    with file_lock(EXPERIMENTS_FILE):
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE) as f:
                return json.load(f)
        return []


def _save_experiments(experiments):
    """Save experiments list to experiments.json."""
    with file_lock(EXPERIMENTS_FILE):
        with open(EXPERIMENTS_FILE, "w") as f:
            json.dump(experiments, f, indent=2)


def _register_cli_experiment(mode, scenario, models, runs, simultaneous=False, parallel=1):
    """Register a CLI-launched eval as an experiment so the dashboard can track it."""
    from uuid import uuid4
    exp = {
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "pid": os.getpid(),
        "completed_at": None,
        "error": None,
        "config": {
            "mode": mode,
            "scenario": scenario,
            "models": models,
            "runs": runs,
            "simultaneous": simultaneous,
            "parallel": parallel,
        },
        "progress": {"current_run": 0, "total_runs": runs},
    }
    experiments = _load_experiments()
    experiments.append(exp)
    _save_experiments(experiments)
    return exp["id"]


def _complete_cli_experiment(exp_id):
    """Mark a CLI-launched experiment as completed."""
    _update_experiment(exp_id, {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


def _update_experiment(exp_id, updates):
    """Update a single experiment by ID."""
    with file_lock(EXPERIMENTS_FILE):
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE) as f:
                experiments = json.load(f)
        else:
            experiments = []
        for exp in experiments:
            if exp["id"] == exp_id:
                exp.update(updates)
                break
        with open(EXPERIMENTS_FILE, "w") as f:
            json.dump(experiments, f, indent=2)


def _get_scenario_metadata():
    """Return list of scenario metadata dicts."""
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    result = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            if data["agents"][0].get("target") is not None:
                scarcity = data.get("scarcity", {})
                scarcity_summary = ", ".join(
                    f"{k} ({v['ratio']:.0%})" for k, v in scarcity.items()
                ) if scarcity else "none"
                result.append({
                    "name": data["name"],
                    "num_agents": len(data["agents"]),
                    "max_rounds": data.get("max_rounds", 10),
                    "description": data.get("description", ""),
                    "scarcity": scarcity_summary,
                })
    return result


# ---- Dashboard HTTP handler ----

# Track active subprocess processes in server memory for cancel
_active_processes = {}  # exp_id -> Popen


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler with API endpoints for experiment management."""

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        if self.path == "/api/experiments":
            self._handle_get_experiments()
        elif self.path == "/api/scenarios":
            self._handle_get_scenarios()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/launch":
            self._handle_launch()
        elif self.path == "/api/cancel":
            self._handle_cancel()
        elif self.path == "/api/suite":
            self._handle_suite()
        else:
            self.send_error(404)

    def _handle_get_experiments(self):
        experiments = _load_experiments()
        # Load results to estimate progress
        all_results = []
        for rf in [RESULTS_FILE, BENCHMARK_RESULTS_FILE]:
            if rf.exists():
                try:
                    with open(rf) as f:
                        all_results.extend(json.load(f))
                except Exception:
                    pass

        dirty = False
        for exp in experiments:
            if exp["status"] == "running":
                pid = exp.get("pid")
                if pid and exp["id"] not in _active_processes:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        exp["status"] = "completed"
                        exp["completed_at"] = datetime.now(timezone.utc).isoformat()
                        dirty = True

                # Estimate progress from results created after experiment started
                started = exp.get("started_at", "")
                if started and exp.get("progress"):
                    count = sum(1 for r in all_results if (r.get("timestamp", "") >= started))
                    exp["progress"]["current_run"] = count
        if dirty:
            _save_experiments(experiments)
        self._send_json(experiments)

    def _handle_get_scenarios(self):
        self._send_json(_get_scenario_metadata())

    def _handle_launch(self):
        try:
            self._do_handle_launch()
        except Exception as e:
            print(f"[server] Launch error: {e}")
            import traceback; traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _do_handle_launch(self):
        body = self._read_body()
        config = body.get("config", {})

        mode = config.get("mode", "eval")
        scenario = config.get("scenario", "gold_rush")
        models = config.get("models", "haiku:6")
        runs = config.get("runs", 1)
        simultaneous = config.get("simultaneous", False)
        parallel = config.get("parallel", 1)
        anchor = config.get("anchor")
        temperature = config.get("temperature", 1.0)

        # Calculate total runs (scenario "all" multiplies by number of eval scenarios)
        if scenario == "all":
            num_scenarios = len(list_scenarios())
            total_runs = runs * num_scenarios
        else:
            total_runs = runs

        # Create experiment entry
        exp_id = f"exp_{int(time.time())}_{random.randint(1000,9999)}"
        exp = {
            "id": exp_id,
            "status": "queued",
            "pid": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "config": {
                "mode": mode,
                "scenario": scenario,
                "models": models,
                "runs": runs,
                "simultaneous": simultaneous,
                "parallel": parallel,
                "anchor": anchor,
            },
            "progress": {"current_run": 0, "total_runs": total_runs},
        }
        experiments = _load_experiments()
        experiments.append(exp)
        _save_experiments(experiments)

        # Build CLI command
        cmd = [sys.executable, str(Path(__file__).resolve())]
        if mode == "benchmark":
            cmd += ["--benchmark"]
            if anchor:
                cmd += ["--anchor", anchor]
            # For benchmark, models should be test models (not anchor)
            cmd += ["--models", models]
            cmd += ["--eval", scenario]
        elif mode == "arena":
            cmd += ["--arena", "--eval", scenario]
        else:
            cmd += ["--eval", scenario, "--models", models]

        cmd += ["--runs", str(runs)]
        if simultaneous:
            cmd += ["--simultaneous"]
        if parallel > 1:
            cmd += ["--parallel", str(parallel)]
        if temperature != 1.0:
            cmd += ["--temperature", str(temperature)]

        # Spawn subprocess
        print(f"[server] Launching experiment {exp_id}: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _active_processes[exp_id] = proc
            _update_experiment(exp_id, {
                "status": "running",
                "pid": proc.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })

            # Start monitoring thread
            t = threading.Thread(
                target=_monitor_experiment,
                args=(exp_id, proc, scenario, total_runs),
                daemon=True,
            )
            t.start()

            print(f"[server] Experiment {exp_id} started (pid={proc.pid})")
            self._send_json({"id": exp_id, "status": "running"})
        except Exception as e:
            print(f"[server] Experiment {exp_id} failed to start: {e}")
            _update_experiment(exp_id, {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            self._send_json({"id": exp_id, "status": "failed", "error": str(e)}, 500)

    def _handle_cancel(self):
        try:
            self._do_handle_cancel()
        except Exception as e:
            print(f"[server] Cancel error: {e}")
            self._send_json({"error": str(e)}, 500)

    def _do_handle_cancel(self):
        body = self._read_body()
        exp_id = body.get("id", "")

        proc = _active_processes.get(exp_id)
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            _active_processes.pop(exp_id, None)

        _update_experiment(exp_id, {
            "status": "cancelled",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        self._send_json({"id": exp_id, "status": "cancelled"})

    def _handle_suite(self):
        try:
            self._do_handle_suite()
        except Exception as e:
            print(f"[server] Suite error: {e}")
            import traceback; traceback.print_exc()
            self._send_json({"error": str(e)}, 500)

    def _do_handle_suite(self):
        body = self._read_body()
        models_str = body.get("models", "sonnet")
        test_models = [m.strip() for m in models_str.split(",") if m.strip()]

        total_runs = len(SUITE_SCENARIOS) * SUITE_RUNS_PER_SCENARIO * len(test_models)

        exp_id = f"exp_{int(time.time())}_{random.randint(1000,9999)}"
        exp = {
            "id": exp_id,
            "status": "queued",
            "pid": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "config": {
                "mode": "suite",
                "scenario": "all (suite)",
                "models": models_str,
                "runs": total_runs,
                "simultaneous": False,
                "parallel": 1,
                "anchor": SUITE_ANCHOR,
            },
            "progress": {"current_run": 0, "total_runs": total_runs},
        }
        experiments = _load_experiments()
        experiments.append(exp)
        _save_experiments(experiments)

        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--suite", "--models", models_str]

        print(f"[server] Launching suite {exp_id}: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _active_processes[exp_id] = proc
            _update_experiment(exp_id, {
                "status": "running",
                "pid": proc.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })

            t = threading.Thread(
                target=_monitor_suite_experiment,
                args=(exp_id, proc, total_runs),
                daemon=True,
            )
            t.start()

            print(f"[server] Suite {exp_id} started (pid={proc.pid})")
            self._send_json({"id": exp_id, "status": "running"})
        except Exception as e:
            print(f"[server] Suite {exp_id} failed to start: {e}")
            _update_experiment(exp_id, {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            self._send_json({"id": exp_id, "status": "failed", "error": str(e)}, 500)


def _monitor_suite_experiment(exp_id, proc, total_runs):
    """Monitor a suite experiment by counting new results across all scenarios."""
    def _count_results():
        count = 0
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE) as f:
                    data = json.load(f)
                count += sum(1 for r in data if r.get("mode") == "suite")
            except Exception:
                pass
        return count

    baseline_count = _count_results()

    while proc.poll() is None:
        time.sleep(2)
        new_count = _count_results() - baseline_count
        _update_experiment(exp_id, {
            "progress": {"current_run": min(new_count, total_runs), "total_runs": total_runs},
        })

    _active_processes.pop(exp_id, None)
    exit_code = proc.returncode
    final_count = _count_results() - baseline_count

    if exit_code == 0:
        _update_experiment(exp_id, {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "progress": {"current_run": final_count, "total_runs": total_runs},
        })
    elif exit_code == -signal.SIGTERM or exit_code == -signal.SIGKILL:
        pass
    else:
        error_msg = f"Process exited with code {exit_code}"
        try:
            output = proc.stdout.read().decode(errors="replace")[-500:]
            if output.strip():
                error_msg += f": {output.strip()}"
        except Exception:
            pass
        _update_experiment(exp_id, {
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def _monitor_experiment(exp_id, proc, scenario, total_runs):
    """Monitor a running experiment subprocess, updating progress."""
    # Count results matching this scenario that exist before we started
    def _count_results():
        count = 0
        for rf in [RESULTS_FILE, BENCHMARK_RESULTS_FILE]:
            if rf.exists():
                try:
                    with open(rf) as f:
                        data = json.load(f)
                    if scenario == "all":
                        count += len(data)
                    else:
                        count += sum(1 for r in data if r.get("scenario") == scenario)
                except Exception:
                    pass
        return count

    baseline_count = _count_results()

    while proc.poll() is None:
        time.sleep(2)
        # Update progress by counting new results
        new_count = _count_results() - baseline_count
        _update_experiment(exp_id, {
            "progress": {"current_run": min(new_count, total_runs), "total_runs": total_runs},
        })

    # Process exited
    _active_processes.pop(exp_id, None)
    exit_code = proc.returncode
    final_count = _count_results() - baseline_count

    if exit_code == 0:
        _update_experiment(exp_id, {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "progress": {"current_run": final_count, "total_runs": total_runs},
        })
    elif exit_code == -signal.SIGTERM or exit_code == -signal.SIGKILL:
        # Already marked cancelled by _handle_cancel
        pass
    else:
        # Try to capture error output
        error_msg = f"Process exited with code {exit_code}"
        try:
            output = proc.stdout.read().decode(errors="replace")[-500:]
            if output.strip():
                error_msg += f": {output.strip()}"
        except Exception:
            pass
        _update_experiment(exp_id, {
            "status": "failed",
            "error": error_msg,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


# ---- Eval Suite mode ----

SUITE_SCENARIOS = ["gold_rush", "water_crisis", "spice_wars", "grand_bazaar"]
SUITE_RUNS_PER_SCENARIO = 10
SUITE_ANCHOR = "haiku"


def run_suite(test_models, verbose):
    """Run standardized eval suite: each test model vs haiku anchor across all scenarios."""
    scenarios = SUITE_SCENARIOS
    runs_per = SUITE_RUNS_PER_SCENARIO
    anchor = SUITE_ANCHOR

    total_per_model = len(scenarios) * runs_per
    total = total_per_model * len(test_models)
    print(f"Eval Suite: {', '.join(test_models)} vs {anchor} anchor")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Runs per scenario: {runs_per}")
    print(f"  Total runs: {total}")
    print()

    results = load_existing_results()
    run_offset = len(results)
    match_num = 0

    suite_entries = {}  # model -> list of entries

    for test_model in test_models:
        print(f"{'='*60}")
        print(f"  Testing: {test_model} vs {anchor}")
        print(f"{'='*60}")
        suite_entries[test_model] = []

        for scenario_name in scenarios:
            scenario = load_scenario(scenario_name)
            num_agents = len(scenario["agents"])
            anchor_count = num_agents // 2
            test_count = num_agents - anchor_count

            model_config = [(anchor, anchor_count), (test_model, test_count)]
            model_config_str = f"{anchor}:{anchor_count},{test_model}:{test_count}"

            print(f"\n--- {scenario_name} ({num_agents} agents: {anchor}:{anchor_count}, {test_model}:{test_count}) ---")

            for run_num in range(runs_per):
                match_num += 1
                run_id = run_offset + match_num
                print(f"\n  Run {run_num + 1}/{runs_per} (match {match_num}/{total})")
                entry = run_single(scenario_name, scenario, model_config, model_config_str,
                                   run_id, verbose, simultaneous=False, live_updates=True,
                                   history_rounds=3)
                entry["mode"] = "suite"
                entry["anchor_model"] = anchor
                _locked_append_result(entry, RESULTS_FILE)
                save_run_file(entry, scenario_name)
                suite_entries[test_model].append(entry)

    # Print suite report card
    print(f"\n{'='*80}")
    print(f"  EVAL SUITE REPORT CARD")
    print(f"{'='*80}")

    elo_ratings = load_ratings()

    for test_model in test_models:
        entries = suite_entries[test_model]
        valid = [e for e in entries if "error" not in e]

        print(f"\n  {test_model} vs {anchor}")
        print(f"  {'-'*40}")

        # Per-scenario breakdown
        wins, losses, draws = 0, 0, 0
        all_test_scores = []
        all_anchor_scores = []

        for scenario_name in scenarios:
            sc_entries = [e for e in valid if e["scenario"] == scenario_name]
            if not sc_entries:
                print(f"    {scenario_name}: no data")
                continue
            test_scores = [e["model_goal_completion"].get(test_model, 0) for e in sc_entries]
            anchor_scores = [e["model_goal_completion"].get(anchor, 0) for e in sc_entries]
            avg_test = sum(test_scores) / len(test_scores) * 100
            avg_anchor = sum(anchor_scores) / len(anchor_scores) * 100
            all_test_scores.extend(test_scores)
            all_anchor_scores.extend(anchor_scores)

            for ts, as_ in zip(test_scores, anchor_scores):
                if ts - as_ >= 0.02:
                    wins += 1
                elif as_ - ts >= 0.02:
                    losses += 1
                else:
                    draws += 1

            print(f"    {scenario_name:<16} {test_model}={avg_test:.1f}%  {anchor}={avg_anchor:.1f}%")

        # Composite
        if all_test_scores:
            composite = sum(all_test_scores) / len(all_test_scores) * 100
            anchor_composite = sum(all_anchor_scores) / len(all_anchor_scores) * 100
            print(f"\n    Composite:       {test_model}={composite:.1f}%  {anchor}={anchor_composite:.1f}%")
            print(f"    W-L-D:           {wins}-{losses}-{draws}")
            elo = elo_ratings.get(test_model, 1500)
            anchor_elo = elo_ratings.get(anchor, 1500)
            print(f"    ELO:             {test_model}={elo:.0f}  {anchor}={anchor_elo:.0f}")

    print(f"\n{'='*80}")
    print(f"\nResults saved to {RESULTS_FILE}")
    print_elo_leaderboard()
    compute_bt_ratings()
    print_bt_leaderboard()


def serve_dashboard():
    """Serve the dashboard with API endpoints on a local HTTP server."""
    import webbrowser

    os.chdir(Path(__file__).parent)
    port = 8080

    print(f"Serving dashboard at http://localhost:{port}/dashboard.html")
    webbrowser.open(f"http://localhost:{port}/dashboard.html")

    with HTTPServer(("", port), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="BarterBench: competitive marketplace benchmark with ELO ratings")
    parser.add_argument("--eval", type=str, default=None,
                        help="Scenario to evaluate: name, 'all' for tournament, or 'procedural' for fresh generated (default for benchmark)")
    parser.add_argument("--models", type=str, default=None,
                        help="Model config: 'haiku:3,opus:3' — counts must match scenario agent count")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per scenario (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full marketplace transcripts")
    parser.add_argument("--serve", action="store_true",
                        help="Serve the dashboard web UI")
    parser.add_argument("--clear", action="store_true",
                        help="Clear previous results and ELO ratings before running")
    parser.add_argument("--list", action="store_true",
                        help="List available marketplace scenarios and strategies")
    parser.add_argument("--elo", action="store_true",
                        help="Show current ELO leaderboard")
    # Arena mode
    parser.add_argument("--arena", action="store_true",
                        help="Arena mode: pit prompt strategies against each other")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Strategies to compete: 'aggressive,cooperative' or omit for all")
    parser.add_argument("--submit", nargs=2, metavar=("NAME", "PROMPT"),
                        help="Submit a new strategy: --submit 'my_strat' 'Your prompt here'")
    # Benchmark mode (hybrid anchor)
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode: test models against a cheap anchor model in one big scenario")
    parser.add_argument("--anchor", type=str, default="haiku",
                        help="Anchor model for benchmark mode (default: haiku)")
    # Eval suite
    parser.add_argument("--suite", action="store_true",
                        help="Run standardized eval suite against haiku anchor across all scenarios")
    # Simultaneous + parallel
    parser.add_argument("--simultaneous", action="store_true",
                        help="Simultaneous mode: all agents act on same snapshot per round")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Run N matches concurrently (default: 1)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM sampling temperature (default: 1.0). Lower = more deterministic. API backend only.")
    # Procedural scenario generation
    parser.add_argument("--generate", action="store_true",
                        help="Generate a procedural scenario instead of using existing ones")
    parser.add_argument("--gen-agents", type=int, default=6,
                        help="Number of agents for generated scenario (default: 6)")
    parser.add_argument("--gen-items", type=int, default=4,
                        help="Number of items for generated scenario (default: 4)")
    parser.add_argument("--gen-scarce", type=int, default=1,
                        help="Number of scarce items for generated scenario (default: 1)")
    parser.add_argument("--gen-seed", type=int, default=None,
                        help="Seed for procedural scenario generation")
    parser.add_argument("--history-rounds", type=int, default=3,
                        help="Number of past rounds to include in agent conversation history (default: 3)")
    parser.add_argument("--resume", nargs="?", const="checkpoint.json", default=None,
                        help="Resume from checkpoint file (default: checkpoint.json)")
    # Analysis & reporting
    parser.add_argument("--scaling-report", action="store_true",
                        help="Print scaling analysis (performance vs model size/cost)")
    parser.add_argument("--behavior-report", action="store_true",
                        help="Print emergent behavior taxonomy report")
    # Cross-provider matrix
    parser.add_argument("--matrix", action="store_true",
                        help="Run pairwise model matrix: all combinations of --models")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard()
        return

    if args.resume:
        resume_run(args.resume, args.verbose)
        return

    if args.scaling_report:
        from analysis import print_scaling_report
        print_scaling_report()
        return

    if args.behavior_report:
        from taxonomy import print_behavior_report
        from analysis import load_results
        entries = load_results()
        if not entries:
            print("No results found. Run some evals first.")
        else:
            print_behavior_report(entries)
        return

    if args.matrix:
        if not args.models:
            print("Error: --matrix requires --models (comma-separated list)")
            print("  Example: --matrix --models haiku,sonnet,opus,hunter,llama-70b --runs 3")
            sys.exit(1)
        model_list = [m.strip().split(":")[0] for m in args.models.split(",")]
        if len(model_list) < 2:
            print("Error: --matrix requires at least 2 models")
            sys.exit(1)
        _runs = args.runs or 1
        run_matrix(model_list, _runs, args.verbose,
                   simultaneous=args.simultaneous, parallel=args.parallel,
                   temperature=args.temperature, history_rounds=args.history_rounds)
        return

    if args.elo:
        print_elo_leaderboard()
        compute_bt_ratings()
        print_bt_leaderboard()
        from arena.runner import print_arena_leaderboard
        print_arena_leaderboard()
        return

    if args.submit:
        from arena.runner import submit_strategy
        submit_strategy(args.submit[0], args.submit[1])
        return

    if args.list:
        print("Available scenarios:")
        for s in list_scenarios():
            scenario = load_scenario(s)
            n = len(scenario["agents"])
            scarcity = scenario.get("scarcity", {})
            scarce_items = ", ".join(f"{k} ({v['ratio']:.0%} supply)" for k, v in scarcity.items()) if scarcity else "none"
            print(f"  {s:<20} {n} agents, {scenario.get('max_rounds', 10)} rounds — scarcity: {scarce_items}")

        from arena.runner import list_strategies
        strats = list_strategies()
        if strats:
            print(f"\nAvailable strategies ({len(strats)}):")
            for s in strats:
                prompt_preview = s["prompt"][:60].replace("\n", " ") + "..." if len(s["prompt"]) > 60 else s["prompt"].replace("\n", " ")
                print(f"  {s['id']:<20} by {s.get('author', '?'):<12} model={s.get('model', 'haiku'):<8} {prompt_preview}")
        return

    if args.clear:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
        reset_ratings()
        reset_bt()
        from arena.runner import reset_arena
        reset_arena()
        print("Cleared previous results and ELO ratings.")

    # ---- Arena mode ----
    if args.arena:
        from arena.runner import run_arena
        scenario = args.eval or "all"
        strat_names = None
        if args.strategies:
            strat_names = [s.strip() for s in args.strategies.split(",")]
        # Parse models for cross-model arena
        models = None
        if args.models:
            models = [m.strip().split(":")[0] for m in args.models.split(",")]
        run_arena(strat_names, scenario, args.runs, args.verbose, models=models,
                  simultaneous=args.simultaneous)
        return

    # ---- Eval suite mode ----
    if args.suite:
        if not args.models:
            print("Error: --models required for suite mode.")
            print("  Example: --suite --models sonnet")
            print("  Example: --suite --models sonnet,opus")
            sys.exit(1)
        test_models = [m.strip().split(":")[0] for m in args.models.split(",")]
        test_models = [m for m in test_models if m != SUITE_ANCHOR]
        if not test_models:
            print(f"Error: --models must include at least one model besides the anchor ({SUITE_ANCHOR}).")
            sys.exit(1)
        run_suite(test_models, args.verbose)
        return

    # ---- Procedural scenario generation ----
    if args.generate:
        from scenario_gen import generate_scenario, save_procedural_scenario
        gen_seed = args.gen_seed if args.gen_seed is not None else random.randint(0, 2**32 - 1)
        scenario = generate_scenario(
            num_agents=args.gen_agents,
            num_items=args.gen_items,
            num_scarce=args.gen_scarce,
            seed=gen_seed,
        )
        path = save_procedural_scenario(scenario, overwrite=True)
        print(f"Generated scenario: {scenario['name']}")
        print(f"  Agents: {len(scenario['agents'])} | Items: {args.gen_items} | Scarce: {args.gen_scarce}")
        print(f"  Seed: {gen_seed}")
        print(f"  Saved to: {path}")

        if args.models:
            # Run eval on the generated scenario immediately
            run_eval(scenario["name"], args.models, args.runs, args.verbose,
                     simultaneous=args.simultaneous, parallel=args.parallel,
                     temperature=args.temperature, history_rounds=args.history_rounds)
        return

    # ---- Benchmark mode (hybrid anchor) ----
    if args.benchmark:
        if not args.models:
            print("Error: --models required for benchmark mode.")
            print("  Example: --benchmark --models sonnet,opus")
            sys.exit(1)
        scenario_name = args.eval or "procedural"
        test_models = [m.strip().split(":")[0] for m in args.models.split(",")]
        test_models = [m for m in test_models if m != args.anchor]
        if not test_models:
            print("Error: --models must include at least one model besides the anchor.")
            sys.exit(1)
        _runs = args.runs or 3
        exp_id = _register_cli_experiment("benchmark", scenario_name, args.models, _runs,
                                          simultaneous=args.simultaneous, parallel=args.parallel)
        run_benchmark(scenario_name, args.anchor, test_models, _runs, args.verbose,
                      simultaneous=args.simultaneous, parallel=args.parallel,
                      temperature=args.temperature, history_rounds=args.history_rounds)
        _complete_cli_experiment(exp_id)
        return

    if not args.eval:
        print("BarterBench: competitive marketplace benchmark with ELO ratings")
        print()
        print("Benchmark mode (hybrid anchor — fast leaderboard):")
        print("  python3 eval.py --benchmark --models sonnet,opus --runs 3")
        print("  python3 eval.py --benchmark --anchor sonnet --models opus,haiku --eval grand_bazaar")
        print()
        print("Pairwise mode (compare models head-to-head with ELO):")
        print("  python3 eval.py --eval <scenario> --models <config>")
        print("  python3 eval.py --eval all --models haiku,opus --runs 3")
        print()
        print("Arena mode (compare prompt strategies across models):")
        print("  python3 eval.py --arena --eval all --runs 3")
        print("  python3 eval.py --arena --models haiku,sonnet,opus --eval all")
        print("  python3 eval.py --submit 'my_strat' 'Trade aggressively'")
        print()
        print("Cross-provider matrix (all pairwise comparisons):")
        print("  python3 eval.py --matrix --models haiku,sonnet,opus,hunter --runs 3")
        print()
        print("Analysis:")
        print("  python3 eval.py --scaling-report    # Performance vs model size/cost")
        print("  python3 eval.py --behavior-report   # Emergent behavior taxonomy")
        print("  python3 eval.py --elo               # View ELO ratings")
        print("  python3 eval.py --list              # List scenarios & strategies")
        print("  python3 eval.py --serve             # Dashboard")
        return

    if not args.models:
        if args.eval == "all":
            print("Error: --models required for tournament. Use two model names.")
            print("  Example: --models haiku,opus")
        else:
            scenario = load_scenario(args.eval)
            n = len(scenario["agents"])
            print(f"Error: --models required. Scenario '{args.eval}' has {n} agents.")
            print(f"  Example: --models haiku:{n//2},opus:{n - n//2}")
        sys.exit(1)

    if args.eval == "all":
        model_parts = [m.strip() for m in args.models.split(",")]
        model_names = [m.split(":")[0] for m in model_parts]
        if len(model_names) != 2:
            print("Error: tournament mode requires exactly 2 models")
            print("  Example: --models haiku,opus")
            sys.exit(1)
        exp_id = _register_cli_experiment("eval", "all", args.models, args.runs,
                                          simultaneous=args.simultaneous, parallel=args.parallel)
        run_tournament(model_names[0], model_names[1], args.runs, args.verbose,
                       simultaneous=args.simultaneous, parallel=args.parallel,
                       temperature=args.temperature, history_rounds=args.history_rounds)
        _complete_cli_experiment(exp_id)
    else:
        exp_id = _register_cli_experiment("eval", args.eval, args.models, args.runs,
                                          simultaneous=args.simultaneous, parallel=args.parallel)
        run_eval(args.eval, args.models, args.runs, args.verbose,
                 simultaneous=args.simultaneous, parallel=args.parallel,
                 temperature=args.temperature, history_rounds=args.history_rounds)
        _complete_cli_experiment(exp_id)


if __name__ == "__main__":
    main()
