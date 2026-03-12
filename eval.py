"""CLI entry point: run marketplace eval across scenarios."""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import MarketAgent
from engine import MarketEngine
from elo import record_match, print_elo_leaderboard, reset_ratings, load_ratings
from scoring import compute_metrics

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_FILE = Path(__file__).parent / "results.json"
RUNS_DIR = Path(__file__).parent / "runs"
BENCHMARK_RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"


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
    """Assign models to agent slots randomly."""
    assignments = []
    for model, count in model_config:
        assignments.extend([model] * count)

    total = sum(c for _, c in model_config)
    if total != num_agents:
        print(f"Error: model config specifies {total} agents but scenario has {num_agents}")
        sys.exit(1)

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

    print("\n" + "=" * 75)
    print("  LEADERBOARD")
    print("=" * 75)
    if elo_ratings:
        print(f"  {'Model':<12} {'ELO':>8} {'Avg Goal%':>10} {'Avg Trades':>11} {'Invalid%':>9} {'Runs':>6}")
    else:
        print(f"  {'Model':<12} {'Avg Goal%':>10} {'Avg Trades':>11} {'Invalid%':>9} {'Runs':>6}")
    print("-" * 75)

    rows = []
    for model, stats in model_stats.items():
        n = len(stats["scores"])
        rows.append((
            model,
            elo_ratings.get(model, 0),
            sum(stats["scores"]) / n * 100,
            sum(stats["trades"]) / n,
            sum(stats["invalids"]) / n * 100,
            n,
        ))
    rows.sort(key=lambda r: (r[1] if r[1] else 0, r[2]), reverse=True)

    for model, elo, score, trades, invalids, n in rows:
        if elo_ratings:
            print(f"  {model:<12} {elo:>8.1f} {score:>9.1f}% {trades:>11.1f} {invalids:>8.1f}% {n:>6}")
        else:
            print(f"  {model:<12} {score:>9.1f}% {trades:>11.1f} {invalids:>8.1f}% {n:>6}")
    print("=" * 75)


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

    print("\n" + "=" * 80)
    print("  BENCHMARK LEADERBOARD")
    print("=" * 80)
    print(f"  {'Model':<14} {'Avg Goal%':>10} {'vs Anchor':>12} {'W-L-D':>10}")
    print("-" * 80)

    rows = []
    for model, stats in model_stats.items():
        n = len(stats["scores"])
        avg = sum(stats["scores"]) / n * 100
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
        rows.append((model, avg, vs_str, wld, model == anchor))

    rows.sort(key=lambda r: r[1], reverse=True)

    for model, avg, vs_str, wld, is_anchor in rows:
        marker = " *" if is_anchor else "  "
        print(f"{marker}{model:<12} {avg:>9.1f}% {vs_str:>12} {wld:>10}")

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


def run_benchmark(scenario_name, anchor, test_models, runs, verbose):
    """Run benchmark mode: test models against anchor in a single scenario."""
    scenario = load_scenario(scenario_name)
    num_agents = len(scenario["agents"])

    config, config_str = build_benchmark_config(anchor, test_models, num_agents)

    print(f"Benchmark Mode: {scenario_name}")
    print(f"  Anchor: {anchor} ({config[0][1]} agents)")
    test_info = ", ".join(f"{m}({c})" for m, c in config[1:])
    print(f"  Test models: {test_info}")
    print(f"  Agents: {num_agents} | Rounds: {scenario.get('max_rounds', 10)} | Runs: {runs}")
    print()

    if BENCHMARK_RESULTS_FILE.exists():
        with open(BENCHMARK_RESULTS_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = []

    run_offset = len(all_results)
    run_entries = []

    for run_num in range(runs):
        run_id = run_offset + run_num + 1
        print(f"  Run {run_num + 1}/{runs}")
        entry = run_single(scenario_name, scenario, config, config_str, run_id, verbose)
        entry["mode"] = "benchmark"
        entry["anchor_model"] = anchor
        run_entries.append(entry)
        all_results.append(entry)

        with open(BENCHMARK_RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        save_run_file(entry, scenario_name)

    print_benchmark_leaderboard(run_entries, anchor)
    print(f"\nResults saved to {BENCHMARK_RESULTS_FILE}")


# ---- Run a single match (model vs model) ----

def run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose):
    """Run a single marketplace match. Returns the result entry."""
    num_agents = len(scenario["agents"])
    model_assignments = assign_models(model_config, num_agents)
    agents = [
        MarketAgent(model_name=model_assignments[i], agent_idx=i)
        for i in range(num_agents)
    ]

    print(f"  [{scenario_name}] models: {' '.join(model_assignments)}")
    print(f"  Running marketplace...", end="", flush=True)

    try:
        engine = MarketEngine(scenario, agents)
        result = engine.run()
        result["scenario_data"] = scenario
        metrics = compute_metrics(result)

        clean_history = list(result.get("history", []))

        entry = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "model_config": model_config_str,
            "model_assignments": model_assignments,
            **metrics,
            "elapsed_seconds": result["elapsed_seconds"],
            "agent_tokens": [
                {"model": agents[i].model_name,
                 "tokens": agents[i].total_input_tokens + agents[i].total_output_tokens}
                for i in range(num_agents)
            ],
            "initial_inventories": result["initial_inventories"],
            "targets": [dict(a.get("target", {})) for a in scenario["agents"]],
            "agent_results": result["agent_results"],
            "trades": result["trades"],
            "history": clean_history,
        }

        status = f"goal={metrics['avg_goal_completion']*100:.0f}% trades={metrics['num_trades']} invalid={metrics['invalid_rate']*100:.0f}%"
        print(f" done ({status})")

        for model, score in metrics["model_goal_completion"].items():
            print(f"    {model}: {score*100:.1f}% goal completion")

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


# ---- Orchestration modes ----

def run_eval(scenario_name, model_config_str, runs, verbose):
    """Run eval for a single scenario."""
    scenario = load_scenario(scenario_name)
    num_agents = len(scenario["agents"])
    model_config = parse_model_config(model_config_str)

    total_assigned = sum(c for _, c in model_config)
    if total_assigned != num_agents:
        print(f"Error: scenario '{scenario_name}' has {num_agents} agents but model config specifies {total_assigned}")
        print(f"  Use --models with correct counts, e.g. --models haiku:{num_agents // 2},opus:{num_agents // 2}")
        sys.exit(1)

    results = load_existing_results()
    run_offset = len(results)

    print(f"Marketplace Eval: {scenario_name}")
    print(f"  Agents: {num_agents} | Models: {model_config_str} | Rounds: {scenario.get('max_rounds', 10)} | Runs: {runs}")
    print()

    for run_num in range(runs):
        run_id = run_offset + run_num + 1
        entry = run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose)
        results.append(entry)
        save_results(results)
        save_run_file(entry, scenario_name)

    print_leaderboard(results)
    print_elo_leaderboard()
    print(f"\nResults saved to {RESULTS_FILE}")


def run_tournament(model_a, model_b, runs_per_scenario, verbose):
    """Run a full tournament: model_a vs model_b across all scenarios."""
    scenarios = list_scenarios()
    if not scenarios:
        print("Error: no scenarios found")
        sys.exit(1)

    results = load_existing_results()
    run_offset = len(results)

    total_matches = len(scenarios) * runs_per_scenario
    print(f"Tournament: {model_a} vs {model_b}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Runs per scenario: {runs_per_scenario}")
    print(f"  Total matches: {total_matches}")
    print()

    match_num = 0
    for scenario_name in scenarios:
        scenario = load_scenario(scenario_name)
        num_agents = len(scenario["agents"])
        half_a, half_b = auto_model_config(scenario)
        model_config = [(model_a, half_a), (model_b, half_b)]
        model_config_str = f"{model_a}:{half_a},{model_b}:{half_b}"

        print(f"--- {scenario_name} ({num_agents} agents, {scenario.get('max_rounds', 10)} rounds) ---")

        for run_num in range(runs_per_scenario):
            match_num += 1
            run_id = run_offset + match_num
            print(f"\n  Match {match_num}/{total_matches}")

            entry = run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose)
            results.append(entry)
            save_results(results)
            save_run_file(entry, scenario_name)

        print()

    print_leaderboard(results)
    print_elo_leaderboard()
    print(f"\nResults saved to {RESULTS_FILE}")


# ---- Dashboard ----

def serve_dashboard():
    """Serve the dashboard on a local HTTP server."""
    import http.server
    import webbrowser

    os.chdir(Path(__file__).parent)
    port = 8080

    print(f"Serving dashboard at http://localhost:{port}/dashboard.html")
    webbrowser.open(f"http://localhost:{port}/dashboard.html")

    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="BarterBench: competitive marketplace benchmark with ELO ratings")
    parser.add_argument("--eval", type=str, default=None,
                        help="Scenario to evaluate, or 'all' for full tournament")
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
    args = parser.parse_args()

    if args.serve:
        serve_dashboard()
        return

    if args.elo:
        print_elo_leaderboard()
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
        run_arena(strat_names, scenario, args.runs, args.verbose, models=models)
        return

    # ---- Benchmark mode (hybrid anchor) ----
    if args.benchmark:
        if not args.models:
            print("Error: --models required for benchmark mode.")
            print("  Example: --benchmark --models sonnet,opus")
            sys.exit(1)
        scenario_name = args.eval or "grand_bazaar"
        test_models = [m.strip().split(":")[0] for m in args.models.split(",")]
        test_models = [m for m in test_models if m != args.anchor]
        if not test_models:
            print("Error: --models must include at least one model besides the anchor.")
            sys.exit(1)
        run_benchmark(scenario_name, args.anchor, test_models, args.runs or 3, args.verbose)
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
        print("Other:")
        print("  python3 eval.py --elo      # View ELO ratings")
        print("  python3 eval.py --list     # List scenarios & strategies")
        print("  python3 eval.py --serve    # Dashboard")
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
        run_tournament(model_names[0], model_names[1], args.runs, args.verbose)
    else:
        run_eval(args.eval, args.models, args.runs, args.verbose)


if __name__ == "__main__":
    main()
