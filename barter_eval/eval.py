"""CLI entry point: run bartering eval across models and scenarios."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from .agent import BarterAgent
from .engine import BarterEngine
from .scoring import compute_metrics

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_FILE = Path(__file__).parent / "results.json"


def load_scenarios(names):
    """Load scenario JSON files. 'all' loads everything in scenarios/."""
    if names == "all":
        files = sorted(SCENARIOS_DIR.glob("*.json"))
    else:
        files = [SCENARIOS_DIR / f"{n}.json" for n in names.split(",")]

    scenarios = []
    for f in files:
        if not f.exists():
            print(f"Warning: scenario {f} not found, skipping")
            continue
        with open(f) as fh:
            scenarios.append(json.load(fh))
    return scenarios


def get_model_pairs(models):
    """Generate all ordered pairs including self-play."""
    return list(product(models, repeat=2))


def load_existing_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def print_leaderboard(results):
    """Print aggregate model performance."""
    if not results:
        return

    # Aggregate by model (across both positions)
    model_stats = {}
    for r in results:
        for key in ["model_a", "model_b"]:
            model = r[key]
            if model not in model_stats:
                model_stats[model] = {"surplus": [], "pareto": [], "deals": [], "score": []}
            idx = 0 if key == "model_a" else 1
            model_stats[model]["surplus"].append(r[f"utility_gain_{idx}"])
            model_stats[model]["pareto"].append(r["pareto_efficiency"])
            model_stats[model]["deals"].append(r["deal_rate"])
            model_stats[model]["score"].append(r["barter_score"])

    print("\n" + "=" * 65)
    print("  LEADERBOARD")
    print("=" * 65)
    print(f"  {'Model':<12} {'Avg Score':>10} {'Avg Surplus':>12} {'Pareto':>8} {'Deal%':>7} {'Runs':>6}")
    print("-" * 65)

    rows = []
    for model, stats in model_stats.items():
        n = len(stats["score"])
        rows.append((
            model,
            sum(stats["score"]) / n,
            sum(stats["surplus"]) / n,
            sum(stats["pareto"]) / n,
            sum(stats["deals"]) / n * 100,
            n,
        ))
    rows.sort(key=lambda r: r[1], reverse=True)

    for model, score, surplus, pareto, deals, n in rows:
        print(f"  {model:<12} {score:>10.4f} {surplus:>12.1f} {pareto:>8.3f} {deals:>6.0f}% {n:>6}")
    print("=" * 65)


def run_eval(models, scenarios_name, runs, verbose):
    scenarios = load_scenarios(scenarios_name)
    if not scenarios:
        print("No scenarios found!")
        sys.exit(1)

    pairs = get_model_pairs(models)
    total = len(scenarios) * len(pairs) * runs
    results = load_existing_results()
    run_offset = len(results)

    print(f"Running barter eval: {len(scenarios)} scenarios × {len(pairs)} model pairs × {runs} runs = {total} evals")
    print()

    count = 0
    for scenario in scenarios:
        for model_a, model_b in pairs:
            for run_num in range(runs):
                count += 1
                label = f"  [{count}/{total}] {scenario['name']}: {model_a} vs {model_b}"
                print(f"{label} ...", end="", flush=True)

                try:
                    agents = [BarterAgent(model_a), BarterAgent(model_b)]
                    engine = BarterEngine(scenario, agents)
                    result = engine.run()
                    metrics = compute_metrics(result, scenario)

                    entry = {
                        "run_id": run_offset + count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "scenario": scenario["name"],
                        "model_a": model_a,
                        "model_b": model_b,
                        **metrics,
                        "elapsed_seconds": result["elapsed_seconds"],
                        "tokens_a": agents[0].total_input_tokens + agents[0].total_output_tokens,
                        "tokens_b": agents[1].total_input_tokens + agents[1].total_output_tokens,
                    }
                    results.append(entry)
                    save_results(results)

                    status = f"surplus={metrics['total_surplus']:.1f} pareto={metrics['pareto_efficiency']:.2f}"
                    print(f" done ({status})")

                    if verbose:
                        for h in result["history"]:
                            tag = "  " if h["agent"] == 0 else "    "
                            print(f"{tag}{h['role']} [{h['action']}]: {h.get('message', '')}")
                        print()

                except Exception as e:
                    print(f" ERROR: {e}")
                    results.append({
                        "run_id": run_offset + count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "scenario": scenario["name"],
                        "model_a": model_a,
                        "model_b": model_b,
                        "error": str(e),
                    })
                    save_results(results)

    print_leaderboard(results)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Dashboard: python -m barter_eval.eval --serve")


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


def main():
    parser = argparse.ArgumentParser(description="Barter Eval: benchmark model bartering capabilities")
    parser.add_argument("--models", nargs="+", default=["haiku", "opus"],
                        help="Models to evaluate (default: haiku opus)")
    parser.add_argument("--scenarios", default="all",
                        help="Scenario names (comma-separated) or 'all'")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per model pair per scenario")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full negotiation transcripts")
    parser.add_argument("--serve", action="store_true",
                        help="Serve the dashboard web UI")
    parser.add_argument("--clear", action="store_true",
                        help="Clear previous results before running")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard()
        return

    if args.clear and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
        print("Cleared previous results.")

    run_eval(args.models, args.scenarios, args.runs, args.verbose)


if __name__ == "__main__":
    main()
