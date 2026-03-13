"""Post-hoc analysis: scaling curves, cost frontiers, efficiency rankings.

Reads results.json and model_registry to produce cross-dimensional analysis.
"""

import json
import math
from pathlib import Path

from model_registry import get_model_info, get_model_size, get_size_tier, compute_dollar_cost


RESULTS_FILE = Path(__file__).parent / "results.json"


def load_results(path=None):
    """Load results from JSON file."""
    p = Path(path) if path else RESULTS_FILE
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def compute_scaling_analysis(results):
    """Analyze how model performance scales with parameter count.

    Returns scatter data, log-linear fit, per-capability scaling,
    and Pareto-optimal models on the size vs performance frontier.
    """
    # Collect per-model aggregated scores + metadata
    model_data = {}
    for entry in results:
        mgc = entry.get("model_goal_completion", {})
        for model, score in mgc.items():
            if model not in model_data:
                info = get_model_info(model)
                model_data[model] = {
                    "scores": [],
                    "parameters_b": info.get("parameters_b"),
                    "family": info.get("family", "unknown"),
                    "provider": info.get("provider", "unknown"),
                    "cost_tier": info.get("cost_tier", "unknown"),
                    "size_tier": get_size_tier(model),
                }
            model_data[model]["scores"].append(score)

    # Build scatter data (only models with known size)
    scatter = []
    for model, data in model_data.items():
        mean_score = sum(data["scores"]) / len(data["scores"])
        size = data["parameters_b"]
        scatter.append({
            "model": model,
            "parameters_b": size,
            "mean_goal_completion": round(mean_score, 4),
            "n_runs": len(data["scores"]),
            "family": data["family"],
            "provider": data["provider"],
            "size_tier": data["size_tier"],
            "cost_tier": data["cost_tier"],
        })

    # Log-linear fit (log(params) vs score) — only for models with known size
    sized = [(math.log(s["parameters_b"]), s["mean_goal_completion"])
             for s in scatter if s["parameters_b"] and s["parameters_b"] > 0]

    fit = None
    if len(sized) >= 3:
        n = len(sized)
        sum_x = sum(x for x, _ in sized)
        sum_y = sum(y for _, y in sized)
        sum_xy = sum(x * y for x, y in sized)
        sum_x2 = sum(x * x for x, _ in sized)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) > 1e-10:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

            # R-squared
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for _, y in sized)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in sized)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            fit = {
                "slope": round(slope, 4),
                "intercept": round(intercept, 4),
                "r_squared": round(r_squared, 4),
                "interpretation": (
                    f"Each 10x increase in parameters → "
                    f"{slope * math.log(10) * 100:.1f}pp goal completion"
                ),
            }

    # Pareto frontier: models that are not dominated on size vs performance
    pareto = []
    for s in sorted(scatter, key=lambda x: (x["parameters_b"] or float("inf"))):
        if s["parameters_b"] is None:
            continue
        # A model is Pareto-optimal if no smaller model has higher performance
        dominated = False
        for other in scatter:
            if (other["parameters_b"] is not None
                    and other["parameters_b"] < s["parameters_b"]
                    and other["mean_goal_completion"] >= s["mean_goal_completion"]):
                dominated = True
                break
        if not dominated:
            pareto.append(s["model"])

    # Per-tier summary
    tier_summary = {}
    for s in scatter:
        tier = s["size_tier"]
        tier_summary.setdefault(tier, []).append(s["mean_goal_completion"])
    for tier, scores in tier_summary.items():
        tier_summary[tier] = {
            "mean": round(sum(scores) / len(scores), 4),
            "n_models": len(scores),
        }

    return {
        "scatter": scatter,
        "log_linear_fit": fit,
        "pareto_frontier": pareto,
        "tier_summary": tier_summary,
        "total_models": len(scatter),
    }


def compute_cost_frontier(results):
    """Analyze cost vs performance frontier.

    Returns Pareto frontier of (total_cost_usd, goal_completion)
    and identifies best-value models.
    """
    model_data = {}
    for entry in results:
        mgc = entry.get("model_goal_completion", {})
        tokens = entry.get("agent_tokens", [])

        # Aggregate tokens per model
        model_tokens = {}
        for at in tokens:
            m = at["model"]
            model_tokens.setdefault(m, {"input": 0, "output": 0})
            model_tokens[m]["input"] += at.get("input_tokens", 0)
            model_tokens[m]["output"] += at.get("output_tokens", 0)

        for model, score in mgc.items():
            if model not in model_data:
                model_data[model] = {"scores": [], "costs": []}
            model_data[model]["scores"].append(score)

            tok = model_tokens.get(model, {"input": 0, "output": 0})
            cost = compute_dollar_cost(model, tok["input"], tok["output"])
            model_data[model]["costs"].append(cost)

    frontier = []
    for model, data in model_data.items():
        mean_score = sum(data["scores"]) / len(data["scores"])
        mean_cost = sum(data["costs"]) / len(data["costs"])
        frontier.append({
            "model": model,
            "mean_goal_completion": round(mean_score, 4),
            "mean_cost_usd": round(mean_cost, 6),
            "cost_per_goal_point": round(mean_cost / mean_score, 6) if mean_score > 0 else float("inf"),
            "n_runs": len(data["scores"]),
        })

    # Sort by cost
    frontier.sort(key=lambda x: x["mean_cost_usd"])

    # Identify Pareto-optimal (best performance at each cost level)
    pareto = []
    best_perf = -1
    for f in frontier:
        if f["mean_goal_completion"] > best_perf:
            pareto.append(f["model"])
            best_perf = f["mean_goal_completion"]

    return {
        "models": frontier,
        "pareto_frontier": pareto,
        "cheapest_model": frontier[0]["model"] if frontier else None,
        "best_value": min(frontier, key=lambda x: x["cost_per_goal_point"])["model"] if frontier else None,
    }


def compute_efficiency_ranking(results):
    """Rank models by token efficiency: tokens per trade, tokens per goal point."""
    model_data = {}
    for entry in results:
        mgc = entry.get("model_goal_completion", {})
        tokens = entry.get("agent_tokens", [])

        model_tokens = {}
        for at in tokens:
            m = at["model"]
            model_tokens.setdefault(m, 0)
            model_tokens[m] += at.get("tokens", 0)

        num_trades = entry.get("num_trades", 0)
        num_agents = len(entry.get("model_assignments", []))

        for model, score in mgc.items():
            if model not in model_data:
                model_data[model] = {"scores": [], "tokens": [], "trades": []}
            model_data[model]["scores"].append(score)
            model_data[model]["tokens"].append(model_tokens.get(model, 0))
            # Attribute trades proportionally
            agent_count = sum(1 for m in entry.get("model_assignments", []) if m == model)
            share = agent_count / num_agents if num_agents > 0 else 1
            model_data[model]["trades"].append(num_trades * share)

    ranking = []
    for model, data in model_data.items():
        total_tokens = sum(data["tokens"])
        total_trades = sum(data["trades"])
        mean_score = sum(data["scores"]) / len(data["scores"])

        tokens_per_trade = total_tokens / total_trades if total_trades > 0 else float("inf")
        tokens_per_goal_point = total_tokens / (mean_score * len(data["scores"])) if mean_score > 0 else float("inf")

        ranking.append({
            "model": model,
            "total_tokens": total_tokens,
            "mean_goal_completion": round(mean_score, 4),
            "tokens_per_trade": round(tokens_per_trade, 1),
            "tokens_per_goal_point": round(tokens_per_goal_point, 1),
            "n_runs": len(data["scores"]),
        })

    ranking.sort(key=lambda x: x["tokens_per_goal_point"])
    return ranking


def print_scaling_report(results=None, path=None):
    """Print a human-readable scaling analysis report."""
    if results is None:
        results = load_results(path)

    if not results:
        print("No results found. Run some evals first.")
        return

    scaling = compute_scaling_analysis(results)
    cost = compute_cost_frontier(results)
    efficiency = compute_efficiency_ranking(results)

    print("\n" + "=" * 70)
    print("SCALING ANALYSIS REPORT")
    print("=" * 70)

    # Model performance by size
    print(f"\nModels analyzed: {scaling['total_models']}")
    print(f"\n{'Model':<20s} {'Size':>8s} {'Tier':<10s} {'Goal%':>8s} {'N':>4s}")
    print("-" * 54)
    for s in sorted(scaling["scatter"], key=lambda x: -(x["mean_goal_completion"])):
        size_str = f"{s['parameters_b']}B" if s["parameters_b"] else "?B"
        print(f"  {s['model']:<18s} {size_str:>8s} {s['size_tier']:<10s} "
              f"{s['mean_goal_completion']*100:>7.1f}% {s['n_runs']:>4d}")

    # Scaling fit
    if scaling["log_linear_fit"]:
        fit = scaling["log_linear_fit"]
        print(f"\nLog-linear fit: R² = {fit['r_squared']:.3f}")
        print(f"  {fit['interpretation']}")

    # Size tier summary
    if scaling["tier_summary"]:
        print(f"\n{'Size Tier':<12s} {'Mean Goal%':>10s} {'N Models':>10s}")
        print("-" * 34)
        for tier in ["baseline", "small", "medium", "large", "frontier", "unknown"]:
            if tier in scaling["tier_summary"]:
                t = scaling["tier_summary"][tier]
                print(f"  {tier:<10s} {t['mean']*100:>9.1f}% {t['n_models']:>10d}")

    # Pareto frontier
    if scaling["pareto_frontier"]:
        print(f"\nPareto-optimal (size vs performance): {', '.join(scaling['pareto_frontier'])}")

    # Cost analysis
    if cost["models"]:
        print(f"\n{'Model':<20s} {'Cost/Run':>10s} {'Goal%':>8s} {'$/GoalPt':>10s}")
        print("-" * 52)
        for c in sorted(cost["models"], key=lambda x: -x["mean_goal_completion"]):
            cost_str = f"${c['mean_cost_usd']:.4f}" if c["mean_cost_usd"] < 100 else f"${c['mean_cost_usd']:.2f}"
            cpg = f"${c['cost_per_goal_point']:.4f}" if c["cost_per_goal_point"] < 100 else "N/A"
            print(f"  {c['model']:<18s} {cost_str:>10s} {c['mean_goal_completion']*100:>7.1f}% {cpg:>10s}")

        if cost["best_value"]:
            print(f"\n  Best value: {cost['best_value']}")

    # Token efficiency
    if efficiency:
        print(f"\n{'Model':<20s} {'Tok/Trade':>10s} {'Tok/GoalPt':>12s} {'Goal%':>8s}")
        print("-" * 54)
        for e in efficiency:
            tpt = f"{e['tokens_per_trade']:.0f}" if e['tokens_per_trade'] < 1e6 else "N/A"
            tpg = f"{e['tokens_per_goal_point']:.0f}" if e['tokens_per_goal_point'] < 1e6 else "N/A"
            print(f"  {e['model']:<18s} {tpt:>10s} {tpg:>12s} {e['mean_goal_completion']*100:>7.1f}%")

    print("\n" + "=" * 70)
