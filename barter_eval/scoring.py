"""Compute evaluation metrics for marketplace runs."""


def compute_metrics(result):
    """Compute metrics from a completed marketplace session.

    Returns per-model aggregates and overall stats.
    """
    agent_results = result["agent_results"]
    history = result["history"]

    # Per-model goal completion
    model_scores = {}
    for ar in agent_results:
        model = ar["model"]
        if model not in model_scores:
            model_scores[model] = []
        model_scores[model].append(ar["goal_completion"])

    model_avg = {}
    for model, scores in model_scores.items():
        model_avg[model] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Overall goal completion
    all_completions = [ar["goal_completion"] for ar in agent_results]
    avg_completion = sum(all_completions) / len(all_completions) if all_completions else 0.0

    # Trade activity
    num_trades = result["num_trades"]
    num_turns = result["num_turns"]

    # Invalid action rate
    invalid_count = sum(1 for h in history if h.get("invalid"))
    non_pass = sum(1 for h in history if h["action"] != "pass_turn")
    invalid_rate = invalid_count / non_pass if non_pass > 0 else 0.0

    # Pass rate (how often agents pass vs take action)
    pass_count = sum(1 for h in history if h["action"] == "pass_turn")
    pass_rate = pass_count / num_turns if num_turns > 0 else 0.0

    # Trade efficiency: trades per round
    rounds_used = max(h["round"] for h in history) + 1 if history else 0
    trades_per_round = num_trades / rounds_used if rounds_used > 0 else 0.0

    # Scarce item capture: how much of each scarce item each model ended up with
    scarce_capture = compute_scarce_capture(result)

    metrics = {
        "avg_goal_completion": round(avg_completion, 4),
        "model_goal_completion": model_avg,
        "num_trades": num_trades,
        "num_turns": num_turns,
        "invalid_rate": round(invalid_rate, 4),
        "pass_rate": round(pass_rate, 4),
        "trades_per_round": round(trades_per_round, 4),
        "per_agent": [
            {"agent": ar["agent_idx"], "model": ar["model"], "goal_completion": ar["goal_completion"]}
            for ar in agent_results
        ],
    }
    if scarce_capture:
        metrics["scarce_capture"] = scarce_capture

    return metrics


def compute_scarce_capture(result):
    """Compute how much of each scarce item each model captured.

    Returns {item: {model: total_qty}} for items in scarcity metadata.
    """
    scenario = result.get("scenario_data", {})
    scarcity = scenario.get("scarcity", {})
    if not scarcity:
        return {}

    agent_results = result["agent_results"]
    capture = {}
    for item in scarcity:
        capture[item] = {}
        for ar in agent_results:
            model = ar["model"]
            qty = ar["final_inventory"].get(item, 0)
            capture[item][model] = capture[item].get(model, 0) + qty

    return capture
