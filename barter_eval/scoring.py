"""Compute evaluation metrics from a completed bartering session."""


def compute_max_surplus(scenario):
    """Maximum possible surplus: every item goes to whoever values it more."""
    agents = scenario["agents"]

    # Collect total items in the economy
    all_items = {}
    for a in agents:
        for item, qty in a["inventory"].items():
            all_items[item] = all_items.get(item, 0) + qty

    # Optimal allocation: each unit goes to the agent who values it more
    optimal_total = 0
    for item, total_qty in all_items.items():
        v0 = agents[0]["valuations"].get(item, 0)
        v1 = agents[1]["valuations"].get(item, 0)
        optimal_total += total_qty * max(v0, v1)

    # Initial total value (before any trades)
    initial_total = sum(
        sum(qty * agents[i]["valuations"].get(item, 0)
            for item, qty in agents[i]["inventory"].items())
        for i in range(2)
    )

    return optimal_total - initial_total


def compute_metrics(result, scenario):
    """Compute all eval metrics from a completed session."""
    iv = result["initial_values"]
    fv = result["final_values"]

    utility_gains = [fv[i] - iv[i] for i in range(2)]
    total_surplus = sum(utility_gains)
    max_surplus = compute_max_surplus(scenario)

    # Pareto efficiency: what fraction of possible gains from trade were captured
    pareto_efficiency = total_surplus / max_surplus if max_surplus > 0 else (1.0 if total_surplus == 0 else 0.0)

    # Deal rate: did any trades happen
    deal_rate = 1.0 if result["num_trades"] > 0 else 0.0

    # Fairness: Gini coefficient of surplus split (0 = equal, 1 = all to one side)
    if total_surplus > 0:
        shares = [max(0, g) / total_surplus for g in utility_gains]
        gini = abs(shares[0] - shares[1])
    else:
        gini = 0.0

    # Invalid trade rate
    invalid_count = sum(1 for h in result["history"] if h.get("invalid"))
    total_proposals = sum(1 for h in result["history"] if h["action"] == "propose_trade")
    invalid_rate = invalid_count / total_proposals if total_proposals > 0 else 0.0

    # Rounds to first agreement
    rounds_to_first = None
    for h in result["history"]:
        if h["action"] == "accept_trade":
            rounds_to_first = h["round"] + 1
            break

    # Composite barter score (normalized 0-1)
    barter_score = (
        0.4 * min(pareto_efficiency, 1.0)
        + 0.3 * deal_rate
        + 0.2 * (1.0 - gini)
        + 0.1 * (1.0 - invalid_rate)
    )

    return {
        "utility_gain_0": utility_gains[0],
        "utility_gain_1": utility_gains[1],
        "total_surplus": total_surplus,
        "max_surplus": max_surplus,
        "pareto_efficiency": round(pareto_efficiency, 4),
        "deal_rate": deal_rate,
        "fairness_gini": round(gini, 4),
        "invalid_rate": round(invalid_rate, 4),
        "rounds_to_first": rounds_to_first,
        "num_trades": result["num_trades"],
        "num_turns": result["num_turns"],
        "barter_score": round(barter_score, 4),
    }
