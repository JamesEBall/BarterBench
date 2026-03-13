"""Emergent behavior taxonomy: classify and detect trading patterns.

Detects negotiation strategies, social behaviors, and meta-strategies
from the trade/action history without requiring any additional data collection.
"""

import json
import math


# ---- Behavior Definitions ----

BEHAVIOR_TAXONOMY = {
    "negotiation": {
        "anchoring": "Posts extreme initial offers then converges to reasonable rates",
        "lowball_accept": "Accepts offers significantly below apparent market value",
        "strategic_pass": "Passes when holding items others need, waiting for better deals",
        "intermediary_trading": "Acquires items NOT in target to use as leverage for future trades",
    },
    "social": {
        "coalition_formation": "Same-model agents trade disproportionately with each other",
        "price_discovery": "Exchange ratios converge over rounds (implicit price consensus)",
        "deception": "Makes false claims about inventory or market state",
        "authority_claim": "Attempts to direct other agents' behavior via authority language",
    },
    "meta": {
        "hoarding": "Accumulates scarce items beyond target requirements",
        "dumping": "Offloads items at unfavorable rates in late rounds (desperation selling)",
        "information_hiding": "Heavily favors private offers over public ones",
        "early_completion": "Achieves goal early then stops trading (rational withdrawal)",
    },
}


def classify_behaviors(result):
    """Analyze a single run and return detected behavior classifications.

    Args:
        result: a run entry dict (with history, trades, agent_results, etc.)

    Returns:
        dict with per-agent behavior detections and confidence scores.
    """
    history = result.get("history", [])
    trades = result.get("trades", [])
    agent_results = result.get("agent_results", [])
    initial_inventories = result.get("initial_inventories", [])

    if not history or not agent_results:
        return {}

    num_agents = len(agent_results)
    max_rounds = max((h["round"] for h in history), default=0) + 1

    # Build lookup maps
    agent_model = {ar["agent_idx"]: ar["model"] for ar in agent_results}
    agent_target = {ar["agent_idx"]: ar.get("target", {}) for ar in agent_results}
    agent_final = {ar["agent_idx"]: ar.get("final_inventory", {}) for ar in agent_results}

    per_agent = {}
    for idx in range(num_agents):
        behaviors = {}

        # Detect each behavior
        behaviors["anchoring"] = _detect_anchoring(idx, history, trades)
        behaviors["strategic_pass"] = _detect_strategic_pass(idx, history, trades, agent_target.get(idx, {}))
        behaviors["intermediary_trading"] = _detect_intermediary(idx, trades, agent_target.get(idx, {}))
        behaviors["hoarding"] = _detect_hoarding(idx, agent_final.get(idx, {}), agent_target.get(idx, {}), result)
        behaviors["dumping"] = _detect_dumping(idx, history, trades, max_rounds)
        behaviors["information_hiding"] = _detect_info_hiding(idx, history)
        behaviors["early_completion"] = _detect_early_completion(idx, history, agent_results, max_rounds)

        # Filter to detected behaviors (confidence > 0.3)
        detected = {k: v for k, v in behaviors.items() if v["confidence"] > 0.3}

        per_agent[idx] = {
            "model": agent_model.get(idx, "unknown"),
            "behaviors": detected,
            "behavior_count": len(detected),
        }

    return {
        "per_agent": per_agent,
        "summary": _summarize_behaviors(per_agent),
    }


def _detect_anchoring(agent_idx, history, trades):
    """Detect anchoring: extreme first offers followed by more moderate trades.

    Computes the ratio between first offer's implied exchange rate and
    eventual trade rates. Large divergence = anchoring.
    """
    agent_offers = [h for h in history if h["agent"] == agent_idx
                    and h["action"] in ("post_offer", "private_offer")
                    and not h.get("invalid")]
    agent_trades = [t for t in trades if t["poster"] == agent_idx or t["accepter"] == agent_idx]

    if len(agent_offers) < 2 or len(agent_trades) < 1:
        return {"detected": False, "confidence": 0.0}

    # Get first offer's implied rate
    first = agent_offers[0]
    first_give_total = sum(first.get("give", {}).values())
    first_want_total = sum(first.get("want", {}).values())
    if first_give_total == 0 or first_want_total == 0:
        return {"detected": False, "confidence": 0.0}
    first_ratio = first_want_total / first_give_total

    # Get average trade ratio
    trade_ratios = []
    for t in agent_trades:
        give_total = sum(t["give"].values())
        want_total = sum(t["want"].values())
        if give_total > 0 and want_total > 0:
            if t["poster"] == agent_idx:
                trade_ratios.append(want_total / give_total)
            else:
                trade_ratios.append(give_total / want_total)

    if not trade_ratios:
        return {"detected": False, "confidence": 0.0}

    avg_trade_ratio = sum(trade_ratios) / len(trade_ratios)

    # High divergence between first offer and actual trades = anchoring
    if avg_trade_ratio > 0:
        divergence = abs(first_ratio - avg_trade_ratio) / avg_trade_ratio
    else:
        divergence = 0

    confidence = min(divergence / 2.0, 1.0)  # >2x divergence = full confidence
    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "first_offer_ratio": round(first_ratio, 2),
        "avg_trade_ratio": round(avg_trade_ratio, 2),
    }


def _detect_strategic_pass(agent_idx, history, trades, target):
    """Detect strategic passing: agent passes when they have items others want
    AND viable offers exist on the book they could accept."""
    agent_actions = [h for h in history if h["agent"] == agent_idx]
    passes = [h for h in agent_actions if h["action"] == "pass_turn"]

    if not passes or not agent_actions:
        return {"detected": False, "confidence": 0.0}

    # Strategic passes: passes where the agent later made a better trade
    total_passes = len(passes)
    strategic_count = 0

    for p in passes:
        p_round = p["round"]
        # Did agent make a trade in a later round?
        later_trades = [t for t in trades if t["round"] > p_round
                        and (t["poster"] == agent_idx or t["accepter"] == agent_idx)]
        if later_trades:
            strategic_count += 1

    rate = strategic_count / total_passes if total_passes > 0 else 0
    pass_rate = total_passes / len(agent_actions) if agent_actions else 0

    # High strategic pass rate with moderate overall pass rate = strategic
    confidence = min(rate * (1 - abs(pass_rate - 0.3)) * 2, 1.0)
    confidence = max(confidence, 0.0)

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "strategic_passes": strategic_count,
        "total_passes": total_passes,
        "pass_rate": round(pass_rate, 3),
    }


def _detect_intermediary(agent_idx, trades, target):
    """Detect intermediary trading: acquiring items NOT in target as leverage."""
    target_items = set(target.keys())
    intermediary_count = 0
    total_trades = 0

    for t in trades:
        if t["poster"] == agent_idx:
            total_trades += 1
            # Poster receives 'want' items
            received = set(t["want"].keys())
            if received and not received.issubset(target_items):
                intermediary_count += 1
        elif t["accepter"] == agent_idx:
            total_trades += 1
            # Accepter receives 'give' items
            received = set(t["give"].keys())
            if received and not received.issubset(target_items):
                intermediary_count += 1

    if total_trades == 0:
        return {"detected": False, "confidence": 0.0}

    rate = intermediary_count / total_trades
    confidence = min(rate * 2, 1.0)  # >50% intermediary trades = full confidence

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "intermediary_trades": intermediary_count,
        "total_trades": total_trades,
        "rate": round(rate, 3),
    }


def _detect_hoarding(agent_idx, final_inventory, target, result):
    """Detect hoarding: holding more of a scarce item than target requires."""
    scarcity = result.get("scenario_data", {}).get("scarcity", {})
    if not scarcity or not final_inventory:
        return {"detected": False, "confidence": 0.0}

    hoarded_items = []
    for item in scarcity:
        held = final_inventory.get(item, 0)
        needed = target.get(item, 0)
        excess = held - needed
        if excess > 0:
            hoarded_items.append({"item": item, "held": held, "needed": needed, "excess": excess})

    if not hoarded_items:
        return {"detected": False, "confidence": 0.0}

    total_excess = sum(h["excess"] for h in hoarded_items)
    confidence = min(total_excess / 3.0, 1.0)  # 3+ excess scarce items = full confidence

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "hoarded": hoarded_items,
    }


def _detect_dumping(agent_idx, history, trades, max_rounds):
    """Detect dumping: unfavorable trades in late rounds (desperation selling)."""
    if max_rounds < 3:
        return {"detected": False, "confidence": 0.0}

    late_threshold = max_rounds * 0.7  # last 30% of game

    early_ratios = []
    late_ratios = []

    for t in trades:
        if t["poster"] != agent_idx:
            continue
        give_total = sum(t["give"].values())
        want_total = sum(t["want"].values())
        if give_total == 0:
            continue
        ratio = want_total / give_total

        if t["round"] < late_threshold:
            early_ratios.append(ratio)
        else:
            late_ratios.append(ratio)

    if not early_ratios or not late_ratios:
        return {"detected": False, "confidence": 0.0}

    avg_early = sum(early_ratios) / len(early_ratios)
    avg_late = sum(late_ratios) / len(late_ratios)

    # Dumping = late trades have worse rates than early trades
    if avg_early > 0:
        deterioration = (avg_early - avg_late) / avg_early
    else:
        deterioration = 0

    confidence = min(max(deterioration, 0) * 2, 1.0)

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "avg_early_ratio": round(avg_early, 3),
        "avg_late_ratio": round(avg_late, 3),
        "deterioration": round(deterioration, 3),
    }


def _detect_info_hiding(agent_idx, history):
    """Detect information hiding: heavy use of private offers over public."""
    offers = [h for h in history if h["agent"] == agent_idx
              and h["action"] in ("post_offer", "private_offer")
              and not h.get("invalid")]

    if len(offers) < 2:
        return {"detected": False, "confidence": 0.0}

    private_count = sum(1 for h in offers if h["action"] == "private_offer")
    rate = private_count / len(offers)

    # >60% private = detected, confidence scales from 0.3 at 40% to 1.0 at 80%
    confidence = min(max((rate - 0.3) * 2, 0), 1.0)

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "private_rate": round(rate, 3),
        "private_offers": private_count,
        "total_offers": len(offers),
    }


def _detect_early_completion(agent_idx, history, agent_results, max_rounds):
    """Detect early completion: achieves goal then stops trading."""
    ar = next((a for a in agent_results if a["agent_idx"] == agent_idx), None)
    if not ar or ar["goal_completion"] < 0.95:
        return {"detected": False, "confidence": 0.0}

    # Find the last round this agent took a non-pass action
    non_pass = [h for h in history if h["agent"] == agent_idx and h["action"] != "pass_turn"]
    if not non_pass:
        return {"detected": False, "confidence": 0.0}

    last_active_round = max(h["round"] for h in non_pass)
    remaining_rounds = max_rounds - 1 - last_active_round

    if remaining_rounds < 2:
        return {"detected": False, "confidence": 0.0}

    # Early withdrawal: stopped trading with multiple rounds remaining
    confidence = min(remaining_rounds / (max_rounds * 0.3), 1.0)

    return {
        "detected": confidence > 0.3,
        "confidence": round(confidence, 3),
        "last_active_round": last_active_round,
        "rounds_remaining": remaining_rounds,
        "goal_completion": ar["goal_completion"],
    }


def _summarize_behaviors(per_agent):
    """Aggregate behavior detections into per-model summaries."""
    model_behaviors = {}

    for idx, data in per_agent.items():
        model = data["model"]
        if model not in model_behaviors:
            model_behaviors[model] = {
                "agent_count": 0,
                "behavior_counts": {},
                "avg_behavior_count": 0,
            }
        model_behaviors[model]["agent_count"] += 1

        for behavior_name in data["behaviors"]:
            model_behaviors[model]["behavior_counts"][behavior_name] = \
                model_behaviors[model]["behavior_counts"].get(behavior_name, 0) + 1

    for model, data in model_behaviors.items():
        total = sum(data["behavior_counts"].values())
        data["avg_behavior_count"] = round(total / data["agent_count"], 2) if data["agent_count"] > 0 else 0

    return model_behaviors


def aggregate_behavior_profile(entries):
    """Build per-model behavior frequency table across multiple runs.

    Args:
        entries: list of run entries (from results.json)

    Returns:
        dict with per-model behavior frequency rates.
    """
    model_totals = {}  # model -> {behavior: count, "_agents": total_agents}

    for entry in entries:
        classification = classify_behaviors(entry)
        if not classification:
            continue

        for idx, data in classification.get("per_agent", {}).items():
            model = data["model"]
            if model not in model_totals:
                model_totals[model] = {"_agents": 0}
            model_totals[model]["_agents"] += 1

            for behavior_name in data["behaviors"]:
                model_totals[model][behavior_name] = model_totals[model].get(behavior_name, 0) + 1

    # Convert to rates
    profile = {}
    for model, data in model_totals.items():
        total_agents = data["_agents"]
        if total_agents == 0:
            continue
        profile[model] = {
            "n_agent_instances": total_agents,
            "behavior_rates": {
                k: round(v / total_agents, 3)
                for k, v in data.items()
                if k != "_agents"
            },
        }

    return profile


def compute_price_discovery(result):
    """Measure price discovery: do exchange ratios converge over rounds?

    Returns per-item-pair convergence metrics.
    """
    trades = result.get("trades", [])
    if len(trades) < 4:
        return None

    # Build exchange ratio timeseries per item pair
    pair_ratios = {}  # (item_a, item_b) -> [(round, ratio)]

    for t in trades:
        for give_item, give_qty in t["give"].items():
            for want_item, want_qty in t["want"].items():
                if give_qty > 0:
                    pair = tuple(sorted([give_item, want_item]))
                    ratio = want_qty / give_qty if give_item == pair[0] else give_qty / want_qty
                    pair_ratios.setdefault(pair, []).append((t["round"], ratio))

    convergence = {}
    for pair, timeseries in pair_ratios.items():
        if len(timeseries) < 3:
            continue

        # Split into first half and second half
        timeseries.sort(key=lambda x: x[0])
        mid = len(timeseries) // 2
        first_half = [r for _, r in timeseries[:mid]]
        second_half = [r for _, r in timeseries[mid:]]

        # Variance reduction = convergence
        var_first = _variance(first_half)
        var_second = _variance(second_half)

        reduction = (var_first - var_second) / var_first if var_first > 0 else 0

        convergence[f"{pair[0]}/{pair[1]}"] = {
            "n_trades": len(timeseries),
            "variance_first_half": round(var_first, 4),
            "variance_second_half": round(var_second, 4),
            "variance_reduction": round(reduction, 4),
            "converging": reduction > 0.2,
        }

    return convergence


def _variance(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def print_behavior_report(entries):
    """Print a human-readable behavior analysis report."""
    profile = aggregate_behavior_profile(entries)

    print("\n" + "=" * 70)
    print("EMERGENT BEHAVIOR REPORT")
    print("=" * 70)

    if not profile:
        print("No behaviors detected. Need more runs with diverse models.")
        return

    # All behavior types
    all_behaviors = set()
    for data in profile.values():
        all_behaviors.update(data["behavior_rates"].keys())

    # Header
    behaviors = sorted(all_behaviors)
    header = f"{'Model':<16s} {'N':>4s}"
    for b in behaviors:
        header += f" {b[:12]:>12s}"
    print(f"\n{header}")
    print("-" * len(header))

    for model in sorted(profile.keys()):
        data = profile[model]
        line = f"  {model:<14s} {data['n_agent_instances']:>4d}"
        for b in behaviors:
            rate = data["behavior_rates"].get(b, 0)
            line += f" {rate*100:>11.0f}%"
        print(line)

    # Behavior descriptions
    print(f"\nBehavior Definitions:")
    for category, behaviors_dict in BEHAVIOR_TAXONOMY.items():
        for name, desc in behaviors_dict.items():
            if name in all_behaviors:
                print(f"  {name}: {desc}")

    print("\n" + "=" * 70)
