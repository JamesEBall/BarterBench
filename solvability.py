"""Scenario solvability analysis: greedy upper bound on achievable welfare."""

import itertools


def _goal_completion(inventory, target):
    """Compute goal completion (0-1) for an agent given inventory and target."""
    if not target:
        return 1.0
    scores = []
    for item, needed in target.items():
        if needed > 0:
            have = inventory.get(item, 0)
            scores.append(min(have / needed, 1.0))
    return sum(scores) / len(scores) if scores else 1.0


def _total_welfare(inventories, targets):
    """Sum of all agents' goal completions."""
    return sum(_goal_completion(inventories[i], targets[i]) for i in range(len(inventories)))


def compute_max_welfare(scenario):
    """Greedy upper bound on total welfare achievable via bilateral trades.

    Iteratively finds the bilateral trade (between any two agents) that
    maximizes total welfare gain, executes it, and repeats until no
    improving trades remain. This is a greedy heuristic — not globally
    optimal, but provides a reasonable upper bound.

    Returns:
        dict with max_welfare, max_avg_completion, is_upper_bound
    """
    agents = scenario["agents"]
    n = len(agents)
    inventories = [dict(a["inventory"]) for a in agents]
    targets = [dict(a.get("target", {})) for a in agents]

    # Collect all item types
    all_items = set()
    for inv in inventories:
        all_items.update(inv.keys())
    for t in targets:
        all_items.update(t.keys())
    all_items = sorted(all_items)

    # Greedy: keep finding the best bilateral trade until no improvement
    max_iters = n * n * 10  # safety bound
    for _ in range(max_iters):
        best_gain = 0
        best_trade = None

        # Try all agent pairs
        for i, j in itertools.combinations(range(n), 2):
            # Try all possible item swaps between i and j
            # Agent i gives item_a to j, agent j gives item_b to i
            for item_a in all_items:
                if inventories[i].get(item_a, 0) <= 0:
                    continue
                for item_b in all_items:
                    if item_a == item_b:
                        continue
                    if inventories[j].get(item_b, 0) <= 0:
                        continue

                    # Try different quantities (1 up to available)
                    max_qty_a = inventories[i].get(item_a, 0)
                    max_qty_b = inventories[j].get(item_b, 0)

                    for qty_a in range(1, max_qty_a + 1):
                        for qty_b in range(1, max_qty_b + 1):
                            # Simulate trade
                            inventories[i][item_a] -= qty_a
                            inventories[j][item_a] = inventories[j].get(item_a, 0) + qty_a
                            inventories[j][item_b] -= qty_b
                            inventories[i][item_b] = inventories[i].get(item_b, 0) + qty_b

                            new_welfare = _total_welfare(inventories, targets)

                            # Undo
                            inventories[i][item_a] += qty_a
                            inventories[j][item_a] -= qty_a
                            inventories[j][item_b] += qty_b
                            inventories[i][item_b] -= qty_b

                            current_welfare = _total_welfare(inventories, targets)
                            gain = new_welfare - current_welfare

                            if gain > best_gain + 1e-9:
                                best_gain = gain
                                best_trade = (i, j, {item_a: qty_a}, {item_b: qty_b})

        if best_trade is None:
            break

        # Execute best trade
        i, j, give, want = best_trade
        for item, qty in give.items():
            inventories[i][item] -= qty
            inventories[j][item] = inventories[j].get(item, 0) + qty
        for item, qty in want.items():
            inventories[j][item] -= qty
            inventories[i][item] = inventories[i].get(item, 0) + qty

    max_welfare = _total_welfare(inventories, targets)
    max_avg = max_welfare / n if n > 0 else 0

    return {
        "max_welfare": round(max_welfare, 4),
        "max_avg_completion": round(max_avg, 4),
        "is_upper_bound": True,
    }
