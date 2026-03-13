"""Procedural scenario generation for BarterBench.

Generates randomized but balanced trading scenarios with configurable
scarcity, agent count, and item variety. Ensures structural validity
(paired roles, supply/demand constraints, solvable goals).
"""

import json
import math
import random
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Item pools for thematic variety
ITEM_POOLS = {
    "resources": ["wheat", "iron", "wood", "stone", "coal", "clay", "fiber", "copper"],
    "luxury": ["gold", "silver", "gems", "silk", "spice", "diamonds", "pearls", "jade"],
    "food": ["grain", "fish", "fruit", "meat", "herbs", "honey", "salt", "oil"],
    "trade_goods": ["cloth", "pottery", "tools", "leather", "glass", "paper", "dye", "rope"],
}


def generate_scenario(
    num_agents=6,
    num_items=4,
    max_rounds=8,
    num_scarce=1,
    scarcity_ratio=0.5,
    theme="mixed",
    seed=None,
    name=None,
):
    """Generate a balanced procedural scenario.

    Args:
        num_agents: total agents (must be even for paired roles)
        num_items: number of distinct items
        max_rounds: trading rounds
        num_scarce: how many items are scarce (demand > supply)
        scarcity_ratio: supply/demand ratio for scarce items (lower = more scarce)
        theme: item theme ("resources", "luxury", "food", "trade_goods", "mixed")
        seed: random seed for reproducibility
        name: scenario name (auto-generated if None)

    Returns:
        dict: valid scenario JSON
    """
    if seed is not None:
        random.seed(seed)

    if num_agents % 2 != 0:
        num_agents += 1  # force even

    num_roles = num_agents // 2  # paired roles

    # Pick items
    if theme == "mixed":
        all_items = []
        for pool in ITEM_POOLS.values():
            all_items.extend(pool)
        random.shuffle(all_items)
    else:
        all_items = list(ITEM_POOLS.get(theme, ITEM_POOLS["resources"]))
        random.shuffle(all_items)

    items = all_items[:num_items]

    # Designate scarce items
    scarce_items = items[:num_scarce]
    normal_items = items[num_scarce:]

    # Generate role templates (each used twice for paired roles)
    role_templates = []
    for role_idx in range(num_roles):
        inventory = {}
        target = {}

        # Each role specializes: has some items, wants others
        # Split items into "have" and "want" sets
        shuffled_items = list(items)
        random.shuffle(shuffled_items)

        # Each role has 1-2 items to offer and wants 1-3 items
        num_have = random.randint(1, max(1, num_items // 2))
        have_items = shuffled_items[:num_have]
        want_items = [i for i in shuffled_items if i not in have_items]

        if not want_items:
            want_items = [shuffled_items[-1]]
            have_items = [i for i in shuffled_items if i != want_items[0]]

        # Set inventory quantities
        for item in items:
            if item in have_items:
                inventory[item] = random.randint(3, 6)
            else:
                inventory[item] = 0

        # Set target quantities
        for item in want_items:
            if item in scarce_items:
                target[item] = random.randint(2, 4)
            else:
                target[item] = random.randint(1, 3)

        # Also want to keep some of what they have
        for item in have_items:
            if random.random() < 0.3:
                target[item] = random.randint(1, 2)

        role_templates.append({"inventory": inventory, "target": target})

    # Build agents (paired: each template used twice)
    agents = []
    for role_idx, template in enumerate(role_templates):
        for copy_idx in range(2):
            agent_idx = role_idx * 2 + copy_idx
            agents.append({
                "role": f"Trader {agent_idx}",
                "inventory": dict(template["inventory"]),
                "target": dict(template["target"]),
            })

    # Compute scarcity metadata
    scarcity = {}
    for item in scarce_items:
        supply = sum(a["inventory"].get(item, 0) for a in agents)
        demand = sum(a["target"].get(item, 0) for a in agents)

        # Adjust supply if scarcity ratio is too high
        if demand > 0 and supply / demand > scarcity_ratio:
            # Reduce supply to meet target scarcity
            target_supply = max(1, int(demand * scarcity_ratio))
            excess = supply - target_supply
            # Remove excess from agents that have it
            holders = [a for a in agents if a["inventory"].get(item, 0) > 0]
            random.shuffle(holders)
            for holder in holders:
                if excess <= 0:
                    break
                can_remove = min(holder["inventory"][item] - 1, excess)
                if can_remove > 0:
                    holder["inventory"][item] -= can_remove
                    excess -= can_remove

            supply = sum(a["inventory"].get(item, 0) for a in agents)

        if demand > 0:
            ratio = round(supply / demand, 2)
            scarcity[item] = {
                "supply": supply,
                "demand": demand,
                "ratio": ratio,
            }

    # Auto-generate name
    if name is None:
        name = f"proc_{num_agents}a_{num_items}i_{'_'.join(scarce_items)}"

    scenario = {
        "name": name,
        "description": f"Procedurally generated: {num_agents} agents, {num_items} items, "
                       f"{num_scarce} scarce. Seed={seed}",
        "difficulty": "medium",
        "max_rounds": max_rounds,
        "agents": agents,
        "scarcity": scarcity,
        "procedural": True,
        "generation_params": {
            "seed": seed,
            "num_agents": num_agents,
            "num_items": num_items,
            "num_scarce": num_scarce,
            "scarcity_ratio": scarcity_ratio,
            "theme": theme,
        },
    }

    return scenario


def save_procedural_scenario(scenario, overwrite=False):
    """Save a procedural scenario to the scenarios directory."""
    SCENARIOS_DIR.mkdir(exist_ok=True)
    path = SCENARIOS_DIR / f"{scenario['name']}.json"
    if path.exists() and not overwrite:
        return path
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2)
    return path


def generate_calibrated_scenario(
    target_difficulty=0.3,
    tolerance=0.15,
    max_attempts=20,
    **kwargs,
):
    """Generate a scenario calibrated to a target difficulty level.

    Uses the solvability solver to estimate difficulty before finalizing.
    Retries with different seeds until difficulty falls within tolerance.

    Args:
        target_difficulty: desired difficulty (0 = trivial, 1 = impossible)
        tolerance: acceptable deviation from target
        max_attempts: max generation attempts
        **kwargs: passed to generate_scenario()

    Returns:
        dict: scenario with difficulty close to target
    """
    from solvability import compute_max_welfare

    rng = random.Random(kwargs.get("seed"))
    best_scenario = None
    best_diff = float("inf")

    for attempt in range(max_attempts):
        seed = rng.randint(0, 2**32 - 1)
        kwargs_copy = dict(kwargs)
        kwargs_copy["seed"] = seed
        scenario = generate_scenario(**kwargs_copy)

        solvability = compute_max_welfare(scenario)
        difficulty = 1 - solvability["max_avg_completion"]

        deviation = abs(difficulty - target_difficulty)
        if deviation < best_diff:
            best_diff = deviation
            best_scenario = scenario
            best_scenario["_calibration"] = {
                "target_difficulty": target_difficulty,
                "actual_difficulty": round(difficulty, 4),
                "solvability": solvability,
                "attempts": attempt + 1,
            }

        if deviation <= tolerance:
            break

    return best_scenario


def generate_suite(count=5, seed=42):
    """Generate a suite of diverse procedural scenarios.

    Creates scenarios with varying agent counts, item counts, and scarcity levels.
    """
    rng = random.Random(seed)
    scenarios = []

    configs = [
        {"num_agents": 4, "num_items": 3, "num_scarce": 1, "scarcity_ratio": 0.5, "theme": "resources"},
        {"num_agents": 6, "num_items": 4, "num_scarce": 1, "scarcity_ratio": 0.6, "theme": "luxury"},
        {"num_agents": 8, "num_items": 5, "num_scarce": 2, "scarcity_ratio": 0.5, "theme": "food"},
        {"num_agents": 10, "num_items": 5, "num_scarce": 2, "scarcity_ratio": 0.4, "theme": "trade_goods"},
        {"num_agents": 12, "num_items": 6, "num_scarce": 3, "scarcity_ratio": 0.5, "theme": "mixed"},
    ]

    for i, cfg in enumerate(configs[:count]):
        s = rng.randint(0, 2**32 - 1)
        scenario = generate_scenario(seed=s, **cfg)
        scenarios.append(scenario)

    return scenarios
