"""Arena mode: prompt strategies compete, optionally across multiple models."""

import fcntl
import json
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path

from ..agent import MarketAgent
from ..engine import MarketEngine
from ..elo import (
    determine_match_result, update_ratings,
    DEFAULT_RATING,
)
from ..scoring import compute_metrics
from ..eval import load_scenario, list_scenarios, auto_model_config

ARENA_DIR = Path(__file__).parent
STRATEGIES_DIR = ARENA_DIR / "strategies"
ARENA_RESULTS = ARENA_DIR / "results.json"
ARENA_ELO = ARENA_DIR / "elo_ratings.json"
ARENA_MATCHES = ARENA_DIR / "match_log.json"
ARENA_RUNS = ARENA_DIR / "runs"


# ---- Strategy management ----

def load_strategy(name):
    """Load a strategy JSON file."""
    f = STRATEGIES_DIR / f"{name}.json"
    if not f.exists():
        print(f"Error: strategy '{name}' not found")
        import sys
        sys.exit(1)
    with open(f) as fh:
        return json.load(fh)


def list_strategies():
    """List all available strategies."""
    strategies = []
    if not STRATEGIES_DIR.exists():
        return strategies
    for f in sorted(STRATEGIES_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            if not any(s["id"] == data["id"] for s in strategies):
                strategies.append(data)
    return strategies


def submit_strategy(name, prompt, author="user", model="haiku"):
    """Create a new strategy file."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    sid = name.lower().replace(" ", "_")
    strategy = {
        "id": sid,
        "name": name,
        "author": author,
        "model": model,
        "prompt": prompt,
    }
    f = STRATEGIES_DIR / f"{sid}.json"
    with open(f, "w") as fh:
        json.dump(strategy, fh, indent=2)
    print(f"Strategy '{name}' saved to {f}")
    return strategy


# ---- Arena persistence (with file locking for parallel runs) ----

def _ensure_dir(path):
    """Ensure parent directory exists before writing."""
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _file_lock(path):
    """Acquire an exclusive file lock for atomic read-modify-write."""
    _ensure_dir(path)
    lock_path = Path(str(path) + ".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _load_json(path, default):
    """Load JSON from path, returning default if missing."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _save_json(path, data):
    """Save JSON to path."""
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_arena_results():
    return _load_json(ARENA_RESULTS, [])


def save_arena_results(results):
    _save_json(ARENA_RESULTS, results)


def load_arena_elo():
    return _load_json(ARENA_ELO, {})


def save_arena_elo(ratings):
    _save_json(ARENA_ELO, ratings)


def load_arena_matches():
    return _load_json(ARENA_MATCHES, [])


def save_arena_matches(log):
    _save_json(ARENA_MATCHES, log)


def reset_arena():
    """Clear all arena data."""
    for f in [ARENA_RESULTS, ARENA_ELO, ARENA_MATCHES]:
        if f.exists():
            f.unlink()


def record_arena_match(entry):
    """Record an arena match and update arena ELO ratings (thread-safe)."""
    sgc = entry.get("strategy_goal_completion", {})
    result = determine_match_result(sgc)
    if result is None:
        return None

    strat_a, strat_b, score_a = result

    # Atomic update of ELO ratings
    with _file_lock(ARENA_ELO):
        ratings = load_arena_elo()
        old_a = ratings.get(strat_a, DEFAULT_RATING)
        old_b = ratings.get(strat_b, DEFAULT_RATING)
        new_a, new_b = update_ratings(old_a, old_b, score_a)
        ratings[strat_a] = new_a
        ratings[strat_b] = new_b
        save_arena_elo(ratings)

    if score_a == 1.0:
        outcome = strat_a
    elif score_a == 0.0:
        outcome = strat_b
    else:
        outcome = "draw"

    match = {
        "run_id": entry.get("run_id"),
        "scenario": entry.get("scenario"),
        "model_a": strat_a,
        "model_b": strat_b,
        "score_a": sgc[strat_a],
        "score_b": sgc[strat_b],
        "winner": outcome,
        "elo_before": {strat_a: old_a, strat_b: old_b},
        "elo_after": {strat_a: new_a, strat_b: new_b},
    }

    # Atomic append to match log
    with _file_lock(ARENA_MATCHES):
        log = load_arena_matches()
        log.append(match)
        save_arena_matches(log)

    return match


# ---- Run a single arena match ----

def run_arena_match(scenario_name, scenario, strat_a, strat_b, run_id, verbose,
                    model_a=None, model_b=None, label_a=None, label_b=None):
    """Run a single arena match between two contestants."""
    num_agents = len(scenario["agents"])
    half_a, half_b = auto_model_config(scenario)

    model_a = model_a or strat_a.get("model", "haiku")
    model_b = model_b or strat_b.get("model", "haiku")
    label_a = label_a or strat_a["id"]
    label_b = label_b or strat_b["id"]

    # Assign agents: use labels as strategy_id so scoring groups correctly
    assignments = [label_a] * half_a + [label_b] * half_b
    random.shuffle(assignments)

    agents = []
    for i in range(num_agents):
        is_a = assignments[i] == label_a
        strat = strat_a if is_a else strat_b
        model = model_a if is_a else model_b
        agents.append(MarketAgent(
            model_name=model,
            agent_idx=i,
            strategy_id=assignments[i],
            strategy_prompt=strat["prompt"],
        ))

    print(f"  [{scenario_name}] {label_a} vs {label_b}: {' '.join(assignments)}")
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
            "mode": "arena",
            "strategy_config": f"{label_a}:{half_a},{label_b}:{half_b}",
            "strategy_assignments": assignments,
            "model_assignments": [agents[i].model_name for i in range(num_agents)],
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

        sgc = metrics.get("strategy_goal_completion", {})
        for sid, score in sgc.items():
            print(f"    {sid}: {score*100:.1f}% goal completion")

        match = record_arena_match(entry)
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
                label = h.get("contestant", h["model"])
                tag = f"    [R{h['round']}] Trader {h['agent']} ({label})"
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
            "mode": "arena",
            "strategy_config": f"{label_a},{label_b}",
            "error": str(e),
        }


# ---- Arena orchestration ----

def run_arena(strategy_names, scenario_name, runs_per_matchup, verbose, models=None):
    """Run arena: all pairwise matchups between strategies.

    If models is provided (list of model names), each strategy is run with each model,
    creating (strategy, model) pairs as contestants.
    """
    if strategy_names:
        strategies = [load_strategy(s) for s in strategy_names]
    else:
        strategies = list_strategies()

    if len(strategies) < 2:
        print("Error: arena requires at least 2 strategies")
        print("  Submit strategies with: --submit 'name' 'prompt text'")
        import sys
        sys.exit(1)

    if scenario_name == "all":
        scenarios = list_scenarios()
    else:
        scenarios = [scenario_name]

    # Build contestant list: each contestant is (strategy, model, label)
    if models and len(models) > 1:
        # Cross-model arena: each strategy x each model = distinct contestants
        contestants = []
        for strat, model in product(strategies, models):
            label = f"{strat['id']}:{model}"
            contestants.append((strat, model, label))

        pairs = list(combinations(contestants, 2))
        total_matches = len(pairs) * len(scenarios) * runs_per_matchup
        print(f"Barter Arena (cross-model)")
        print(f"  Strategies: {', '.join(s['id'] for s in strategies)}")
        print(f"  Models: {', '.join(models)}")
        print(f"  Contestants: {len(contestants)} (strategy x model)")
        print(f"  Scenarios: {', '.join(scenarios)}")
        print(f"  Total matches: {total_matches}")
    else:
        # Same-model arena: contestant = strategy (all use default model)
        contestants_a = [(s, s.get("model", "haiku"), s["id"]) for s in strategies]
        pairs = list(combinations(contestants_a, 2))
        total_matches = len(pairs) * len(scenarios) * runs_per_matchup
        print(f"Barter Arena")
        print(f"  Strategies: {', '.join(s['id'] for s in strategies)}")
        print(f"  Scenarios: {', '.join(scenarios)}")
        print(f"  Matchups: {len(pairs)} pairs x {len(scenarios)} scenarios x {runs_per_matchup} runs = {total_matches} matches")
    print()

    results = load_arena_results()
    run_offset = len(results)
    match_num = 0

    for scenario_n in scenarios:
        scenario = load_scenario(scenario_n)
        print(f"--- {scenario_n} ({len(scenario['agents'])} agents, {scenario.get('max_rounds', 10)} rounds) ---")

        for (strat_a, model_a, label_a), (strat_b, model_b, label_b) in pairs:
            for run_num in range(runs_per_matchup):
                match_num += 1
                run_id = run_offset + match_num
                print(f"\n  Match {match_num}/{total_matches}")

                entry = run_arena_match(scenario_n, scenario, strat_a, strat_b, run_id, verbose,
                                        model_a=model_a, model_b=model_b, label_a=label_a, label_b=label_b)

                # Atomic append to shared results file
                with _file_lock(ARENA_RESULTS):
                    results = load_arena_results()
                    results.append(entry)
                    save_arena_results(results)

                ARENA_RUNS.mkdir(parents=True, exist_ok=True)
                run_file = ARENA_RUNS / f"run_{entry['run_id']:04d}_{scenario_n}.json"
                with open(run_file, "w") as f:
                    json.dump(entry, f, indent=2)

        print()

    print_arena_leaderboard()
    print(f"\nArena results saved to {ARENA_RESULTS}")


def print_arena_leaderboard():
    """Print arena ELO leaderboard."""
    ratings = load_arena_elo()
    log = load_arena_matches()

    if not ratings:
        print("\nNo arena matches yet.")
        return

    match_counts = {}
    win_counts = {}
    for m in log:
        for s in [m["model_a"], m["model_b"]]:
            match_counts[s] = match_counts.get(s, 0) + 1
        if m["winner"] != "draw":
            win_counts[m["winner"]] = win_counts.get(m["winner"], 0) + 1

    print("\n" + "=" * 65)
    print("  BARTER ARENA LEADERBOARD")
    print("=" * 65)
    print(f"  {'Strategy':<20} {'ELO':>8} {'W':>5} {'L':>5} {'D':>5} {'Matches':>8}")
    print("-" * 65)

    rows = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for strat, elo in rows:
        matches = match_counts.get(strat, 0)
        wins = win_counts.get(strat, 0)
        losses = 0
        draws = 0
        for m in log:
            if m["model_a"] == strat or m["model_b"] == strat:
                if m["winner"] == "draw":
                    draws += 1
                elif m["winner"] != strat:
                    losses += 1
        print(f"  {strat:<20} {elo:>8.1f} {wins:>5} {losses:>5} {draws:>5} {matches:>8}")

    print("=" * 65)
