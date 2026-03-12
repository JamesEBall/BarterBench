"""Bradley-Terry Maximum Likelihood Estimation for pairwise model rankings.

More statistically principled than incremental ELO — fits all match data
simultaneously via iterative MLE. Produces strength parameters on a log scale.
"""

import json
import math
from pathlib import Path

MATCH_LOG = Path(__file__).parent / "match_log.json"
BT_FILE = Path(__file__).parent / "bt_ratings.json"


def load_match_log():
    """Load match history."""
    if MATCH_LOG.exists():
        with open(MATCH_LOG) as f:
            return json.load(f)
    return []


def fit_bradley_terry(matches, max_iter=100, tol=1e-6):
    """Fit Bradley-Terry model via iterative MLE.

    Args:
        matches: list of dicts with model_a, model_b, winner (model name or 'draw')
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        dict of {model: strength} where strength is on log scale, normalized so mean=0.
    """
    if not matches:
        return {}

    # Collect all models
    models = set()
    for m in matches:
        models.add(m["model_a"])
        models.add(m["model_b"])
    models = sorted(models)

    if len(models) < 2:
        return {m: 0.0 for m in models}

    # Initialize strengths (log scale, uniform)
    gamma = {m: 1.0 for m in models}

    # Count wins and matchups
    # For draws, each player gets 0.5 wins
    for iteration in range(max_iter):
        new_gamma = {}
        for i in models:
            # W_i = number of wins for model i (draws count as 0.5)
            w_i = 0.0
            # Sum over all matches involving i of 1 / (gamma_i + gamma_j)
            denom_sum = 0.0

            for m in matches:
                if m["model_a"] != i and m["model_b"] != i:
                    continue

                opponent = m["model_b"] if m["model_a"] == i else m["model_a"]

                if m["winner"] == i:
                    w_i += 1.0
                elif m["winner"] == "draw":
                    w_i += 0.5

                denom_sum += 1.0 / (gamma[i] + gamma[opponent])

            new_gamma[i] = w_i / denom_sum if denom_sum > 0 else gamma[i]

        # Normalize so geometric mean = 1
        log_mean = sum(math.log(max(g, 1e-10)) for g in new_gamma.values()) / len(models)
        norm_factor = math.exp(log_mean)
        for m in models:
            new_gamma[m] /= norm_factor

        # Check convergence
        max_change = max(abs(new_gamma[m] - gamma[m]) for m in models)
        gamma = new_gamma
        if max_change < tol:
            break

    # Convert to log scale (like ELO: 400 * log10(strength) centered at 1500)
    ratings = {}
    for m in models:
        ratings[m] = round(1500 + 400 * math.log10(max(gamma[m], 1e-10)), 1)

    return ratings


def compute_bt_ratings():
    """Compute Bradley-Terry ratings from the full match log."""
    matches = load_match_log()
    if not matches:
        return {}
    ratings = fit_bradley_terry(matches)
    # Save to disk
    with open(BT_FILE, "w") as f:
        json.dump(ratings, f, indent=2)
    return ratings


def load_bt_ratings():
    """Load cached Bradley-Terry ratings."""
    if BT_FILE.exists():
        with open(BT_FILE) as f:
            return json.load(f)
    return {}


def print_bt_leaderboard():
    """Print Bradley-Terry leaderboard (recomputed from match log)."""
    matches = load_match_log()
    if not matches:
        return

    ratings = fit_bradley_terry(matches)

    # Count stats
    match_counts = {}
    win_counts = {}
    for m in matches:
        for model in [m["model_a"], m["model_b"]]:
            match_counts[model] = match_counts.get(model, 0) + 1
        if m["winner"] != "draw":
            win_counts[m["winner"]] = win_counts.get(m["winner"], 0) + 1

    print("\n" + "=" * 60)
    print("  BRADLEY-TERRY RATINGS (MLE)")
    print("=" * 60)
    print(f"  {'Model':<12} {'BT Rating':>10} {'Win%':>7} {'Matches':>8}")
    print("-" * 60)

    rows = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for model, rating in rows:
        total = match_counts.get(model, 0)
        wins = win_counts.get(model, 0)
        win_pct = (wins / total * 100) if total > 0 else 0.0
        print(f"  {model:<12} {rating:>10.1f} {win_pct:>6.1f}% {total:>8}")

    print("=" * 60)
    print("  BT ratings use MLE over all matches (more stable than incremental ELO)")


def reset_bt():
    """Clear Bradley-Terry data."""
    if BT_FILE.exists():
        BT_FILE.unlink()
