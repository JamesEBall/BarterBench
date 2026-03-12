"""ELO rating system for BarterBench pairwise model comparison."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

ELO_FILE = Path(__file__).parent / "elo_ratings.json"
MATCH_LOG = Path(__file__).parent / "match_log.json"

DEFAULT_RATING = 1500
K_FACTOR = 32


@contextmanager
def _file_lock(path):
    """Acquire an exclusive file lock for atomic read-modify-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def expected_score(rating_a, rating_b):
    """Expected score for player A given ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(rating_a, rating_b, score_a):
    """Update ELO ratings after a match.

    score_a: 1.0 = A wins, 0.0 = A loses, 0.5 = draw
    Returns (new_rating_a, new_rating_b).
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1 - ea
    score_b = 1 - score_a

    new_a = rating_a + K_FACTOR * (score_a - ea)
    new_b = rating_b + K_FACTOR * (score_b - eb)
    return round(new_a, 1), round(new_b, 1)


def determine_match_result(model_goal_completion):
    """Determine winner from a run's per-model goal completions.

    Returns (model_a, model_b, score_a) where score_a is 1.0/0.0/0.5.
    Returns None if not exactly 2 models in the run.
    """
    models = list(model_goal_completion.keys())
    if len(models) != 2:
        return None

    a, b = models[0], models[1]
    sa, sb = model_goal_completion[a], model_goal_completion[b]

    # Win threshold: 2 percentage points to avoid noise draws
    diff = sa - sb
    if abs(diff) < 0.02:
        score_a = 0.5
    elif diff > 0:
        score_a = 1.0
    else:
        score_a = 0.0

    return a, b, score_a


def load_ratings():
    """Load ELO ratings from disk."""
    if ELO_FILE.exists():
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}


def save_ratings(ratings):
    """Save ELO ratings to disk."""
    with open(ELO_FILE, "w") as f:
        json.dump(ratings, f, indent=2)


def load_match_log():
    """Load match history."""
    if MATCH_LOG.exists():
        with open(MATCH_LOG) as f:
            return json.load(f)
    return []


def save_match_log(log):
    """Save match history."""
    with open(MATCH_LOG, "w") as f:
        json.dump(log, f, indent=2)


def record_match(run_entry, key_field="model_goal_completion"):
    """Process a completed run and update ELO ratings (thread-safe).

    key_field: which result field to use for pairwise comparison.
    Use "strategy_goal_completion" for arena mode.
    Returns the match record or None if not a valid 2-way matchup.
    """
    mgc = run_entry.get(key_field, run_entry.get("model_goal_completion", {}))
    result = determine_match_result(mgc)
    if result is None:
        return None

    model_a, model_b, score_a = result

    # Atomic update of ELO ratings
    with _file_lock(ELO_FILE):
        ratings = load_ratings()
        old_a = ratings.get(model_a, DEFAULT_RATING)
        old_b = ratings.get(model_b, DEFAULT_RATING)
        new_a, new_b = update_ratings(old_a, old_b, score_a)
        ratings[model_a] = new_a
        ratings[model_b] = new_b
        save_ratings(ratings)

    # Determine outcome label
    if score_a == 1.0:
        outcome = model_a
    elif score_a == 0.0:
        outcome = model_b
    else:
        outcome = "draw"

    match = {
        "run_id": run_entry.get("run_id"),
        "scenario": run_entry.get("scenario"),
        "model_a": model_a,
        "model_b": model_b,
        "score_a": mgc[model_a],
        "score_b": mgc[model_b],
        "winner": outcome,
        "elo_before": {model_a: old_a, model_b: old_b},
        "elo_after": {model_a: new_a, model_b: new_b},
    }

    # Atomic append to match log
    with _file_lock(MATCH_LOG):
        log = load_match_log()
        log.append(match)
        save_match_log(log)

    return match


def print_elo_leaderboard():
    """Print ELO leaderboard."""
    ratings = load_ratings()
    log = load_match_log()

    if not ratings:
        return

    # Count matches per model
    match_counts = {}
    win_counts = {}
    for m in log:
        for model in [m["model_a"], m["model_b"]]:
            match_counts[model] = match_counts.get(model, 0) + 1
        if m["winner"] != "draw":
            win_counts[m["winner"]] = win_counts.get(m["winner"], 0) + 1

    print("\n" + "=" * 60)
    print("  ELO RATINGS")
    print("=" * 60)
    print(f"  {'Model':<12} {'ELO':>8} {'W':>5} {'L':>5} {'D':>5} {'Matches':>8}")
    print("-" * 60)

    rows = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for model, elo in rows:
        matches = match_counts.get(model, 0)
        wins = win_counts.get(model, 0)
        # Count losses and draws
        losses = 0
        draws = 0
        for m in log:
            if m["model_a"] == model or m["model_b"] == model:
                if m["winner"] == "draw":
                    draws += 1
                elif m["winner"] != model:
                    losses += 1
        print(f"  {model:<12} {elo:>8.1f} {wins:>5} {losses:>5} {draws:>5} {matches:>8}")

    print("=" * 60)


def reset_ratings():
    """Clear all ELO data."""
    if ELO_FILE.exists():
        ELO_FILE.unlink()
    if MATCH_LOG.exists():
        MATCH_LOG.unlink()
