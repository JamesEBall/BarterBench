"""Match log and online ELO for BarterBench.

The match log (``match_log.json``) is the source of truth for every rating.
Each record keeps the continuous margin and the per-agent scores, so ratings
can be recomputed with any method (see ``ratings.py``). The online ELO kept
here is order-dependent and is reported only as a live indicator; the
leaderboard uses Bradley-Terry with bootstrap intervals.
"""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from .paths import ELO_FILE, MATCH_LOG  # noqa: E402

DEFAULT_RATING = 1500
K_FACTOR = 32
DRAW_THRESHOLD = 0.02  # |margin| below this is recorded as a draw for W/L/D ratings


@contextmanager
def file_lock(path):
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
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(rating_a, rating_b, score_a):
    """score_a: 1.0 = A wins, 0.0 = A loses, 0.5 = draw. Returns (new_a, new_b)."""
    ea = expected_score(rating_a, rating_b)
    eb = 1 - ea
    score_b = 1 - score_a
    new_a = rating_a + K_FACTOR * (score_a - ea)
    new_b = rating_b + K_FACTOR * (score_b - eb)
    return round(new_a, 1), round(new_b, 1)


def determine_match_result(model_goal_completion, draw_threshold=DRAW_THRESHOLD):
    """Return (model_a, model_b, score_a) or None if not exactly two contestants."""
    models = list(model_goal_completion.keys())
    if len(models) != 2:
        if len(models) > 2:
            import sys
            print(f"[elo] Skipping rating update: {len(models)} contestants in match (need exactly 2): {models}",
                  file=sys.stderr)
        return None
    a, b = models[0], models[1]
    diff = model_goal_completion[a] - model_goal_completion[b]
    if abs(diff) < draw_threshold:
        score_a = 0.5
    elif diff > 0:
        score_a = 1.0
    else:
        score_a = 0.0
    return a, b, score_a


def load_ratings():
    if ELO_FILE.exists():
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}


def save_ratings(ratings):
    with open(ELO_FILE, "w") as f:
        json.dump(ratings, f, indent=2)


def load_match_log():
    if MATCH_LOG.exists():
        with open(MATCH_LOG) as f:
            return json.load(f)
    return []


def save_match_log(log):
    with open(MATCH_LOG, "w") as f:
        json.dump(log, f, indent=2)


def _agent_scores_by_contestant(run_entry, key_field):
    """{contestant: [per-agent goal completions]} from a run entry."""
    out = {}
    label_key = "strategy_id" if key_field == "strategy_goal_completion" else None
    for pa in run_entry.get("agent_results", run_entry.get("per_agent", [])):
        label = pa.get(label_key) if label_key else None
        label = label or pa.get("contestant") or pa.get("model")
        out.setdefault(label, []).append(pa["goal_completion"])
    return out


def record_match(run_entry, key_field="model_goal_completion", update_elo=True):
    """Append a completed two-contestant run to the match log and update online ELO.

    Returns the match record or None if the run is not a two-way matchup.
    """
    mgc = run_entry.get(key_field, run_entry.get("model_goal_completion", {}))
    result = determine_match_result(mgc)
    if result is None:
        return None
    model_a, model_b, score_a = result

    old_a = old_b = new_a = new_b = None
    if update_elo:
        with file_lock(ELO_FILE):
            ratings = load_ratings()
            old_a = ratings.get(model_a, DEFAULT_RATING)
            old_b = ratings.get(model_b, DEFAULT_RATING)
            new_a, new_b = update_ratings(old_a, old_b, score_a)
            ratings[model_a] = new_a
            ratings[model_b] = new_b
            save_ratings(ratings)

    outcome = model_a if score_a == 1.0 else model_b if score_a == 0.0 else "draw"
    agent_scores = _agent_scores_by_contestant(run_entry, key_field)
    slots = {}
    for i, name in enumerate(run_entry.get("model_assignments", [])):
        slots.setdefault(name, []).append(i)

    match = {
        "run_id": run_entry.get("run_id"),
        "scenario": run_entry.get("scenario"),
        "seed": run_entry.get("seed"),
        "mode": run_entry.get("mode"),
        "communication": run_entry.get("communication"),
        "conceal": run_entry.get("conceal", False),
        "model_a": model_a,
        "model_b": model_b,
        "score_a": mgc[model_a],
        "score_b": mgc[model_b],
        "margin": round(mgc[model_a] - mgc[model_b], 4),
        "agent_scores": agent_scores,
        "slots": slots,
        "winner": outcome,
    }
    if update_elo:
        match["elo_before"] = {model_a: old_a, model_b: old_b}
        match["elo_after"] = {model_a: new_a, model_b: new_b}

    with file_lock(MATCH_LOG):
        log = load_match_log()
        log.append(match)
        save_match_log(log)
    return match


def print_elo_leaderboard():
    """Online ELO (order-dependent). For the real leaderboard see ratings.print_rating_table."""
    ratings = load_ratings()
    log = load_match_log()
    if not ratings:
        return
    stats = {}
    for m in log:
        for model in (m["model_a"], m["model_b"]):
            s = stats.setdefault(model, {"w": 0, "l": 0, "d": 0, "n": 0})
            s["n"] += 1
        if m["winner"] == "draw":
            stats[m["model_a"]]["d"] += 1
            stats[m["model_b"]]["d"] += 1
        else:
            loser = m["model_b"] if m["winner"] == m["model_a"] else m["model_a"]
            stats.setdefault(m["winner"], {"w": 0, "l": 0, "d": 0, "n": 0})["w"] += 1
            stats.setdefault(loser, {"w": 0, "l": 0, "d": 0, "n": 0})["l"] += 1

    print("\n" + "=" * 60)
    print("  ONLINE ELO (order-dependent, live indicator only)")
    print("=" * 60)
    print(f"  {'Model':<20} {'ELO':>8} {'W':>5} {'L':>5} {'D':>5} {'Matches':>8}")
    print("-" * 60)
    for model, elo in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        s = stats.get(model, {"w": 0, "l": 0, "d": 0, "n": 0})
        print(f"  {model:<20} {elo:>8.1f} {s['w']:>5} {s['l']:>5} {s['d']:>5} {s['n']:>8}")
    print("=" * 60)


def reset_ratings():
    if ELO_FILE.exists():
        ELO_FILE.unlink()
    if MATCH_LOG.exists():
        MATCH_LOG.unlink()
