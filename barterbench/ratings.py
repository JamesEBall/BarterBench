"""Rating systems for pairwise match logs, with uncertainty.

Primary: Bradley-Terry maximum likelihood with a weak prior, plus bootstrap
confidence intervals over matches.

Secondary:
- Massey least-squares ratings on the continuous score margin (goal-completion
  difference), which uses the information the win/draw/loss threshold throws away.
- Online ELO averaged over many random match orderings, with the spread across
  orderings reported, so the order-dependence of ELO is measured, not hidden.

All functions take a list of match dicts with at least ``model_a``, ``model_b``
and ``winner`` (a model name or ``"draw"``). Massey additionally needs
``score_a`` and ``score_b`` (or ``margin``).

Pure Python, no numpy, so it runs anywhere the engine runs.
"""

import math
import random

ELO_K = 32
ELO_BASE = 1500
BT_SCALE = 400  # rating = 1500 + 400*log10(gamma), ELO-like scale


# ---- helpers ----

def _models_in(matches):
    s = set()
    for m in matches:
        s.add(m["model_a"])
        s.add(m["model_b"])
    return sorted(s)


def _margin(m):
    if "margin" in m and m["margin"] is not None:
        return float(m["margin"])
    if m.get("score_a") is not None and m.get("score_b") is not None:
        return float(m["score_a"]) - float(m["score_b"])
    if m["winner"] == "draw":
        return 0.0
    return 1.0 if m["winner"] == m["model_a"] else -1.0


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


# ---- Bradley-Terry ----

def fit_bradley_terry(matches, prior=0.5, max_iter=500, tol=1e-8):
    """Bradley-Terry MLE via the MM algorithm (Hunter 2004), draws as half wins.

    ``prior`` adds that many pseudo-wins and pseudo-losses for every model against
    a fixed reference of strength 1. It keeps undefeated or winless models finite
    and shrinks tiny samples toward the centre. Returns {model: rating} on the
    1500 + 400*log10(gamma) scale with geometric mean gamma = 1.
    """
    if not matches:
        return {}
    models = _models_in(matches)
    if len(models) < 2:
        return {m: float(ELO_BASE) for m in models}

    wins = {m: prior for m in models}
    pair_games = {}  # (i, j) -> number of games between i and j
    for m in matches:
        a, b = m["model_a"], m["model_b"]
        if m["winner"] == a:
            wins[a] += 1.0
        elif m["winner"] == b:
            wins[b] += 1.0
        else:
            wins[a] += 0.5
            wins[b] += 0.5
        pair_games[(a, b)] = pair_games.get((a, b), 0) + 1

    gamma = {m: 1.0 for m in models}
    for _ in range(max_iter):
        new_gamma = {}
        for i in models:
            denom = 0.0
            for (a, b), n in pair_games.items():
                if a == i:
                    denom += n / (gamma[i] + gamma[b])
                elif b == i:
                    denom += n / (gamma[i] + gamma[a])
            # prior games vs reference of strength 1 (2*prior games, prior wins)
            denom += 2 * prior / (gamma[i] + 1.0)
            new_gamma[i] = wins[i] / denom if denom > 0 else gamma[i]
        log_mean = sum(math.log(max(g, 1e-12)) for g in new_gamma.values()) / len(models)
        norm = math.exp(log_mean)
        for m in models:
            new_gamma[m] /= norm
        change = max(abs(math.log(max(new_gamma[m], 1e-12)) - math.log(max(gamma[m], 1e-12))) for m in models)
        gamma = new_gamma
        if change < tol:
            break
    return {m: round(ELO_BASE + BT_SCALE * math.log10(max(gamma[m], 1e-12)), 1) for m in models}


def bootstrap_bradley_terry(matches, n_boot=500, seed=0, prior=0.5, ci=0.95):
    """Resample matches with replacement and refit BT. Returns per-model
    {rating, ci_lower, ci_upper, se, n_matches} plus the point estimate."""
    point = fit_bradley_terry(matches, prior=prior)
    if not matches or len(point) < 2:
        return {m: {"rating": r, "ci_lower": r, "ci_upper": r, "se": 0.0, "n_matches": 0}
                for m, r in point.items()}
    rng = random.Random(seed)
    models = _models_in(matches)
    samples = {m: [] for m in models}
    n = len(matches)
    for _ in range(n_boot):
        resampled = [matches[rng.randrange(n)] for _ in range(n)]
        fit = fit_bradley_terry(resampled, prior=prior)
        for m in models:
            # A model absent from a resample is given the centre rating (prior).
            samples[m].append(fit.get(m, ELO_BASE))
    alpha = (1 - ci) / 2
    counts = {m: 0 for m in models}
    for m in matches:
        counts[m["model_a"]] += 1
        counts[m["model_b"]] += 1
    out = {}
    for m in models:
        vals = sorted(samples[m])
        mean = sum(vals) / len(vals)
        se = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        out[m] = {
            "rating": point[m],
            "ci_lower": round(_percentile(vals, alpha), 1),
            "ci_upper": round(_percentile(vals, 1 - alpha), 1),
            "se": round(se, 1),
            "n_matches": counts[m],
        }
    return out


def pairwise_win_probability(bt_ratings, a, b):
    """P(a beats b) implied by BT ratings."""
    ra, rb = bt_ratings.get(a, ELO_BASE), bt_ratings.get(b, ELO_BASE)
    return 1.0 / (1.0 + 10 ** ((rb - ra) / BT_SCALE))


# ---- Massey (margin-based least squares) ----

def massey_ratings(matches, ridge=1e-6):
    """Least-squares ratings r such that r_a - r_b ≈ margin for each match, with
    sum(r) = 0. Margins are goal-completion differences, so ratings are in goal
    completion units (×100 for percentage points). Pure-Python Gaussian elimination."""
    if not matches:
        return {}
    models = _models_in(matches)
    k = len(models)
    if k < 2:
        return {m: 0.0 for m in models}
    idx = {m: i for i, m in enumerate(models)}
    # Normal equations M r = p
    M = [[0.0] * k for _ in range(k)]
    p = [0.0] * k
    for m in matches:
        a, b = idx[m["model_a"]], idx[m["model_b"]]
        d = _margin(m)
        M[a][a] += 1; M[b][b] += 1
        M[a][b] -= 1; M[b][a] -= 1
        p[a] += d; p[b] -= d
    for i in range(k):
        M[i][i] += ridge
    # Replace last equation with sum(r) = 0 (Massey's constraint)
    M[k - 1] = [1.0] * k
    p[k - 1] = 0.0
    r = _solve(M, p)
    if r is None:
        return {m: 0.0 for m in models}
    return {m: round(r[idx[m]], 4) for m in models}


def _solve(A, b):
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            return None
        A[col], A[pivot] = A[pivot], A[col]
        for r in range(n):
            if r != col:
                f = A[r][col] / A[col][col]
                if f:
                    for c in range(col, n + 1):
                        A[r][c] -= f * A[col][c]
    return [A[i][n] / A[i][i] for i in range(n)]


# ---- ELO averaged over orderings ----

def elo_online(matches, k_factor=ELO_K):
    ratings = {}
    for m in matches:
        a, b = m["model_a"], m["model_b"]
        ra, rb = ratings.get(a, ELO_BASE), ratings.get(b, ELO_BASE)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        sa = 1.0 if m["winner"] == a else 0.0 if m["winner"] == b else 0.5
        ratings[a] = ra + k_factor * (sa - ea)
        ratings[b] = rb + k_factor * ((1 - sa) - (1 - ea))
    return ratings


def elo_permutation_average(matches, n_perm=200, seed=0, k_factor=ELO_K):
    """ELO under many random match orders. ``spread`` is the 95% interval of
    final ratings across orderings — how much the usual single-order ELO depends
    on arbitrary sequencing."""
    if not matches:
        return {}
    rng = random.Random(seed)
    models = _models_in(matches)
    acc = {m: [] for m in models}
    order = list(matches)
    for _ in range(n_perm):
        rng.shuffle(order)
        r = elo_online(order, k_factor)
        for m in models:
            acc[m].append(r.get(m, ELO_BASE))
    out = {}
    for m in models:
        vals = sorted(acc[m])
        out[m] = {
            "elo_mean": round(sum(vals) / len(vals), 1),
            "elo_spread_lower": round(_percentile(vals, 0.025), 1),
            "elo_spread_upper": round(_percentile(vals, 0.975), 1),
        }
    return out


# ---- Combined report ----

def rating_report(matches, n_boot=500, n_perm=200, seed=0):
    """Everything the leaderboard needs, from one match list."""
    bt = bootstrap_bradley_terry(matches, n_boot=n_boot, seed=seed)
    massey = massey_ratings(matches)
    elo = elo_permutation_average(matches, n_perm=n_perm, seed=seed)
    record = {}
    for m in matches:
        for side in ("model_a", "model_b"):
            record.setdefault(m[side], {"wins": 0, "losses": 0, "draws": 0, "matches": 0})
            record[m[side]]["matches"] += 1
        if m["winner"] == "draw":
            record[m["model_a"]]["draws"] += 1
            record[m["model_b"]]["draws"] += 1
        else:
            loser = m["model_b"] if m["winner"] == m["model_a"] else m["model_a"]
            record[m["winner"]]["wins"] += 1
            record[loser]["losses"] += 1
    rows = []
    for model in sorted(bt, key=lambda x: -bt[x]["rating"]):
        rows.append({
            "model": model,
            **bt[model],
            "massey_margin": massey.get(model, 0.0),
            **elo.get(model, {}),
            **record.get(model, {}),
        })
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return {"bradley_terry": rows, "n_matches": len(matches)}


def print_rating_table(report):
    rows = report["bradley_terry"]
    if not rows:
        return
    print("\n" + "=" * 100)
    print(f"  RATINGS  ({report['n_matches']} matches)  —  Bradley-Terry with 95% bootstrap CI (primary)")
    print("=" * 100)
    print(f"  {'#':>2} {'Model':<20} {'BT':>7} {'95% CI':>16} {'Massey':>8} {'ELO(avg)':>9} {'ELO spread':>14} {'W-L-D':>10} {'N':>4}")
    print("-" * 100)
    for r in rows:
        ci = f"[{r['ci_lower']:.0f}, {r['ci_upper']:.0f}]"
        spread = f"[{r.get('elo_spread_lower', 0):.0f}, {r.get('elo_spread_upper', 0):.0f}]"
        wld = f"{r.get('wins', 0)}-{r.get('losses', 0)}-{r.get('draws', 0)}"
        print(f"  {r['rank']:>2} {r['model']:<20} {r['rating']:>7.0f} {ci:>16} {r['massey_margin']*100:>+7.1f}pp "
              f"{r.get('elo_mean', 0):>9.0f} {spread:>14} {wld:>10} {r.get('matches', 0):>4}")
    print("-" * 100)
    print("  Massey: least-squares rating on goal-completion margin (percentage points vs field average).")
    print("  ELO(avg): online ELO averaged over random match orders; spread = 95% range across orders.")
    print("=" * 100)
