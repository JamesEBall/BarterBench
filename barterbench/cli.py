"""BarterBench command line.

    python -m barterbench --suite --models qwen3-8b,phi-4
    python -m barterbench --matrix --models a,b,c --runs 5 --heldout 3
    python -m barterbench --selfplay qwen3-8b --runs 10
    python -m barterbench --eval gold_rush --models qwen3-8b:3,heuristic:3
    python -m barterbench --report | --list | --registry | --resume <checkpoint>

Every match seed is a SHA-256 of (scenario, run id, model config, communication
mode, conceal flag); results, the match log and ratings live under results/.
"""

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from . import __version__
from .agents import build_agent, is_scripted, load_endpoints
from .engine import MarketEngine, COMMUNICATION_MODES
from .matchlog import record_match, load_match_log, file_lock, print_elo_leaderboard, reset_ratings
from .paths import SCENARIOS_DIR, RESULTS_DIR, RUNS_DIR, RESULTS_FILE, RATINGS_FILE, REPORT_FILE, CHECKPOINT_DIR
from .ratings import rating_report, print_rating_table
from .registry import get_model_info, compute_dollar_cost, compute_gpu_cost, BASELINES, get_registry_summary
from .scenarios import generate_scenario, save_procedural_scenario
from .scoring import compute_metrics, compute_cost_efficiency, compute_slot_stats, compute_aggregate_statistics

SUITE_SCENARIOS = ["gold_rush", "water_crisis", "spice_wars", "grand_bazaar"]
SUITE_RUNS_PER_SCENARIO = 10
SUITE_ANCHORS = ["random", "heuristic"]
SUITE_HELDOUT = 2
SUITE_SELFPLAY_RUNS = 6

# Set from the CLI; read by run_single()
RUN_OPTIONS = {"communication": "full", "conceal": False, "endpoints": None}


# ---- helpers ----

def _seed_for(*parts):
    """Deterministic 32-bit seed, stable across processes (unlike hash())."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def load_scenario(name):
    f = SCENARIOS_DIR / f"{name}.json"
    if not f.exists():
        sys.exit(f"Error: scenario '{name}' not found at {f}")
    return json.loads(f.read_text())


def list_scenarios():
    out = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        if data["agents"][0].get("target") is not None:
            out.append(data["name"])
    return out


def parse_model_config(config_str):
    """'a:3,b:3' -> [('a', 3), ('b', 3)]; bare names count 1."""
    config = []
    for p in config_str.split(","):
        if ":" in p:
            model, count = p.split(":")
            config.append((model.strip(), int(count.strip())))
        else:
            config.append((p.strip(), 1))
    return config


def assign_models(model_config, num_agents, rng):
    """Stratified slot assignment: for two contestants on an even board, each paired
    role (0&1, 2&3, ...) gets one agent of each side. Otherwise a seeded shuffle."""
    assignments = []
    for model, count in model_config:
        assignments.extend([model] * count)
    if len(assignments) != num_agents:
        sys.exit(f"Error: model config specifies {len(assignments)} agents but scenario has {num_agents}")
    models = sorted(set(assignments))
    if len(models) == 2 and num_agents % 2 == 0 and assignments.count(models[0]) == num_agents // 2:
        result = []
        for _ in range(0, num_agents, 2):
            pair = list(models)
            rng.shuffle(pair)
            result.extend(pair)
        return result
    rng.shuffle(assignments)
    return assignments


def _locked_append_result(entry, results_file=RESULTS_FILE):
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(results_file):
        results = json.loads(results_file.read_text()) if results_file.exists() else []
        results.append(entry)
        results_file.write_text(json.dumps(results, indent=2))
    return results


def load_existing_results():
    return json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []


def save_run_file(entry, scenario_name):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"run_{entry['run_id']:04d}_{scenario_name}.json").write_text(json.dumps(entry, indent=2))


def _generate_scenario_for_eval(num_agents, run_seed, max_rounds=10):
    num_items = max(3, num_agents // 2)
    num_scarce = max(1, num_items // 3)
    sc = generate_scenario(num_agents=num_agents, num_items=num_items, num_scarce=num_scarce,
                           seed=run_seed, max_rounds=max_rounds)
    sc["name"] = f"{sc['name']}_s{run_seed % 100000}"
    sc["_generated"] = True
    sc["_seed"] = run_seed
    return sc


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=str(SCENARIOS_DIR.parent)).decode().strip()[:12]
    except Exception:
        return "unknown"


# ---- one match ----

def run_single(scenario_name, scenario, model_config, model_config_str, run_id, verbose=False,
               simultaneous=False, temperature=1.0, history_rounds=-1, mode=None,
               communication=None, conceal=None, contestant_labels=None):
    """Run one marketplace match and return the result entry.

    ``contestant_labels`` maps a config name to a scoring label (self-play uses the same
    model under two labels; agents never see labels or model names)."""
    communication = RUN_OPTIONS["communication"] if communication is None else communication
    conceal = RUN_OPTIONS["conceal"] if conceal is None else conceal
    endpoints = RUN_OPTIONS["endpoints"]

    seed = _seed_for(scenario_name, run_id, model_config_str, communication, conceal)
    rng = random.Random(seed)
    num_agents = len(scenario["agents"])
    model_assignments = assign_models(model_config, num_agents, rng)
    agents = []
    for i, name in enumerate(model_assignments):
        label = (contestant_labels or {}).get(name)
        real_model = name.split("@")[0]
        agents.append(build_agent(
            real_model, i, scenario["agents"][i], scenario.get("auction_enabled", False),
            seed=seed + i, num_agents=num_agents, temperature=temperature,
            history_rounds=history_rounds, conceal=conceal, communication=communication,
            endpoints=endpoints, strategy_id=label,
        ))

    flags = [f for f, on in (("simultaneous", simultaneous), (f"comm={communication}", communication != "full"),
                             ("conceal", conceal)) if on]
    print(f"  [{scenario_name}]{(' [' + ', '.join(flags) + ']') if flags else ''} models: {' '.join(model_assignments)}")
    print("  Running marketplace...", end="", flush=True)

    try:
        engine = MarketEngine(scenario, agents, simultaneous=simultaneous, run_id=run_id, seed=seed,
                              communication=communication)
        result = engine.run()
        result["scenario_data"] = scenario
        metrics = compute_metrics(result)

        entry = {
            "run_id": run_id,
            "seed": seed,
            "temperature": temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "mode": mode,
            "communication": communication,
            "conceal": conceal,
            "model_config": model_config_str,
            "model_assignments": model_assignments,
            "turn_orders": result.get("turn_orders", []),
            **metrics,
            "elapsed_seconds": result["elapsed_seconds"],
            "agent_reliability": [{"model": a.model_name, "agent_idx": i, **a.reliability()} for i, a in enumerate(agents)],
            "agent_tokens": [
                {"model": a.model_name, "agent_idx": i,
                 "input_tokens": a.total_input_tokens, "output_tokens": a.total_output_tokens,
                 "tokens": a.total_input_tokens + a.total_output_tokens,
                 "cost_usd": compute_dollar_cost(a.model_name, a.total_input_tokens, a.total_output_tokens)}
                for i, a in enumerate(agents)
            ],
            "agent_latencies": [
                {"model": a.model_name, "agent_idx": i,
                 "mean_latency": sum(a.turn_latencies) / len(a.turn_latencies) if a.turn_latencies else 0,
                 "max_latency": max(a.turn_latencies) if a.turn_latencies else 0,
                 "num_turns": len(a.turn_latencies)}
                for i, a in enumerate(agents)
            ],
            "model_versions": {a.model_name: getattr(a, "resolved_model", a.model_name)
                               for a in agents if not is_scripted(a.model_name)},
            "model_metadata": {m: get_model_info(m) for m in set(model_assignments)},
            "gpu_cost_usd_estimate": compute_gpu_cost(result["elapsed_seconds"]),
            "initial_inventories": result["initial_inventories"],
            "targets": [dict(a.get("target", {})) for a in scenario["agents"]],
            "agent_results": result["agent_results"],
            "trades": result["trades"],
            "public_messages": result.get("public_messages", []),
            "history": result["history"],
            "reproducibility": {
                "python_version": sys.version.split()[0],
                "barterbench_version": __version__,
                "git_sha": _git_sha(),
                "seed": seed,
                "seed_inputs": [scenario_name, run_id, model_config_str, communication, conceal],
                "temperature": temperature,
                "history_rounds": history_rounds,
                "simultaneous": simultaneous,
                "structured_output_modes": sorted({a.reliability().get("structured_mode", "") for a in agents}),
            },
        }
        cost_eff = compute_cost_efficiency(entry)
        if cost_eff:
            entry["cost_efficiency"] = cost_eff

        print(f" done (goal={metrics['avg_goal_completion']*100:.0f}% trades={metrics['num_trades']} "
              f"invalid={metrics['invalid_rate']*100:.0f}%)")
        for model, score in metrics["model_goal_completion"].items():
            print(f"    {model}: {score*100:.1f}% goal completion")

        key = "strategy_goal_completion" if mode == "selfplay" else "model_goal_completion"
        match = record_match(entry, key_field=key, update_elo=(mode != "selfplay"))
        if match:
            w = match["winner"]
            margin = match["margin"] * 100 * (-1 if w == match["model_b"] else 1)
            print(f"    Result: {'draw' if w == 'draw' else w + ' wins'} (margin {margin:+.1f}pp)")

        bad = {m: r for m, r in metrics.get("reliability", {}).items()
               if r["parse_error_rate"] > 0.05 or r["fallback_rate"] > 0}
        if bad:
            print("    [warn] harness reliability: " + ", ".join(
                f"{m} parse_err={r['parse_error_rate']*100:.0f}% fallback={r['fallback_rate']*100:.0f}%"
                for m, r in bad.items()))

        if verbose:
            print()
            for h in result["history"]:
                tag = f"    [R{h['round']}] Trader {h['agent']} ({h['model']})"
                inv = " [INVALID]" if h.get("invalid") else ""
                if h["action"] in ("post_offer", "private_offer"):
                    kind = "WHISPER" if h["action"] == "private_offer" else "POST"
                    print(f"{tag} {kind}: give {h.get('give', {})} want {h.get('want', {})}{inv} — {h.get('message', '')}")
                elif h["action"] == "accept_offer":
                    print(f"{tag} ACCEPT #{h.get('offer_id', '?')}{inv} — {h.get('message', '')}")
                else:
                    print(f"{tag} {h['action'].upper()}{inv} — {h.get('message', '')}")
            print()
        return entry

    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "scenario": scenario_name,
                "mode": mode, "model_config": model_config_str, "error": str(e)}


def _run_jobs(jobs, fn, parallel):
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            return list(pool.map(fn, jobs))
    return [fn(j) for j in jobs]


class _RunIds:
    def __init__(self):
        self.offset = len(load_existing_results())
        self.n = 0
        self.lock = threading.Lock()

    def next(self):
        with self.lock:
            self.n += 1
            return self.offset + self.n


# ---- modes ----

def run_eval(scenario_name, model_config_str, runs, verbose, simultaneous=False, parallel=1,
             temperature=1.0, history_rounds=-1):
    """One scenario, 'all' named scenarios, or 'procedural' (fresh seeded scenario per run)."""
    model_config = parse_model_config(model_config_str)
    total = sum(c for _, c in model_config)
    names = list_scenarios() if scenario_name == "all" else [scenario_name]
    ids = _RunIds()
    jobs = [(n, r) for n in names for r in range(runs)]

    def _one(job):
        name, r = job
        run_id = ids.next()
        if name == "procedural":
            sc = _generate_scenario_for_eval(total, _seed_for("eval", model_config_str, run_id))
            name = sc["name"]
        else:
            sc = load_scenario(name)
            if len(sc["agents"]) != total:
                sys.exit(f"Error: scenario '{name}' has {len(sc['agents'])} agents but config specifies {total}")
        entry = run_single(name, sc, model_config, model_config_str, run_id, verbose, simultaneous=simultaneous,
                           temperature=temperature, history_rounds=history_rounds, mode="eval")
        _locked_append_result(entry)
        save_run_file(entry, name)
        return entry

    print(f"Eval: {', '.join(names)} | models {model_config_str} | runs {runs}")
    _run_jobs(jobs, _one, parallel)
    print_leaderboard(load_existing_results())
    print_ratings()


def run_matrix(models, runs_per_pair, verbose, parallel=1, temperature=1.0, history_rounds=-1,
               scenarios=None, heldout=SUITE_HELDOUT, mode="matrix"):
    """Round-robin: every pair on each fixed scenario plus ``heldout`` procedural scenarios
    drawn from a fixed master seed (identical for every pair)."""
    pairs = list(combinations(models, 2))
    scenario_list = [(n, load_scenario(n)) for n in (scenarios or SUITE_SCENARIOS)]
    for k in range(heldout):
        sc = _generate_scenario_for_eval(8, _seed_for("heldout", k))
        scenario_list.append((sc["name"], sc))
    ids = _RunIds()
    print(f"Round-robin: {', '.join(models)} | {len(pairs)} pairs × {len(scenario_list)} scenarios "
          f"({heldout} held-out) × {runs_per_pair} runs = {len(pairs) * len(scenario_list) * runs_per_pair} matches")

    all_entries = []
    for a, b in pairs:
        print(f"\n--- {a} vs {b} ---")
        jobs = [(n, sc, r) for n, sc in scenario_list for r in range(runs_per_pair)]

        def _one(job, a=a, b=b):
            name, sc, r = job
            n = len(sc["agents"])
            ha, hb = n // 2, n - n // 2
            entry = run_single(name, sc, [(a, ha), (b, hb)], f"{a}:{ha},{b}:{hb}", ids.next(), verbose,
                               temperature=temperature, history_rounds=history_rounds, mode=mode)
            _locked_append_result(entry)
            save_run_file(entry, name)
            return entry

        pair_entries = _run_jobs(jobs, _one, parallel)
        stats = compute_aggregate_statistics([e for e in pair_entries if not e.get("error")])
        if stats:
            for m, s in stats["per_model"].items():
                print(f"    {m}: {s['mean']*100:.1f}% ± {s['std']*100:.1f}% ({s['n_runs']} runs)")
            for key, c in stats.get("pairwise", {}).items():
                print(f"    {key}: diff={c['mean_diff']*100:+.1f}pp CI=[{c['ci_lower']*100:+.1f}, {c['ci_upper']*100:+.1f}] "
                      f"p={c['p_value']:.3f}{' *' if c['significant'] else ''}")
        all_entries.extend(pair_entries)
    if mode == "matrix":
        print_ratings()
    return all_entries


def run_selfplay(model, scenario_names, runs, verbose, parallel=1, temperature=1.0, history_rounds=-1):
    """Model vs itself under hidden labels: harness fairness check and collusion baseline."""
    labels = {f"{model}@A": f"{model}@A", f"{model}@B": f"{model}@B"}
    ids = _RunIds()
    jobs = [(n, r) for n in scenario_names for r in range(runs)]

    def _one(job):
        name, r = job
        sc = load_scenario(name)
        n = len(sc["agents"])
        ha, hb = n // 2, n - n // 2
        entry = run_single(name, sc, [(f"{model}@A", ha), (f"{model}@B", hb)], f"{model}@A:{ha},{model}@B:{hb}",
                           ids.next(), verbose, temperature=temperature, history_rounds=history_rounds,
                           mode="selfplay", contestant_labels=labels)
        entry["selfplay_model"] = model
        _locked_append_result(entry)
        save_run_file(entry, name)
        return entry

    print(f"\n--- Self-play: {model} vs {model} ({len(jobs)} matches) ---")
    entries = _run_jobs(jobs, _one, parallel)
    summarise_selfplay(entries, model)
    return entries


def summarise_selfplay(entries, model=None, quiet=False):
    valid = [e for e in entries if not e.get("error") and e.get("strategy_goal_completion")]
    if not valid:
        return None
    a_wins = b_wins = draws = 0
    margins = []
    for e in valid:
        keys = sorted(e["strategy_goal_completion"])
        if len(keys) != 2:
            continue
        d = e["strategy_goal_completion"][keys[0]] - e["strategy_goal_completion"][keys[1]]
        margins.append(d)
        if abs(d) < 0.02:
            draws += 1
        elif d > 0:
            a_wins += 1
        else:
            b_wins += 1
    n = len(margins)
    mean = sum(margins) / n if n else 0.0
    se = (sum((m - mean) ** 2 for m in margins) / max(n - 1, 1)) ** 0.5 / max(n, 1) ** 0.5
    slots = compute_slot_stats(valid)
    summary = {"model": model, "matches": n, "label_a_wins": a_wins, "label_b_wins": b_wins, "draws": draws,
               "mean_margin_a_minus_b": round(mean, 4), "margin_se": round(se, 4),
               "position_gc_correlation": slots.get("position_gc_correlation"), "by_slot": slots.get("by_slot")}
    if not quiet:
        print(f"  Self-play {model}: A {a_wins} – B {b_wins} – draws {draws}  mean margin {mean*100:+.1f}pp ± {se*100:.1f}  "
              f"turn-position/GC corr = {slots.get('position_gc_correlation')}")
        if n >= 6 and se > 0 and abs(mean) > 2 * se:
            print("  [warn] label asymmetry exceeds 2 SE: check slot assignment / scenario symmetry")
    return summary


def run_suite(test_models, verbose, parallel=1, runs_per=SUITE_RUNS_PER_SCENARIO, heldout=SUITE_HELDOUT,
              selfplay_runs=SUITE_SELFPLAY_RUNS, anchors=None, temperature=1.0, history_rounds=-1):
    """Anchors → round-robin (fixed + held-out) → self-play, all on fixed seeds."""
    anchors = anchors or SUITE_ANCHORS
    scenarios = SUITE_SCENARIOS
    n_anchor = len(test_models) * len(anchors) * len(scenarios) * runs_per
    n_rr = (len(test_models) * (len(test_models) - 1) // 2) * (len(scenarios) + heldout) * runs_per
    print(f"Eval Suite v2: {', '.join(test_models)}")
    print(f"  Stage 1 anchors    : {', '.join(anchors)} × {len(scenarios)} scenarios × {runs_per} runs = {n_anchor}")
    print(f"  Stage 2 round-robin: {len(scenarios)} fixed + {heldout} held-out × {runs_per} runs = {n_rr}")
    print(f"  Stage 3 self-play  : {selfplay_runs} per model = {len(test_models) * selfplay_runs}")
    print(f"  communication={RUN_OPTIONS['communication']} conceal={RUN_OPTIONS['conceal']} temperature={temperature}\n")

    ids = _RunIds()
    anchor_entries = {m: [] for m in test_models}
    jobs = [(m, a, s, r) for m in test_models for a in anchors for s in scenarios for r in range(runs_per)]

    def _anchor(job):
        m, a, name, r = job
        sc = load_scenario(name)
        n = len(sc["agents"])
        ha, hb = n // 2, n - n // 2
        entry = run_single(name, sc, [(a, ha), (m, hb)], f"{a}:{ha},{m}:{hb}", ids.next(), verbose,
                           temperature=temperature, history_rounds=history_rounds, mode="suite")
        entry["anchor_model"] = a
        _locked_append_result(entry)
        save_run_file(entry, name)
        return m, entry

    print(f"{'='*60}\n  Stage 1: anchors\n{'='*60}")
    for m, entry in _run_jobs(jobs, _anchor, parallel):
        anchor_entries[m].append(entry)

    if len(test_models) >= 2:
        print(f"\n{'='*60}\n  Stage 2: round-robin\n{'='*60}")
        run_matrix(test_models, runs_per, verbose, parallel=parallel, temperature=temperature,
                   history_rounds=history_rounds, scenarios=scenarios, heldout=heldout, mode="suite")

    selfplay = {}
    if selfplay_runs > 0:
        print(f"\n{'='*60}\n  Stage 3: self-play\n{'='*60}")
        primary = ["spice_wars"] if "spice_wars" in scenarios else scenarios[:1]
        for m in test_models:
            selfplay[m] = summarise_selfplay(run_selfplay(m, primary, selfplay_runs, verbose, parallel=parallel,
                                                          temperature=temperature, history_rounds=history_rounds),
                                             m, quiet=True)

    print(f"\n{'='*80}\n  EVAL SUITE REPORT CARD\n{'='*80}")
    for m in test_models:
        valid = [e for e in anchor_entries[m] if not e.get("error")]
        print(f"\n  {m}\n  {'-'*40}")
        for a in anchors:
            t, s_, w, l, d = [], [], 0, 0, 0
            for e in valid:
                if e.get("anchor_model") != a:
                    continue
                ts, as_ = e["model_goal_completion"].get(m, 0), e["model_goal_completion"].get(a, 0)
                t.append(ts); s_.append(as_)
                w, l, d = (w + 1, l, d) if ts - as_ >= 0.02 else (w, l + 1, d) if as_ - ts >= 0.02 else (w, l, d + 1)
            if t:
                print(f"    vs {a:<10} {m}={sum(t)/len(t)*100:.1f}%  {a}={sum(s_)/len(s_)*100:.1f}%   W-L-D {w}-{l}-{d}")
        if selfplay.get(m):
            s = selfplay[m]
            print(f"    self-play     A-B-D {s['label_a_wins']}-{s['label_b_wins']}-{s['draws']}  "
                  f"margin {s['mean_margin_a_minus_b']*100:+.1f}pp ± {s['margin_se']*100:.1f}")
    print(f"\n{'='*80}\n")
    print_full_report()


def resume_run(checkpoint_path, verbose=False):
    engine, start_round, init, elapsed = MarketEngine.from_checkpoint(checkpoint_path)
    name = engine.scenario.get("name", "unknown")
    assignments = [a.model_name for a in engine.agents]
    print(f"Resuming {name} from round {start_round}/{engine.max_rounds}: {' '.join(assignments)}")
    result = engine.run(start_round=start_round, initial_inventories=init, elapsed_offset=elapsed)
    result["scenario_data"] = engine.scenario
    metrics = compute_metrics(result)
    cfg = ",".join(f"{m}:{assignments.count(m)}" for m in dict.fromkeys(assignments))
    entry = {"run_id": engine.run_id, "seed": engine.seed, "timestamp": datetime.now(timezone.utc).isoformat(),
             "scenario": name, "mode": "resumed", "communication": engine.communication, "model_config": cfg,
             "model_assignments": assignments, "resumed_from_round": start_round, **metrics,
             "elapsed_seconds": result["elapsed_seconds"], "initial_inventories": result["initial_inventories"],
             "targets": [dict(t) for t in engine.targets], "agent_results": result["agent_results"],
             "trades": result["trades"], "history": result["history"]}
    _locked_append_result(entry)
    save_run_file(entry, name)
    record_match(entry)
    for m, s in metrics["model_goal_completion"].items():
        print(f"    {m}: {s*100:.1f}% goal completion")


# ---- reporting ----

def print_leaderboard(results):
    stats = {}
    for r in results:
        if r.get("error"):
            continue
        for m, s in r.get("model_goal_completion", {}).items():
            stats.setdefault(m, []).append(s)
    if not stats:
        return
    print(f"\n{'='*70}\n  GOAL COMPLETION (all runs)\n{'='*70}")
    print(f"  {'Model':<20} {'Mean':>8} {'± 95% CI':>9} {'StdDev':>8} {'Runs':>6}")
    for m, scores in sorted(stats.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        n = len(scores)
        mean = sum(scores) / n * 100
        std = math.sqrt(sum((s * 100 - mean) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
        ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        print(f"  {m:<20} {mean:>7.1f}% {('±%.1f%%' % ci) if n > 1 else 'n/a':>9} {std:>7.1f}% {n:>6}")
    print("=" * 70)


def _non_selfplay_matches():
    return [m for m in load_match_log() if m.get("mode") != "selfplay"]


def print_ratings():
    matches = _non_selfplay_matches()
    if not matches:
        return
    report = rating_report(matches)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RATINGS_FILE.write_text(json.dumps(report, indent=2))
    print_rating_table(report)
    llm_only = [m for m in matches if m["model_a"] not in BASELINES and m["model_b"] not in BASELINES]
    if llm_only and len(llm_only) < len(matches):
        print("\n  LLM-only matches (anchors excluded):")
        print_rating_table(rating_report(llm_only))


def build_report_data():
    valid = [e for e in load_existing_results() if not e.get("error")]
    with_pm = [e for e in valid if e.get("process_metrics")]
    matches = _non_selfplay_matches()
    ratings = rating_report(matches)
    llm_only = [m for m in matches if m["model_a"] not in BASELINES and m["model_b"] not in BASELINES]

    gc = {}
    for e in valid:
        for m, s in e.get("model_goal_completion", {}).items():
            gc.setdefault(m, []).append(s)

    metric_fields = [
        ("trade_efficiency_ratio", "trade_efficiency_ratio", "per_model"),
        ("information_security_score", "iss_gratuitous", "per_model"),
        ("information_security_score", "iss_gratuitous_active", "per_model_active"),
        ("information_security_score", "iss_structural", "per_model_structural"),
        ("information_security_score", "iss_quantity", "per_model_quantity"),
        ("offer_execution_rate", "offer_execution_rate", "per_model"),
        ("trade_relevance_rate", "trade_relevance_rate", "per_model"),
    ]
    pm_acc = {}
    for e in with_pm:
        for key, out, field in metric_fields:
            for m, v in (e["process_metrics"].get(key, {}).get(field) or {}).items():
                if v is not None:
                    pm_acc.setdefault(m, {}).setdefault(out, []).append(v)

    rel_acc = {}
    for e in valid:
        for m, r in (e.get("reliability") or {}).items():
            for k, v in r.items():
                if k != "turns":
                    rel_acc.setdefault(m, {}).setdefault(k, []).append(v)

    versions = {}
    for e in valid:
        for m, v in (e.get("model_versions") or {}).items():
            versions.setdefault(m, v)

    per_model = {}
    for m in sorted(set(gc) | set(pm_acc)):
        scores = gc.get(m, [])
        mean = sum(scores) / len(scores) if scores else None
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) if len(scores) > 1 else None
        per_model[m] = {
            "goal_completion_mean": round(mean, 4) if mean is not None else None,
            "goal_completion_std": round(std, 4) if std is not None else None,
            "n_runs": len(scores),
            "process_metrics": {k: round(sum(v) / len(v), 4) for k, v in pm_acc.get(m, {}).items()},
            "reliability": {k: round(sum(v) / len(v), 4) for k, v in rel_acc.get(m, {}).items()},
            "metadata": get_model_info(m),
            "model_version": versions.get(m),
        }

    # Disclosure by ablation condition (descriptive only)
    by_condition = {}
    for cond in sorted({(e.get("communication", "full"), bool(e.get("conceal"))) for e in with_pm}):
        sub = [e for e in with_pm if (e.get("communication", "full"), bool(e.get("conceal"))) == cond]
        iss = {"gratuitous": [], "structural": [], "quantity": []}
        gcm = {}
        for e in sub:
            blk = e["process_metrics"].get("information_security_score", {})
            for k, src in (("gratuitous", "overall"), ("structural", "overall_structural"), ("quantity", "overall_quantity")):
                if blk.get(src) is not None:
                    iss[k].append(blk[src])
            for m, s in e.get("model_goal_completion", {}).items():
                gcm.setdefault(m, []).append(s)
        by_condition[f"communication={cond[0]},conceal={cond[1]}"] = {
            "n_runs": len(sub),
            "iss": {k: (round(sum(v) / len(v), 4) if v else None) for k, v in iss.items()},
            "goal_completion_by_model": {m: round(sum(v) / len(v), 4) for m, v in gcm.items()},
        }

    selfplay = {}
    for e in valid:
        if e.get("mode") == "selfplay" and e.get("selfplay_model"):
            selfplay.setdefault(e["selfplay_model"], []).append(e)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "barterbench_version": __version__,
        "dataset": {"valid_runs": len(valid), "contestants": sorted(gc),
                    "scenarios": sorted({e.get("scenario") for e in valid if e.get("scenario")})},
        "leaderboard": {"bradley_terry_all_matches": ratings["bradley_terry"],
                        "bradley_terry_llm_only": rating_report(llm_only)["bradley_terry"] if llm_only else [],
                        "n_matches": ratings["n_matches"]},
        "per_model": per_model,
        "slot_fairness": compute_slot_stats(valid),
        "selfplay": {m: summarise_selfplay(es, m, quiet=True) for m, es in selfplay.items()},
        "disclosure_by_condition": by_condition,
        "note": "Disclosure rates are descriptive. Compare --conceal and --communication runs before "
                "calling disclosure a failure.",
    }


def print_full_report():
    print_ratings()
    data = build_report_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(data, indent=2))
    pm = data["per_model"]
    if not pm:
        print("  No results yet.")
        return
    W = 100
    print(f"\n{'='*W}\n  PROCESS METRICS\n{'='*W}")
    cols = [("goal_completion_mean", "GoalC%", "pct"), ("trade_efficiency_ratio", "TER", "ratio"),
            ("iss_gratuitous", "ISS", "pct"), ("iss_structural", "ISS_str", "pct"), ("iss_quantity", "ISS_qty", "pct"),
            ("offer_execution_rate", "OER", "pct"), ("trade_relevance_rate", "TRR", "pct")]
    print(f"  {'Model':<18s}" + "".join(f" {lbl:>8s}" for _, lbl, _ in cols))
    for m, d in pm.items():
        row = f"  {m:<18s}"
        for key, _, kind in cols:
            v = d["goal_completion_mean"] if key == "goal_completion_mean" else d["process_metrics"].get(key)
            row += f" {'n/a':>8s}" if v is None else (f" {v:>8.3f}" if kind == "ratio" else f" {v*100:>7.1f}%")
        print(row)
    print("  ISS: gratuitous disclosure of target items in messages beyond the offer's own want field (1 = never).")
    print("  ISS_str: structural disclosure via offers (unavoidable). ISS_qty: target quantities stated in messages.")
    print("  Agents that never message are excluded from message-based scores, not credited.")

    print(f"\n{'='*W}\n  HARNESS RELIABILITY\n{'='*W}")
    print(f"  {'Model':<18s} {'parse_err':>10s} {'truncated':>10s} {'fallback':>9s} {'invalid':>8s} {'pass':>7s}")
    for m, d in pm.items():
        r = d["reliability"]
        if not r:
            continue
        flag = "  <-- check" if r.get("parse_error_rate", 0) > 0.05 or r.get("fallback_rate", 0) > 0.01 else ""
        print(f"  {m:<18s} {r.get('parse_error_rate', 0)*100:>9.1f}% {r.get('truncation_rate', 0)*100:>9.1f}% "
              f"{r.get('fallback_rate', 0)*100:>8.1f}% {r.get('invalid_rate', 0)*100:>7.1f}% {r.get('pass_rate', 0)*100:>6.1f}%{flag}")

    sf, sp = data["slot_fairness"], data["selfplay"]
    print(f"\n{'='*W}\n  FAIRNESS\n{'='*W}")
    if sf.get("position_gc_correlation") is not None:
        print(f"  Turn position vs goal completion (Pearson r): {sf['position_gc_correlation']:+.3f}")
    for m, s in sp.items():
        if s:
            print(f"  Self-play {m:<16s} A-B-D {s['label_a_wins']}-{s['label_b_wins']}-{s['draws']}  "
                  f"margin {s['mean_margin_a_minus_b']*100:+.1f}pp ± {s['margin_se']*100:.1f}")

    print(f"\n{'='*W}\n  DISCLOSURE BY CONDITION\n{'='*W}")
    for cond, blk in data["disclosure_by_condition"].items():
        f = lambda v: f"{v*100:.1f}%" if v is not None else "n/a"
        print(f"  [{cond}] {blk['n_runs']} runs — gratuitous={f(blk['iss']['gratuitous'])} "
              f"structural={f(blk['iss']['structural'])} quantity={f(blk['iss']['quantity'])}")
    print(f"  {data['note']}")
    print(f"  Report: {REPORT_FILE}\n{'='*W}\n")


# ---- entry point ----

def main(argv=None):
    p = argparse.ArgumentParser(prog="barterbench", description="Competitive barter-market benchmark for local LLMs")
    p.add_argument("--suite", action="store_true", help="anchors → round-robin (fixed + held-out) → self-play")
    p.add_argument("--matrix", action="store_true", help="round-robin over --models only")
    p.add_argument("--selfplay", metavar="MODEL", help="model vs itself under hidden labels")
    p.add_argument("--eval", metavar="SCENARIO", help="one scenario, 'all', or 'procedural' (needs --models with counts)")
    p.add_argument("--models", help="'qwen3-8b,phi-4' for suite/matrix; 'qwen3-8b:3,heuristic:3' for --eval")
    p.add_argument("--runs", type=int, default=None, help="runs per scenario (suite default 10, others 1)")
    p.add_argument("--heldout", type=int, default=SUITE_HELDOUT, help="held-out procedural scenarios (default 2)")
    p.add_argument("--selfplay-runs", type=int, default=SUITE_SELFPLAY_RUNS, help="self-play matches per model in --suite")
    p.add_argument("--anchors", default=",".join(SUITE_ANCHORS), help="comma-separated scripted anchors for --suite")
    p.add_argument("--communication", choices=list(COMMUNICATION_MODES), default="full")
    p.add_argument("--conceal", action="store_true", help="instruct agents not to disclose targets (ablation)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--history-rounds", type=int, default=-1, help="rounds of own history shown (-1 = all)")
    p.add_argument("--simultaneous", action="store_true", help="all agents act on a frozen snapshot each round")
    p.add_argument("--parallel", type=int, default=1, help="concurrent matches")
    p.add_argument("--endpoint", action="append", default=[], metavar="ALIAS=URL[::SERVED]",
                   help="map a model alias to a local OpenAI-compatible endpoint (repeatable)")
    p.add_argument("--verbose", "-v", action="store_true", help="print full transcripts")
    p.add_argument("--report", action="store_true", help="ratings, process metrics, reliability, fairness")
    p.add_argument("--json", action="store_true", help="with --report: machine-readable JSON to stdout")
    p.add_argument("--list", action="store_true", help="list scenarios")
    p.add_argument("--registry", action="store_true", help="list local models with VRAM and vLLM hints")
    p.add_argument("--generate", action="store_true", help="generate and save a procedural scenario")
    p.add_argument("--gen-agents", type=int, default=6)
    p.add_argument("--gen-items", type=int, default=4)
    p.add_argument("--gen-scarce", type=int, default=1)
    p.add_argument("--gen-seed", type=int, default=0)
    p.add_argument("--resume", metavar="CHECKPOINT", help=f"resume a match from results/checkpoints/")
    p.add_argument("--clear", action="store_true", help="delete results/ (results, match log, ratings) before running")
    p.add_argument("--version", action="version", version=f"barterbench {__version__}")
    args = p.parse_args(argv)

    RUN_OPTIONS["communication"] = args.communication
    RUN_OPTIONS["conceal"] = args.conceal
    endpoints = load_endpoints()
    for spec in args.endpoint:
        alias, _, rest = spec.partition("=")
        url, _, served = rest.partition("::")
        if not alias or not url:
            sys.exit(f"Error: --endpoint expects ALIAS=URL[::SERVED], got {spec!r}")
        endpoints[alias.strip()] = {"base_url": url.strip(), **({"model": served.strip()} if served else {})}
    RUN_OPTIONS["endpoints"] = endpoints

    if args.clear:
        import shutil
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)
        reset_ratings()
        print("Cleared results/")

    if args.registry:
        print(get_registry_summary())
        if endpoints:
            print("\nEndpoints:")
            for a, ep in endpoints.items():
                print(f"  {a:20s} {ep.get('base_url')}  {ep.get('model', '')}")
        return
    if args.list:
        for s in list_scenarios():
            sc = load_scenario(s)
            scarce = ", ".join(f"{k} ({v['ratio']:.0%})" for k, v in sc.get("scarcity", {}).items()) or "none"
            print(f"  {s:<20} {len(sc['agents'])} agents, {sc.get('max_rounds', 10)} rounds — scarcity: {scarce}")
        return
    if args.report:
        if args.json:
            print(json.dumps(build_report_data(), indent=2))
        else:
            print_full_report()
            print_elo_leaderboard()
        return
    if args.resume:
        resume_run(args.resume, args.verbose)
        return
    if args.generate:
        sc = generate_scenario(num_agents=args.gen_agents, num_items=args.gen_items, num_scarce=args.gen_scarce,
                               seed=args.gen_seed)
        print(f"Generated {sc['name']} → {save_procedural_scenario(sc, overwrite=True)}")
        return

    def _names():
        if not args.models:
            sys.exit("Error: --models required")
        return [m.strip().split(":")[0] for m in args.models.split(",")]

    if args.suite:
        models = [m for m in _names() if m not in BASELINES]
        if not models:
            sys.exit(f"Error: --models needs at least one LLM; {BASELINES} are anchors")
        run_suite(models, args.verbose, parallel=args.parallel, runs_per=args.runs or SUITE_RUNS_PER_SCENARIO,
                  heldout=args.heldout, selfplay_runs=args.selfplay_runs,
                  anchors=[a.strip() for a in args.anchors.split(",") if a.strip()],
                  temperature=args.temperature, history_rounds=args.history_rounds)
    elif args.matrix:
        models = _names()
        if len(models) < 2:
            sys.exit("Error: --matrix needs at least two models")
        scen = None if (not args.eval or args.eval == "all") else [args.eval]
        run_matrix(models, args.runs or 1, args.verbose, parallel=args.parallel, temperature=args.temperature,
                   history_rounds=args.history_rounds, scenarios=scen, heldout=args.heldout)
    elif args.selfplay:
        scen = SUITE_SCENARIOS if (not args.eval or args.eval == "all") else [args.eval]
        run_selfplay(args.selfplay, scen, args.runs or 1, args.verbose, parallel=args.parallel,
                     temperature=args.temperature, history_rounds=args.history_rounds)
    elif args.eval:
        if not args.models:
            sys.exit("Error: --eval needs --models with counts, e.g. qwen3-8b:3,heuristic:3")
        run_eval(args.eval, args.models, args.runs or 1, args.verbose, simultaneous=args.simultaneous,
                 parallel=args.parallel, temperature=args.temperature, history_rounds=args.history_rounds)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
