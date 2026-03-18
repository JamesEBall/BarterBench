#!/usr/bin/env python3
"""
Build and publish BarterBench dataset to Kaggle.
Usage: python3 scripts/publish_kaggle.py [--new]
  --new   Create the dataset for the first time (default: update existing)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STAGING = ROOT / "kaggle_dataset"
SCENARIOS_SRC = ROOT / "scenarios"


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def build_leaderboard_csv(results_data, bt_data, elo_data, out_path):
    """Generate leaderboard.csv from results + ratings."""
    model_stats = {}

    for entry in results_data:
        if "error" in entry:
            continue
        mgc = entry.get("model_goal_completion", {})
        for model, score in mgc.items():
            if model not in model_stats:
                model_stats[model] = {"scores": [], "wins": 0, "losses": 0, "draws": 0}
            model_stats[model]["scores"].append(score * 100)

    rows = []
    for model, stats in model_stats.items():
        scores = stats["scores"]
        avg = sum(scores) / len(scores) if scores else 0
        n = len(scores)
        elo = (elo_data or {}).get(model, {})
        elo_rating = elo.get("rating", 1500) if isinstance(elo, dict) else elo
        bt_rating = (bt_data or {}).get(model, None)
        rows.append({
            "model": model,
            "avg_goal_completion_pct": round(avg, 1),
            "n_games": n,
            "elo_rating": round(elo_rating, 1),
            "bt_rating": round(bt_rating, 4) if bt_rating is not None else "",
        })

    rows.sort(key=lambda r: r["avg_goal_completion_pct"], reverse=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "avg_goal_completion_pct", "n_games", "elo_rating", "bt_rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  leaderboard.csv: {len(rows)} models")


def build_staging():
    """Assemble the kaggle_dataset/ staging directory."""
    STAGING.mkdir(exist_ok=True)

    # Load source data
    results_data = load_json(ROOT / "results.json") or []
    bt_data = load_json(ROOT / "bt_ratings.json")
    elo_data = load_json(ROOT / "elo_ratings.json")

    print("Building staging directory...")

    # Leaderboard CSV
    build_leaderboard_csv(results_data, bt_data, elo_data, STAGING / "leaderboard.csv")

    # Copy JSON data files
    for fname in ["results.json", "bt_ratings.json", "elo_ratings.json"]:
        src = ROOT / fname
        if src.exists():
            shutil.copy(src, STAGING / fname)
            print(f"  copied {fname}")

    # Copy scenarios
    scenarios_dst = STAGING / "scenarios"
    if scenarios_dst.exists():
        shutil.rmtree(scenarios_dst)
    shutil.copytree(SCENARIOS_SRC, scenarios_dst)
    print(f"  copied scenarios/")

    print(f"Staging complete: {STAGING}")


def kaggle_publish(new=False):
    env = os.environ.copy()
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        # Write kaggle.json for CLI
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        # Parse token — Kaggle CLI expects username+key in kaggle.json
        # KAGGLE_API_TOKEN format is just the key; username comes from config
        username = "jameseball"
        kaggle_json.write_text(json.dumps({"username": username, "key": token}))
        kaggle_json.chmod(0o600)

    if new:
        cmd = ["kaggle", "datasets", "create", "-p", str(STAGING), "--dir-mode", "zip"]
        print("Creating new Kaggle dataset...")
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(STAGING), "-m", "Auto-update via CI", "--dir-mode", "zip"]
        print("Updating Kaggle dataset...")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode == 0:
        print(result.stdout)
        print("Done.")
    else:
        print("ERROR:", result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", action="store_true", help="Create dataset (first time)")
    args = parser.parse_args()

    build_staging()
    kaggle_publish(new=args.new)
