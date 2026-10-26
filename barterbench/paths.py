"""Filesystem layout. Everything generated lands under results/ (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_DIR = ROOT / "scenarios"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_FILE = RESULTS_DIR / "results.json"
MATCH_LOG = RESULTS_DIR / "match_log.json"
ELO_FILE = RESULTS_DIR / "elo_ratings.json"
RATINGS_FILE = RESULTS_DIR / "ratings.json"
REPORT_FILE = RESULTS_DIR / "report.json"
ENDPOINTS_FILE = ROOT / "endpoints.json"
