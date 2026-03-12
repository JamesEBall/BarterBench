# Barter Eval — Outer Loop Agent Instructions

You are an autonomous agent iterating on a bartering evaluation benchmark.
Your goal: make the eval better at discriminating between model capabilities.

## Setup

1. Read all files in `barter_eval/` to understand the codebase
2. Read existing `barter_eval/results.json` to understand past results
3. Read `results.tsv` (repo root) to understand past iterations

## The Loop

Repeat forever:

### 1. Analyze
- Look at current results: which models score similarly? Which scenarios are too easy/hard?
- Identify what the eval is NOT capturing (e.g., deception, multi-step planning, fairness)
- Check `results.tsv` for what's been tried before — don't repeat failed approaches

### 2. Hypothesize
- Form a specific hypothesis: "Adding scenario X will better separate haiku from opus because..."
- Or: "Changing the scoring weights will improve discriminability because..."
- Or: "Adding a new metric (e.g., deception detection) will reveal differences in..."

### 3. Modify
You may change any file in `barter_eval/` EXCEPT this file. Common modifications:
- Add/edit scenarios in `scenarios/` (new item sets, valuations, asymmetries)
- Modify `scoring.py` (new metrics, weight changes)
- Modify `agent.py` (prompt engineering, tool definitions)
- Modify `engine.py` (negotiation rules, round structure)

### 4. Commit
```bash
git add barter_eval/
git commit -m "barter_eval: <description of change>"
```

### 5. Run Eval
```bash
uv run python -m barter_eval --models haiku opus --runs 2 --clear
```

### 6. Evaluate
Extract key metrics from results.json:
- **discriminability**: variance of barter_score across models (higher = better)
- **stability**: std dev of barter_score within same model pair across runs (lower = better)
- **coverage**: do different scenarios test different capabilities?

### 7. Record
Append to `results.tsv` (tab-separated, repo root):
```
commit	discriminability	stability	status	description
```

### 8. Keep or Discard
- If discriminability improved OR a new capability is tested: **keep** the commit
- Otherwise: `git reset --hard HEAD~1` and try something different

## Key Principles

- **Discriminability is the north star**: the eval should clearly rank models
- **Stability matters**: results should be reproducible across runs
- **Simplicity**: prefer simple scenario changes over complex code changes
- **Coverage**: ensure scenarios test distinct capabilities (basic trading, strategic deception, multi-step planning, fairness)
- **Never stop**: always have a next hypothesis to test
