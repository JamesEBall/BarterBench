# Barter Eval

A benchmark for evaluating LLM bartering capabilities. Inspired by [autoresearch](https://github.com/karpathy/autoresearch) — same autonomous iteration loop, but instead of optimizing a training script, we iteratively improve an eval that measures how well different Claude models negotiate and trade.

## What it does

Two Claude agents are placed in a bartering scenario with private inventories and valuations. They negotiate through structured tool use (propose, accept, reject, end) across multiple rounds. The eval measures how much economic surplus they capture and how efficiently they trade.

## Quick start

```bash
# Run eval: haiku vs opus across all scenarios
python3 -m barter_eval --models haiku opus --runs 1

# Run a specific scenario
python3 -m barter_eval --models haiku opus --scenarios fruit_trade

# View the dashboard (chart + leaderboard + replays)
python3 -m barter_eval --serve
# → open http://localhost:8080/dashboard.html

# Verbose mode: print full negotiation transcripts
python3 -m barter_eval --models haiku opus -v

# Clear previous results
python3 -m barter_eval --models haiku opus --clear
```

## Architecture

```
barter_eval/
├── agent.py          # Claude API/CLI wrapper with tool_use
├── engine.py         # Bartering session engine
├── scoring.py        # Metrics: surplus, Pareto efficiency, fairness
├── eval.py           # CLI entry point
├── dashboard.html    # Combined dashboard + replay viewer
├── program.md        # Outer loop agent instructions
├── results.json      # Eval results (auto-generated)
└── scenarios/
    ├── fruit_trade.json       # Easy: complementary fruit valuations
    ├── resource_scarcity.json # Medium: unequal endowments
    ├── hidden_gem.json        # Hard: extreme info asymmetry
    └── complex_market.json    # Expert: 6 items, complex planning
```

## Scenarios

| Scenario | Items | Difficulty | What it tests |
|---|---|---|---|
| fruit_trade | 3 | Easy | Basic trading, value recognition |
| resource_scarcity | 3 | Medium | Negotiation under asymmetry |
| hidden_gem | 3 | Hard | Strategic deception, info hiding |
| complex_market | 6 | Expert | Multi-step planning, optimization |

## Metrics

| Metric | Description |
|---|---|
| **Barter Score** | Composite (0-1): 40% Pareto + 30% deal rate + 20% fairness + 10% validity |
| **Pareto Efficiency** | Fraction of maximum possible surplus captured |
| **Total Surplus** | Combined value gained by both agents |
| **Deal Rate** | Whether any trades completed |
| **Fairness (Gini)** | How evenly surplus was split |
| **Invalid Rate** | Proportion of invalid trade proposals |

## How it works

1. Each agent gets a system prompt with their private inventory and valuations
2. Agents alternate turns, using structured actions: `propose_trade`, `accept_trade`, `reject_trade`, `end_negotiation`
3. The engine validates trades, updates inventories, and tracks history
4. After the session ends, scoring computes all metrics
5. Results are saved to `results.json` and displayed on the dashboard

## Auth

- **With API key**: Set `ANTHROPIC_API_KEY` — uses the Anthropic SDK with tool_use
- **With Claude Code OAuth**: No API key needed — falls back to `claude` CLI

## The outer loop

Like autoresearch, this is designed for autonomous iteration. A Claude Code agent can:

1. Analyze current eval results
2. Hypothesize improvements (new scenarios, better scoring, prompt changes)
3. Modify the eval code
4. Run evals and measure discriminability between models
5. Keep changes that improve the eval, discard the rest

See `program.md` for the full outer loop instructions.

## First results (haiku vs opus)

```
Model       Score    Pareto   Deal%   Key finding
opus        0.773    0.648    100%    Better strategic planning
haiku       0.744    0.616    100%    More agreeable, less optimal
```

Biggest gap on `hidden_gem` scenario: opus achieves 88% Pareto vs haiku's 40% when information asymmetry matters most.
