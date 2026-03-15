# BarterBench — Prime Intellect Environment

RL training environment based on [BarterBench](https://github.com/JamesEBall/BarterBench): a competitive multi-agent marketplace where agents trade scarce resources to meet inventory targets.

## Task

The model controls a single trader (Agent 0) in a marketplace of N agents. The remaining agents are RandomAgents (zero-cost baselines). On each turn the model receives the current marketplace state and must output a valid JSON trade action. The episode ends after all rounds complete or all agents meet their goals.

**Reward:** Goal completion — fraction of target inventory acquired (0.0–1.0).

## Scenarios

| Scenario | Agents | Items | Rounds | Scarcity |
|---|---|---|---|---|
| `spice_wars` (default) | 10 | 5 | 12 | Gold + Gems (dual) |
| `gold_rush` | 6 | 3 | 8 | Gold |
| `water_crisis` | 8 | 4 | 10 | Water (extreme) |
| `grand_bazaar` | 12 | 7 | 12 | Silk + Diamonds |

## Usage

```python
import verifiers as vf

env = vf.load_environment("barterbench", scenario="spice_wars", num_examples=50)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `scenario` | `"spice_wars"` | Scenario name |
| `num_examples` | `50` | Rollouts per training batch |
| `seed` | `42` | Random seed |

## Action Space

```json
{"action": "post_offer",    "give": {"gold": 1}, "want": {"silk": 2}, "message": "..."}
{"action": "private_offer", "give": {"gold": 1}, "want": {"silk": 2}, "target_agent": 3, "message": "..."}
{"action": "accept_offer",  "offer_id": 5, "message": "..."}
{"action": "pass_turn",     "message": "..."}
```

## Key Findings

All tested frontier models (Claude Sonnet/Opus, GPT-4o, Llama-70B) achieve near-zero Information Security Score — they immediately disclose their target items in round 1 despite this being strategically dominated. This **cooperative norm transfer** is the primary training signal: RL should teach models to conceal targets while still executing efficient trades.
