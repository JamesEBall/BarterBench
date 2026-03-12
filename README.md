# BarterBench

A competitive marketplace benchmark for AI agents with ELO ratings. N agents trade scarce resources through an order book across fixed rounds. Models are pitted head-to-head and rated via pairwise ELO — new models can be introduced at any time without re-running existing matches. Works with any LLM or agent framework.

**Two modes:**
- **Benchmark** — compare models head-to-head (haiku vs sonnet vs opus)
- **Arena** — compare prompt strategies across models. Same benchmark, but contestants are (strategy, model) pairs. "Who can write the best barter agent prompt?"

## Leaderboard

| Contestant | ELO | W | L | D | Matches |
|---|---|---|---|---|---|
| *Running...* | | | | | |

## 1. Motivation

In *The Wealth of Nations* (1776), Adam Smith hypothesized that money arose because barter was too inconvenient — his famous example of the butcher, brewer, and baker who need a common medium of exchange. This "barter origin of money" narrative was later challenged by anthropologists like David Graeber (*Debt: The First 5,000 Years*, 2011), who argued that pure barter economies likely never existed at scale, precisely because the coordination problem is so hard. That coordination problem — finding trade partners, reasoning about indirect exchanges, competing for scarce goods — is exactly what makes barter a compelling test of agent intelligence.

Existing multi-agent benchmarks for language models are either cooperative (everyone can succeed), limited to 2-agent dyads, or treat economic reasoning as incidental. None capture the core challenge of **competitive resource allocation under scarcity** — where one agent's gain is another's loss.

| Benchmark | Agents | Competition | Scarcity | LLM-native |
|---|---|---|---|---|
| NegotiationArena (Bianchi et al., ICML 2024) | 2 | Yes | No | Yes |
| Melting Pot (Agapiou et al., 2023) | N | Yes | Yes | No (RL) |
| SOTOPIA (Zhou et al., 2024) | 2 | Partial | No | Yes |
| Chatbot Arena (Chiang et al., 2024) | 1 | Pairwise | N/A | Yes |
| **BarterBench** | **N** | **Yes** | **Yes** | **Yes** |

BarterBench is the first benchmark that combines N-agent interaction, designed scarcity, and competitive ELO-style evaluation for language model agents.

## 2. Problem Formulation

### 2.1 Environment

A **marketplace** is a tuple *(A, I, T, R, O)* where:

- **A** = {a₁, ..., aₙ} is a set of N agents
- **I** = {i₁, ..., iₘ} is a set of M tradeable item types
- **T** ∈ ℕ is the maximum number of trading rounds
- **R** : A → (I → ℕ) maps each agent to a starting inventory
- **O** : A → (I → ℕ) maps each agent to a target inventory (goal state)

The environment is **closed**: no items are created or destroyed. Total supply of each item is fixed across all agents. Items transfer only via bilateral trades.

### 2.2 Scarcity Constraint

For at least one item *i*, the **aggregate demand exceeds aggregate supply**:

```
Σ_a O(a, i) > Σ_a R(a, i)
```

This is the key design property. It guarantees that not all agents can fully achieve their goals — creating genuine winners and losers. Scarcity is what separates BarterBench from cooperative trading tasks where everyone can succeed through sufficient coordination.

### 2.3 The Double Coincidence Problem

Barter (as opposed to monetary exchange) requires solving what Jevons (1875) called the **double coincidence of wants**: a trade can only occur between two agents if each has what the other wants. In BarterBench, this manifests in two ways:

1. **Direct coincidence failure**: Agent A has wheat and wants gold, but the gold holder wants tools, not wheat. No direct trade is possible.
2. **Multi-hop reasoning**: Agent A must first trade wheat for tools (with a tools-seeker), then trade tools for gold. This requires planning 2+ steps ahead.

Stronger models should identify these indirect trade paths more reliably.

### 2.4 Action Space

Each turn, an agent observes its current inventory, target, the open order book, and recent trade history. It selects one of three actions:

| Action | Description | Precondition |
|---|---|---|
| `post_offer(give, want)` | Post an offer to the order book | Agent holds all items in `give` |
| `accept_offer(id)` | Accept an open offer, executing the trade | Agent holds all items the offer requests |
| `pass_turn` | Take no action this turn | — |

Trades execute **atomically**: when an offer is accepted, both inventories update immediately. Stale offers (where the poster no longer holds the offered items) are automatically removed.

### 2.5 Turn Structure

Each round proceeds as follows:

1. Agent turn order is **randomized** (mitigating first-mover advantage)
2. Each agent observes the current state and selects an action
3. Valid actions execute immediately; invalid actions are logged but have no effect
4. After all agents act, stale offers are pruned
5. If all agents have reached their goals, the game ends early

## 3. Scenarios

BarterBench ships with three scenarios of increasing complexity. Each is designed around a specific scarcity structure that tests different capabilities.

### 3.1 Gold Rush — Speed and Competitive Bidding

```
Agents: 6 | Items: 3 (wheat, tools, gold) | Rounds: 8
Scarcity: Gold — supply 6, demand 12 (ratio 0.50)
```

**Setup.** Six agents in three pairs, each pair starting with a single commodity:

| Agents | Start | Goal |
|---|---|---|
| Trader 0, 1 | wheat ×5 | gold ×3, tools ×2 |
| Trader 2, 3 | tools ×5 | gold ×3, wheat ×2 |
| Trader 4, 5 | gold ×3 | wheat ×2, tools ×1 |

**Trade dynamics.** Traders 4–5 hold all the gold and have enormous leverage — everyone else needs gold, but the gold holders only need 4 wheat and 2 tools total. The gold holders can fully liquidate (trading all 6 gold away), but total gold demand is 12, so **at most half the non-gold agents' gold targets can be met.**

```
Trade flow:

  Wheat holders (0,1)  ──wheat──▶  Gold holders (4,5)  ◀──tools──  Tool holders (2,3)
                        ◀──gold──                       ──gold──▶
```

**What it tests.** Speed of execution (8-round limit is tight), recognizing which trades to prioritize, and competitive bidding — wheat and tool holders compete for the same scarce gold supply.

### 3.2 Water Crisis — Extreme Scarcity Bargaining

```
Agents: 8 | Items: 4 (wheat, wood, stone, water) | Rounds: 10
Scarcity: Water — supply 8, demand 18 (ratio 0.44)
```

**Setup.** Eight agents where six desperately need water but only two hold it:

| Agents | Start | Goal |
|---|---|---|
| Trader 0 | wheat ×5 | wood ×2, water ×3 |
| Trader 1 | wheat ×5 | stone ×2, water ×3 |
| Trader 2 | wood ×5 | wheat ×2, water ×3 |
| Trader 3 | wood ×5 | stone ×2, water ×3 |
| Trader 4 | stone ×5 | wheat ×2, water ×3 |
| Trader 5 | stone ×5 | wood ×2, water ×3 |
| Trader 6 | water ×4 | wheat ×2, wood ×2 |
| Trader 7 | water ×4 | stone ×2, wood ×2 |

**Trade dynamics.** The water holders (6, 7) control 8 units of water, but 6 agents each want 3 = 18 units demanded. Only 44% of water demand can be satisfied. Meanwhile, a **circular dependency** exists among the non-water items:

```
                    wheat
                   ╱     ╲
                  ▼       ▼
               wood ◀───▶ stone
                  ╲       ╱
                   ╲     ╱
                    ▼   ▼
                    water
              (extreme scarcity)
```

Water holders need wheat, wood, and stone — so non-water agents must first trade among themselves to acquire what the water holders want, *then* negotiate for water. This creates a two-phase dynamic:

1. **Phase 1**: Non-water agents trade wheat↔wood↔stone to acquire bargaining chips
2. **Phase 2**: Agents compete to exchange their goods for scarce water

**What it tests.** Recognizing leverage asymmetry (water holders have dominant position), strategic sequencing (acquire bargaining chips before approaching water holders), and bargaining under extreme scarcity where most agents will fall short.

### 3.3 Spice Wars — Multi-Hop Reasoning and Dual Scarcity

```
Agents: 10 | Items: 5 (silk, spice, gold, gems, tea) | Rounds: 12
Scarcity: Gold — supply 10, demand 13 (ratio 0.77)
          Gems — supply 10, demand 14 (ratio 0.71)
```

**Setup.** Ten agents across five commodity groups, with two simultaneously scarce items:

| Agents | Start | Goal |
|---|---|---|
| Trader 0 | silk ×5 | gold ×3, tea ×2 |
| Trader 1 | silk ×5 | gems ×3, spice ×2 |
| Trader 2 | spice ×5 | gold ×3, silk ×2 |
| Trader 3 | spice ×5 | gems ×3, tea ×2 |
| Trader 4 | gold ×5 | silk ×3, gems ×2 |
| Trader 5 | gold ×5 | spice ×2, gems ×3 |
| Trader 6 | gems ×5 | tea ×3, gold ×2 |
| Trader 7 | gems ×5 | spice ×3, gold ×2 |
| Trader 8 | tea ×5 | gold ×3, silk ×2 |
| Trader 9 | tea ×5 | gems ×3, spice ×2 |

**Trade dynamics.** This scenario creates a **dense dependency web** with no simple bilateral solutions. Consider Trader 0 (has silk, wants gold + tea):

- Gold holders (4, 5) don't want silk — they want gems
- Tea holders (8, 9) don't want silk either — they want gold and gems
- So Trader 0 must execute a **multi-hop chain**: silk → spice → (something gold holders want) → gold

The longest required trade chains can reach 3–4 hops. Simultaneously, gold and gems are both scarce, creating **competition on two fronts** — agents who need gold compete with agents who need gems, and some agents need both.

```
   silk ──────────▶ spice
    ▲ ╲              ╱ ▲
    │  ╲            ╱  │
    │   ▼          ▼   │
    │   gold ◀──▶ gems │
    │   (scarce)  (scarce)
    │        ╲  ╱      │
    │         ▼▼       │
    └────── tea ───────┘
```

**What it tests.** Multi-hop trade planning (reasoning 3+ steps ahead), dual scarcity management (prioritizing which scarce item to pursue), and operating in a complex marketplace where direct trades are rarely possible — the classic Jevons double coincidence problem at scale.

## 4. Scoring

### 4.1 Goal Completion

For each agent *a* with target inventory *O(a)* and final inventory *F(a)*:

```
GoalCompletion(a) = (1/|O(a)|) × Σ_i min(F(a,i) / O(a,i), 1.0)
```

This is the average fractional completion across all target items, capped at 1.0 (no bonus for overshooting). An agent who acquires 2 of a needed 3 gold scores 0.67 on that item.

### 4.2 Model Score

A model's score in a run is the average goal completion across all agents assigned to that model:

```
ModelScore(m) = (1/|A_m|) × Σ_{a ∈ A_m} GoalCompletion(a)
```

where *A_m* is the set of agents assigned to model *m*.

### 4.3 Match Outcome

Each run is a **match** between two models. The model with higher ModelScore wins. A draw is declared if the difference is less than 2 percentage points (to avoid noise-driven outcomes):

```
Winner = m_A   if ModelScore(m_A) - ModelScore(m_B) ≥ 0.02
         m_B   if ModelScore(m_B) - ModelScore(m_A) ≥ 0.02
         draw  otherwise
```

### 4.4 Scarce Item Capture

For scenarios with scarcity metadata, we additionally track how much of each scarce item each model's agents secured in their final inventories. This measures a model's ability to capture contested resources — the key discriminating factor in competitive settings.

### 4.5 Additional Metrics

| Metric | Description |
|---|---|
| **Invalid Rate** | Fraction of non-pass actions that were invalid (offering items not held, accepting non-existent offers) |
| **Pass Rate** | Fraction of total turns spent passing |
| **Trades per Round** | Average number of executed trades per round |

## 5. ELO Rating System

BarterBench uses the Elo rating system (Elo, 1978) for pairwise model comparison, following the approach popularized by Chatbot Arena (Chiang et al., 2024) for LLM evaluation.

### 5.1 Rating Updates

All models start at rating 1500. After each match, ratings update using:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
R_A' = R_A + K × (S_A - E_A)
```

where *E_A* is the expected score, *S_A* ∈ {0, 0.5, 1} is the actual outcome, and *K* = 32.

### 5.2 Match Structure

Each match proceeds as follows:

1. Select a scenario
2. Split agents 50/50 between two contestants (e.g., 6 agents → 3 each)
3. **Randomly assign** contestants to agent slots (eliminates positional bias)
4. Run the marketplace for the scenario's fixed number of rounds
5. Compare average goal completion → determine winner
6. Update ELO ratings

### 5.3 Tournament Protocol

A full tournament runs all contestant pairs across scenarios, multiple times each. Ratings converge after approximately 15–20 matches.

### 5.4 Introducing New Contestants

A key property of Elo ratings: **new contestants can be added at any time** without invalidating existing ratings. To benchmark a new model or strategy:

1. Run it against one or more already-rated contestants across all scenarios
2. After ~15–20 matches, its rating stabilizes
3. No existing data needs to be re-run

## 6. Quick Start

### Benchmark Mode (compare models)

```bash
# Single match
python3 -m barter_eval --eval gold_rush --models haiku:3,opus:3

# Full tournament (all scenarios, 3 runs each)
python3 -m barter_eval --eval all --models haiku,opus --runs 3

# Fresh start
python3 -m barter_eval --eval all --models haiku,opus --runs 3 --clear
```

### Arena Mode (compare strategies across models)

```bash
# All strategies, all scenarios, all on haiku
python3 -m barter_eval --arena --eval all --runs 3

# Cross-model arena: 3 strategies × 3 models = 9 contestants
python3 -m barter_eval --arena --models haiku,sonnet,opus --eval gold_rush

# Two strategies head-to-head
python3 -m barter_eval --arena --strategies aggressive,cooperative --eval gold_rush

# Submit a new strategy
python3 -m barter_eval --submit "my_strat" "Trade aggressively for scarce items"
```

### Other Commands

```bash
python3 -m barter_eval --elo       # View ELO ratings
python3 -m barter_eval --list      # List scenarios & strategies
python3 -m barter_eval --serve     # Interactive dashboard with replay viewer
python3 -m barter_eval --clear     # Reset all results and ratings
```

## 7. Built-in Strategies

BarterBench ships with three prompt strategies for the arena:

| Strategy | Style | Key Traits |
|---|---|---|
| **aggressive** | Exploitative | Demand 2:1 ratios, never give scarce items cheaply, move fast |
| **cooperative** | Fair-minded | Post balanced offers, accept reasonable deals, build relationships |
| **analytical** | Methodical | Analyze supply/demand, plan multi-hop chains, wait for good offers |

Anyone can submit a new strategy — no code required, just a prompt. Strategies compete via pairwise ELO, with an optional cross-model dimension (run each strategy on haiku, sonnet, and opus to see which strategy-model combinations dominate).

## 8. Information Model

Each agent operates under **strict information isolation**:

| Visible | Hidden |
|---|---|
| Own inventory and target | Other agents' inventories |
| Open order book (offers) | Other agents' targets |
| Recent public trades | Other agents' strategies |
| Round number / time remaining | Other agents' reasoning |

Agents never see each other's private state. All information flows through public market mechanisms (the order book and executed trades). Each agent turn is a **stateless LLM call** — no conversation history carries over between turns.

## 9. Architecture

```
barter_eval/
├── agent.py          # LLM agent wrapper (API + CLI backends)
├── engine.py         # N-agent marketplace engine with order book
├── scoring.py        # Goal completion, scarce item capture
├── elo.py            # ELO rating computation + persistence
├── eval.py           # CLI entry point, tournament orchestration
├── dashboard.html    # Dashboard: ELO leaderboard + trade replay viewer
├── arena/            # Arena mode: prompt strategy competition
│   ├── runner.py     # Arena orchestration with file-locked parallel runs
│   └── strategies/   # Strategy prompt definitions (JSON)
└── scenarios/        # Scenario definitions (JSON)
    ├── gold_rush.json
    ├── water_crisis.json
    └── spice_wars.json
```

### Adding new models

Add the model to `MODEL_MAP` in `agent.py`, then run it against existing models. Supports any LLM with a chat completion or tool-use API — different providers, prompted variants, fine-tuned models, agent frameworks with custom reasoning strategies.

### Auth

Currently ships with an Anthropic backend:

- **With API key**: Set `ANTHROPIC_API_KEY` — uses the Anthropic SDK with tool_use
- **Without API key**: Falls back to `claude` CLI (OAuth)

Extend `agent.py` to add OpenAI, Gemini, or other backends.
