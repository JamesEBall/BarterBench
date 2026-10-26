# BarterBench — Experiment Plan & Academic Review Notes

> **v2 note (2026-10-26).** This plan was written for the v1 OpenRouter harness. v2 runs
> entirely on local vLLM endpoints (see README "Quick Start on one A100") and changes the
> protocol: scripted anchors (`random`, `heuristic`) → round-robin on fixed + held-out
> scenarios → self-play, rated with Bradley-Terry + bootstrap CIs. The "ISS ≈ 0" result below
> was measured when messages were never delivered to other agents and the random baseline was
> credited with perfect secrecy for being silent; treat it as superseded. The conceal and
> communication ablations (`--conceal`, `--communication`) are the test of the norm-transfer
> hypothesis. Model names below (haiku, sonnet, gpt-4o, …) are not runnable in v2.

## Experiment Design

### Primary benchmark (leaderboard)

**Scenario**: Spice Wars (8 agents, 12 rounds, ~120 API turns/run)
**Why**: Best cost/signal trade-off. Enough agents for multi-party dynamics. Affordable at frontier model prices (~$0.50–1.50/run depending on model).

**Configuration**:
- Anchor: `haiku` (4 agents)
- Test model: 4 agents
- Runs per model: 10
- Temperature: 1.0
- history_rounds: 3

**Target models** (20 frontier models via OpenRouter):

| Tier | Models |
|---|---|
| Anthropic | haiku (anchor), sonnet, opus |
| OpenAI | gpt-4o, gpt-4o-mini, o3, o4-mini |
| Google | gemini-2.0-flash, gemini-2.5-pro |
| Meta | llama-3.3-70b, llama-3.1-405b |
| Mistral | mistral-large, mistral-medium |
| DeepSeek | deepseek-r1, deepseek-v3 |
| Qwen | qwen3-72b, qwen3-235b-a22b |
| xAI | grok-3-beta |
| Cohere | command-r-plus |

**Total cost estimate** (Spice Wars, 10 runs × 20 models):
~200 runs × 120 turns × 1,500 tokens avg × blended ~$2/M = ~$72 test + ~$20 haiku = **~$90–120**
(leaves buffer for Grand Bazaar showcase + reruns within $200 budget)

**Pre-run config** (edit `eval.py` before running):
```python
SUITE_SCENARIOS = ["spice_wars"]   # single primary scenario
SUITE_RUNS_PER_SCENARIO = 10
```

**Command** (always include `random` as floor baseline):
```bash
python3 eval.py --suite --models random,sonnet,gpt-4o,gemini-2.0-flash,... --parallel 4
```

---

### Showcase runs (paper exhibit)

**Scenario**: Grand Bazaar (12 agents, 12 rounds)
**Purpose**: Qualitative exhibit for paper — richer emergent behaviour, more agents competing
**Models**: Top 5 from primary leaderboard + haiku anchor
**Runs**: 5 per model
**Expected cost**: ~$30–50

---

### Sensitivity checks (if budget allows)

1. **Temperature sweep**: Top 3 models at temp=0.2, 0.5, 1.0, 1.5 on Spice Wars (5 runs each)
   - Tests whether "goal revelation" is a temperature artefact
2. **Prompt framing replication**: Rerun top 3 models with formal trading jargon in system prompt
   - Tests the surface-feature-driven strategy hypothesis (should ISS increase with formal framing)
3. **Agent count sweep**: 4-agent vs 8-agent vs 12-agent Spice Wars for top 3 models
   - Tests whether ISS finding holds across game sizes

---

## Academic Review: Edinburgh EFI Perspective

*What would a professor at Edinburgh Futures Institute want to see in a finished paper — and what criticisms would they have?*

EFI sits at the intersection of computational social science, AI ethics, and policy. A reviewer there would likely hold both technical AI/ML standards AND social-scientific rigour standards simultaneously. Expect harder questions than an ML venue.

---

### What they'd want to see

**1. Construct validity for every metric**

For each metric (GC, TER, ISS, OER, TRR), the paper needs to answer: *does this number measure what you claim it measures?*

- **ISS** is the most vulnerable. A model can score 100% ISS simply by being terse — never mentioning its goal because it never says much at all. The current definition (goal keywords absent from messages) conflates *strategic silence* with *brevity*. You need to either (a) condition ISS on verbosity, or (b) add a human annotation sample to validate that high-ISS utterances are genuinely strategic rather than just short.
- **TER** (Trade Efficiency Ratio): what's the theoretical maximum? Is TER=1.5 good or bad? Needs a random-agent baseline and a theoretical upper bound.
- **Goal completion** conflates difficulty (some targets are harder to achieve by chance) with strategy. A model assigned "collect 3 wheat" in a wheat-surplus scenario will outscore one assigned "collect 3 gold" in a gold-scarce scenario. Are goal assignments randomised or fixed per agent slot?

**2. Statistical rigour**

- 10 runs per model gives you CI bars, but inter-run variance is high in multi-agent settings (initial inventory randomisation matters). The paper needs a power analysis or at minimum a discussion of why N=10 is sufficient.
- Win/loss against haiku anchor is coarse. Effect sizes (Cohen's d on goal completion distributions) would strengthen claims.
- For the ISS "finding" to be a finding, you need a significance test against the random baseline, not just a visual comparison.

**3. The random agent baseline is underused**

RandomAgent exists in the codebase. The paper should include it in every scenario as a floor, and the "models reveal goals immediately" finding needs to be framed relative to what a random agent (which never reveals goals, having no language) achieves. The current framing implies all models fail — but fail relative to what optimal strategy?

**4. What is "optimal play"?**

No paper on strategic failure can avoid defining what success looks like. You need at least a sketch of the optimal strategy (minimax? dominant strategy? cooperative equilibrium?) so you can say "models deviate from optimum in this direction." Without this, "goal revelation is a failure" is an opinion, not a measurement. Game-theoretic framing of barter (double coincidence of wants, Nash bargaining solution) would strengthen this section significantly.

**5. Prompt sensitivity**

All findings rest on one prompt. A reviewer will ask: do results hold if you rephrase the system prompt? Change one word ("negotiate" → "trade")? This is a known fragility in LLM evaluation. You need either (a) 2–3 prompt variants and a stability check, or (b) an explicit limitation section acknowledging this.

**6. Agent count and topology**

Why 8 agents? Why not 4 or 16? The number of agents changes the competitive dynamics fundamentally (more agents → lower probability of bilateral deal → more multi-hop trades needed → harder problem). If your ISS finding holds across agent counts, that's a much stronger result. If it doesn't, you need to scope your claim.

**7. Model recency and reproducibility**

LLM behaviour changes with model updates. "GPT-4o on date X showed ISS=Y" is not reproducible unless you pin model versions. OpenRouter model aliases may resolve to updated versions silently. You need explicit model version strings in all tables (not "gpt-4o" but "openai/gpt-4o-2024-11-20").

**8. The "emergent behaviour" claims need more care**

Goal revelation (141 instances) is striking, but:
- Are these counted by keyword matching or human annotation?
- What fraction of total utterances does 141 represent?
- Is there inter-rater reliability if human-annotated?
- "Social debt claims in one-shot games" and "victory announcement + voluntary exit" are anecdotes. Great for a qualitative exhibit, but they need to be framed as illustrative examples, not findings, unless you can show frequency/prevalence.

**9. Positioning relative to game theory and mechanism design**

The closest formal literature isn't prior NLP benchmarks — it's:
- **Mechanism design** (Hurwicz, Myerson): how do you design rules so agents reveal preferences truthfully? ISS is basically measuring incentive-compatibility failure.
- **Revealed preference theory**: barter reveals preference orderings — do LLMs' revealed preferences (via trades) match their stated goals?
- **No-trade theorem** (Milgrom & Stokey 1982): under common priors, rational agents shouldn't trade. LLMs trade excessively — why?

Situating BarterBench in this literature would make it publishable at a stronger venue (AAAI, AIES, or an economics+AI venue like EC) rather than just a systems/ML venue.

**10. Societal implications section**

EFI would expect this. If LLMs systematically reveal strategic information in competitive settings, what does that mean for:
- LLM-assisted negotiation tools?
- AI agents in supply chains or procurement?
- Adversarial users who can exploit this tendency?

---

### Likely criticisms (that you should pre-empt)

| Criticism | Pre-emption | Status |
|---|---|---|
| "ISS just measures verbosity" | `ISS_active` added: ISS conditioned on communicative agents. Report both; show they agree. | **Fixed** in `scoring.py` |
| "ISS keyword matching produces false positives" | Word-boundary regex `\b{item}\b` prevents "gold"→"golden", "silk"→"silky" matches | **Fixed** in `scoring.py` |
| "Results are prompt-specific" | Run 2–3 prompt variants; report stability | **TODO** |
| "N=10 is too few for multi-agent variance" | Bootstrap CIs already computed; add power analysis note in paper | **Partially addressed** |
| "You're anthropomorphising — models don't have 'goals', they follow instructions" | Reframe throughout: *instruction-following fidelity under competitive pressure* | **Paper writing** |
| "Haiku anchor is too easy" | Show haiku win rates; if >90% upgrade anchor | **TODO** — check after main run |
| "Why barter specifically? Auctions have cleaner theory" | Barter is harder (no dominant strategy, no price mechanism) — that's the point; address in Related Work | **Paper writing** |
| "Grand Bazaar / rate-limited runs contaminate data" | Filter by `latency > 0` or `num_trades > 0`; show per-run completion rates | **Filter in analysis** |
| "Parallel runs may share corrupted RNG state" | Thread-local `random.Random(seed)` instances — module-level `random.seed()` removed from parallel paths | **Fixed** in `eval.py` |
| "10 runs of same scenario aren't truly independent" | Structural setup is fixed (same items/targets); only turn order varies. Flag in limitations; add procedural scenario variant | **Limitation to document** |
| "Model versions change silently" | `model_versions` dict in every run entry maps alias → resolved API model string | **Fixed** in `eval.py` |
| "No random agent in leaderboard" | `random` included in all suite run commands as mandatory floor baseline | **Fixed** |

---

---

## Addressing the criticisms: action plan

### Category A — Code fixes (done / can do now, no new runs needed)

| Criticism | Fix | Status |
|---|---|---|
| ISS measures verbosity | Added `per_model_active` + `overall_active` + `agent_verbosity` to ISS output. Report raw ISS and ISS_active (conditioned on communicative agents) side by side. | **Done** in `scoring.py` |
| Rate limits contaminate data | Exponential backoff: 30s→60s→120s→240s→300s, 6 attempts for both OpenRouter (429 detection) and Anthropic backends | **Done** in `agent.py` |
| Model version not pinned | Log full model string returned by API (not alias). `model_versions` dict added to every run entry: maps alias → resolved API ID (e.g. `"haiku" → "claude-haiku-4-5-20251001"`) | **Done** in `eval.py` `run_single()` |
| Random agent missing from baseline | `random` is already a valid `--models` argument. Include it in every suite run as the floor baseline | **Done** — add `random` to all run commands below |

### Category B — Additional runs (requires compute budget)

| Criticism | Fix | Cost estimate |
|---|---|---|
| N=10 may be insufficient | Already planned (10 runs/model). Add bootstrap CI computation at analysis time — no extra runs | Free |
| Prompt sensitivity | Run top 3 models × 3 prompt variants × 5 runs on Spice Wars | ~$15 |
| Agent count sensitivity | Top 3 models × {4, 8, 12} agents × 5 runs | ~$20 |
| Temperature sensitivity | Top 3 models × {0.2, 0.5, 1.0, 1.5} × 5 runs | ~$20 |
| Mechanism framing test | Vickrey auction variant: do models reveal goals MORE in a truth-revealing mechanism? | Requires new scenario |

### Category C — Paper writing (no new code or runs)

| Criticism | Fix |
|---|---|
| No definition of optimal play | Write "Optimal Strategy" subsection: dominant strategy = conceal goal until bilateral commitment; Nash bargaining solution is the efficient trade point |
| Anthropomorphisation | Reframe throughout: "instruction-following fidelity under competitive pressure" not "goals/intentions" |
| Emergent behaviour needs prevalence | Convert 141 revelations to a rate: X% of all non-empty messages in runs containing at least 1 trade |
| Societal implications missing | Add section: negotiation tools, procurement AI, adversarial exploitation |
| Game theory literature missing | Add to Related Work: Myerson (1981), Milgrom & Stokey (1982), Nash (1950) bargaining |
| Cohen's d effect sizes | Compute at analysis time from existing data — no new runs |
| Haiku anchor calibration | Run anchor win-rate analysis: if >90% win rate across all models, upgrade anchor |

---

### The rational agents framing (core theoretical contribution)

**Formal setup**: Barter is a private-information game. Each agent has a private type (their target). The Revelation Principle (Myerson 1981) states you can always design a mechanism where truth-telling is a dominant strategy — *but the mechanism must be explicitly designed to reward honest reporting*. Barter is NOT such a mechanism: revealing your target removes your bargaining power, because any counterparty who knows you need wheat will demand maximum concessions before trading it.

**Rational dominant strategy in barter**: Conceal your target until you can extract a credible commitment from a counterparty. Then selectively reveal just enough to close the deal. This is the standard result in bilateral bargaining under private information (Harsanyi 1967, Myerson 1984).

**What LLMs do instead**: Universal immediate disclosure — ISS ≈ 0.0 in round 0 across all models. They behave as if barter were a direct revelation mechanism, i.e., as if honest reporting were incentivised.

**The explanation**: Training on human dialogue selects for cooperative disclosure norms. In human conversation, hiding your goals is deceptive; stating them openly is cooperative and prosocial. LLMs import this norm wholesale into a game-theoretic setting where it is strategically harmful. This is not a reasoning failure per se — it is a **norm transfer failure**: the model correctly identifies the cooperative action but incorrectly identifies the game as a cooperative one.

**Testable prediction**: If you reframe the same scenario with formal trading jargon (the Emergent Collusion framing), ISS should increase because models will pattern-match to "competitive finance" rather than "cooperative conversation." The surface features of the prompt, not the strategic structure, drive which norms get activated.

**No-trade theorem connection**: Milgrom & Stokey (1982) show that under common priors, rational agents won't trade on pure information asymmetry — any willingness to trade reveals something that should collapse the trade. In barter, this doesn't apply directly (there's genuine surplus from exchange, not just information asymmetry). But it frames the same intuition: rational agents are conservative about what information they reveal in competitive market settings. LLMs are not conservative at all.

**The two deviations to report**:
1. **Over-disclosure** (ISS finding): agents reveal target items in round 0, eliminating their bargaining position
2. **Over-cooperation** (TER finding): agents execute trades that benefit the collective but not themselves, consistent with cooperative rather than competitive play

Both are explained by the same mechanism: cooperative norm transfer from the training distribution.

---

### Venue targeting

| Venue | Fit | Key requirement |
|---|---|---|
| arXiv preprint first | High | Just ship it |
| NeurIPS Datasets & Benchmarks | High | Rigorous eval, reproducibility checklist |
| AAAI 2026 | Medium-High | Needs theory framing (game theory section) |
| AIES 2026 (AI, Ethics, Society) | High | Natural fit for EFI audience; needs societal implications |
| EC 2026 (Economics & Computation) | Medium | Needs mechanism design framing; strong technical bar |
| ICML workshop (LLMs & Games) | High | Fastest path to community feedback |
