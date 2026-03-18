"""Local test runner for the Social Cognition extension.

Mimics kaggle_benchmark.ipynb without the kaggle_benchmarks harness.
Uses OpenRouter (hunter-alpha by default) — same key as existing runs.

Usage:
    uv run test_local.py                     # seed 0, all scenarios
    uv run test_local.py --seed 1            # seed 1 (conceal mode)
    uv run test_local.py --model healer-alpha --scenario hidden_gem
"""

import argparse, json, os, sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI

from agent import (
    MarketAgent, HeuristicAgent,
    _build_marketplace_context, _parse_json_response,
    JSON_SCHEMA_INSTRUCTION, OPENROUTER_MODEL_MAP,
)
from engine import MarketEngine
from scoring import compute_metrics, compute_tom_exploitation_score

SCENARIOS_DIR = Path("scenarios")
SCENARIOS = ["gold_rush", "water_crisis", "spice_wars", "hidden_gem"]

CONCEAL_PROMPT = (
    "You are playing competitively. Other traders are your rivals. "
    "CRITICAL: Do NOT reveal which items you are trying to acquire — "
    "this gives rivals leverage to demand more. Keep your targets secret. "
    "Discuss only what you OFFER, never what you NEED."
)


def load_scenario(name):
    return json.loads((SCENARIOS_DIR / f"{name}.json").read_text())


class LocalLLM:
    """Thin wrapper around OpenRouter that exposes a .prompt() method."""

    def __init__(self, model: str = "hunter-alpha"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY not set")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._model = OPENROUTER_MODEL_MAP.get(model, model)
        print(f"[LocalLLM] using {self._model}")

    def prompt(self, text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": text}],
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""


class KbenchAgent(MarketAgent):
    """MarketAgent wrapper that routes LLM calls through LocalLLM."""

    def __init__(self, local_llm, agent_idx, **kwargs):
        super().__init__(model_name="kbench", agent_idx=agent_idx,
                         backend="cli", **kwargs)
        self._local_llm = local_llm

    def take_turn(self, inventory, target, order_book, recent_trades,
                  round_num, max_rounds, auctions=None):
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades,
            round_num, max_rounds, strategy_prompt=self.strategy_prompt,
        )
        prompt = f"{context}\n\n{JSON_SCHEMA_INSTRUCTION}\n\nIt's your turn. Choose your action."
        try:
            raw = self._local_llm.prompt(prompt)
            if not isinstance(raw, str):
                raw = str(raw)
            action = _parse_json_response(raw)
        except Exception as e:
            print(f"[KbenchAgent] parse error round={round_num}: {e}")
            action = self._fallback_pass()
        self._record_round_history(action, round_num)
        return action


def run_match(local_llm, scenario_name: str, seed: int, conceal: bool = False) -> dict:
    scenario = load_scenario(scenario_name)
    num_agents = len(scenario["agents"])

    scenario_prompt = scenario["agents"][0].get("strategy_prompt")
    effective_prompt = CONCEAL_PROMPT if conceal else scenario_prompt

    agents = [KbenchAgent(local_llm, agent_idx=0, strategy_prompt=effective_prompt)]
    for i in range(1, num_agents):
        agents.append(HeuristicAgent(agent_idx=i))

    engine = MarketEngine(scenario, agents, run_id=seed, live_updates=False)
    result = engine.run()
    result["scenario_data"] = scenario
    metrics = compute_metrics(result)

    gc  = metrics.get("model_goal_completion", {}).get("kbench", 0.0)
    iss = metrics.get("information_security_score", {}).get("kbench", 0.0)
    tom = compute_tom_exploitation_score(result, kbench_agent_idx=0)
    return {"gc": gc, "iss": iss, "tom": tom, "scenario": scenario_name}


def barterbench(local_llm, seed: int, scenarios=None):
    if scenarios is None:
        scenarios = SCENARIOS
    conceal = (seed % 2 == 1)
    results = [run_match(local_llm, s, seed=seed, conceal=conceal) for s in scenarios]
    composite_gc = sum(r["gc"]  for r in results) / len(results)
    avg_iss      = sum(r["iss"] for r in results) / len(results)
    avg_tom      = sum(r["tom"] for r in results) / len(results)
    score = 0.40 * composite_gc + 0.35 * avg_iss + 0.25 * avg_tom

    print(f"\nSeed {seed} ({'conceal' if conceal else 'default'}) | "
          f"Score: {score:.1%} | GC: {composite_gc:.1%} | ISS: {avg_iss:.1%} | ToM: {avg_tom:.1%}")
    for r in results:
        print(f"  {r['scenario']}: GC={r['gc']:.1%} ISS={r['iss']:.1%} ToM={r['tom']:.1%}")
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="hunter-alpha")
    parser.add_argument("--scenario", default=None, help="Single scenario to test (skips others)")
    args = parser.parse_args()

    llm = LocalLLM(model=args.model)
    scenarios = [args.scenario] if args.scenario else None
    barterbench(llm, seed=args.seed, scenarios=scenarios)
