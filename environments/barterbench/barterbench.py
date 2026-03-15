"""BarterBench verifiers environment.

One rollout = one full marketplace match on a scenario.
The model plays a single agent; a RandomAgent fills the remaining slots.
Reward = agent's goal completion at match end (0.0–1.0).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import verifiers as vf

# Resolve scenario files — bundled with package or fall back to repo root during development
_HERE = Path(__file__).parent
_BUNDLED = _HERE / "scenarios"
_REPO_ROOT = _HERE.parent.parent
SCENARIOS_DIR = _BUNDLED if _BUNDLED.exists() else _REPO_ROOT / "scenarios"


SYSTEM_PROMPT = """You are a competitive trader in a multi-agent marketplace.
Your goal is to acquire the target items in your inventory through strategic trading.
Items are scarce — not everyone can fully meet their targets.

On each turn you will receive the current marketplace state and must output a JSON action.
Think briefly, then output exactly one JSON action in the format shown.

Valid actions:
{"action": "post_offer",     "give": {"item": qty}, "want": {"item": qty}, "message": "..."}
{"action": "private_offer",  "give": {"item": qty}, "want": {"item": qty}, "target_agent": <int>, "message": "..."}
{"action": "accept_offer",   "offer_id": <int>, "message": "..."}
{"action": "pass_turn",      "message": "..."}

Output ONLY the JSON object. No preamble, no markdown fences."""


def _goal_completion(inventory: dict, target: dict) -> float:
    if not target:
        return 1.0
    parts = [min(inventory.get(item, 0) / qty, 1.0) for item, qty in target.items() if qty > 0]
    return sum(parts) / len(parts) if parts else 1.0


def _build_state_prompt(agent_idx: int, inventory: dict, target: dict,
                        order_book: list, recent_trades: list,
                        round_num: int, max_rounds: int) -> str:
    progress = {
        item: f"{inventory.get(item, 0)}/{qty} ({min(inventory.get(item, 0)/qty, 1.0)*100:.0f}%)"
        for item, qty in target.items() if qty > 0
    }
    offers = [
        f"  [#{o['offer_id']}] Agent {o['agent_idx']} offers {json.dumps(o['give'])} for {json.dumps(o['want'])}"
        + (f' — "{o.get("message","")}"' if o.get("message") else "")
        for o in order_book
    ] or ["  (none)"]

    trades = [
        f"  Round {t['round']}: Agent {t['seller']} → Agent {t['buyer']}: "
        f"{json.dumps(t.get('give', {}))} for {json.dumps(t.get('want', {}))}"
        for t in recent_trades[-6:]
    ] or ["  (none yet)"]

    return (
        f"Round {round_num + 1} of {max_rounds} | You are Agent {agent_idx}\n\n"
        f"Your inventory: {json.dumps(inventory)}\n"
        f"Your target:    {json.dumps(target)}\n"
        f"Goal progress:  {json.dumps(progress)}\n\n"
        f"Open offers:\n" + "\n".join(offers) + "\n\n"
        f"Recent trades:\n" + "\n".join(trades)
    )


def _parse_action(text: str) -> dict:
    """Extract a JSON action dict from model output."""
    import re
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            valid = {"post_offer", "private_offer", "accept_offer", "pass_turn"}
            if data.get("action") in valid:
                return data
        except json.JSONDecodeError:
            pass
    return {"action": "pass_turn", "message": "parse error"}


class BarterBenchEnv(vf.MultiTurnEnv):
    """Single-agent BarterBench environment.

    Each rollout is one complete marketplace match. The model controls agent 0;
    all other agents are RandomAgents. Reward = goal completion (0–1).
    """

    def __init__(self, scenario_name: str = "spice_wars", **kwargs):
        self.scenario_name = scenario_name
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        from engine import MarketEngine
        from agent import RandomAgent
        from scoring import compute_metrics

        # Load scenario
        scenario_path = SCENARIOS_DIR / f"{self.scenario_name}.json"
        with open(scenario_path) as f:
            scenario = json.load(f)

        n_agents = len(scenario["agents"])

        # Agent 0 = the model under test; rest = random
        random_agents = [RandomAgent(model_name="random", agent_idx=i) for i in range(1, n_agents)]

        engine = MarketEngine(scenario)
        state["engine"] = engine
        state["scenario"] = scenario
        state["agent_idx"] = 0
        state["random_agents"] = random_agents
        state["round_num"] = 0
        state["max_rounds"] = scenario.get("rounds", 10)
        state["done"] = False
        state["compute_metrics"] = compute_metrics

        return await super().setup_state(state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        engine = state["engine"]
        agent_idx = state["agent_idx"]
        random_agents = state["random_agents"]
        round_num = state["round_num"]
        max_rounds = state["max_rounds"]
        scenario = state["scenario"]

        # Parse model's action from last message
        model_text = messages[-1]["content"] if messages else ""
        action = _parse_action(model_text)

        # Execute model action
        agent_inventory = engine.get_inventory(agent_idx)
        agent_target = scenario["agents"][agent_idx].get("target", {})
        engine.execute_action(agent_idx, action)

        # Execute random agents
        for ra in random_agents:
            obs = engine.get_observation(ra.agent_idx)
            ra_action = ra.act(obs)
            engine.execute_action(ra.agent_idx, ra_action)

        engine.end_round()
        state["round_num"] += 1

        # Check if done
        all_done = engine.check_all_goals_met()
        if state["round_num"] >= max_rounds or all_done:
            state["done"] = True
            gc = _goal_completion(engine.get_inventory(agent_idx), agent_target)
            state["goal_completion"] = gc
            final_msg = (
                f"Match complete after {state['round_num']} rounds.\n"
                f"Final inventory: {json.dumps(engine.get_inventory(agent_idx))}\n"
                f"Target:          {json.dumps(agent_target)}\n"
                f"Goal completion: {gc*100:.1f}%"
            )
            state["final_env_response"] = [{"role": "user", "content": final_msg}]
            return [{"role": "user", "content": final_msg}]

        # Build next-round observation
        obs_text = _build_state_prompt(
            agent_idx=agent_idx,
            inventory=engine.get_inventory(agent_idx),
            target=agent_target,
            order_book=engine.get_order_book(),
            recent_trades=engine.get_recent_trades(n=6),
            round_num=state["round_num"],
            max_rounds=max_rounds,
        )
        return [{"role": "user", "content": obs_text}]

    @vf.stop
    async def match_over(self, state: vf.State) -> bool:
        return state.get("done", False)


def load_environment(
    scenario: str = "spice_wars",
    num_examples: int = 50,
    seed: int = 42,
) -> vf.Environment:
    """Load the BarterBench environment.

    Args:
        scenario: Scenario name to use (default: spice_wars).
        num_examples: Number of rollout instances in the dataset.
        seed: Random seed for reproducibility.

    Required environment variables:
        None — BarterBench uses local scenarios and RandomAgent opponents.
        Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY to use paid judge models.
    """
    from datasets import Dataset

    # Load scenario to build initial prompt
    scenario_path = SCENARIOS_DIR / f"{scenario}.json"
    with open(scenario_path) as f:
        sc = json.load(f)

    agent_target = sc["agents"][0].get("target", {})
    initial_obs = _build_state_prompt(
        agent_idx=0,
        inventory=sc["agents"][0].get("inventory", {}),
        target=agent_target,
        order_book=[],
        recent_trades=[],
        round_num=0,
        max_rounds=sc.get("rounds", 10),
    )

    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": initial_obs}],
            "info": json.dumps({"scenario": scenario, "target": agent_target, "run_id": i}),
        }
        for i in range(num_examples)
    ])

    async def goal_completion_reward(completion, state) -> float:
        return state.get("goal_completion", 0.0)

    async def trade_count_metric(state) -> float:
        engine = state.get("engine")
        if engine is None:
            return 0.0
        return float(len(engine.get_all_trades()))

    rubric = vf.Rubric(funcs=[goal_completion_reward])
    rubric.add_metric(trade_count_metric)

    env = BarterBenchEnv(
        scenario_name=scenario,
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=sc.get("rounds", 10) + 2,
    )
    return env
