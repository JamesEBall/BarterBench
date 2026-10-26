"""Engine invariants: determinism, visibility, escrow, communication, conservation, checkpoints."""

import json
import sys
import os
import tempfile
from pathlib import Path

import pytest

from barterbench.engine import MarketEngine
from barterbench.agents import RandomAgent, HeuristicAgent, build_agent


class ScriptAgent:
    """Replays a list of actions; records every observation it was given."""

    def __init__(self, agent_idx, actions=None, name="script"):
        self.agent_idx = agent_idx
        self.model_name = name
        self.strategy_id = None
        self.actions = list(actions or [])
        self.observations = []
        self.turn_latencies = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def contestant_name(self):
        return self.model_name

    def reliability(self):
        return {}

    def get_state(self):
        return {}

    def set_state(self, state):
        pass

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds,
                  auctions=None, messages=None, **kw):
        self.observations.append({"round": round_num, "inventory": dict(inventory), "order_book": order_book,
                                  "recent_trades": recent_trades, "messages": messages})
        if self.actions:
            return self.actions.pop(0)
        return {"action": "pass_turn", "message": ""}


def scenario(n=4, rounds=3, **extra):
    agents = [{"role": f"Trader {i}", "inventory": {"gold": 3, "wheat": 3}, "target": {"gold": 1, "wheat": 1}}
              for i in range(n)]
    return {"name": "t", "max_rounds": rounds, "agents": agents, **extra}


# ---- Determinism ----

def test_same_seed_same_turn_orders_and_outcome():
    def play(seed):
        sc = scenario(n=6, rounds=6)
        sc["agents"] = [{"role": f"T{i}", "inventory": {"a": 4, "b": 0}, "target": {"b": 2}} if i % 2 == 0
                        else {"role": f"T{i}", "inventory": {"a": 0, "b": 4}, "target": {"a": 2}} for i in range(6)]
        agents = [RandomAgent(i, seed=seed + i, num_agents=6) for i in range(6)]
        eng = MarketEngine(sc, agents, seed=seed)
        res = eng.run()
        return res["turn_orders"], res["trades"], [a["goal_completion"] for a in res["agent_results"]]

    assert play(123) == play(123)
    assert play(123) != play(124)


def test_global_random_module_not_consumed():
    import random
    random.seed(0)
    before = random.random()
    random.seed(0)
    eng = MarketEngine(scenario(), [ScriptAgent(i) for i in range(4)], seed=7)
    eng.run()
    assert random.random() == before, "engine must not touch the global random module"


# ---- Private offers ----

def test_private_offer_only_acceptable_by_recipient():
    acts0 = [{"action": "private_offer", "give": {"gold": 1}, "want": {"wheat": 1}, "target_agent": 2, "message": "psst"}]
    acts1 = [{"action": "pass_turn", "message": ""},
             {"action": "accept_offer", "offer_id": 1, "message": "sniped"}]
    acts2 = [{"action": "pass_turn", "message": ""},
             {"action": "pass_turn", "message": ""},
             {"action": "accept_offer", "offer_id": 1, "message": "ok"}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1, acts1), ScriptAgent(2, acts2)]
    eng = MarketEngine(scenario(n=3, rounds=3), agents, seed=1)
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1, 2], init, 0)
    eng._run_sequential_round(1, [1, 0, 2], init, 0)   # agent 1 tries to snipe
    assert len(eng.trades) == 0
    snipe = [h for h in eng.history if h["agent"] == 1 and h["action"] == "accept_offer"][0]
    assert snipe["invalid"] is True
    eng._run_sequential_round(2, [2, 0, 1], init, 0)   # recipient accepts
    assert len(eng.trades) == 1 and eng.trades[0]["accepter"] == 2 and eng.trades[0]["private"] is True


def test_private_offer_invisible_to_third_party():
    acts0 = [{"action": "private_offer", "give": {"gold": 1}, "want": {"wheat": 1}, "target_agent": 2, "message": "secret"}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1), ScriptAgent(2)]
    eng = MarketEngine(scenario(n=3, rounds=1), agents, seed=1)
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1, 2], init, 0)
    assert agents[1].observations[0]["order_book"] == []
    assert len(agents[2].observations[0]["order_book"]) == 1
    assert agents[2].observations[0]["order_book"][0]["message"] == "secret"


# ---- Communication ----

def test_messages_are_delivered_in_full_mode():
    acts0 = [{"action": "post_offer", "give": {"gold": 1}, "want": {"wheat": 1}, "message": "selling gold cheap"},
             {"action": "pass_turn", "message": "anyone need wheat?"}]
    acts1 = [{"action": "pass_turn", "message": ""}, {"action": "accept_offer", "offer_id": 1, "message": "deal"}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1, acts1), ScriptAgent(2)]
    eng = MarketEngine(scenario(n=3, rounds=2), agents, seed=1, communication="full")
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1, 2], init, 0)
    # Agent 2 sees the offer with its message
    assert agents[2].observations[0]["order_book"][0]["message"] == "selling gold cheap"
    eng._run_sequential_round(1, [1, 0, 2], init, 0)
    obs2 = agents[2].observations[1]
    assert obs2["recent_trades"][0]["message"] == "deal"
    kinds = {(m["agent"], m["action"]) for m in obs2["messages"]}
    assert (1, "accept_offer") in kinds and (0, "pass_turn") in kinds


def test_offers_only_mode_strips_all_text():
    acts0 = [{"action": "post_offer", "give": {"gold": 1}, "want": {"wheat": 1}, "message": "selling gold cheap"},
             {"action": "pass_turn", "message": "hello"}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1)]
    eng = MarketEngine(scenario(n=2, rounds=2), agents, seed=1, communication="offers_only")
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1], init, 0)
    eng._run_sequential_round(1, [0, 1], init, 0)
    for obs in agents[1].observations:
        assert all("message" not in o for o in obs["order_book"])
        assert obs["messages"] == []
    # History still records what was said, for analysis
    assert eng.history[0]["message"] == "selling gold cheap"


def test_invalid_communication_mode_rejected():
    with pytest.raises(ValueError):
        MarketEngine(scenario(), [ScriptAgent(i) for i in range(4)], communication="loud")


# ---- Escrow ----

def test_bid_items_are_escrowed():
    sc = scenario(n=2, rounds=3, auction_enabled=True)
    acts0 = [{"action": "start_auction", "give": {"gold": 1}, "message": ""}]
    acts1 = [{"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 3}, "message": ""},
             {"action": "post_offer", "give": {"wheat": 1}, "want": {"gold": 1}, "message": ""}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1, acts1)]
    eng = MarketEngine(sc, agents, seed=1)
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1], init, 0)
    assert eng.reserved[1] == {"wheat": 3}
    eng._run_sequential_round(1, [1, 0], init, 0)
    post = [h for h in eng.history if h["action"] == "post_offer"][0]
    assert post["invalid"] is True, "escrowed wheat must not be offerable"
    assert eng.inventories[1]["wheat"] == 3  # nothing moved


def test_escrow_released_on_cancel_and_expiry():
    sc = scenario(n=2, rounds=2, auction_enabled=True)
    acts0 = [{"action": "start_auction", "give": {"gold": 1}, "message": ""},
             {"action": "close_auction", "auction_id": 1, "reject_all": True, "message": ""}]
    acts1 = [{"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 2}, "message": ""}]
    agents = [ScriptAgent(0, acts0), ScriptAgent(1, acts1)]
    eng = MarketEngine(sc, agents, seed=1)
    res = eng.run()
    assert all(not r for r in eng.reserved)
    assert res["num_trades"] == 0

    # Expiry path
    acts0 = [{"action": "start_auction", "give": {"gold": 1}, "message": ""}]
    acts1 = [{"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 2}, "message": ""}]
    eng = MarketEngine(scenario(n=2, rounds=2, auction_enabled=True),
                       [ScriptAgent(0, acts0), ScriptAgent(1, acts1)], seed=1)
    eng.run()
    assert eng.active_auctions[0]["status"] == "expired"
    assert all(not r for r in eng.reserved)


def test_rebid_replaces_previous_escrow():
    sc = scenario(n=2, rounds=3, auction_enabled=True)
    acts0 = [{"action": "start_auction", "give": {"gold": 1}, "message": ""},
             {"action": "pass_turn", "message": ""},
             {"action": "close_auction", "auction_id": 1, "accepted_bid_idx": 0, "message": ""}]
    acts1 = [{"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 1}, "message": ""},
             {"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 2}, "message": ""}]
    eng = MarketEngine(sc, [ScriptAgent(0, acts0), ScriptAgent(1, acts1)], seed=1)
    init = [dict(i) for i in eng.inventories]
    eng._run_sequential_round(0, [0, 1], init, 0)
    eng._run_sequential_round(1, [1, 0], init, 0)
    assert eng.reserved[1] == {"wheat": 2}
    eng._run_sequential_round(2, [0, 1], init, 0)
    assert len(eng.trades) == 1 and eng.trades[0]["want"] == {"wheat": 2}
    assert all(not r for r in eng.reserved)


# ---- Conservation & bookkeeping ----

def test_item_conservation_and_turn_positions():
    sc = scenario(n=6, rounds=5)
    # Make goals unmet at the start so the match runs all rounds
    for i, a in enumerate(sc["agents"]):
        a["inventory"] = {"gold": 4, "wheat": 0} if i % 2 == 0 else {"gold": 0, "wheat": 4}
        a["target"] = {"wheat": 3} if i % 2 == 0 else {"gold": 3}
    agents = [RandomAgent(i, seed=i, num_agents=6) if i % 2 else HeuristicAgent(i) for i in range(6)]
    eng = MarketEngine(sc, agents, seed=3)
    before = eng.total_supply()
    res = eng.run()
    assert eng.total_supply() == before
    positions = {(h["round"], h["turn_position"]) for h in res["history"]}
    assert len(positions) == len(res["history"])  # every (round, position) unique
    assert all(0 <= h["turn_position"] < 6 for h in res["history"])
    assert len(res["turn_orders"]) == 5


def test_scripted_agents_send_no_messages():
    sc = scenario(n=4, rounds=3)
    agents = [RandomAgent(0, seed=1, num_agents=4), RandomAgent(1, seed=2, num_agents=4),
              HeuristicAgent(2), HeuristicAgent(3)]
    eng = MarketEngine(sc, agents, seed=3)
    res = eng.run()
    assert all(h["message"] == "" for h in res["history"])


def test_build_agent_factory_for_baselines():
    a = build_agent("random", 0, {}, False, seed=1, num_agents=4)
    b = build_agent("heuristic", 1, {}, False, seed=1, num_agents=4)
    assert a.model_name == "random" and b.model_name == "heuristic"


# ---- Checkpoint ----

def test_checkpoint_restores_rng_and_resumes_identically():
    def make(seed, cp_dir):
        sc = scenario(n=4, rounds=6)
        sc["agents"] = [{"role": f"T{i}", "inventory": {"a": 4, "b": 0}, "target": {"b": 2}} if i % 2 == 0
                        else {"role": f"T{i}", "inventory": {"a": 0, "b": 4}, "target": {"a": 2}} for i in range(4)]
        agents = [RandomAgent(i, seed=seed + i, num_agents=4) for i in range(4)]
        return MarketEngine(sc, agents, seed=seed, run_id=99, checkpoint_dir=cp_dir)

    with tempfile.TemporaryDirectory() as d:
        full = make(5, d).run()
        # Run 3 rounds, checkpoint, then resume from disk
        eng = make(5, d)
        init = [dict(i) for i in eng.inventories]
        for r in range(3):
            order = list(range(4)); eng.rng.shuffle(order); eng.turn_orders.append(order)
            eng._run_sequential_round(r, order, init, 0)
            eng._write_checkpoint(r, init, 0)
        cp = Path(d) / "checkpoint_99.json"
        assert cp.exists()
        resumed, start_round, init2, _ = MarketEngine.from_checkpoint(
            str(cp), agent_factory=lambda name, i, sa, auc: RandomAgent(i, num_agents=4))
        assert start_round == 3
        res = resumed.run(start_round=start_round, initial_inventories=init2)
        assert res["turn_orders"] == full["turn_orders"]
        assert res["trades"] == full["trades"]
