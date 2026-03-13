"""Tests for RandomAgent baseline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import RandomAgent, VALID_ACTIONS


def test_random_agent_creation():
    """RandomAgent creates with correct defaults."""
    agent = RandomAgent(agent_idx=0, seed=42)
    assert agent.model_name == "random"
    assert agent.agent_idx == 0
    assert agent.contestant_name == "random"
    assert agent.total_input_tokens == 0
    assert agent.total_output_tokens == 0
    assert agent.backend == "random"


def test_random_agent_valid_actions():
    """RandomAgent always produces valid action types."""
    agent = RandomAgent(agent_idx=0, seed=42)
    inventory = {"wheat": 5, "gold": 3}
    target = {"iron": 2, "gold": 1}
    order_book = [
        {"id": 1, "poster": 1, "give": {"iron": 1}, "want": {"wheat": 2}},
    ]

    for _ in range(50):
        result = agent.take_turn(inventory, target, order_book, [], 0, 10)
        assert result["action"] in VALID_ACTIONS
        assert "message" in result


def test_random_agent_never_offers_more_than_has():
    """RandomAgent respects inventory constraints."""
    agent = RandomAgent(agent_idx=0, seed=42)
    inventory = {"wheat": 2}
    target = {"gold": 3}
    order_book = []

    for _ in range(100):
        result = agent.take_turn(inventory, target, order_book, [], 0, 10)
        if result["action"] in ("post_offer", "private_offer"):
            for item, qty in result.get("give", {}).items():
                assert qty <= inventory.get(item, 0), \
                    f"Offered {qty} {item} but only has {inventory.get(item, 0)}"


def test_random_agent_accepts_affordable_offers():
    """When RandomAgent accepts, it only accepts offers it can afford."""
    agent = RandomAgent(agent_idx=0, seed=42)
    inventory = {"wheat": 5}
    target = {"gold": 2}
    order_book = [
        {"id": 1, "poster": 1, "give": {"gold": 1}, "want": {"wheat": 2}},
        {"id": 2, "poster": 2, "give": {"iron": 1}, "want": {"diamonds": 10}},  # Can't afford
    ]

    accepted_ids = set()
    for _ in range(200):
        result = agent.take_turn(inventory, target, order_book, [], 0, 10)
        if result["action"] == "accept_offer":
            accepted_ids.add(result["offer_id"])

    # Should only accept offer #1 (can afford wheat), never #2 (no diamonds)
    assert 2 not in accepted_ids, "RandomAgent accepted an unaffordable offer"


def test_random_agent_doesnt_accept_own_offers():
    """RandomAgent should not accept offers it posted."""
    agent = RandomAgent(agent_idx=0, seed=42)
    inventory = {"wheat": 5}
    target = {"gold": 2}
    order_book = [
        {"id": 1, "poster": 0, "give": {"wheat": 1}, "want": {"gold": 1}},  # Own offer
    ]

    for _ in range(100):
        result = agent.take_turn(inventory, target, order_book, [], 0, 10)
        if result["action"] == "accept_offer":
            assert result["offer_id"] != 1, "RandomAgent accepted its own offer"


def test_random_agent_zero_tokens():
    """RandomAgent uses zero API tokens."""
    agent = RandomAgent(agent_idx=0, seed=42)
    for _ in range(10):
        agent.take_turn({"wheat": 5}, {"gold": 1}, [], [], 0, 10)
    assert agent.total_input_tokens == 0
    assert agent.total_output_tokens == 0


def test_random_agent_latency_tracking():
    """RandomAgent records latency for each turn."""
    agent = RandomAgent(agent_idx=0, seed=42)
    for _ in range(5):
        agent.take_turn({"wheat": 5}, {"gold": 1}, [], [], 0, 10)
    assert len(agent.turn_latencies) == 5
    # Random agent should be very fast (< 1ms)
    for lat in agent.turn_latencies:
        assert lat < 0.01


def test_random_agent_state_serialization():
    """get_state/set_state round-trips correctly."""
    agent = RandomAgent(agent_idx=0, seed=42)
    for _ in range(3):
        agent.take_turn({"wheat": 5}, {"gold": 1}, [], [], 0, 10)

    state = agent.get_state()
    assert "turn_latencies" in state
    assert "rng_state" in state

    # Create new agent and restore
    agent2 = RandomAgent(agent_idx=0)
    agent2.set_state(state)
    assert len(agent2.turn_latencies) == 3


def test_random_agent_reproducible():
    """Same seed produces same actions."""
    results1 = []
    results2 = []

    for seed in [42, 42]:
        agent = RandomAgent(agent_idx=0, seed=seed)
        results = []
        for r in range(5):
            result = agent.take_turn({"wheat": 5, "gold": 2}, {"iron": 3}, [], [], r, 10)
            results.append(result["action"])
        if not results1:
            results1 = results
        else:
            results2 = results

    assert results1 == results2
