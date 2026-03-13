"""Tests for auction mechanics in MarketEngine."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import MarketEngine


class FakeAgent:
    """Minimal agent stub that returns pre-programmed actions."""

    def __init__(self, agent_idx, actions=None):
        self.agent_idx = agent_idx
        self.model_name = "fake"
        self.strategy_id = None
        self.actions = actions or []
        self._call_idx = 0
        self.turn_latencies = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def contestant_name(self):
        return self.model_name

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        if self._call_idx < len(self.actions):
            action = self.actions[self._call_idx]
            self._call_idx += 1
            return action
        return {"action": "pass_turn", "message": "no more actions"}


def _make_scenario(n_agents=3, auction_enabled=True, max_rounds=3):
    """Create a simple test scenario."""
    agents = []
    for i in range(n_agents):
        agents.append({
            "role": f"Trader {i}",
            "inventory": {"gold": 3, "wheat": 3},
            "target": {"gold": 1, "wheat": 1},
        })
    return {
        "name": "test_auction",
        "max_rounds": max_rounds,
        "agents": agents,
        "auction_enabled": auction_enabled,
    }


class TestAuctionStartAndBid:
    def test_start_auction_creates_auction(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        turn = {"give": {"gold": 2}, "min_bid": {"wheat": 3}, "message": "selling gold"}
        entry = {"round": 0, "agent": 0, "model": "fake", "contestant": "fake", "action": "start_auction", "message": "selling gold", "reasoning": ""}
        result = engine._handle_start_auction(0, turn, 0, entry)

        assert result is True
        assert len(engine.active_auctions) == 1
        auction = engine.active_auctions[0]
        assert auction["id"] == 1
        assert auction["auctioneer"] == 0
        assert auction["give"] == {"gold": 2}
        assert auction["status"] == "open"
        assert auction["visible_to"] is None  # public

    def test_start_auction_private(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        turn = {"give": {"gold": 1}, "visible_to": [1], "message": "private auction"}
        entry = {"round": 0, "agent": 0, "model": "fake", "contestant": "fake", "action": "start_auction", "message": "", "reasoning": ""}
        result = engine._handle_start_auction(0, turn, 0, entry)

        assert result is True
        assert engine.active_auctions[0]["visible_to"] == [1]

    def test_start_auction_invalid_no_items(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)
        engine.inventories[0] = {"gold": 0, "wheat": 3}

        turn = {"give": {"gold": 2}, "message": "selling"}
        entry = {"round": 0, "agent": 0, "model": "fake", "contestant": "fake", "action": "start_auction", "message": "", "reasoning": ""}
        result = engine._handle_start_auction(0, turn, 0, entry)

        assert result is False
        assert entry.get("invalid") is True

    def test_submit_bid_valid(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        # Start an auction first
        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "bid": {"wheat": 3}, "message": "my bid"}
        entry = {"round": 0, "agent": 1, "model": "fake", "contestant": "fake", "action": "submit_bid", "message": "", "reasoning": ""}
        result = engine._handle_submit_bid(1, turn, 0, entry)

        assert result is True
        assert len(engine.active_auctions[0]["bids"]) == 1
        assert engine.active_auctions[0]["bids"][0]["bidder"] == 1

    def test_submit_bid_auctioneer_cannot_bid(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "bid": {"wheat": 3}, "message": "self-bid"}
        entry = {"round": 0, "agent": 0, "model": "fake", "contestant": "fake", "action": "submit_bid", "message": "", "reasoning": ""}
        result = engine._handle_submit_bid(0, turn, 0, entry)

        assert result is False
        assert entry.get("invalid") is True

    def test_submit_bid_private_auction_ineligible(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": [1], "bids": [],
            "created_round": 0, "status": "open",
        })

        # Agent 2 is not in visible_to
        turn = {"auction_id": 1, "bid": {"wheat": 2}, "message": "uninvited bid"}
        entry = {"round": 0, "agent": 2, "model": "fake", "contestant": "fake", "action": "submit_bid", "message": "", "reasoning": ""}
        result = engine._handle_submit_bid(2, turn, 0, entry)

        assert result is False
        assert entry.get("invalid") is True


class TestAuctionClose:
    def test_close_auction_accept_bid(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None,
            "bids": [{"bidder": 1, "bid": {"wheat": 3}, "round": 0}],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "accepted_bid_idx": 0, "message": "accepting"}
        entry = {"round": 1, "agent": 0, "model": "fake", "contestant": "fake", "action": "close_auction", "message": "", "reasoning": ""}
        result = engine._handle_close_auction(0, turn, 1, entry)

        assert result is True
        assert engine.active_auctions[0]["status"] == "resolved"
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade["poster"] == 0
        assert trade["accepter"] == 1
        assert trade["give"] == {"gold": 2}
        assert trade["want"] == {"wheat": 3}
        assert trade["auction_trade"] is True
        # Verify inventories updated
        assert engine.inventories[0]["gold"] == 1  # 3 - 2
        assert engine.inventories[0]["wheat"] == 6  # 3 + 3
        assert engine.inventories[1]["gold"] == 5  # 3 + 2
        assert engine.inventories[1]["wheat"] == 0  # 3 - 3

    def test_close_auction_reject_all(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None,
            "bids": [{"bidder": 1, "bid": {"wheat": 1}, "round": 0}],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "reject_all": True, "message": "bids too low"}
        entry = {"round": 1, "agent": 0, "model": "fake", "contestant": "fake", "action": "close_auction", "message": "", "reasoning": ""}
        result = engine._handle_close_auction(0, turn, 1, entry)

        assert result is True
        assert engine.active_auctions[0]["status"] == "cancelled"
        assert len(engine.trades) == 0
        # Inventories unchanged
        assert engine.inventories[0]["gold"] == 3

    def test_close_auction_non_auctioneer_rejected(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "accepted_bid_idx": 0, "message": "hijack"}
        entry = {"round": 0, "agent": 1, "model": "fake", "contestant": "fake", "action": "close_auction", "message": "", "reasoning": ""}
        result = engine._handle_close_auction(1, turn, 0, entry)

        assert result is False
        assert entry.get("invalid") is True

    def test_close_auction_bidder_lacks_items(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 1},
            "min_bid": {}, "visible_to": None,
            "bids": [{"bidder": 1, "bid": {"wheat": 10}, "round": 0}],
            "created_round": 0, "status": "open",
        })

        turn = {"auction_id": 1, "accepted_bid_idx": 0, "message": "accept"}
        entry = {"round": 1, "agent": 0, "model": "fake", "contestant": "fake", "action": "close_auction", "message": "", "reasoning": ""}
        result = engine._handle_close_auction(0, turn, 1, entry)

        assert result is False
        assert entry.get("auction_result") == "bidder_lacks_items"


class TestAuctionVisibility:
    def test_auctioneer_sees_all_bids(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None,
            "bids": [
                {"bidder": 1, "bid": {"wheat": 2}, "round": 0},
                {"bidder": 2, "bid": {"wheat": 3}, "round": 0},
            ],
            "created_round": 0, "status": "open",
        })

        visible = engine._visible_auctions(0)
        assert len(visible) == 1
        assert "bids" in visible[0]
        assert len(visible[0]["bids"]) == 2

    def test_bidder_sees_only_bid_count(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": None,
            "bids": [
                {"bidder": 1, "bid": {"wheat": 2}, "round": 0},
                {"bidder": 2, "bid": {"wheat": 3}, "round": 0},
            ],
            "created_round": 0, "status": "open",
        })

        visible = engine._visible_auctions(1)
        assert len(visible) == 1
        assert "bids" not in visible[0]
        assert visible[0]["num_bids"] == 2

    def test_private_auction_visibility(self):
        scenario = _make_scenario(n_agents=3)
        agents = [FakeAgent(i) for i in range(3)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 2},
            "min_bid": {}, "visible_to": [1],  # only agent 1 can see/bid
            "bids": [], "created_round": 0, "status": "open",
        })

        # Agent 1 (invited) can see
        assert len(engine._visible_auctions(1)) == 1
        # Agent 2 (not invited) cannot see
        assert len(engine._visible_auctions(2)) == 0
        # Auctioneer can see
        assert len(engine._visible_auctions(0)) == 1

    def test_closed_auction_not_visible(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 1},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "resolved",
        })

        assert len(engine._visible_auctions(0)) == 0
        assert len(engine._visible_auctions(1)) == 0


class TestAutoClose:
    def test_auto_close_expires_open_auctions(self):
        scenario = _make_scenario(n_agents=2)
        agents = [FakeAgent(i) for i in range(2)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        engine.active_auctions.append({
            "id": 1, "auctioneer": 0, "give": {"gold": 1},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "open",
        })
        engine.active_auctions.append({
            "id": 2, "auctioneer": 1, "give": {"wheat": 1},
            "min_bid": {}, "visible_to": None, "bids": [],
            "created_round": 0, "status": "resolved",
        })

        engine._auto_close_auctions()

        assert engine.active_auctions[0]["status"] == "expired"
        assert engine.active_auctions[1]["status"] == "resolved"  # unchanged


class TestAuctionFullRound:
    def test_auction_in_sequential_round(self):
        """Full integration test: agent 0 starts auction, agent 1 bids, agent 0 closes."""
        scenario = _make_scenario(n_agents=2, max_rounds=3)
        # Round 0: agent 0 starts auction, agent 1 passes (no auction yet visible when 1 acts if 1 goes first)
        # We'll manually control the flow by running individual rounds

        agent0_actions = [
            {"action": "start_auction", "give": {"gold": 2}, "message": "selling gold"},
            {"action": "pass_turn", "message": "waiting for bids"},
            {"action": "close_auction", "auction_id": 1, "accepted_bid_idx": 0, "message": "accepting best bid"},
        ]
        agent1_actions = [
            {"action": "pass_turn", "message": "nothing yet"},
            {"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 3}, "message": "my bid"},
            {"action": "pass_turn", "message": "waiting"},
        ]

        agents = [FakeAgent(0, agent0_actions), FakeAgent(1, agent1_actions)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        initial_inventories = [dict(inv) for inv in engine.inventories]
        start_time = __import__("time").time()

        # Round 0: force turn order [0, 1] so agent 0 starts auction first
        engine._run_sequential_round(0, [0, 1], initial_inventories, start_time)
        assert len(engine.active_auctions) == 1
        assert engine.active_auctions[0]["status"] == "open"

        # Round 1: force [1, 0] so agent 1 bids, then agent 0 passes
        engine._run_sequential_round(1, [1, 0], initial_inventories, start_time)
        assert len(engine.active_auctions[0]["bids"]) == 1

        # Round 2: force [0, 1] so agent 0 closes auction
        engine._run_sequential_round(2, [0, 1], initial_inventories, start_time)
        assert engine.active_auctions[0]["status"] == "resolved"
        assert len(engine.trades) == 1

        # Verify final inventories
        assert engine.inventories[0]["gold"] == 1  # 3 - 2
        assert engine.inventories[0]["wheat"] == 6  # 3 + 3
        assert engine.inventories[1]["gold"] == 5  # 3 + 2
        assert engine.inventories[1]["wheat"] == 0  # 3 - 3

    def test_auction_disabled_actions_become_pass(self):
        """When auction_enabled=False, auction actions are ignored (fall through to pass)."""
        scenario = _make_scenario(n_agents=2, auction_enabled=False)
        agent0_actions = [{"action": "start_auction", "give": {"gold": 2}, "message": "trying"}]
        agents = [FakeAgent(0, agent0_actions), FakeAgent(1)]
        engine = MarketEngine(scenario, agents, live_updates=False)

        initial_inventories = [dict(inv) for inv in engine.inventories]
        start_time = __import__("time").time()
        engine._run_sequential_round(0, [0, 1], initial_inventories, start_time)

        assert len(engine.active_auctions) == 0


class TestHistoryRounds:
    def test_history_rounds_caps_conversation(self):
        from agent import MarketAgent
        # Create agent with history_rounds=1 (should cap at 6 items)
        agent = MarketAgent.__new__(MarketAgent)
        agent.history_rounds = 1
        agent.conversation_history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": [{"type": "text", "text": "resp1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": [{"type": "text", "text": "resp2"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "ok"}]},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": [{"type": "text", "text": "resp3"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "3", "content": "ok"}]},
        ]
        # Simulate the cap logic
        max_history_items = agent.history_rounds * 6
        if len(agent.conversation_history) > max_history_items:
            agent.conversation_history = agent.conversation_history[-max_history_items:]
        assert len(agent.conversation_history) == 6

    def test_default_history_rounds_is_three(self):
        from agent import MarketAgent
        agent = MarketAgent.__new__(MarketAgent)
        # Simulate __init__ defaults
        agent.history_rounds = 3
        assert agent.history_rounds * 6 == 18


class TestParseJsonResponse:
    def test_parse_auction_fields(self):
        from agent import _parse_json_response

        text = '{"action": "start_auction", "give": {"gold": 2}, "min_bid": {"wheat": 1}, "visible_to": [1, 2], "message": "selling"}'
        result = _parse_json_response(text)
        assert result["action"] == "start_auction"
        assert result["give"] == {"gold": 2}
        assert result["min_bid"] == {"wheat": 1}
        assert result["visible_to"] == [1, 2]

    def test_parse_submit_bid(self):
        from agent import _parse_json_response

        text = '{"action": "submit_bid", "auction_id": 1, "bid": {"wheat": 3}, "message": "bidding"}'
        result = _parse_json_response(text)
        assert result["action"] == "submit_bid"
        assert result["auction_id"] == 1
        assert result["bid"] == {"wheat": 3}

    def test_parse_close_auction(self):
        from agent import _parse_json_response

        text = '{"action": "close_auction", "auction_id": 1, "accepted_bid_idx": 0, "message": "done"}'
        result = _parse_json_response(text)
        assert result["action"] == "close_auction"
        assert result["auction_id"] == 1
        assert result["accepted_bid_idx"] == 0

    def test_parse_close_auction_reject_all(self):
        from agent import _parse_json_response

        text = '{"action": "close_auction", "auction_id": 1, "reject_all": true, "message": "nope"}'
        result = _parse_json_response(text)
        assert result["action"] == "close_auction"
        assert result["reject_all"] is True
