"""Bartering engine: runs a multi-round negotiation session between two agents."""

import json
import time


class BarterEngine:
    def __init__(self, scenario, agents):
        self.scenario = scenario
        self.agents = agents
        self.max_rounds = scenario.get("max_rounds", 10)
        self.inventories = [dict(s["inventory"]) for s in scenario["agents"]]
        self.valuations = [s["valuations"] for s in scenario["agents"]]
        self.roles = [s["role"] for s in scenario["agents"]]
        self.history = []
        self.trades_executed = []
        self.pending_trade = None
        self.consecutive_ends = 0

    def _compute_value(self, agent_idx, inventory=None):
        inv = inventory or self.inventories[agent_idx]
        vals = self.valuations[agent_idx]
        return sum(qty * vals.get(item, 0) for item, qty in inv.items())

    def _is_valid_proposal(self, proposer, give, receive):
        responder = 1 - proposer
        for item, qty in give.items():
            if not isinstance(qty, (int, float)) or qty <= 0:
                return False
            if self.inventories[proposer].get(item, 0) < qty:
                return False
        for item, qty in receive.items():
            if not isinstance(qty, (int, float)) or qty <= 0:
                return False
            if self.inventories[responder].get(item, 0) < qty:
                return False
        if not give and not receive:
            return False
        return True

    def _execute_trade(self, trade):
        proposer = trade["proposer"]
        responder = 1 - proposer
        for item, qty in trade["give"].items():
            self.inventories[proposer][item] = self.inventories[proposer].get(item, 0) - int(qty)
            self.inventories[responder][item] = self.inventories[responder].get(item, 0) + int(qty)
        for item, qty in trade["receive"].items():
            self.inventories[responder][item] = self.inventories[responder].get(item, 0) - int(qty)
            self.inventories[proposer][item] = self.inventories[proposer].get(item, 0) + int(qty)
        self.trades_executed.append(trade)

    def run(self):
        initial_values = [self._compute_value(i) for i in range(2)]
        start_time = time.time()
        stop = False

        for round_num in range(self.max_rounds):
            if stop:
                break
            for agent_idx in range(2):
                # Determine what pending trade this agent can respond to
                visible_pending = None
                if self.pending_trade and self.pending_trade["proposer"] != agent_idx:
                    # Flip give/receive from the responder's perspective
                    visible_pending = {
                        "proposer_role": self.roles[self.pending_trade["proposer"]],
                        "give": self.pending_trade["receive"],  # what proposer wants = what responder gives
                        "receive": self.pending_trade["give"],  # what proposer gives = what responder receives
                    }

                turn = self.agents[agent_idx].take_turn(
                    role=self.roles[agent_idx],
                    inventory=self.inventories[agent_idx],
                    valuations=self.valuations[agent_idx],
                    history=self.history,
                    pending_trade=visible_pending,
                )

                entry = {
                    "round": round_num,
                    "agent": agent_idx,
                    "role": self.roles[agent_idx],
                    **turn,
                }

                action = turn["action"]

                if action == "propose_trade":
                    give = turn.get("give", {})
                    receive = turn.get("receive", {})
                    if self._is_valid_proposal(agent_idx, give, receive):
                        self.pending_trade = {
                            "proposer": agent_idx,
                            "give": give,
                            "receive": receive,
                        }
                    else:
                        entry["invalid"] = True
                    self.consecutive_ends = 0

                elif action == "accept_trade":
                    if self.pending_trade and self.pending_trade["proposer"] != agent_idx:
                        self._execute_trade(self.pending_trade)
                        self.pending_trade = None
                    self.consecutive_ends = 0

                elif action == "reject_trade":
                    self.pending_trade = None
                    self.consecutive_ends = 0

                elif action == "end_negotiation":
                    self.consecutive_ends += 1
                    if self.consecutive_ends >= 2:
                        self.history.append(entry)
                        stop = True
                        break

                self.history.append(entry)

        elapsed = time.time() - start_time
        final_values = [self._compute_value(i) for i in range(2)]

        return {
            "scenario": self.scenario["name"],
            "initial_inventories": [dict(self.scenario["agents"][i]["inventory"]) for i in range(2)],
            "final_inventories": [dict(inv) for inv in self.inventories],
            "initial_values": initial_values,
            "final_values": final_values,
            "num_trades": len(self.trades_executed),
            "num_turns": len(self.history),
            "elapsed_seconds": round(elapsed, 2),
            "history": self.history,
        }
