"""Marketplace engine: runs N-agent trading with an order book."""

import random
import time


class MarketEngine:
    """N-agent marketplace with order book, round-robin turns, fixed round limit."""

    def __init__(self, scenario, agents):
        self.scenario = scenario
        self.agents = agents
        self.num_agents = len(agents)
        self.max_rounds = scenario.get("max_rounds", 10)
        self.inventories = [dict(a["inventory"]) for a in scenario["agents"]]
        self.targets = [dict(a.get("target", {})) for a in scenario["agents"]]
        self.order_book = []  # list of open offers
        self.next_offer_id = 1
        self.history = []  # all actions taken
        self.trades = []   # executed trades
        self.round_log = []  # per-round summary

    def _has_items(self, agent_idx, items):
        """Check if agent has all specified items."""
        for item, qty in items.items():
            if not isinstance(qty, (int, float)) or qty <= 0:
                return False
            if self.inventories[agent_idx].get(item, 0) < qty:
                return False
        return bool(items)

    def _execute_trade(self, poster_idx, accepter_idx, give, want):
        """Transfer items between two agents."""
        # poster gives `give` and receives `want`
        for item, qty in give.items():
            self.inventories[poster_idx][item] = self.inventories[poster_idx].get(item, 0) - int(qty)
            self.inventories[accepter_idx][item] = self.inventories[accepter_idx].get(item, 0) + int(qty)
        for item, qty in want.items():
            self.inventories[accepter_idx][item] = self.inventories[accepter_idx].get(item, 0) - int(qty)
            self.inventories[poster_idx][item] = self.inventories[poster_idx].get(item, 0) + int(qty)

    def _visible_order_book(self, agent_idx):
        """Return order book entries visible to this agent (all open offers)."""
        return [
            {"id": o["id"], "poster": o["poster"], "give": o["give"], "want": o["want"]}
            for o in self.order_book
        ]

    def _remove_stale_offers(self):
        """Remove offers where the poster no longer has the items."""
        self.order_book = [
            o for o in self.order_book
            if self._has_items(o["poster"], o["give"])
        ]

    def goal_completion(self, agent_idx):
        """Compute goal completion for an agent (0-1)."""
        target = self.targets[agent_idx]
        if not target:
            return 1.0
        inv = self.inventories[agent_idx]
        scores = []
        for item, needed in target.items():
            if needed > 0:
                have = inv.get(item, 0)
                scores.append(min(have / needed, 1.0))
        return sum(scores) / len(scores) if scores else 1.0

    def run(self):
        start_time = time.time()
        initial_inventories = [dict(inv) for inv in self.inventories]

        for round_num in range(self.max_rounds):
            # Randomize turn order each round
            turn_order = list(range(self.num_agents))
            random.shuffle(turn_order)

            round_actions = []
            for agent_idx in turn_order:
                self._remove_stale_offers()

                visible_book = self._visible_order_book(agent_idx)
                turn = self.agents[agent_idx].take_turn(
                    inventory=self.inventories[agent_idx],
                    target=self.targets[agent_idx],
                    order_book=visible_book,
                    recent_trades=self.trades[-10:],
                    round_num=round_num,
                    max_rounds=self.max_rounds,
                )

                action = turn["action"]
                entry = {
                    "round": round_num,
                    "agent": agent_idx,
                    "model": self.agents[agent_idx].model_name,
                    "action": action,
                    "message": turn.get("message", ""),
                }

                if action == "post_offer":
                    give = turn.get("give", {})
                    want = turn.get("want", {})
                    if self._has_items(agent_idx, give) and give and want:
                        offer = {
                            "id": self.next_offer_id,
                            "poster": agent_idx,
                            "give": give,
                            "want": want,
                        }
                        self.order_book.append(offer)
                        entry["offer_id"] = self.next_offer_id
                        entry["give"] = give
                        entry["want"] = want
                        self.next_offer_id += 1
                    else:
                        entry["invalid"] = True
                        entry["give"] = give
                        entry["want"] = want

                elif action == "accept_offer":
                    offer_id = turn.get("offer_id")
                    matched = None
                    for o in self.order_book:
                        if o["id"] == offer_id and o["poster"] != agent_idx:
                            matched = o
                            break

                    if matched and self._has_items(matched["poster"], matched["give"]) \
                            and self._has_items(agent_idx, matched["want"]):
                        self._execute_trade(matched["poster"], agent_idx, matched["give"], matched["want"])
                        trade_record = {
                            "round": round_num,
                            "offer_id": matched["id"],
                            "poster": matched["poster"],
                            "accepter": agent_idx,
                            "give": matched["give"],
                            "want": matched["want"],
                        }
                        self.trades.append(trade_record)
                        self.order_book = [o for o in self.order_book if o["id"] != offer_id]
                        entry["offer_id"] = offer_id
                        entry["trade"] = trade_record
                    else:
                        entry["invalid"] = True
                        entry["offer_id"] = offer_id

                elif action == "pass_turn":
                    pass  # nothing to do

                self.history.append(entry)
                round_actions.append(entry)

            # Check if all agents have reached their goals
            all_complete = all(self.goal_completion(i) >= 1.0 for i in range(self.num_agents))
            if all_complete:
                break

        elapsed = time.time() - start_time

        # Compute per-agent results
        agent_results = []
        for i in range(self.num_agents):
            agent_results.append({
                "agent_idx": i,
                "model": self.agents[i].model_name,
                "initial_inventory": initial_inventories[i],
                "final_inventory": dict(self.inventories[i]),
                "target": self.targets[i],
                "goal_completion": round(self.goal_completion(i), 4),
            })

        return {
            "scenario": self.scenario["name"],
            "scenario_data": self.scenario,
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "num_trades": len(self.trades),
            "num_turns": len(self.history),
            "elapsed_seconds": round(elapsed, 2),
            "agent_results": agent_results,
            "trades": self.trades,
            "history": self.history,
            "initial_inventories": initial_inventories,
        }
