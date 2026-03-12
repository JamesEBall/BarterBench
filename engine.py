"""Marketplace engine: runs N-agent trading with an order book."""

import copy
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class MarketEngine:
    """N-agent marketplace with order book, round-robin turns, fixed round limit."""

    def __init__(self, scenario, agents, simultaneous=False, run_id=None, live_updates=True):
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
        self.simultaneous = simultaneous
        self.run_id = run_id
        self.live_updates = live_updates

    def _has_items(self, agent_idx, items):
        """Check if agent has all specified items."""
        for item, qty in items.items():
            if not isinstance(qty, (int, float)) or qty <= 0:
                return False
            if self.inventories[agent_idx].get(item, 0) < qty:
                return False
        return bool(items)

    def _has_items_snapshot(self, inventory_dict, items):
        """Check if an inventory dict has all specified items (for simultaneous mode)."""
        for item, qty in items.items():
            if not isinstance(qty, (int, float)) or qty <= 0:
                return False
            if inventory_dict.get(item, 0) < qty:
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
        """Return order book entries visible to this agent.

        Public offers are visible to everyone. Private offers (whispers)
        are only visible to the poster and the target agent.
        """
        visible = []
        for o in self.order_book:
            if o.get("visible_to") is not None:
                # Private offer — only visible to poster and target
                if agent_idx != o["poster"] and agent_idx != o["visible_to"]:
                    continue
            entry = {"id": o["id"], "poster": o["poster"], "give": o["give"], "want": o["want"]}
            if o.get("visible_to") is not None:
                entry["private"] = True
                entry["visible_to"] = o["visible_to"]
            visible.append(entry)
        return visible

    def _remove_stale_offers(self):
        """Remove offers where the poster no longer has the items."""
        self.order_book = [
            o for o in self.order_book
            if self._has_items(o["poster"], o["give"])
        ]

    def _write_live(self, initial_inventories, start_time, status="running"):
        """Write current match state to live_<scenario>.json for dashboard streaming."""
        try:
            live_dir = Path(__file__).parent / "arena"
            live_dir.mkdir(parents=True, exist_ok=True)
            scenario_name = self.scenario.get("name", "match").replace(" ", "_")
            live_file = live_dir / f"live_{scenario_name}.json"
            agent_results = []
            for i in range(self.num_agents):
                ar = {
                    "agent_idx": i,
                    "model": self.agents[i].model_name,
                    "final_inventory": dict(self.inventories[i]),
                    "target": self.targets[i],
                    "goal_completion": round(self.goal_completion(i), 4),
                }
                if self.agents[i].strategy_id:
                    ar["strategy_id"] = self.agents[i].strategy_id
                agent_results.append(ar)

            live = {
                "status": status,
                "scenario": self.scenario.get("name", ""),
                "elapsed_seconds": round(time.time() - start_time, 1),
                "num_agents": self.num_agents,
                "max_rounds": self.max_rounds,
                "num_trades": len(self.trades),
                "model_assignments": [a.model_name for a in self.agents],
                "strategy_assignments": [a.contestant_name for a in self.agents],
                "initial_inventories": initial_inventories,
                "targets": [dict(t) for t in self.targets],
                "agent_results": agent_results,
                "trades": self.trades,
                "history": self.history,
            }
            with open(live_file, "w") as f:
                json.dump(live, f)
        except Exception:
            pass  # non-critical, don't break the match

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

    def _run_sequential_round(self, round_num, turn_order, initial_inventories, start_time):
        """Run one round of sequential (round-robin) play."""
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
                "contestant": self.agents[agent_idx].contestant_name,
                "action": action,
                "message": turn.get("message", ""),
                "reasoning": turn.get("reasoning", ""),
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

            elif action == "private_offer":
                give = turn.get("give", {})
                want = turn.get("want", {})
                target = turn.get("target_agent")
                valid_target = (isinstance(target, int) and 0 <= target < self.num_agents
                                and target != agent_idx)
                if self._has_items(agent_idx, give) and give and want and valid_target:
                    offer = {
                        "id": self.next_offer_id,
                        "poster": agent_idx,
                        "give": give,
                        "want": want,
                        "visible_to": target,
                    }
                    self.order_book.append(offer)
                    entry["offer_id"] = self.next_offer_id
                    entry["give"] = give
                    entry["want"] = want
                    entry["target_agent"] = target
                    entry["private"] = True
                    self.next_offer_id += 1
                else:
                    entry["invalid"] = True
                    entry["give"] = give
                    entry["want"] = want
                    entry["target_agent"] = target
                    entry["private"] = True

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
                    if matched.get("visible_to") is not None:
                        trade_record["private"] = True
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
            if self.live_updates:
                self._write_live(initial_inventories, start_time)

        return round_actions

    def _run_simultaneous_round(self, round_num, turn_order, initial_inventories, start_time):
        """Run one round of simultaneous play: all agents act on the same frozen snapshot."""
        # 1. Freeze state at round start
        frozen_inventories = copy.deepcopy(self.inventories)
        frozen_order_book = copy.deepcopy(self.order_book)

        # 2. Parallel LLM calls — each agent sees the frozen snapshot
        def _call_agent(agent_idx):
            visible_book = []
            for o in frozen_order_book:
                if o.get("posted_this_round"):
                    continue  # not visible yet
                if o.get("visible_to") is not None:
                    if agent_idx != o["poster"] and agent_idx != o["visible_to"]:
                        continue
                entry = {"id": o["id"], "poster": o["poster"], "give": o["give"], "want": o["want"]}
                if o.get("visible_to") is not None:
                    entry["private"] = True
                    entry["visible_to"] = o["visible_to"]
                visible_book.append(entry)

            return self.agents[agent_idx].take_turn(
                inventory=dict(frozen_inventories[agent_idx]),
                target=self.targets[agent_idx],
                order_book=visible_book,
                recent_trades=self.trades[-10:],
                round_num=round_num,
                max_rounds=self.max_rounds,
            )

        with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
            futures = {agent_idx: pool.submit(_call_agent, agent_idx) for agent_idx in turn_order}
            turns = {agent_idx: futures[agent_idx].result() for agent_idx in turn_order}

        # 3. Resolve actions in shuffled turn_order priority
        round_actions = []
        for agent_idx in turn_order:
            turn = turns[agent_idx]
            action = turn["action"]
            entry = {
                "round": round_num,
                "agent": agent_idx,
                "model": self.agents[agent_idx].model_name,
                "contestant": self.agents[agent_idx].contestant_name,
                "action": action,
                "message": turn.get("message", ""),
                "reasoning": turn.get("reasoning", ""),
                "simultaneous": True,
            }

            if action == "post_offer":
                give = turn.get("give", {})
                want = turn.get("want", {})
                # Validate against frozen inventory
                if self._has_items_snapshot(frozen_inventories[agent_idx], give) and give and want:
                    offer = {
                        "id": self.next_offer_id,
                        "poster": agent_idx,
                        "give": give,
                        "want": want,
                        "posted_this_round": True,
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

            elif action == "private_offer":
                give = turn.get("give", {})
                want = turn.get("want", {})
                target = turn.get("target_agent")
                valid_target = (isinstance(target, int) and 0 <= target < self.num_agents
                                and target != agent_idx)
                if self._has_items_snapshot(frozen_inventories[agent_idx], give) and give and want and valid_target:
                    offer = {
                        "id": self.next_offer_id,
                        "poster": agent_idx,
                        "give": give,
                        "want": want,
                        "visible_to": target,
                        "posted_this_round": True,
                    }
                    self.order_book.append(offer)
                    entry["offer_id"] = self.next_offer_id
                    entry["give"] = give
                    entry["want"] = want
                    entry["target_agent"] = target
                    entry["private"] = True
                    self.next_offer_id += 1
                else:
                    entry["invalid"] = True
                    entry["give"] = give
                    entry["want"] = want
                    entry["target_agent"] = target
                    entry["private"] = True

            elif action == "accept_offer":
                offer_id = turn.get("offer_id")
                matched = None
                for o in self.order_book:
                    if o["id"] == offer_id and o["poster"] != agent_idx:
                        if not o.get("posted_this_round"):
                            matched = o
                        break

                # Validate poster inventory against real (mutating) state to catch double-spends
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
                    if matched.get("visible_to") is not None:
                        trade_record["private"] = True
                    self.trades.append(trade_record)
                    self.order_book = [o for o in self.order_book if o["id"] != offer_id]
                    entry["offer_id"] = offer_id
                    entry["trade"] = trade_record
                else:
                    entry["invalid"] = True
                    entry["conflict"] = True
                    entry["offer_id"] = offer_id

            elif action == "pass_turn":
                pass  # nothing to do

            self.history.append(entry)
            round_actions.append(entry)

        # 4. Clear posted_this_round flags so new offers become visible next round
        for o in self.order_book:
            o.pop("posted_this_round", None)

        # Remove stale offers after resolution
        self._remove_stale_offers()

        # Write live file once per round (not per agent)
        if self.live_updates:
            self._write_live(initial_inventories, start_time)

        return round_actions

    def run(self):
        start_time = time.time()
        initial_inventories = [dict(inv) for inv in self.inventories]

        for round_num in range(self.max_rounds):
            # Randomize turn order each round
            turn_order = list(range(self.num_agents))
            random.shuffle(turn_order)

            if self.simultaneous:
                self._run_simultaneous_round(round_num, turn_order, initial_inventories, start_time)
            else:
                self._run_sequential_round(round_num, turn_order, initial_inventories, start_time)

            # Check if all agents have reached their goals
            all_complete = all(self.goal_completion(i) >= 1.0 for i in range(self.num_agents))
            if all_complete:
                break

        # Mark live file as completed
        self._write_live(initial_inventories, start_time, status="completed")

        elapsed = time.time() - start_time

        # Compute per-agent results
        agent_results = []
        for i in range(self.num_agents):
            ar = {
                "agent_idx": i,
                "model": self.agents[i].model_name,
                "initial_inventory": initial_inventories[i],
                "final_inventory": dict(self.inventories[i]),
                "target": self.targets[i],
                "goal_completion": round(self.goal_completion(i), 4),
            }
            if self.agents[i].strategy_id:
                ar["strategy_id"] = self.agents[i].strategy_id
            agent_results.append(ar)

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
            "simultaneous": self.simultaneous,
        }
