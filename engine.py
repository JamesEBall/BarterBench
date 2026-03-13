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
        self.auction_enabled = scenario.get("auction_enabled", False)
        self.active_auctions = []  # list of active auctions
        self.next_auction_id = 1

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

    # ---- Auction mechanics ----

    def _visible_auctions(self, agent_idx):
        """Return auctions visible to this agent."""
        visible = []
        for a in self.active_auctions:
            if a["status"] != "open":
                continue
            if a["visible_to"] is not None and agent_idx not in a["visible_to"] and agent_idx != a["auctioneer"]:
                continue
            entry = {
                "auction_id": a["id"],
                "auctioneer": a["auctioneer"],
                "give": a["give"],
                "min_bid": a.get("min_bid", {}),
                "private": a["visible_to"] is not None,
                "created_round": a["created_round"],
            }
            # Only the auctioneer sees all bids
            if agent_idx == a["auctioneer"]:
                entry["bids"] = [{"bidder": b["bidder"], "bid": b["bid"], "round": b["round"]}
                                 for b in a["bids"]]
                entry["num_bids"] = len(a["bids"])
            else:
                # Other agents only know how many bids exist
                entry["num_bids"] = len(a["bids"])
            visible.append(entry)
        return visible

    def _handle_start_auction(self, agent_idx, turn, round_num, entry):
        """Handle start_auction action. Returns True if valid."""
        give = turn.get("give", {})
        visible_to = turn.get("visible_to")  # list of agent indices or None for public
        min_bid = turn.get("min_bid", {})

        if not give or not self._has_items(agent_idx, give):
            entry["invalid"] = True
            entry["give"] = give
            return False

        # Validate visible_to
        if visible_to is not None:
            if not isinstance(visible_to, list):
                entry["invalid"] = True
                return False
            visible_to = [v for v in visible_to if isinstance(v, int) and 0 <= v < self.num_agents and v != agent_idx]
            if not visible_to:
                visible_to = None  # fallback to public

        auction = {
            "id": self.next_auction_id,
            "auctioneer": agent_idx,
            "give": give,
            "min_bid": min_bid,
            "visible_to": visible_to,
            "bids": [],
            "created_round": round_num,
            "status": "open",
        }
        self.active_auctions.append(auction)
        entry["auction_id"] = self.next_auction_id
        entry["give"] = give
        entry["min_bid"] = min_bid
        if visible_to is not None:
            entry["visible_to"] = visible_to
            entry["private"] = True
        self.next_auction_id += 1
        return True

    def _handle_submit_bid(self, agent_idx, turn, round_num, entry):
        """Handle submit_bid action. Returns True if valid."""
        auction_id = turn.get("auction_id")
        bid = turn.get("bid", {})

        # Find the auction
        auction = None
        for a in self.active_auctions:
            if a["id"] == auction_id and a["status"] == "open":
                auction = a
                break

        if auction is None:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        # Validate: bidder is not the auctioneer
        if agent_idx == auction["auctioneer"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        # Validate: bidder is eligible (in visible_to or public)
        if auction["visible_to"] is not None and agent_idx not in auction["visible_to"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        # Validate: bidder has the items
        if not bid or not self._has_items(agent_idx, bid):
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            entry["bid"] = bid
            return False

        auction["bids"].append({
            "bidder": agent_idx,
            "bid": bid,
            "round": round_num,
        })
        entry["auction_id"] = auction_id
        entry["bid"] = bid
        return True

    def _handle_close_auction(self, agent_idx, turn, round_num, entry):
        """Handle close_auction action. Returns True if valid."""
        auction_id = turn.get("auction_id")
        accepted_bid_idx = turn.get("accepted_bid_idx")
        reject_all = turn.get("reject_all", False)

        # Find the auction
        auction = None
        for a in self.active_auctions:
            if a["id"] == auction_id and a["status"] == "open":
                auction = a
                break

        if auction is None:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        # Only the auctioneer can close
        if agent_idx != auction["auctioneer"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        entry["auction_id"] = auction_id

        if reject_all:
            # Cancel auction — no trade
            auction["status"] = "cancelled"
            entry["auction_result"] = "cancelled"
            return True

        # Accept a specific bid
        if accepted_bid_idx is None or not isinstance(accepted_bid_idx, int):
            entry["invalid"] = True
            return False

        if accepted_bid_idx < 0 or accepted_bid_idx >= len(auction["bids"]):
            entry["invalid"] = True
            return False

        winning_bid = auction["bids"][accepted_bid_idx]
        winner = winning_bid["bidder"]
        bid_items = winning_bid["bid"]

        # Validate both parties still have the items
        if not self._has_items(auction["auctioneer"], auction["give"]):
            entry["invalid"] = True
            entry["auction_result"] = "auctioneer_lacks_items"
            auction["status"] = "failed"
            return False

        if not self._has_items(winner, bid_items):
            entry["invalid"] = True
            entry["auction_result"] = "bidder_lacks_items"
            auction["status"] = "failed"
            return False

        # Execute the trade
        self._execute_trade(auction["auctioneer"], winner, auction["give"], bid_items)
        trade_record = {
            "round": round_num,
            "auction_id": auction_id,
            "poster": auction["auctioneer"],
            "accepter": winner,
            "give": auction["give"],
            "want": bid_items,
            "auction_trade": True,
        }
        self.trades.append(trade_record)
        auction["status"] = "resolved"
        entry["trade"] = trade_record
        entry["auction_result"] = "resolved"
        entry["winning_bidder"] = winner
        return True

    def _auto_close_auctions(self):
        """Auto-close any remaining open auctions at end of match (no trade)."""
        for a in self.active_auctions:
            if a["status"] == "open":
                a["status"] = "expired"

    def _write_checkpoint(self, round_num, initial_inventories, elapsed_so_far):
        """Write full resumable state to checkpoint.json after each completed round."""
        try:
            checkpoint = {
                "round_num": round_num + 1,  # next round to run
                "scenario": self.scenario,
                "inventories": [dict(inv) for inv in self.inventories],
                "initial_inventories": initial_inventories,
                "targets": [dict(t) for t in self.targets],
                "order_book": copy.deepcopy(self.order_book),
                "next_offer_id": self.next_offer_id,
                "active_auctions": copy.deepcopy(self.active_auctions),
                "next_auction_id": self.next_auction_id,
                "trades": self.trades,
                "history": self.history,
                "simultaneous": self.simultaneous,
                "run_id": self.run_id,
                "elapsed_seconds": round(elapsed_so_far, 2),
                "auction_enabled": self.auction_enabled,
                "model_assignments": [a.model_name for a in self.agents],
                "agent_states": [a.get_state() for a in self.agents],
            }
            cp_file = Path(__file__).parent / "checkpoint.json"
            import fcntl
            with open(cp_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(checkpoint, f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            import sys
            print(f"[warn] checkpoint write failed: {e}", file=sys.stderr)

    @classmethod
    def from_checkpoint(cls, path="checkpoint.json"):
        """Reconstruct engine + agents from a checkpoint file."""
        from agent import MarketAgent
        cp_path = Path(path) if Path(path).is_absolute() else Path(__file__).parent / path
        with open(cp_path) as f:
            cp = json.load(f)

        scenario = cp["scenario"]
        model_assignments = cp["model_assignments"]
        agent_states = cp["agent_states"]
        auction_enabled = cp.get("auction_enabled", False)

        # Reconstruct agents
        agents = []
        for i, model_name in enumerate(model_assignments):
            strategy_id = scenario["agents"][i].get("strategy_id") if i < len(scenario.get("agents", [])) else None
            strategy_prompt = scenario["agents"][i].get("strategy_prompt") if i < len(scenario.get("agents", [])) else None
            a = MarketAgent(model_name, i, auction_enabled=auction_enabled,
                            strategy_id=strategy_id, strategy_prompt=strategy_prompt)
            a.set_state(agent_states[i])
            agents.append(a)

        # Reconstruct engine (bypass __init__ scenario parsing)
        engine = cls.__new__(cls)
        engine.scenario = scenario
        engine.agents = agents
        engine.num_agents = len(agents)
        engine.max_rounds = scenario.get("max_rounds", 10)
        engine.inventories = [dict(inv) for inv in cp["inventories"]]
        engine.targets = [dict(t) for t in cp["targets"]]
        engine.order_book = cp["order_book"]
        engine.next_offer_id = cp["next_offer_id"]
        engine.history = cp["history"]
        engine.trades = cp["trades"]
        engine.round_log = []
        engine.simultaneous = cp.get("simultaneous", False)
        engine.run_id = cp.get("run_id")
        engine.live_updates = True
        engine.auction_enabled = auction_enabled
        engine.active_auctions = cp.get("active_auctions", [])
        engine.next_auction_id = cp.get("next_auction_id", 1)

        start_round = cp["round_num"]
        initial_inventories = cp["initial_inventories"]
        elapsed_so_far = cp.get("elapsed_seconds", 0)

        return engine, start_round, initial_inventories, elapsed_so_far

    def _write_live(self, initial_inventories, start_time, status="running"):
        """Write current match state to live_match.json for dashboard streaming.

        Thread-safe: uses fcntl file locking so parallel runs don't corrupt the file.
        """
        try:
            live_file = Path(__file__).parent / "live_match.json"
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
                "run_id": self.run_id,
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
            import fcntl
            with open(live_file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(live, f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            import sys
            print(f"[warn] live update write failed: {e}", file=sys.stderr)

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
            visible_auctions = self._visible_auctions(agent_idx) if self.auction_enabled else None
            turn = self.agents[agent_idx].take_turn(
                inventory=self.inventories[agent_idx],
                target=self.targets[agent_idx],
                order_book=visible_book,
                recent_trades=self.trades[-10:],
                round_num=round_num,
                max_rounds=self.max_rounds,
                auctions=visible_auctions,
            )

            action = turn["action"]
            agent = self.agents[agent_idx]
            latency = agent.turn_latencies[-1] if agent.turn_latencies else 0
            entry = {
                "round": round_num,
                "agent": agent_idx,
                "model": agent.model_name,
                "contestant": agent.contestant_name,
                "action": action,
                "message": turn.get("message", ""),
                "reasoning": turn.get("reasoning", ""),
                "latency_seconds": latency,
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

            elif action == "start_auction" and self.auction_enabled:
                self._handle_start_auction(agent_idx, turn, round_num, entry)

            elif action == "submit_bid" and self.auction_enabled:
                self._handle_submit_bid(agent_idx, turn, round_num, entry)

            elif action == "close_auction" and self.auction_enabled:
                self._handle_close_auction(agent_idx, turn, round_num, entry)

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

        frozen_auctions = copy.deepcopy(self.active_auctions) if self.auction_enabled else None

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

            visible_auctions = self._visible_auctions(agent_idx) if self.auction_enabled else None

            return self.agents[agent_idx].take_turn(
                inventory=dict(frozen_inventories[agent_idx]),
                target=self.targets[agent_idx],
                order_book=visible_book,
                recent_trades=self.trades[-10:],
                round_num=round_num,
                max_rounds=self.max_rounds,
                auctions=visible_auctions,
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

            elif action == "start_auction" and self.auction_enabled:
                self._handle_start_auction(agent_idx, turn, round_num, entry)

            elif action == "submit_bid" and self.auction_enabled:
                self._handle_submit_bid(agent_idx, turn, round_num, entry)

            elif action == "close_auction" and self.auction_enabled:
                self._handle_close_auction(agent_idx, turn, round_num, entry)

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

    def run(self, start_round=0, initial_inventories=None, elapsed_offset=0):
        start_time = time.time()
        if initial_inventories is None:
            initial_inventories = [dict(inv) for inv in self.inventories]

        for round_num in range(start_round, self.max_rounds):
            # Randomize turn order each round
            turn_order = list(range(self.num_agents))
            random.shuffle(turn_order)

            effective_start = start_time - elapsed_offset
            if self.simultaneous:
                self._run_simultaneous_round(round_num, turn_order, initial_inventories, effective_start)
            else:
                self._run_sequential_round(round_num, turn_order, initial_inventories, effective_start)

            # Checkpoint after each completed round
            elapsed = time.time() - start_time + elapsed_offset
            self._write_checkpoint(round_num, initial_inventories, elapsed)

            # Check if all agents have reached their goals
            all_complete = all(self.goal_completion(i) >= 1.0 for i in range(self.num_agents))
            if all_complete:
                break

        # Auto-close any remaining open auctions
        if self.auction_enabled:
            self._auto_close_auctions()

        # Write final status so dashboard knows this run is done
        effective_start = start_time - elapsed_offset
        if self.live_updates:
            self._write_live(initial_inventories, effective_start, status="complete")

        elapsed = time.time() - start_time + elapsed_offset

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
