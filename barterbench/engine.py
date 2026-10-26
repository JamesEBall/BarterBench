"""Marketplace engine: runs N-agent trading with an order book.

Design notes (v2):
- All randomness flows through ``self.rng`` (a ``random.Random`` seeded per run).
  Turn order, and therefore the whole match given deterministic agents, is
  reproducible from ``seed``. RNG state is checkpointed and restored.
- Messages are real communication. Offers carry their message, trades carry the
  accepter's message, and pass-turn messages go to a public feed that every agent
  sees. Private-offer messages are visible only to the recipient. The
  ``communication`` setting ("full" | "offers_only" | "none") is an ablation knob.
- Private offers can only be accepted by their recipient.
- Auction bids are escrowed: items pledged in an open bid cannot be offered,
  spent, or pledged elsewhere until the auction resolves.
- Every history entry records the agent's position in that round's turn order so
  slot/position effects can be measured post hoc.
"""

import copy
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .paths import CHECKPOINT_DIR

COMMUNICATION_MODES = ("full", "offers_only", "none")
PUBLIC_FEED_LIMIT = 8   # most recent public messages shown to agents
RECENT_TRADES_LIMIT = 10


class MarketEngine:
    """N-agent marketplace with order book, randomized turn order, fixed round limit."""

    def __init__(self, scenario, agents, simultaneous=False, run_id=None,
                 seed=None, communication="full", checkpoint_dir=None):
        if communication not in COMMUNICATION_MODES:
            raise ValueError(f"communication must be one of {COMMUNICATION_MODES}, got {communication!r}")
        self.scenario = scenario
        self.agents = agents
        self.num_agents = len(agents)
        self.max_rounds = scenario.get("max_rounds", 10)
        self.inventories = [dict(a["inventory"]) for a in scenario["agents"]]
        self.targets = [dict(a.get("target", {})) for a in scenario["agents"]]
        self.reserved = [{} for _ in range(self.num_agents)]  # auction escrow
        self.order_book = []  # list of open offers
        self.next_offer_id = 1
        self.history = []  # all actions taken
        self.trades = []   # executed trades
        self.public_messages = []  # public broadcast feed: {round, agent, action, message}
        self.round_log = []  # per-round summary
        self.simultaneous = simultaneous
        self.run_id = run_id
        self.auction_enabled = scenario.get("auction_enabled", False)
        self.active_auctions = []  # list of active auctions
        self.next_auction_id = 1
        self.seed = seed
        self.rng = random.Random(seed)
        self.communication = communication
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.turn_orders = []  # per-round turn orders (for audit)

    # ---- Inventory helpers ----

    def _available(self, agent_idx, item):
        """Units of ``item`` the agent can actually commit (inventory minus escrow)."""
        return self.inventories[agent_idx].get(item, 0) - self.reserved[agent_idx].get(item, 0)

    def _has_items(self, agent_idx, items):
        """Check if agent has all specified items free of escrow."""
        if not isinstance(items, dict):
            return False
        for item, qty in items.items():
            if not isinstance(qty, (int, float)) or isinstance(qty, bool) or qty <= 0:
                return False
            if self._available(agent_idx, item) < qty:
                return False
        return bool(items)

    def _has_items_snapshot(self, inventory_dict, items):
        """Check if an inventory dict has all specified items (for simultaneous mode)."""
        if not isinstance(items, dict):
            return False
        for item, qty in items.items():
            if not isinstance(qty, (int, float)) or isinstance(qty, bool) or qty <= 0:
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

    def total_supply(self):
        """Total units of every item across all agents (conservation invariant)."""
        totals = {}
        for inv in self.inventories:
            for item, qty in inv.items():
                totals[item] = totals.get(item, 0) + qty
        return totals

    # ---- Visibility ----

    def _offer_view(self, o):
        entry = {"id": o["id"], "poster": o["poster"], "give": o["give"], "want": o["want"],
                 "round": o.get("round")}
        if o.get("visible_to") is not None:
            entry["private"] = True
            entry["visible_to"] = o["visible_to"]
        if self.communication == "full" and o.get("message"):
            entry["message"] = o["message"]
        return entry

    def _can_see_offer(self, agent_idx, o):
        if o.get("visible_to") is None:
            return True
        return agent_idx == o["poster"] or agent_idx == o["visible_to"]

    def _can_accept_offer(self, agent_idx, o):
        """Only the recipient of a private offer may accept it; never the poster."""
        if o["poster"] == agent_idx:
            return False
        if o.get("visible_to") is not None and o["visible_to"] != agent_idx:
            return False
        return True

    def _visible_order_book(self, agent_idx, book=None):
        """Return order book entries visible to this agent."""
        book = self.order_book if book is None else book
        return [self._offer_view(o) for o in book if self._can_see_offer(agent_idx, o)]

    def _visible_trades(self):
        """Public trade record. Messages are included only in full-communication mode."""
        out = []
        for t in self.trades[-RECENT_TRADES_LIMIT:]:
            v = {k: t[k] for k in ("round", "poster", "accepter", "give", "want") if k in t}
            if t.get("private"):
                v["private"] = True
            if self.communication == "full" and t.get("message"):
                v["message"] = t["message"]
            out.append(v)
        return out

    def _visible_messages(self, agent_idx):
        """Public feed (pass-turn broadcasts and accept messages) visible to everyone."""
        if self.communication != "full":
            return []
        return list(self.public_messages[-PUBLIC_FEED_LIMIT:])

    def _remove_stale_offers(self):
        """Remove offers where the poster no longer has the (unreserved) items."""
        self.order_book = [
            o for o in self.order_book
            if self._has_items(o["poster"], o["give"])
        ]

    def _broadcast(self, round_num, agent_idx, action, message):
        if self.communication != "full" or not message:
            return
        self.public_messages.append({
            "round": round_num, "agent": agent_idx, "action": action, "message": message,
        })

    # ---- Auction mechanics ----

    def _reserve(self, agent_idx, items):
        for item, qty in items.items():
            self.reserved[agent_idx][item] = self.reserved[agent_idx].get(item, 0) + int(qty)

    def _release(self, agent_idx, items):
        for item, qty in items.items():
            left = self.reserved[agent_idx].get(item, 0) - int(qty)
            if left > 0:
                self.reserved[agent_idx][item] = left
            else:
                self.reserved[agent_idx].pop(item, None)

    def _release_all_bids(self, auction):
        for b in auction["bids"]:
            if not b.get("released"):
                self._release(b["bidder"], b["bid"])
                b["released"] = True

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
            if agent_idx == a["auctioneer"]:
                entry["bids"] = [{"bidder": b["bidder"], "bid": b["bid"], "round": b["round"]}
                                 for b in a["bids"]]
                entry["num_bids"] = len(a["bids"])
            else:
                entry["num_bids"] = len(a["bids"])
            visible.append(entry)
        return visible

    def _handle_start_auction(self, agent_idx, turn, round_num, entry):
        """Handle start_auction action. Returns True if valid."""
        give = turn.get("give", {})
        visible_to = turn.get("visible_to")
        min_bid = turn.get("min_bid", {})

        if not give or not self._has_items(agent_idx, give):
            entry["invalid"] = True
            entry["give"] = give
            return False

        if visible_to is not None:
            if not isinstance(visible_to, list):
                entry["invalid"] = True
                return False
            visible_to = [v for v in visible_to if isinstance(v, int) and 0 <= v < self.num_agents and v != agent_idx]
            if not visible_to:
                visible_to = None

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
        """Handle submit_bid action. Bid items are escrowed. Returns True if valid."""
        auction_id = turn.get("auction_id")
        bid = turn.get("bid", {})

        auction = None
        for a in self.active_auctions:
            if a["id"] == auction_id and a["status"] == "open":
                auction = a
                break

        if auction is None:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        if agent_idx == auction["auctioneer"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        if auction["visible_to"] is not None and agent_idx not in auction["visible_to"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        # A new bid on the same auction replaces the bidder's previous bid (and its escrow).
        previous = [b for b in auction["bids"] if b["bidder"] == agent_idx and not b.get("released")]
        for b in previous:
            self._release(agent_idx, b["bid"])
            b["released"] = True
            b["superseded"] = True

        if not bid or not self._has_items(agent_idx, bid):
            # Restore escrow of a superseded bid if the new one is invalid
            for b in previous:
                self._reserve(agent_idx, b["bid"])
                b["released"] = False
                b.pop("superseded", None)
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            entry["bid"] = bid
            return False

        self._reserve(agent_idx, bid)
        auction["bids"].append({
            "bidder": agent_idx,
            "bid": bid,
            "round": round_num,
            "released": False,
        })
        entry["auction_id"] = auction_id
        entry["bid"] = bid
        return True

    def _handle_close_auction(self, agent_idx, turn, round_num, entry):
        """Handle close_auction action. Returns True if valid."""
        auction_id = turn.get("auction_id")
        accepted_bid_idx = turn.get("accepted_bid_idx")
        reject_all = turn.get("reject_all", False)

        auction = None
        for a in self.active_auctions:
            if a["id"] == auction_id and a["status"] == "open":
                auction = a
                break

        if auction is None:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        if agent_idx != auction["auctioneer"]:
            entry["invalid"] = True
            entry["auction_id"] = auction_id
            return False

        entry["auction_id"] = auction_id

        if reject_all:
            self._release_all_bids(auction)
            auction["status"] = "cancelled"
            entry["auction_result"] = "cancelled"
            return True

        live_bids = [b for b in auction["bids"] if not b.get("superseded")]
        if accepted_bid_idx is None or isinstance(accepted_bid_idx, bool) or not isinstance(accepted_bid_idx, int):
            entry["invalid"] = True
            return False

        if accepted_bid_idx < 0 or accepted_bid_idx >= len(live_bids):
            entry["invalid"] = True
            return False

        winning_bid = live_bids[accepted_bid_idx]
        winner = winning_bid["bidder"]
        bid_items = winning_bid["bid"]

        if not self._has_items(auction["auctioneer"], auction["give"]):
            self._release_all_bids(auction)
            entry["invalid"] = True
            entry["auction_result"] = "auctioneer_lacks_items"
            auction["status"] = "failed"
            return False

        # Winner's items are escrowed, so this cannot fail unless escrow was bypassed.
        self._release_all_bids(auction)
        if not self._has_items(winner, bid_items):
            entry["invalid"] = True
            entry["auction_result"] = "bidder_lacks_items"
            auction["status"] = "failed"
            return False

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
        if turn.get("message") and self.communication == "full":
            trade_record["message"] = turn["message"]
        self.trades.append(trade_record)
        auction["status"] = "resolved"
        entry["trade"] = trade_record
        entry["auction_result"] = "resolved"
        entry["winning_bidder"] = winner
        return True

    def _auto_close_auctions(self):
        """Auto-close any remaining open auctions at end of match (no trade, release escrow)."""
        for a in self.active_auctions:
            if a["status"] == "open":
                self._release_all_bids(a)
                a["status"] = "expired"

    # ---- Checkpointing ----

    def checkpoint_path(self):
        name = "checkpoint.json" if self.run_id is None else f"checkpoint_{self.run_id}.json"
        return self.checkpoint_dir / name

    def _write_checkpoint(self, round_num, initial_inventories, elapsed_so_far):
        """Write full resumable state after each completed round."""
        try:
            checkpoint = {
                "round_num": round_num + 1,  # next round to run
                "scenario": self.scenario,
                "inventories": [dict(inv) for inv in self.inventories],
                "reserved": [dict(r) for r in self.reserved],
                "initial_inventories": initial_inventories,
                "targets": [dict(t) for t in self.targets],
                "order_book": copy.deepcopy(self.order_book),
                "next_offer_id": self.next_offer_id,
                "active_auctions": copy.deepcopy(self.active_auctions),
                "next_auction_id": self.next_auction_id,
                "trades": self.trades,
                "history": self.history,
                "public_messages": self.public_messages,
                "turn_orders": self.turn_orders,
                "simultaneous": self.simultaneous,
                "run_id": self.run_id,
                "seed": self.seed,
                "rng_state": _rng_state_to_json(self.rng.getstate()),
                "communication": self.communication,
                "elapsed_seconds": round(elapsed_so_far, 2),
                "auction_enabled": self.auction_enabled,
                "model_assignments": [a.model_name for a in self.agents],
                "agent_states": [a.get_state() for a in self.agents],
            }
            cp_file = self.checkpoint_path()
            cp_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(checkpoint, f)
            tmp.replace(cp_file)
        except Exception as e:
            import sys
            print(f"[warn] checkpoint write failed: {e}", file=sys.stderr)

    @classmethod
    def from_checkpoint(cls, path="checkpoint.json", agent_factory=None):
        """Reconstruct engine + agents from a checkpoint file.

        ``agent_factory(model_name, agent_idx, scenario_agent, auction_enabled)`` builds
        agents; defaults to ``agent.build_agent``.
        """
        if agent_factory is None:
            from .agents import build_agent as agent_factory
        cp_path = Path(path) if Path(path).is_absolute() else CHECKPOINT_DIR / path
        with open(cp_path) as f:
            cp = json.load(f)

        scenario = cp["scenario"]
        model_assignments = cp["model_assignments"]
        agent_states = cp["agent_states"]
        auction_enabled = cp.get("auction_enabled", False)

        agents = []
        for i, model_name in enumerate(model_assignments):
            sa = scenario["agents"][i] if i < len(scenario.get("agents", [])) else {}
            a = agent_factory(model_name, i, sa, auction_enabled)
            a.set_state(agent_states[i])
            agents.append(a)

        engine = cls(scenario, agents, simultaneous=cp.get("simultaneous", False),
                     run_id=cp.get("run_id"), seed=cp.get("seed"),
                     communication=cp.get("communication", "full"))
        engine.inventories = [dict(inv) for inv in cp["inventories"]]
        engine.reserved = [dict(r) for r in cp.get("reserved", [{} for _ in agents])]
        engine.targets = [dict(t) for t in cp["targets"]]
        engine.order_book = cp["order_book"]
        engine.next_offer_id = cp["next_offer_id"]
        engine.history = cp["history"]
        engine.trades = cp["trades"]
        engine.public_messages = cp.get("public_messages", [])
        engine.turn_orders = cp.get("turn_orders", [])
        engine.active_auctions = cp.get("active_auctions", [])
        engine.next_auction_id = cp.get("next_auction_id", 1)
        if cp.get("rng_state") is not None:
            engine.rng.setstate(_rng_state_from_json(cp["rng_state"]))

        start_round = cp["round_num"]
        initial_inventories = cp["initial_inventories"]
        elapsed_so_far = cp.get("elapsed_seconds", 0)
        return engine, start_round, initial_inventories, elapsed_so_far

    # ---- Scoring ----

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

    # ---- Turn handling ----

    def _observe(self, agent_idx, inventory=None, book=None):
        """Build the observation passed to an agent."""
        return {
            "inventory": dict(inventory if inventory is not None else self.inventories[agent_idx]),
            "target": self.targets[agent_idx],
            "order_book": self._visible_order_book(agent_idx, book),
            "recent_trades": self._visible_trades(),
            "auctions": self._visible_auctions(agent_idx) if self.auction_enabled else None,
            "messages": self._visible_messages(agent_idx),
            "reserved": dict(self.reserved[agent_idx]),
        }

    def _call_agent(self, agent_idx, round_num, obs):
        return self.agents[agent_idx].take_turn(
            inventory=obs["inventory"],
            target=obs["target"],
            order_book=obs["order_book"],
            recent_trades=obs["recent_trades"],
            round_num=round_num,
            max_rounds=self.max_rounds,
            auctions=obs["auctions"],
            messages=obs["messages"],
        )

    def _base_entry(self, round_num, agent_idx, turn, position):
        agent = self.agents[agent_idx]
        entry = {
            "round": round_num,
            "agent": agent_idx,
            "turn_position": position,
            "model": agent.model_name,
            "contestant": agent.contestant_name,
            "action": turn["action"],
            "message": turn.get("message", "") or "",
            "reasoning": turn.get("reasoning", "") or "",
        }
        if agent.turn_latencies:
            entry["latency_seconds"] = agent.turn_latencies[-1]
        for flag in ("parse_error", "truncated", "fallback"):
            if turn.get(flag):
                entry[flag] = True
        return entry

    def _apply_offer(self, agent_idx, turn, round_num, entry, has_items):
        give = turn.get("give", {})
        want = turn.get("want", {})
        private = turn["action"] == "private_offer"
        target = turn.get("target_agent")
        valid_target = (not private) or (isinstance(target, int) and not isinstance(target, bool)
                                         and 0 <= target < self.num_agents and target != agent_idx)
        entry["give"] = give
        entry["want"] = want
        if private:
            entry["target_agent"] = target
            entry["private"] = True
        if has_items(give) and give and isinstance(want, dict) and want and valid_target \
                and all(isinstance(q, (int, float)) and not isinstance(q, bool) and q > 0 for q in want.values()):
            offer = {
                "id": self.next_offer_id,
                "poster": agent_idx,
                "give": give,
                "want": want,
                "round": round_num,
            }
            if private:
                offer["visible_to"] = target
            if self.communication == "full" and turn.get("message"):
                offer["message"] = turn["message"]
            self.order_book.append(offer)
            entry["offer_id"] = self.next_offer_id
            self.next_offer_id += 1
            return offer
        entry["invalid"] = True
        return None

    def _apply_accept(self, agent_idx, turn, round_num, entry, posted_this_round=None):
        offer_id = turn.get("offer_id")
        entry["offer_id"] = offer_id
        matched = None
        for o in self.order_book:
            if o["id"] == offer_id:
                if posted_this_round is not None and o["id"] in posted_this_round:
                    break  # not acceptable until next round (simultaneous mode)
                if self._can_accept_offer(agent_idx, o):
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
            if self.communication == "full" and turn.get("message"):
                trade_record["message"] = turn["message"]
            self.trades.append(trade_record)
            self.order_book = [o for o in self.order_book if o["id"] != offer_id]
            entry["trade"] = trade_record
            return trade_record
        entry["invalid"] = True
        return None

    def _apply_action(self, agent_idx, turn, round_num, entry, has_items, posted_this_round=None):
        action = turn["action"]
        if action in ("post_offer", "private_offer"):
            offer = self._apply_offer(agent_idx, turn, round_num, entry, has_items)
            if offer is not None and posted_this_round is not None:
                posted_this_round.add(offer["id"])
        elif action == "accept_offer":
            trade = self._apply_accept(agent_idx, turn, round_num, entry, posted_this_round)
            if trade is not None:
                self._broadcast(round_num, agent_idx, "accept_offer", turn.get("message"))
        elif action == "start_auction" and self.auction_enabled:
            self._handle_start_auction(agent_idx, turn, round_num, entry)
            self._broadcast(round_num, agent_idx, "start_auction", turn.get("message"))
        elif action == "submit_bid" and self.auction_enabled:
            self._handle_submit_bid(agent_idx, turn, round_num, entry)
        elif action == "close_auction" and self.auction_enabled:
            self._handle_close_auction(agent_idx, turn, round_num, entry)
        elif action == "pass_turn":
            self._broadcast(round_num, agent_idx, "pass_turn", turn.get("message"))
        else:
            # Unknown action or auction action with auctions disabled → treated as a pass.
            entry["action"] = "pass_turn"
            entry["unrecognized_action"] = action

    def _run_sequential_round(self, round_num, turn_order, initial_inventories, start_time):
        """Run one round of sequential play."""
        round_actions = []
        for position, agent_idx in enumerate(turn_order):
            self._remove_stale_offers()
            obs = self._observe(agent_idx)
            turn = self._call_agent(agent_idx, round_num, obs)
            entry = self._base_entry(round_num, agent_idx, turn, position)
            self._apply_action(agent_idx, turn, round_num, entry,
                               has_items=lambda items, i=agent_idx: self._has_items(i, items))
            self.history.append(entry)
            round_actions.append(entry)
        return round_actions

    def _run_simultaneous_round(self, round_num, turn_order, initial_inventories, start_time):
        """Run one round of simultaneous play: all agents act on the same frozen snapshot."""
        frozen_inventories = copy.deepcopy(self.inventories)
        frozen_order_book = copy.deepcopy(self.order_book)

        observations = {
            i: self._observe(i, inventory=frozen_inventories[i], book=frozen_order_book)
            for i in turn_order
        }

        with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
            futures = {i: pool.submit(self._call_agent, i, round_num, observations[i]) for i in turn_order}
            turns = {i: futures[i].result() for i in turn_order}

        # Resolve in shuffled priority order. Offers posted this round cannot be accepted
        # until next round; accepts are validated against live inventories so the first
        # agent in priority order wins any contested offer.
        posted_this_round = set()
        round_actions = []
        for position, agent_idx in enumerate(turn_order):
            turn = turns[agent_idx]
            entry = self._base_entry(round_num, agent_idx, turn, position)
            entry["simultaneous"] = True
            self._apply_action(
                agent_idx, turn, round_num, entry,
                has_items=lambda items, i=agent_idx: self._has_items_snapshot(frozen_inventories[i], items)
                and self._has_items(i, items),
                posted_this_round=posted_this_round,
            )
            if entry.get("invalid") and turn["action"] == "accept_offer":
                entry["conflict"] = True
            self.history.append(entry)
            round_actions.append(entry)

        self._remove_stale_offers()
        return round_actions

    def run(self, start_round=0, initial_inventories=None, elapsed_offset=0):
        start_time = time.time()
        if initial_inventories is None:
            initial_inventories = [dict(inv) for inv in self.inventories]
        supply_before = self.total_supply()

        for round_num in range(start_round, self.max_rounds):
            turn_order = list(range(self.num_agents))
            self.rng.shuffle(turn_order)
            self.turn_orders.append(list(turn_order))

            effective_start = start_time - elapsed_offset
            if self.simultaneous:
                self._run_simultaneous_round(round_num, turn_order, initial_inventories, effective_start)
            else:
                self._run_sequential_round(round_num, turn_order, initial_inventories, effective_start)

            elapsed = time.time() - start_time + elapsed_offset
            self._write_checkpoint(round_num, initial_inventories, elapsed)

            if all(self.goal_completion(i) >= 1.0 for i in range(self.num_agents)):
                break

        if self.auction_enabled:
            self._auto_close_auctions()

        assert self.total_supply() == supply_before, "item conservation violated"
        assert all(not r for r in self.reserved), "escrow not fully released at match end"

        elapsed = time.time() - start_time + elapsed_offset

        agent_results = []
        for i in range(self.num_agents):
            ar = {
                "agent_idx": i,
                "model": self.agents[i].model_name,
                "contestant": self.agents[i].contestant_name,
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
            "public_messages": self.public_messages,
            "turn_orders": self.turn_orders,
            "initial_inventories": initial_inventories,
            "simultaneous": self.simultaneous,
            "seed": self.seed,
            "communication": self.communication,
        }


# ---- RNG state (de)serialisation ----

def _rng_state_to_json(state):
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _rng_state_from_json(data):
    version, internal, gauss = data
    return (version, tuple(internal), gauss)
