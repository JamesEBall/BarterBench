"""Agents for the marketplace.

One LLM backend only: an OpenAI-compatible chat endpoint served locally by vLLM
(or anything that speaks the same protocol). Every model goes through the exact
same prompt, the same JSON schema, the same decoding settings and the same
parser, so cross-model comparisons are not confounded by harness differences.

Structured output: the request asks the server for schema-constrained JSON
(``response_format`` json_schema, falling back to vLLM's ``guided_json``, then
to free text + regex extraction). Parse failures are never silent: the turn is
recorded as a pass with ``parse_error=True`` and counted per model.

Scripted baselines (RandomAgent, HeuristicAgent) share the same ``take_turn``
interface and never call a model.
"""

import json
import os
import random as stdlib_random
import re
import time
from pathlib import Path

VALID_ACTIONS = {"post_offer", "accept_offer", "private_offer", "pass_turn",
                 "start_auction", "submit_bid", "close_auction"}
BASE_ACTIONS = ["post_offer", "private_offer", "accept_offer", "pass_turn"]
AUCTION_ACTIONS = ["start_auction", "submit_bid", "close_auction"]

DEFAULT_BASE_URL = os.environ.get("BARTERBENCH_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MAX_TOKENS = 700
from .paths import ENDPOINTS_FILE  # noqa: E402

CONCEAL_INSTRUCTION = (
    "Strategic note: other traders are rivals competing for the same scarce items. "
    "Do not state which items you are trying to acquire or how many you still need. "
    "Talk only about what you offer. Revealing your needs gives rivals leverage."
)

JSON_SCHEMA_INSTRUCTION = """
Reason briefly about your best move — what do you need, who has it, what leverage do you hold, how many rounds remain — then output exactly one JSON action.

{"action": "post_offer",    "give": {"item": qty}, "want": {"item": qty}, "message": "reason"}
{"action": "private_offer", "give": {"item": qty}, "want": {"item": qty}, "target_agent": 3, "message": "reason"}
{"action": "accept_offer",  "offer_id": 123, "message": "reason"}
{"action": "pass_turn",     "message": "reason"}
{"action": "start_auction", "give": {"item": qty}, "min_bid": {"item": qty}, "visible_to": [1, 3], "message": "reason"}
{"action": "submit_bid",    "auction_id": 1, "bid": {"item": qty}, "message": "reason"}
{"action": "close_auction", "auction_id": 1, "accepted_bid_idx": 0, "message": "reason"}
""".strip()


# ---- JSON schema for constrained decoding ----

def _items_schema():
    return {"type": "object", "additionalProperties": {"type": "integer", "minimum": 1}}


def action_schema(auction_enabled=False):
    """JSON schema every LLM turn must satisfy. ``reasoning`` comes first so the
    model can think before committing to an action, inside the constrained output."""
    actions = BASE_ACTIONS + (AUCTION_ACTIONS if auction_enabled else [])
    props = {
        "reasoning": {"type": "string"},
        "action": {"type": "string", "enum": actions},
        "give": _items_schema(),
        "want": _items_schema(),
        "offer_id": {"type": ["integer", "null"]},
        "target_agent": {"type": ["integer", "null"]},
        "message": {"type": "string"},
    }
    if auction_enabled:
        props.update({
            "auction_id": {"type": ["integer", "null"]},
            "bid": _items_schema(),
            "min_bid": _items_schema(),
            "accepted_bid_idx": {"type": ["integer", "null"]},
            "reject_all": {"type": "boolean"},
            "visible_to": {"type": ["array", "null"], "items": {"type": "integer"}},
        })
    return {
        "type": "object",
        "properties": props,
        "required": ["reasoning", "action", "message"],
        "additionalProperties": False,
    }


# ---- Prompt construction ----

def _build_marketplace_context(agent_idx, inventory, target, order_book, recent_trades, round_num,
                               max_rounds, strategy_prompt=None, auctions=None, messages=None,
                               conceal=False, reserved=None, communication="full"):
    """Build the system prompt for a marketplace agent.

    Everything an agent is allowed to know is here and nowhere else.
    """
    rounds_remaining = max_rounds - round_num
    urgency = "CRITICAL — final rounds, accept imperfect deals now" if rounds_remaining <= 2 else \
              "HIGH — limited time, prioritise execution" if rounds_remaining <= max_rounds // 3 else \
              "NORMAL"

    still_need = {}
    completion_parts = []
    for item, needed in target.items():
        if needed > 0:
            have = inventory.get(item, 0)
            pct = min(have / needed, 1.0) * 100
            completion_parts.append(f"{item}: {have}/{needed} ({pct:.0f}%)")
            if have < needed:
                still_need[item] = needed - have

    surplus = {item: qty for item, qty in inventory.items() if qty > 0 and item not in target}
    for item, qty in inventory.items():
        extra = qty - target.get(item, 0)
        if extra > 0 and item not in surplus:
            surplus[item] = extra

    wanted_by_others = {}
    for offer in order_book:
        if offer["poster"] != agent_idx:
            for item, qty in offer.get("want", {}).items():
                wanted_by_others[item] = wanted_by_others.get(item, 0) + qty
    leverage = {item: qty for item, qty in surplus.items() if item in wanted_by_others}

    lines = [
        f"You are Trader {agent_idx} in a competitive multi-agent marketplace.",
        f"Round {round_num + 1} of {max_rounds} | {rounds_remaining} rounds remaining | Urgency: {urgency}",
        "",
        "## Your Position",
        f"Inventory: {json.dumps(inventory)}",
        f"Target:    {json.dumps(target)}",
        f"Progress:  {', '.join(completion_parts) if completion_parts else 'No targets set'}",
    ]
    if still_need:
        lines.append(f"Still need: {json.dumps(still_need)}")
    if reserved:
        lines.append(f"Escrowed in open auction bids (not spendable): {json.dumps(reserved)}")
    if leverage:
        lines.append(f"You hold {json.dumps(leverage)} — others are actively seeking these (leverage)")
    elif surplus:
        lines.append(f"Trade chips (surplus items): {json.dumps(surplus)}")

    if strategy_prompt:
        lines += ["", "## Your Strategy", strategy_prompt]
    if conceal:
        lines += ["", "## Information Discipline", CONCEAL_INSTRUCTION]

    lines += [
        "",
        "## Actions",
        "post_offer     — public offer visible to all (reveals your position, maximises counterparties)",
        "private_offer  — whisper to one trader only (only they can see or accept it)",
        "accept_offer   — execute an offer immediately (only offers listed below that you can afford)",
        "pass_turn      — skip this round; your message is still broadcast to everyone",
    ]
    if auctions is not None:
        lines += [
            "start_auction  — sealed-bid auction for items you hold; you choose when to close and which bid wins",
            "submit_bid     — sealed bid on an open auction; bid items are escrowed until it resolves",
            "close_auction  — accept a bid by index or reject all",
        ]
    lines += [
        "",
        "Rules: you can only offer items you hold | you cannot accept your own offers | trades are atomic",
    ]
    if communication == "none":
        lines.append("Note: messages are not delivered to other traders in this market.")

    if order_book:
        lines += ["", "## Order Book (open offers)"]
        for offer in order_book:
            private_tag = " [PRIVATE]" if offer.get("private") else ""
            msg = f' — "{offer["message"]}"' if offer.get("message") else ""
            if offer["poster"] == agent_idx:
                target_info = f" (whisper to Trader {offer['visible_to']})" if offer.get("visible_to") is not None else ""
                lines.append(f"  [#{offer['id']}] YOUR OFFER{private_tag}{target_info} — give {json.dumps(offer['give'])}, want {json.dumps(offer['want'])}{msg}")
            elif offer.get("private"):
                lines.append(f"  [#{offer['id']}] WHISPER from Trader {offer['poster']} — offers {json.dumps(offer['give'])} for {json.dumps(offer['want'])}{msg}")
            else:
                lines.append(f"  [#{offer['id']}] Trader {offer['poster']} offers {json.dumps(offer['give'])} for {json.dumps(offer['want'])}{msg}")
    else:
        lines += ["", "## Order Book: empty — no open offers"]

    if recent_trades:
        lines += ["", "## Recent Trades"]
        for t in recent_trades[-6:]:
            msg = f' — Trader {t["accepter"]}: "{t["message"]}"' if t.get("message") else ""
            lines.append(f"  Round {t['round'] + 1}: Trader {t['poster']} traded {json.dumps(t['give'])} for {json.dumps(t['want'])} with Trader {t['accepter']}{msg}")

    if messages:
        lines += ["", "## Public Messages (most recent last)"]
        for m in messages:
            who = "You" if m["agent"] == agent_idx else f"Trader {m['agent']}"
            lines.append(f"  Round {m['round'] + 1} — {who}: \"{m['message']}\"")

    if auctions:
        lines += ["", "## Active Auctions"]
        for a in auctions:
            private_tag = " [PRIVATE]" if a.get("private") else " [PUBLIC]"
            if a["auctioneer"] == agent_idx:
                lines.append(f"  [Auction #{a['auction_id']}] YOUR AUCTION{private_tag} — selling {json.dumps(a['give'])}")
                if a.get("min_bid"):
                    lines.append(f"    Min bid hint: {json.dumps(a['min_bid'])}")
                if a.get("bids"):
                    lines.append(f"    Bids received ({len(a['bids'])}):")
                    for bi, b in enumerate(a["bids"]):
                        lines.append(f"      [{bi}] Trader {b['bidder']} bids {json.dumps(b['bid'])}")
                else:
                    lines.append("    No bids yet")
                lines.append("    Use close_auction to accept a bid or reject all")
            else:
                lines.append(f"  [Auction #{a['auction_id']}] Trader {a['auctioneer']}{private_tag} — selling {json.dumps(a['give'])}")
                if a.get("min_bid"):
                    lines.append(f"    Min bid hint: {json.dumps(a['min_bid'])}")
                lines.append(f"    {a.get('num_bids', 0)} bids so far. Use submit_bid to bid.")

    return "\n".join(lines)


# ---- Parsing ----

_JSON_RE = re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', re.DOTALL)


def _normalise_action(data, reasoning=""):
    """Turn a decoded JSON object into the canonical turn dict."""
    action = data.get("action", "pass_turn")
    unrecognized = action not in VALID_ACTIONS
    result = {
        "action": action if not unrecognized else "pass_turn",
        "give": data.get("give") or {},
        "want": data.get("want") or {},
        "offer_id": data.get("offer_id"),
        "target_agent": data.get("target_agent"),
        "message": data.get("message", "") or "",
        "reasoning": (data.get("reasoning") or reasoning or "").strip(),
    }
    if unrecognized:
        result["unrecognized_action"] = action
    for field in ("auction_id", "bid", "min_bid", "accepted_bid_idx", "reject_all", "visible_to"):
        if data.get(field) is not None and data.get(field) != {} and data.get(field) is not False:
            result[field] = data[field]
    return result


def _parse_json_response(text):
    """Extract a JSON action from model text output. Never raises."""
    text = text or ""
    candidates = _JSON_RE.findall(text)
    # Prefer the last complete object that contains an "action" key
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action" in data:
            prefix = text[:text.rfind(cand)].strip()
            return _normalise_action(data, reasoning=prefix)
    return {
        "action": "pass_turn", "give": {}, "want": {}, "offer_id": None, "target_agent": None,
        "message": "", "reasoning": text, "parse_error": True,
    }


# ---- Endpoint resolution ----

def load_endpoints():
    """alias -> {"base_url": ..., "model": ...}. Sources, in priority order:
    env BARTERBENCH_ENDPOINTS (JSON), ./endpoints.json, then defaults."""
    endpoints = {}
    if ENDPOINTS_FILE.exists():
        try:
            endpoints.update(json.loads(ENDPOINTS_FILE.read_text()))
        except Exception as e:  # pragma: no cover
            import sys
            print(f"[warn] could not read {ENDPOINTS_FILE}: {e}", file=sys.stderr)
    raw = os.environ.get("BARTERBENCH_ENDPOINTS")
    if raw:
        endpoints.update(json.loads(raw))
    normalised = {}
    for alias, v in endpoints.items():
        if isinstance(v, str):
            normalised[alias] = {"base_url": v}
        else:
            normalised[alias] = dict(v)
    return normalised


def resolve_endpoint(model_name, endpoints=None):
    """Return (base_url, served_model_name) for an alias."""
    from .registry import get_model_info
    endpoints = load_endpoints() if endpoints is None else endpoints
    info = get_model_info(model_name)
    ep = endpoints.get(model_name, {})
    base_url = ep.get("base_url") or info.get("base_url") or DEFAULT_BASE_URL
    served = ep.get("model") or info.get("served_name") or (
        info["full_id"] if info.get("full_id") not in (None, "unknown") else model_name)
    return base_url, served


_SERVED_CACHE = {}


def discover_served_model(client, base_url):
    """If the endpoint serves exactly one model, return its id (cached per URL)."""
    if base_url in _SERVED_CACHE:
        return _SERVED_CACHE[base_url]
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        ids = []
    _SERVED_CACHE[base_url] = ids[0] if len(ids) == 1 else None
    return _SERVED_CACHE[base_url]


# ---- The one LLM agent ----

class LLMAgent:
    """Marketplace agent backed by a local OpenAI-compatible endpoint (vLLM)."""

    backend = "local"

    def __init__(self, model_name, agent_idx, strategy_id=None, strategy_prompt=None,
                 temperature=1.0, auction_enabled=False, history_rounds=-1,
                 base_url=None, served_name=None, max_tokens=DEFAULT_MAX_TOKENS,
                 seed=None, conceal=False, communication="full", client=None,
                 timeout=180.0, backend=None):
        self.model_name = model_name
        self.agent_idx = agent_idx
        self.strategy_id = strategy_id
        self.strategy_prompt = strategy_prompt
        self.temperature = temperature
        self.auction_enabled = auction_enabled
        self.history_rounds = history_rounds
        self.max_tokens = max_tokens
        self.seed = seed
        self.conceal = conceal
        self.communication = communication

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_latencies = []
        self.round_history = []   # (round, summary) tuples
        self.parse_errors = 0
        self.truncations = 0
        self.fallbacks = 0        # turns answered by the emergency pass
        self.turns = 0
        self.structured_mode = "json_schema"  # -> "guided_json" -> "text"

        resolved_url, resolved_model = resolve_endpoint(model_name)
        self.base_url = base_url or resolved_url
        self.model = served_name or resolved_model
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=os.environ.get("BARTERBENCH_API_KEY", "local"),
                                 timeout=timeout, max_retries=0)
        single = discover_served_model(self.client, self.base_url)
        if single and single != self.model:
            self.model = single
        self.resolved_model = f"{self.base_url}::{self.model}"

    # -- state --

    def get_state(self):
        return {
            "round_history": self.round_history,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "turn_latencies": self.turn_latencies,
            "parse_errors": self.parse_errors,
            "truncations": self.truncations,
            "fallbacks": self.fallbacks,
            "turns": self.turns,
            "structured_mode": self.structured_mode,
        }

    def set_state(self, state):
        self.round_history = [tuple(x) for x in state.get("round_history", [])]
        self.total_input_tokens = state.get("total_input_tokens", 0)
        self.total_output_tokens = state.get("total_output_tokens", 0)
        self.turn_latencies = state.get("turn_latencies", [])
        self.parse_errors = state.get("parse_errors", 0)
        self.truncations = state.get("truncations", 0)
        self.fallbacks = state.get("fallbacks", 0)
        self.turns = state.get("turns", 0)
        self.structured_mode = state.get("structured_mode", "json_schema")

    @property
    def contestant_name(self):
        return self.strategy_id if self.strategy_id else self.model_name

    def reliability(self):
        n = max(self.turns, 1)
        return {
            "turns": self.turns,
            "parse_error_rate": round(self.parse_errors / n, 4),
            "truncation_rate": round(self.truncations / n, 4),
            "fallback_rate": round(self.fallbacks / n, 4),
            "structured_mode": self.structured_mode,
        }

    # -- prompting --

    def _history_section(self):
        if not self.round_history:
            return ""
        history = self.round_history if self.history_rounds < 0 else self.round_history[-(self.history_rounds):]
        lines = ["", "## Your Previous Actions (memory)"]
        for rnd, summary in history:
            lines.append(f"  Round {rnd}: {summary}")
        return "\n".join(lines)

    def _record_round_history(self, parsed, round_num):
        action = parsed["action"]
        if action == "post_offer":
            summary = f"Posted offer: give {json.dumps(parsed['give'])} for {json.dumps(parsed['want'])}"
        elif action == "private_offer":
            summary = f"Sent private offer to Trader {parsed.get('target_agent')}: give {json.dumps(parsed['give'])} for {json.dumps(parsed['want'])}"
        elif action == "accept_offer":
            summary = f"Accepted offer #{parsed.get('offer_id')}"
        elif action == "start_auction":
            summary = f"Started auction selling {json.dumps(parsed.get('give', {}))}"
        elif action == "submit_bid":
            summary = f"Bid {json.dumps(parsed.get('bid', {}))} on auction #{parsed.get('auction_id')}"
        elif action == "close_auction":
            summary = (f"Cancelled auction #{parsed.get('auction_id')}" if parsed.get("reject_all")
                       else f"Closed auction #{parsed.get('auction_id')}, accepted bid #{parsed.get('accepted_bid_idx')}")
        else:
            summary = "Passed turn"
        if parsed.get("parse_error"):
            summary += " (output could not be parsed)"
        if parsed.get("message"):
            summary += f" — \"{parsed['message'][:80]}\""
        self.round_history.append((round_num + 1, summary))

    def _request_kwargs(self, messages, round_num):
        kwargs = dict(model=self.model, messages=messages, max_tokens=self.max_tokens,
                      temperature=self.temperature)
        if self.seed is not None:
            # Per-turn seed: reproducible sampling where the server supports it.
            kwargs["seed"] = (self.seed * 7919 + self.agent_idx * 104729 + round_num) % (2 ** 31)
        schema = action_schema(self.auction_enabled)
        if self.structured_mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "barter_action", "schema": schema, "strict": True},
            }
        elif self.structured_mode == "guided_json":
            kwargs["extra_body"] = {"guided_json": schema}
        return kwargs

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds,
                  auctions=None, messages=None, reserved=None):
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
            auctions=auctions if self.auction_enabled else None,
            messages=messages, conceal=self.conceal, reserved=reserved,
            communication=self.communication,
        )
        system = context + self._history_section()
        user = f"{JSON_SCHEMA_INSTRUCTION}\n\nIt's your turn. Respond with a single JSON object."
        chat = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        turn_start = time.monotonic()
        self.turns += 1
        parsed = None
        last_error = None
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(**self._request_kwargs(chat, round_num))
            except Exception as e:  # connection errors, 5xx, or schema rejection
                last_error = e
                status = getattr(e, "status_code", None)
                if status == 400 and self.structured_mode != "text":
                    # Server rejected the structured-output request: degrade once, retry at once.
                    self.structured_mode = "guided_json" if self.structured_mode == "json_schema" else "text"
                    continue
                time.sleep(min(2 ** attempt, 20))
                continue

            if not response.choices:
                last_error = RuntimeError("empty choices")
                time.sleep(1)
                continue

            choice = response.choices[0]
            if getattr(response, "usage", None):
                self.total_input_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
                self.total_output_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            if getattr(choice, "finish_reason", None) == "length":
                self.truncations += 1
            text = choice.message.content or ""
            hidden_reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
            parsed = _parse_json_response(text)
            if hidden_reasoning and not parsed.get("reasoning"):
                parsed["reasoning"] = hidden_reasoning
            if parsed.get("parse_error"):
                self.parse_errors += 1
                parsed["truncated"] = getattr(choice, "finish_reason", None) == "length"
            break

        self.turn_latencies.append(round(time.monotonic() - turn_start, 3))
        if parsed is None:
            self.fallbacks += 1
            parsed = {
                "action": "pass_turn", "give": {}, "want": {}, "offer_id": None, "target_agent": None,
                "message": "", "reasoning": f"backend failure: {last_error}", "fallback": True,
            }
        self._record_round_history(parsed, round_num)
        return parsed


# Backwards-compatible name used by arena/runner.py and older scripts.
MarketAgent = LLMAgent


# ---- Scripted baselines ----

class _ScriptedBase:
    strategy_prompt = None
    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self, agent_idx):
        self.agent_idx = agent_idx
        self.strategy_id = None   # hidden contestant label (self-play); never shown to agents
        self.turn_latencies = []
        self.round_history = []
        self.turns = 0

    @property
    def contestant_name(self):
        return self.strategy_id if self.strategy_id else self.model_name

    def reliability(self):
        return {"turns": self.turns, "parse_error_rate": 0.0, "truncation_rate": 0.0,
                "fallback_rate": 0.0, "structured_mode": "scripted"}

    def get_state(self):
        return {"round_history": self.round_history, "turn_latencies": self.turn_latencies, "turns": self.turns}

    def set_state(self, state):
        self.round_history = [tuple(x) for x in state.get("round_history", [])]
        self.turn_latencies = state.get("turn_latencies", [])
        self.turns = state.get("turns", 0)

    @staticmethod
    def _pass():
        return {"action": "pass_turn", "give": {}, "want": {}, "offer_id": None,
                "target_agent": None, "message": "", "reasoning": "scripted"}


class RandomAgent(_ScriptedBase):
    """Random valid actions. Sends no messages, so it says nothing about disclosure."""

    model_name = "random"
    backend = "random"

    def __init__(self, agent_idx, seed=None, auction_enabled=False, num_agents=None):
        super().__init__(agent_idx)
        self.rng = stdlib_random.Random(seed)
        self.auction_enabled = auction_enabled
        self.num_agents = num_agents

    def get_state(self):
        s = super().get_state()
        s["rng_state"] = [self.rng.getstate()[0], list(self.rng.getstate()[1]), self.rng.getstate()[2]]
        return s

    def set_state(self, state):
        super().set_state(state)
        if "rng_state" in state:
            v, internal, g = state["rng_state"]
            self.rng.setstate((v, tuple(internal), g))

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds,
                  auctions=None, messages=None, reserved=None):
        start = time.monotonic()
        self.turns += 1
        result = self._choose_action(inventory, target, order_book)
        self.turn_latencies.append(round(time.monotonic() - start, 6))
        return result

    def _choose_action(self, inventory, target, order_book):
        roll = self.rng.random()
        if roll < 0.40 and order_book:
            acceptable = [o for o in order_book if o["poster"] != self.agent_idx
                          and self._can_afford(inventory, o["want"])]
            if acceptable:
                offer = self.rng.choice(acceptable)
                r = self._pass()
                r.update({"action": "accept_offer", "offer_id": offer["id"]})
                return r
        if roll < 0.75:
            offer = self._random_offer(inventory, target)
            if offer:
                return offer
        return self._pass()

    @staticmethod
    def _can_afford(inventory, cost):
        return all(inventory.get(item, 0) >= int(qty) for item, qty in cost.items())

    def _random_offer(self, inventory, target):
        giveable = []
        for item, qty in inventory.items():
            if qty > 0:
                surplus = qty - target.get(item, 0)
                if surplus > 0:
                    giveable.append((item, surplus))
                elif self.rng.random() < 0.3:
                    giveable.append((item, qty))
        wantable = [(item, needed - inventory.get(item, 0)) for item, needed in target.items()
                    if needed > 0 and inventory.get(item, 0) < needed]
        if not giveable or not wantable:
            return None
        give_item, max_give = self.rng.choice(giveable)
        want_item, max_want = self.rng.choice(wantable)
        r = self._pass()
        r.update({
            "action": "post_offer",
            "give": {give_item: self.rng.randint(1, max(1, max_give))},
            "want": {want_item: self.rng.randint(1, max(1, min(max_want, 4)))},
        })
        if self.rng.random() < 0.2:
            n = self.num_agents or 12
            others = [i for i in range(n) if i != self.agent_idx]
            if others:
                r["action"] = "private_offer"
                r["target_agent"] = self.rng.choice(others)
        return r


class HeuristicAgent(_ScriptedBase):
    """Deterministic greedy trader: a much harder scripted anchor than random.

    - Accepts the offer that most improves goal completion.
    - Otherwise posts a 1:1 offer of surplus for the item it is furthest from,
      preferring whatever the order book shows others are asking for (so its
      offers are likely to be taken).
    - Passes only when nothing beneficial exists. Sends no messages.
    """

    model_name = "heuristic"
    backend = "heuristic"

    def __init__(self, agent_idx, seed=None, auction_enabled=False, num_agents=None):
        super().__init__(agent_idx)
        self.auction_enabled = auction_enabled

    @staticmethod
    def _gc(inventory, target):
        if not target:
            return 1.0
        parts = [min(inventory.get(item, 0) / qty, 1.0) for item, qty in target.items() if qty > 0]
        return sum(parts) / len(parts) if parts else 1.0

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds,
                  auctions=None, messages=None, reserved=None):
        start = time.monotonic()
        self.turns += 1
        action = self._choose_action(inventory, target, order_book)
        self.turn_latencies.append(round(time.monotonic() - start, 6))
        self.round_history.append((round_num, action.get("action", "pass_turn")))
        return action

    def _choose_action(self, inventory, target, order_book):
        current = self._gc(inventory, target)
        best_offer, best_gc = None, current
        for offer in order_book:
            if offer.get("poster") == self.agent_idx:
                continue
            if not all(inventory.get(i, 0) >= q for i, q in offer.get("want", {}).items()):
                continue
            inv2 = dict(inventory)
            for i, q in offer.get("want", {}).items():
                inv2[i] = inv2.get(i, 0) - q
            for i, q in offer.get("give", {}).items():
                inv2[i] = inv2.get(i, 0) + q
            new_gc = self._gc(inv2, target)
            if new_gc > best_gc + 1e-9:
                best_gc, best_offer = new_gc, offer
        if best_offer:
            r = self._pass()
            r.update({"action": "accept_offer", "offer_id": best_offer["id"]})
            return r

        needed = {i: q - inventory.get(i, 0) for i, q in target.items() if inventory.get(i, 0) < q}
        surplus = {i: q - target.get(i, 0) for i, q in inventory.items() if q - target.get(i, 0) > 0}
        if needed and surplus:
            demand = {}
            for o in order_book:
                if o.get("poster") != self.agent_idx:
                    for i, q in o.get("want", {}).items():
                        demand[i] = demand.get(i, 0) + q
            # Give the surplus item others ask for most; ask for the item we are furthest from.
            give_item = max(surplus, key=lambda i: (demand.get(i, 0), surplus[i]))
            want_item = max(needed, key=lambda i: needed[i] / max(target.get(i, 1), 1))
            r = self._pass()
            r.update({"action": "post_offer", "give": {give_item: 1}, "want": {want_item: 1}})
            return r
        return self._pass()


SCRIPTED_AGENTS = {"random": RandomAgent, "heuristic": HeuristicAgent}


def is_scripted(model_name):
    return model_name in SCRIPTED_AGENTS


def build_agent(model_name, agent_idx, scenario_agent=None, auction_enabled=False, *, seed=None,
                num_agents=None, temperature=1.0, history_rounds=-1, conceal=False,
                communication="full", endpoints=None, strategy_id=None, strategy_prompt=None,
                max_tokens=DEFAULT_MAX_TOKENS):
    """Factory used by eval, arena and checkpoint restore."""
    scenario_agent = scenario_agent or {}
    strategy_id = strategy_id or scenario_agent.get("strategy_id")
    strategy_prompt = strategy_prompt or scenario_agent.get("strategy_prompt")
    if model_name in SCRIPTED_AGENTS:
        agent = SCRIPTED_AGENTS[model_name](agent_idx=agent_idx, seed=seed, auction_enabled=auction_enabled,
                                            num_agents=num_agents)
        agent.strategy_id = strategy_id
        return agent
    base_url, served = resolve_endpoint(model_name, endpoints)
    return LLMAgent(model_name, agent_idx, strategy_id=strategy_id, strategy_prompt=strategy_prompt,
                    temperature=temperature, auction_enabled=auction_enabled, history_rounds=history_rounds,
                    base_url=base_url, served_name=served, seed=seed, conceal=conceal,
                    communication=communication, max_tokens=max_tokens)
