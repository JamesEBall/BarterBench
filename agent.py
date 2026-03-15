"""Agent wrapper for marketplace — supports Anthropic API, OpenRouter (OpenAI-compatible), and claude CLI."""

import json
import os
import random as stdlib_random
import re
import subprocess
import time

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

MODEL_MAP = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}

CLI_MODEL_MAP = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
}

# OpenRouter model aliases — short names for convenience
# All free-tier models with tool calling support as of Mar 2026
OPENROUTER_MODEL_MAP = {
    # --- Frontier free models ---
    "hunter":           "openrouter/hunter-alpha",              # 1T params, 1M ctx
    "hunter-alpha":     "openrouter/hunter-alpha",
    "healer":           "openrouter/healer-alpha",              # 256K ctx
    "healer-alpha":     "openrouter/healer-alpha",
    # --- Big open-weight models ---
    "nemotron-120b":    "nvidia/nemotron-3-super-120b-a12b:free",  # 120B MoE, 256K ctx
    "gpt-oss-120b":    "openai/gpt-oss-120b:free",             # 120B, 128K ctx
    "qwen3-coder":     "qwen/qwen3-coder:free",                # 480B MoE, 256K ctx
    "llama-70b":       "meta-llama/llama-3.3-70b-instruct:free",  # 70B, 128K ctx
    "trinity-large":   "arcee-ai/trinity-large-preview:free",  # MoE, 131K ctx
    # --- Mid-size models ---
    "step-flash":      "stepfun/step-3.5-flash:free",          # 196B MoE, 256K ctx
    "qwen3-80b":       "qwen/qwen3-next-80b-a3b-instruct:free", # 80B MoE, 256K ctx
    "nemotron-30b":    "nvidia/nemotron-3-nano-30b-a3b:free",  # 30B MoE, 256K ctx
    "gemma-27b":       "google/gemma-3-27b-it:free",           # 27B, 128K ctx
    "trinity-mini":    "arcee-ai/trinity-mini:free",           # 26B MoE, 128K ctx
    "mistral-small":   "mistralai/mistral-small-3.1-24b-instruct:free",  # 24B, 128K ctx
    "gpt-oss-20b":     "openai/gpt-oss-20b:free",             # 20B, 128K ctx
    "glm-4.5":         "z-ai/glm-4.5-air:free",               # 128K ctx
    # --- Paid models (need OPENROUTER_API_KEY with credits) ---
    "gpt4o":           "openai/gpt-4o",
    "gpt4o-mini":      "openai/gpt-4o-mini",
    "gemini-pro":      "google/gemini-2.5-pro-preview",
    "gemini-flash":    "google/gemini-2.5-flash-preview",
    "deepseek":        "deepseek/deepseek-chat-v3-0324",
}

# Claude model paths when routing through OpenRouter (used when OPENROUTER_API_KEY is set)
ANTHROPIC_VIA_OPENROUTER = {
    "haiku":  "anthropic/claude-haiku-4-5-20251001",
    "sonnet": "anthropic/claude-sonnet-4-6",
    "opus":   "anthropic/claude-opus-4-6",
}

VALID_ACTIONS = {"post_offer", "accept_offer", "private_offer", "pass_turn",
                 "start_auction", "submit_bid", "close_auction"}

JSON_SCHEMA_INSTRUCTION = """
Think briefly, then output a JSON action:

{"action": "post_offer", "give": {"item": qty}, "want": {"item": qty}, "message": "reason"}
{"action": "private_offer", "give": {"item": qty}, "want": {"item": qty}, "target_agent": 3, "message": "reason"}
{"action": "accept_offer", "offer_id": 123, "message": "reason"}
{"action": "pass_turn", "message": "reason"}
{"action": "start_auction", "give": {"item": qty}, "min_bid": {"item": qty}, "visible_to": [1, 3], "message": "reason"}
{"action": "submit_bid", "auction_id": 1, "bid": {"item": qty}, "message": "reason"}
{"action": "close_auction", "auction_id": 1, "accepted_bid_idx": 0, "message": "reason"}
""".strip()

MARKETPLACE_TOOLS = [
    {
        "name": "post_offer",
        "description": "Post a new offer on the marketplace order book. Other traders can see and accept it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "give": {"type": "object", "description": "Items you offer to give, e.g. {\"apples\": 2}"},
                "want": {"type": "object", "description": "Items you want in return, e.g. {\"oranges\": 3}"},
                "message": {"type": "string", "description": "Public message to other traders"},
            },
            "required": ["give", "want", "message"],
        },
    },
    {
        "name": "accept_offer",
        "description": "Accept an existing offer from the order book. The trade executes immediately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offer_id": {"type": "integer", "description": "The ID of the offer to accept"},
                "message": {"type": "string", "description": "Public message to other traders"},
            },
            "required": ["offer_id", "message"],
        },
    },
    {
        "name": "private_offer",
        "description": "Send a private offer (whisper) to a specific trader. Only they can see and accept it — other traders cannot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "give": {"type": "object", "description": "Items you offer to give, e.g. {\"apples\": 2}"},
                "want": {"type": "object", "description": "Items you want in return, e.g. {\"oranges\": 3}"},
                "target_agent": {"type": "integer", "description": "The trader index to send this private offer to"},
                "message": {"type": "string", "description": "Private message to the target trader"},
            },
            "required": ["give", "want", "target_agent", "message"],
        },
    },
    {
        "name": "pass_turn",
        "description": "Pass your turn without taking any action this round.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Optional message"},
            },
            "required": ["message"],
        },
    },
]

AUCTION_TOOLS = [
    {
        "name": "start_auction",
        "description": "Start a sealed-bid auction for items you own. You decide when to close and which bid to accept. Set visible_to for a private auction (only listed traders can bid) or omit for public.",
        "input_schema": {
            "type": "object",
            "properties": {
                "give": {"type": "object", "description": "Items to auction, e.g. {\"gold\": 2}"},
                "min_bid": {"type": "object", "description": "Suggested minimum bid (hint to bidders)"},
                "visible_to": {"type": "array", "items": {"type": "integer"}, "description": "Agent indices who can bid (omit for public auction)"},
                "message": {"type": "string", "description": "Auction announcement"},
            },
            "required": ["give", "message"],
        },
    },
    {
        "name": "submit_bid",
        "description": "Submit a sealed bid for an active auction. Your bid is secret — only the auctioneer can see it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "auction_id": {"type": "integer", "description": "The auction to bid on"},
                "bid": {"type": "object", "description": "Items you offer as payment, e.g. {\"wheat\": 5}"},
                "message": {"type": "string", "description": "Message to the auctioneer"},
            },
            "required": ["auction_id", "bid", "message"],
        },
    },
    {
        "name": "close_auction",
        "description": "Close an auction you started. Accept a bid by index or reject all bids to cancel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "auction_id": {"type": "integer", "description": "The auction to close"},
                "accepted_bid_idx": {"type": "integer", "description": "Index of the bid to accept (0-based)"},
                "reject_all": {"type": "boolean", "description": "Set to true to cancel the auction without accepting any bid"},
                "message": {"type": "string", "description": "Message about your decision"},
            },
            "required": ["auction_id", "message"],
        },
    },
]


def _build_marketplace_context(agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds, strategy_prompt=None, auctions=None):
    """Build the system prompt for a marketplace agent."""
    # Goal completion so far
    completion_parts = []
    for item, needed in target.items():
        if needed > 0:
            have = inventory.get(item, 0)
            pct = min(have / needed, 1.0) * 100
            completion_parts.append(f"{item}: {have}/{needed} ({pct:.0f}%)")

    lines = [
        f"You are Trader {agent_idx} in a competitive multi-agent marketplace.",
        f"Round {round_num + 1} of {max_rounds}.",
        "",
        "## Your Goal",
        "Acquire items to match your TARGET inventory through trading.",
        "Some items are SCARCE (demand exceeds supply across all traders), so not everyone can fully reach their goal.",
        "",
        f"Your current inventory: {json.dumps(inventory)}",
        f"Your target inventory:  {json.dumps(target)}",
        f"Goal progress: {', '.join(completion_parts) if completion_parts else 'No targets'}",
    ]

    if strategy_prompt:
        lines.append("")
        lines.append("## Your Strategy")
        lines.append(strategy_prompt)

    lines.extend([
        "",
        "## Rules",
        "1. You can POST a public offer (visible to all traders on the order book)",
        "2. You can send a PRIVATE OFFER (whisper) to a specific trader — only they can see it",
        "3. You can ACCEPT an existing offer from the order book (public or private)",
        "4. You can PASS if no good trades are available",
        "5. Trades execute immediately when accepted — both inventories update",
        "6. You can only offer items you currently have (quantity > 0)",
        "7. You cannot accept your own offers",
    ])

    if order_book:
        lines.append("")
        lines.append("## Order Book (open offers you can accept)")
        for offer in order_book:
            private_tag = " [PRIVATE]" if offer.get("private") else ""
            if offer["poster"] == agent_idx:
                target_info = f" (whisper to Trader {offer['visible_to']})" if offer.get("visible_to") is not None else ""
                lines.append(f"  [#{offer['id']}] YOUR OFFER{private_tag}{target_info} — give {json.dumps(offer['give'])}, want {json.dumps(offer['want'])}")
            else:
                if offer.get("private"):
                    lines.append(f"  [#{offer['id']}] 🤫 WHISPER from Trader {offer['poster']} — offers {json.dumps(offer['give'])} for {json.dumps(offer['want'])}")
                else:
                    lines.append(f"  [#{offer['id']}] Trader {offer['poster']} offers {json.dumps(offer['give'])} for {json.dumps(offer['want'])}")
    else:
        lines.append("")
        lines.append("## Order Book: empty — no open offers")

    if recent_trades:
        lines.append("")
        lines.append("## Recent Trades")
        for t in recent_trades[-6:]:
            lines.append(f"  Round {t['round']}: Trader {t['poster']} traded {json.dumps(t['give'])} ↔ {json.dumps(t['want'])} with Trader {t['accepter']}")

    if auctions:
        lines.append("")
        lines.append("## Active Auctions")
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
                    lines.append(f"    No bids yet")
                lines.append(f"    Use close_auction to accept a bid or reject all")
            else:
                lines.append(f"  [Auction #{a['auction_id']}] Trader {a['auctioneer']}{private_tag} — selling {json.dumps(a['give'])}")
                if a.get("min_bid"):
                    lines.append(f"    Min bid hint: {json.dumps(a['min_bid'])}")
                lines.append(f"    {a.get('num_bids', 0)} bids so far. Use submit_bid to bid.")

    return "\n".join(lines)


def _anthropic_tools_to_openai(tools):
    """Convert Anthropic tool format to OpenAI function-calling format."""
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return openai_tools


def _parse_json_response(text):
    """Extract JSON action from model text output."""
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = data.get("action", "pass_turn")
            if action not in VALID_ACTIONS:
                action = "pass_turn"
            result = {
                "action": action,
                "give": data.get("give", {}),
                "want": data.get("want", {}),
                "offer_id": data.get("offer_id"),
                "target_agent": data.get("target_agent"),
                "message": data.get("message", ""),
                "reasoning": text[:text.find(json_match.group())].strip(),
            }
            # Auction fields
            if data.get("auction_id") is not None:
                result["auction_id"] = data["auction_id"]
            if data.get("bid"):
                result["bid"] = data["bid"]
            if data.get("min_bid"):
                result["min_bid"] = data["min_bid"]
            if data.get("accepted_bid_idx") is not None:
                result["accepted_bid_idx"] = data["accepted_bid_idx"]
            if data.get("reject_all"):
                result["reject_all"] = data["reject_all"]
            if data.get("visible_to") is not None:
                result["visible_to"] = data["visible_to"]
            return result
        except json.JSONDecodeError:
            pass
    return {
        "action": "pass_turn",
        "give": {},
        "want": {},
        "offer_id": None,
        "message": "Could not determine action",
        "reasoning": text,
        "parse_error": True,
    }


class MarketAgent:
    """Marketplace agent that calls Claude via API or CLI."""

    def __init__(self, model_name: str, agent_idx: int, backend: str = "auto",
                 strategy_id: str = None, strategy_prompt: str = None,
                 temperature: float = 1.0, auction_enabled: bool = False,
                 history_rounds: int = -1):
        self.model_name = model_name
        self.agent_idx = agent_idx
        self.strategy_id = strategy_id
        self.strategy_prompt = strategy_prompt
        self.temperature = temperature
        self.auction_enabled = auction_enabled
        self.history_rounds = history_rounds
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_latencies = []  # per-turn wall clock seconds
        self.conversation_history = []  # stateful: accumulate turns within a match (API)
        self.round_history = []  # stateful: (round, action_summary) tuples (CLI)

        # Determine backend: OpenRouter first (covers all models), then Anthropic API, then CLI
        is_openrouter = model_name in OPENROUTER_MODEL_MAP or model_name.startswith("openrouter/") or "/" in model_name
        if backend == "auto":
            if is_openrouter:
                self.backend = "openrouter"
            elif os.environ.get("OPENROUTER_API_KEY"):
                self.backend = "openrouter"  # route Claude models through OpenRouter
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.backend = "api"
            else:
                self.backend = "cli"
        else:
            self.backend = backend

        if self.backend == "openrouter":
            from openai import OpenAI
            self.model = OPENROUTER_MODEL_MAP.get(model_name) or ANTHROPIC_VIA_OPENROUTER.get(model_name, model_name)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            )
        elif self.backend == "api":
            import anthropic
            self.model = MODEL_MAP.get(model_name, model_name)
            self.client = anthropic.Anthropic()
        else:
            self.cli_model = CLI_MODEL_MAP.get(model_name, model_name)

    def get_state(self):
        """Serialize agent state for checkpointing."""
        state = {
            "round_history": self.round_history,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "turn_latencies": self.turn_latencies,
        }
        if self.backend == "api" and self.conversation_history:
            state["conversation_history"] = self.conversation_history
        return state

    def set_state(self, state):
        """Restore agent state from checkpoint."""
        self.round_history = [tuple(x) for x in state.get("round_history", [])]
        self.total_input_tokens = state.get("total_input_tokens", 0)
        self.total_output_tokens = state.get("total_output_tokens", 0)
        self.turn_latencies = state.get("turn_latencies", [])
        if self.backend == "api":
            self.conversation_history = state.get("conversation_history", [])

    @property
    def contestant_name(self):
        """Return the name used for scoring — strategy_id in arena mode, model_name otherwise."""
        return self.strategy_id if self.strategy_id else self.model_name

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        if self.backend == "openrouter":
            return self._turn_openrouter(inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=auctions)
        elif self.backend == "api":
            return self._turn_api(inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=auctions)
        else:
            return self._turn_cli(inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=auctions)

    def _turn_api(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        import anthropic
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
            auctions=auctions if self.auction_enabled else None,
        )

        # Build messages: include conversation history for statefulness
        messages = list(self.conversation_history)
        messages.append({"role": "user", "content": "It's your turn. Choose your action."})

        turn_start = time.monotonic()
        for attempt in range(6):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=self.temperature,
                    system=context,
                    messages=messages,
                    tools=MARKETPLACE_TOOLS + (AUCTION_TOOLS if self.auction_enabled else []),
                    tool_choice={"type": "any"},
                )
                break
            except anthropic.RateLimitError:
                if attempt < 5:
                    wait = min(30 * (2 ** attempt), 300)
                    time.sleep(wait)
                else:
                    raise
        self.turn_latencies.append(round(time.monotonic() - turn_start, 3))

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        reasoning = ""
        tool_name = "pass_turn"
        tool_input = {"message": ""}
        tool_use_id = None

        for block in response.content:
            if block.type == "text":
                reasoning = block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

        # Record this exchange in conversation history for statefulness
        # Keep system context fresh each round but preserve the agent's reasoning thread
        self.conversation_history.append({"role": "user", "content": f"It's your turn. Choose your action."})
        # Reconstruct assistant response content
        assistant_content = []
        if reasoning:
            assistant_content.append({"type": "text", "text": reasoning})
        if tool_use_id:
            assistant_content.append({"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input})
        if assistant_content:
            self.conversation_history.append({"role": "assistant", "content": assistant_content})
            # Add tool result to close the tool_use turn
            if tool_use_id:
                self.conversation_history.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                                 "content": f"Action executed: {tool_name}"}]
                })

        # Cap history to control token usage (-1 = keep all rounds)
        if self.history_rounds >= 0:
            max_history_items = self.history_rounds * 6
            if len(self.conversation_history) > max_history_items:
                self.conversation_history = self.conversation_history[-max_history_items:]

        result = {
            "action": tool_name,
            "give": tool_input.get("give", {}),
            "want": tool_input.get("want", {}),
            "offer_id": tool_input.get("offer_id"),
            "target_agent": tool_input.get("target_agent"),
            "message": tool_input.get("message", ""),
            "reasoning": reasoning,
        }
        # Auction fields
        if tool_input.get("auction_id") is not None:
            result["auction_id"] = tool_input["auction_id"]
        if tool_input.get("bid"):
            result["bid"] = tool_input["bid"]
        if tool_input.get("min_bid"):
            result["min_bid"] = tool_input["min_bid"]
        if tool_input.get("accepted_bid_idx") is not None:
            result["accepted_bid_idx"] = tool_input["accepted_bid_idx"]
        if tool_input.get("reject_all"):
            result["reject_all"] = tool_input["reject_all"]
        if tool_input.get("visible_to") is not None:
            result["visible_to"] = tool_input["visible_to"]
        return result

    def _turn_openrouter(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        """OpenRouter backend — uses OpenAI-compatible API with tool calling."""
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
            auctions=auctions if self.auction_enabled else None,
        )

        # Build OpenAI-format tools
        tools_anthropic = MARKETPLACE_TOOLS + (AUCTION_TOOLS if self.auction_enabled else [])
        tools_openai = _anthropic_tools_to_openai(tools_anthropic)

        # Build messages with conversation history
        history_section = self._build_round_history_section()
        messages = [{"role": "system", "content": f"{context}{history_section}"}]
        messages.append({"role": "user", "content": f"{JSON_SCHEMA_INSTRUCTION}\n\nIt's your turn. Choose your action."})

        turn_start = time.monotonic()
        response = None
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=self.temperature,
                    messages=messages,
                    tools=tools_openai,
                    tool_choice="required",
                )
                if response and response.choices:
                    break
                # Response came back empty — retry
                if attempt < max_attempts - 1:
                    time.sleep(1)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower() or (
                    hasattr(e, "status_code") and getattr(e, "status_code", None) == 429
                )
                if is_rate_limit and attempt < max_attempts - 1:
                    wait = min(30 * (2 ** attempt), 300)  # 30s, 60s, 120s, 240s, 300s cap
                    time.sleep(wait)
                elif attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Final attempt: fall back to plain text (no tools)
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            max_tokens=512,
                            temperature=self.temperature,
                            messages=messages,
                        )
                        text = (response.choices[0].message.content or "") if response and response.choices else ""
                        if response and response.usage:
                            self.total_input_tokens += response.usage.prompt_tokens
                            self.total_output_tokens += response.usage.completion_tokens
                        parsed = _parse_json_response(text)
                        self._record_round_history(parsed, round_num)
                        return parsed
                    except Exception:
                        return self._fallback_pass()

        if not response or not response.choices:
            self.turn_latencies.append(round(time.monotonic() - turn_start, 3))
            return self._fallback_pass()

        self.turn_latencies.append(round(time.monotonic() - turn_start, 3))
        choice = response.choices[0]
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        reasoning = ""
        tool_name = "pass_turn"
        tool_input = {"message": ""}

        msg = choice.message
        if msg.content:
            reasoning = msg.content
        # Some models (hunter-alpha) return reasoning in a separate field
        if hasattr(msg, 'reasoning') and msg.reasoning:
            reasoning = msg.reasoning

        # Check for tool calls (OpenAI format)
        if msg.tool_calls and len(msg.tool_calls) > 0:
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            try:
                tool_input = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_input = {"message": "Parse error"}
        elif msg.content:
            # No tool call — try to parse JSON from content
            parsed = _parse_json_response(msg.content)
            self._record_round_history(parsed, round_num)
            return parsed

        result = {
            "action": tool_name if tool_name in VALID_ACTIONS else "pass_turn",
            "give": tool_input.get("give", {}),
            "want": tool_input.get("want", {}),
            "offer_id": tool_input.get("offer_id"),
            "target_agent": tool_input.get("target_agent"),
            "message": tool_input.get("message", ""),
            "reasoning": reasoning,
        }
        # Auction fields
        for field in ("auction_id", "bid", "min_bid", "accepted_bid_idx", "reject_all", "visible_to"):
            if tool_input.get(field) is not None:
                result[field] = tool_input[field]

        self._record_round_history(result, round_num)
        return result

    def _fallback_pass(self):
        return {
            "action": "pass_turn", "give": {}, "want": {}, "offer_id": None,
            "message": "Agent failed to respond", "reasoning": "",
        }

    def _record_round_history(self, parsed, round_num):
        """Record action summary for CLI/OpenRouter round history."""
        action = parsed["action"]
        if action == "post_offer":
            summary = f"Posted offer: give {json.dumps(parsed['give'])} for {json.dumps(parsed['want'])}"
        elif action == "private_offer":
            summary = f"Sent private offer to Trader {parsed.get('target_agent')}: give {json.dumps(parsed['give'])} for {json.dumps(parsed['want'])}"
        elif action == "accept_offer":
            summary = f"Accepted offer #{parsed.get('offer_id')}"
        elif action == "start_auction":
            summary = f"Started auction selling {json.dumps(parsed['give'])}"
        elif action == "submit_bid":
            summary = f"Bid {json.dumps(parsed.get('bid', {}))} on auction #{parsed.get('auction_id')}"
        elif action == "close_auction":
            if parsed.get("reject_all"):
                summary = f"Cancelled auction #{parsed.get('auction_id')}"
            else:
                summary = f"Closed auction #{parsed.get('auction_id')}, accepted bid #{parsed.get('accepted_bid_idx')}"
        else:
            summary = "Passed turn"
        if parsed.get("message"):
            summary += f" — \"{parsed['message'][:80]}\""
        self.round_history.append((round_num + 1, summary))

    def _build_round_history_section(self):
        """Build a prompt section summarizing this agent's previous actions."""
        if not self.round_history:
            return ""
        lines = ["\n## Your Previous Actions (memory)"]
        history = self.round_history if self.history_rounds < 0 else self.round_history[-(self.history_rounds * 2):]
        for rnd, summary in history:
            lines.append(f"  Round {rnd}: {summary}")
        return "\n".join(lines)

    def _turn_cli(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
            auctions=auctions if self.auction_enabled else None,
        )
        # Inject round history for statefulness
        history_section = self._build_round_history_section()
        prompt = f"{context}{history_section}\n\n{JSON_SCHEMA_INSTRUCTION}\n\nIt's your turn. Choose your action."

        turn_start = time.monotonic()
        for attempt in range(3):
            try:
                env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
                result = subprocess.run(
                    ["claude", "-p", prompt, "--model", self.cli_model],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self.turn_latencies.append(round(time.monotonic() - turn_start, 3))
                    parsed = _parse_json_response(result.stdout.strip())
                    self._record_round_history(parsed, round_num)
                    return parsed
                if attempt < 2:
                    time.sleep(1)
            except subprocess.TimeoutExpired:
                if attempt < 2:
                    time.sleep(1)
                else:
                    break

        self.turn_latencies.append(round(time.monotonic() - turn_start, 3))
        return {
            "action": "pass_turn",
            "give": {},
            "want": {},
            "offer_id": None,
            "message": "Agent failed to respond",
            "reasoning": "",
        }


class RandomAgent:
    """Baseline agent that makes random valid actions. Zero API calls, zero cost.

    Provides a lower bound for benchmark comparison — any useful model
    should consistently outperform random play.
    """

    def __init__(self, agent_idx, seed=None, auction_enabled=False):
        self.model_name = "random"
        self.agent_idx = agent_idx
        self.strategy_id = None
        self.strategy_prompt = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.turn_latencies = []
        self.round_history = []
        self.backend = "random"
        self.rng = stdlib_random.Random(seed)
        self.auction_enabled = auction_enabled

    @property
    def contestant_name(self):
        return "random"

    def get_state(self):
        return {
            "round_history": self.round_history,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "turn_latencies": self.turn_latencies,
            "rng_state": list(self.rng.getstate()),
        }

    def set_state(self, state):
        self.round_history = [tuple(x) for x in state.get("round_history", [])]
        self.turn_latencies = state.get("turn_latencies", [])
        if "rng_state" in state:
            rng_state = state["rng_state"]
            # Reconstruct tuple format for Random.setstate
            self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        start = time.monotonic()
        result = self._choose_action(inventory, target, order_book)
        self.turn_latencies.append(round(time.monotonic() - start, 6))
        return result

    def _choose_action(self, inventory, target, order_book):
        """Pick a random valid action weighted toward useful behavior.

        Strategy: 40% try accept, 35% post offer, 25% pass.
        Falls back to pass if no valid action is possible.
        """
        roll = self.rng.random()

        # 40% — try to accept an existing offer
        if roll < 0.40 and order_book:
            acceptable = [o for o in order_book if o["poster"] != self.agent_idx
                          and self._can_afford(inventory, o["want"])]
            if acceptable:
                offer = self.rng.choice(acceptable)
                return {
                    "action": "accept_offer",
                    "give": {},
                    "want": {},
                    "offer_id": offer["id"],
                    "message": "Random baseline: accepting available offer",
                    "reasoning": "random",
                }

        # 35% — post a random offer (give something we have, want something we need)
        if roll < 0.75:
            offer = self._random_offer(inventory, target)
            if offer:
                return offer

        # 25% — pass (or fallback)
        return {
            "action": "pass_turn",
            "give": {},
            "want": {},
            "offer_id": None,
            "message": "Random baseline: passing",
            "reasoning": "random",
        }

    def _can_afford(self, inventory, cost):
        """Check if inventory has enough items to pay cost."""
        for item, qty in cost.items():
            if inventory.get(item, 0) < int(qty):
                return False
        return True

    def _random_offer(self, inventory, target):
        """Generate a random but valid offer."""
        # Items we have and can give (qty > 0 and not needed for target)
        giveable = []
        for item, qty in inventory.items():
            if qty > 0:
                target_need = target.get(item, 0)
                surplus = qty - target_need
                if surplus > 0:
                    giveable.append((item, surplus))
                elif qty > 0 and self.rng.random() < 0.3:
                    # Sometimes offer items we need (suboptimal but adds noise)
                    giveable.append((item, qty))

        # Items we want (in our target but don't have enough)
        wantable = []
        for item, needed in target.items():
            if needed > 0 and inventory.get(item, 0) < needed:
                wantable.append((item, needed - inventory.get(item, 0)))

        if not giveable or not wantable:
            return None

        give_item, max_give = self.rng.choice(giveable)
        want_item, max_want = self.rng.choice(wantable)

        give_qty = self.rng.randint(1, max(1, max_give))
        want_qty = self.rng.randint(1, max(1, min(max_want, 4)))

        is_private = self.rng.random() < 0.2
        action = "private_offer" if is_private else "post_offer"

        result = {
            "action": action,
            "give": {give_item: give_qty},
            "want": {want_item: want_qty},
            "offer_id": None,
            "message": f"Random baseline: offering {give_item} for {want_item}",
            "reasoning": "random",
        }
        if is_private:
            # Pick a random other agent (we don't know num_agents, so just pick 0-11)
            result["target_agent"] = self.rng.randint(0, 11)

        return result
