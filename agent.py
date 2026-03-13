"""Agent wrapper for marketplace — supports Anthropic API (with key) and claude CLI (OAuth)."""

import json
import os
import re
import subprocess
import time

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
                 temperature: float = 1.0, auction_enabled: bool = False):
        self.model_name = model_name
        self.agent_idx = agent_idx
        self.strategy_id = strategy_id
        self.strategy_prompt = strategy_prompt
        self.temperature = temperature
        self.auction_enabled = auction_enabled
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.conversation_history = []  # stateful: accumulate turns within a match (API)
        self.round_history = []  # stateful: (round, action_summary) tuples (CLI)

        if backend == "auto":
            self.backend = "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
        else:
            self.backend = backend

        if self.backend == "api":
            import anthropic
            self.model = MODEL_MAP.get(model_name, model_name)
            self.client = anthropic.Anthropic()
        else:
            self.cli_model = CLI_MODEL_MAP.get(model_name, model_name)

    @property
    def contestant_name(self):
        """Return the name used for scoring — strategy_id in arena mode, model_name otherwise."""
        return self.strategy_id if self.strategy_id else self.model_name

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds, auctions=None):
        if self.backend == "api":
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

        for attempt in range(3):
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
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

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

        # Cap history to last 6 exchanges (3 rounds) to control token usage
        max_history_items = 18  # 6 exchanges × 3 messages each
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

    def _build_round_history_section(self):
        """Build a prompt section summarizing this agent's previous actions."""
        if not self.round_history:
            return ""
        lines = ["\n## Your Previous Actions (memory)"]
        for rnd, summary in self.round_history[-6:]:  # last 6 rounds max
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
                    parsed = _parse_json_response(result.stdout.strip())
                    # Record action for future rounds
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
                    return parsed
                if attempt < 2:
                    time.sleep(1)
            except subprocess.TimeoutExpired:
                if attempt < 2:
                    time.sleep(1)
                else:
                    break

        return {
            "action": "pass_turn",
            "give": {},
            "want": {},
            "offer_id": None,
            "message": "Agent failed to respond",
            "reasoning": "",
        }
