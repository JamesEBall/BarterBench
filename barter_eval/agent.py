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

VALID_ACTIONS = {"post_offer", "accept_offer", "pass_turn"}

JSON_SCHEMA_INSTRUCTION = """
Respond with ONLY a JSON object (no other text) in this exact format:

To post a new offer on the order book:
{"action": "post_offer", "give": {"item": qty, ...}, "want": {"item": qty, ...}, "message": "your message"}

To accept an existing offer from the order book (use the offer_id):
{"action": "accept_offer", "offer_id": 123, "message": "your message"}

To pass your turn (do nothing this round):
{"action": "pass_turn", "message": "your message"}
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


def _build_marketplace_context(agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds, strategy_prompt=None):
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
        "Acquire items to match your TARGET inventory. You are competing against other traders —",
        "some items are SCARCE (demand exceeds supply), so not everyone can reach their goal.",
        "Trade aggressively to secure what you need before others do.",
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
        "1. You can POST an offer (give items you have, request items you need)",
        "2. You can ACCEPT an existing offer from the order book",
        "3. You can PASS if no good trades are available",
        "4. Trades execute immediately when accepted — both inventories update",
        "5. You can only offer items you currently have (quantity > 0)",
        "6. You cannot accept your own offers",
        "7. Be strategic: scarce items have leverage — don't give them away cheaply",
        "8. Think about multi-hop trades: trade what you have for intermediary items to reach your goal",
    ])

    if order_book:
        lines.append("")
        lines.append("## Order Book (open offers you can accept)")
        for offer in order_book:
            if offer["poster"] == agent_idx:
                lines.append(f"  [#{offer['id']}] YOUR OFFER — give {json.dumps(offer['give'])}, want {json.dumps(offer['want'])}")
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
            return {
                "action": action,
                "give": data.get("give", {}),
                "want": data.get("want", {}),
                "offer_id": data.get("offer_id"),
                "message": data.get("message", ""),
                "reasoning": text[:text.find(json_match.group())].strip(),
            }
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
                 strategy_id: str = None, strategy_prompt: str = None):
        self.model_name = model_name
        self.agent_idx = agent_idx
        self.strategy_id = strategy_id
        self.strategy_prompt = strategy_prompt
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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

    def take_turn(self, inventory, target, order_book, recent_trades, round_num, max_rounds):
        if self.backend == "api":
            return self._turn_api(inventory, target, order_book, recent_trades, round_num, max_rounds)
        else:
            return self._turn_cli(inventory, target, order_book, recent_trades, round_num, max_rounds)

    def _turn_api(self, inventory, target, order_book, recent_trades, round_num, max_rounds):
        import anthropic
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
        )

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=context,
                    messages=[{"role": "user", "content": "It's your turn. Choose your action."}],
                    tools=MARKETPLACE_TOOLS,
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

        for block in response.content:
            if block.type == "text":
                reasoning = block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

        return {
            "action": tool_name,
            "give": tool_input.get("give", {}),
            "want": tool_input.get("want", {}),
            "offer_id": tool_input.get("offer_id"),
            "message": tool_input.get("message", ""),
            "reasoning": reasoning,
        }

    def _turn_cli(self, inventory, target, order_book, recent_trades, round_num, max_rounds):
        context = _build_marketplace_context(
            self.agent_idx, inventory, target, order_book, recent_trades, round_num, max_rounds,
            strategy_prompt=self.strategy_prompt,
        )
        prompt = f"{context}\n\n{JSON_SCHEMA_INSTRUCTION}\n\nIt's your turn. Choose your action."

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
                    return _parse_json_response(result.stdout.strip())
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
