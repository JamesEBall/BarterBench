"""Agent wrapper — supports both Anthropic API (with key) and claude CLI (OAuth)."""

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

# CLI model names (what `claude --model` expects)
CLI_MODEL_MAP = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
}

VALID_ACTIONS = {"propose_trade", "accept_trade", "reject_trade", "end_negotiation"}

JSON_SCHEMA_INSTRUCTION = """
Respond with ONLY a JSON object (no other text) in this exact format:

For proposing a trade:
{"action": "propose_trade", "give": {"item": qty, ...}, "receive": {"item": qty, ...}, "message": "your message"}

For accepting a pending trade:
{"action": "accept_trade", "message": "your message"}

For rejecting a pending trade:
{"action": "reject_trade", "message": "your message"}

For ending negotiation:
{"action": "end_negotiation", "message": "your message"}
""".strip()

BARTER_TOOLS = [
    {
        "name": "propose_trade",
        "description": "Propose a trade to the other trader.",
        "input_schema": {
            "type": "object",
            "properties": {
                "give": {"type": "object", "description": "Items you offer to give, e.g. {\"apples\": 2}"},
                "receive": {"type": "object", "description": "Items you want in return, e.g. {\"oranges\": 3}"},
                "message": {"type": "string", "description": "Message to the other trader"},
            },
            "required": ["give", "receive", "message"],
        },
    },
    {
        "name": "accept_trade",
        "description": "Accept the current pending trade proposal.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to the other trader"}},
            "required": ["message"],
        },
    },
    {
        "name": "reject_trade",
        "description": "Reject the current pending trade proposal.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Reason for rejection"}},
            "required": ["message"],
        },
    },
    {
        "name": "end_negotiation",
        "description": "End the negotiation. No more trades after this.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Final message"}},
            "required": ["message"],
        },
    },
]


def _build_context(role, inventory, valuations, history, pending_trade):
    current_value = sum(qty * valuations.get(item, 0) for item, qty in inventory.items())

    lines = [
        f"You are {role} in a multi-round bartering negotiation.",
        "",
        "## Your Private Information (DO NOT reveal exact numbers)",
        f"- Your inventory: {json.dumps(inventory)}",
        f"- Your valuations per item: {json.dumps(valuations)}",
        f"- Your current portfolio value: {current_value}",
        "",
        "## Rules",
        "1. Propose trades, accept/reject proposals, or end the negotiation",
        "2. You can only offer items you currently have (quantity > 0)",
        "3. Your valuations are PRIVATE — be strategic about what you reveal",
        "4. Your goal: MAXIMIZE your total portfolio value through smart trades",
        "5. A trade only executes when both parties agree",
        "6. Think carefully about what the other trader might value",
    ]

    if history:
        lines.append("")
        lines.append("## Negotiation History")
        for entry in history:
            role_name = entry["role"]
            action = entry["action"]
            msg = entry.get("message", "")
            if action == "propose_trade":
                give = entry.get("give", {})
                receive = entry.get("receive", {})
                invalid = " [INVALID - items not available]" if entry.get("invalid") else ""
                lines.append(f"  {role_name}: \"{msg}\"")
                lines.append(f"    → Proposed: give {json.dumps(give)}, want {json.dumps(receive)}{invalid}")
            elif action == "accept_trade":
                lines.append(f"  {role_name}: \"{msg}\"")
                lines.append(f"    → ACCEPTED the trade (inventories updated)")
            elif action == "reject_trade":
                lines.append(f"  {role_name}: \"{msg}\"")
                lines.append(f"    → REJECTED the trade")
            elif action == "end_negotiation":
                lines.append(f"  {role_name}: \"{msg}\"")
                lines.append(f"    → Ended negotiation")

    if pending_trade:
        lines.append("")
        lines.append("## Pending Trade Proposal (awaiting your response)")
        lines.append(f"  From: {pending_trade['proposer_role']}")
        lines.append(f"  They give you: {json.dumps(pending_trade['receive'])}")
        lines.append(f"  They want from you: {json.dumps(pending_trade['give'])}")
        lines.append("  You may: accept_trade, reject_trade, or propose_trade (counter-offer)")
    else:
        lines.append("")
        lines.append("## No pending proposal — you may propose_trade or end_negotiation")

    return "\n".join(lines)


def _parse_json_response(text):
    """Extract JSON action from model text output."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            action = data.get("action", "end_negotiation")
            if action not in VALID_ACTIONS:
                action = "end_negotiation"
            return {
                "action": action,
                "give": data.get("give", {}),
                "receive": data.get("receive", {}),
                "message": data.get("message", ""),
                "reasoning": text[:text.find(json_match.group())].strip(),
            }
        except json.JSONDecodeError:
            pass
    # Fallback: couldn't parse
    return {
        "action": "end_negotiation",
        "give": {},
        "receive": {},
        "message": "Could not determine action",
        "reasoning": text,
        "parse_error": True,
    }


class BarterAgent:
    """Bartering agent that calls Claude via API or CLI."""

    def __init__(self, model_name: str, backend: str = "auto"):
        self.model_name = model_name
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

    def take_turn(self, role, inventory, valuations, history, pending_trade):
        if self.backend == "api":
            return self._turn_api(role, inventory, valuations, history, pending_trade)
        else:
            return self._turn_cli(role, inventory, valuations, history, pending_trade)

    def _turn_api(self, role, inventory, valuations, history, pending_trade):
        import anthropic
        context = _build_context(role, inventory, valuations, history, pending_trade)

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=context,
                    messages=[{"role": "user", "content": "It's your turn. Choose your action."}],
                    tools=BARTER_TOOLS,
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
        tool_name = "end_negotiation"
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
            "receive": tool_input.get("receive", {}),
            "message": tool_input.get("message", ""),
            "reasoning": reasoning,
        }

    def _turn_cli(self, role, inventory, valuations, history, pending_trade):
        context = _build_context(role, inventory, valuations, history, pending_trade)
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
            "action": "end_negotiation",
            "give": {},
            "receive": {},
            "message": "Agent failed to respond",
            "reasoning": "",
        }
