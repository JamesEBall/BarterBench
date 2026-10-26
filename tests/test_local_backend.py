"""End-to-end test of the local backend against a mock OpenAI-compatible server.

Exercises: schema-constrained request shape, parsing, reliability counters,
structured-output degradation on HTTP 400, parse-error accounting, per-turn seeds,
and a full engine match driven through the real OpenAI client.
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from barterbench.agents import LLMAgent, action_schema, _parse_json_response, build_agent
from barterbench.engine import MarketEngine

openai = pytest.importorskip("openai")


class MockVLLM(BaseHTTPRequestHandler):
    """Minimal /v1/models + /v1/chat/completions. Behaviour is configured per server."""

    requests = []
    behaviour = {"reject_json_schema": False, "reply": None, "finish_reason": "stop"}

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        MockVLLM.requests.append(req)
        if MockVLLM.behaviour["reject_json_schema"] and "response_format" in req:
            self._send(400, {"error": {"message": "response_format not supported"}})
            return
        reply = MockVLLM.behaviour["reply"]
        if callable(reply):
            reply = reply(req)
        self._send(200, {
            "id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
            "choices": [{"index": 0, "finish_reason": MockVLLM.behaviour["finish_reason"],
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })


@pytest.fixture
def server():
    MockVLLM.requests = []
    MockVLLM.behaviour = {"reject_json_schema": False, "finish_reason": "stop",
                          "reply": json.dumps({"reasoning": "pass", "action": "pass_turn", "message": "hi"})}
    httpd = HTTPServer(("127.0.0.1", 0), MockVLLM)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield base
    httpd.shutdown()


def _agent(base, **kw):
    import barterbench.agents as agent_mod
    agent_mod._SERVED_CACHE.clear()
    return LLMAgent("mock", 0, base_url=base, served_name="whatever", seed=42, **kw)


def test_request_carries_schema_seed_and_single_served_model(server):
    a = _agent(server)
    out = a.take_turn({"gold": 1}, {"wheat": 1}, [], [], 0, 5, messages=[])
    assert out["action"] == "pass_turn" and out["message"] == "hi" and out["reasoning"] == "pass"
    req = MockVLLM.requests[-1]
    assert req["model"] == "mock-model", "single served model should be auto-discovered"
    assert req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["schema"] == action_schema(False)
    assert "seed" in req and isinstance(req["seed"], int)
    assert req["messages"][0]["role"] == "system" and "Trader 0" in req["messages"][0]["content"]
    assert a.total_input_tokens == 100 and a.total_output_tokens == 20
    assert a.reliability()["parse_error_rate"] == 0.0


def test_per_turn_seed_changes_with_round_and_agent(server):
    a = _agent(server)
    a.take_turn({}, {}, [], [], 0, 5)
    a.take_turn({}, {}, [], [], 1, 5)
    s0, s1 = MockVLLM.requests[-2]["seed"], MockVLLM.requests[-1]["seed"]
    assert s0 != s1
    b = _agent(server); b.agent_idx = 1
    b.take_turn({}, {}, [], [], 0, 5)
    assert MockVLLM.requests[-1]["seed"] != s0


def test_degrades_to_guided_json_when_schema_rejected(server):
    MockVLLM.behaviour["reject_json_schema"] = True
    a = _agent(server)
    out = a.take_turn({}, {}, [], [], 0, 5)
    assert out["action"] == "pass_turn"
    assert a.structured_mode == "guided_json"
    ok = [r for r in MockVLLM.requests if "response_format" not in r]
    assert ok and ok[-1]["guided_json"] == action_schema(False)
    # Sticky: next turn goes straight to guided_json
    n = len(MockVLLM.requests)
    a.take_turn({}, {}, [], [], 1, 5)
    assert len(MockVLLM.requests) == n + 1 and "response_format" not in MockVLLM.requests[-1]


def test_parse_error_is_counted_not_hidden(server):
    MockVLLM.behaviour["reply"] = "I think I will trade some gold maybe"
    MockVLLM.behaviour["finish_reason"] = "length"
    a = _agent(server)
    out = a.take_turn({}, {}, [], [], 0, 5)
    assert out["action"] == "pass_turn" and out["parse_error"] is True and out["truncated"] is True
    r = a.reliability()
    assert r["parse_error_rate"] == 1.0 and r["truncation_rate"] == 1.0
    assert "could not be parsed" in a.round_history[-1][1]


def test_text_with_surrounding_prose_still_parses():
    text = 'Thinking... {"action": "post_offer", "give": {"gold": 1}, "want": {"wheat": 2}, "message": "m"} done'
    out = _parse_json_response(text)
    assert out["action"] == "post_offer" and out["want"] == {"wheat": 2} and "Thinking" in out["reasoning"]
    nested = '{"reasoning": "x", "action": "private_offer", "give": {"a": 1}, "want": {"b": {"c": 1}}, "target_agent": 2, "message": ""}'
    assert _parse_json_response(nested)["action"] == "private_offer"
    assert _parse_json_response("no json here")["parse_error"] is True
    assert _parse_json_response('{"action": "dance", "message": ""}')["unrecognized_action"] == "dance"


def test_full_match_through_real_client(server):
    """Two mock LLM agents (always post gold for wheat / accept #1) vs two heuristics."""
    def reply(req):
        system = req["messages"][0]["content"]
        if system.startswith("You are Trader 0"):
            return json.dumps({"reasoning": "r", "action": "post_offer", "give": {"gold": 1},
                               "want": {"wheat": 1}, "message": "gold for wheat"})
        if "[#1]" in system:
            return json.dumps({"reasoning": "r", "action": "accept_offer", "offer_id": 1, "message": "taking it"})
        return json.dumps({"reasoning": "r", "action": "pass_turn", "message": "waiting"})

    MockVLLM.behaviour["reply"] = reply
    import barterbench.agents as agent_mod
    agent_mod._SERVED_CACHE.clear()
    endpoints = {"mock": {"base_url": server, "model": "mock-model"}}
    sc = {"name": "mock", "max_rounds": 3, "agents": [
        {"role": "T0", "inventory": {"gold": 3, "wheat": 0}, "target": {"wheat": 2}},
        {"role": "T1", "inventory": {"gold": 0, "wheat": 3}, "target": {"gold": 2}},
        {"role": "T2", "inventory": {"gold": 1, "wheat": 1}, "target": {"gold": 1}},
        {"role": "T3", "inventory": {"gold": 1, "wheat": 1}, "target": {"wheat": 1}},
    ]}
    agents = [build_agent("mock" if i < 2 else "heuristic", i, sc["agents"][i], False, seed=i,
                          num_agents=4, endpoints=endpoints) for i in range(4)]
    eng = MarketEngine(sc, agents, seed=11)
    res = eng.run()
    assert res["num_turns"] == 12
    assert eng.total_supply() == {"gold": 5, "wheat": 5}
    llm_turns = [h for h in res["history"] if h["model"] == "mock"]
    assert llm_turns and all(not h.get("parse_error") for h in llm_turns)
    # Messages reached the public record
    assert any(m["message"] for m in res["public_messages"]) or any(t.get("message") for t in res["trades"])
    assert agents[0].resolved_model.endswith("::mock-model")
