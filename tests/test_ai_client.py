"""Unit test AIClient với endpoint giả lập (httpx.MockTransport) – không gọi mạng thật."""
from __future__ import annotations

import json

import httpx
import pytest

from ai_client import Action, AIClient, AIDecision, extract_json, parse_decision

SNAPSHOT = {"symbol": "EURUSD", "price": {"bid": 1.1, "ask": 1.1001}}
GOOD = {"action": "BUY", "confidence": 0.8, "stop_loss": 1.098, "take_profit": 1.104, "reasoning": "uptrend"}


def openai_body(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def make_client(handler, **kw) -> AIClient:
    kw.setdefault("max_retries", 3)
    return AIClient(url="https://ai.test/v1/chat/completions", api_key="k", model="m",
                    backoff_base=0.001, transport=httpx.MockTransport(handler), **kw)


# ---------------------------------------------------------------- parsing
def test_parse_plain_json():
    d = parse_decision(json.dumps(GOOD))
    assert d.action is Action.BUY and d.confidence == 0.8 and d.stop_loss == 1.098


def test_parse_code_fence_and_noise():
    text = "Here is my answer:\n```json\n" + json.dumps(GOOD) + "\n```\nGood luck"
    assert parse_decision(text).action is Action.BUY


def test_parse_embedded_braces():
    assert extract_json('blah {"action":"HOLD","confidence":0.1} trailing')["action"] == "HOLD"


def test_parse_normalizes_fields():
    d = parse_decision('{"action":"sell","confidence":75,"stop_loss":"1.2","take_profit":0,"reasoning":null}')
    assert d.action is Action.SELL
    assert d.confidence == pytest.approx(0.75)
    assert d.stop_loss == 1.2 and d.take_profit is None and d.reasoning == ""


def test_parse_nested_decision_key():
    assert parse_decision(json.dumps({"decision": GOOD})).action is Action.BUY


@pytest.mark.parametrize("bad", [
    "", "not json", '{"action":"YOLO","confidence":0.5}', '{"action":"BUY"}',
    '{"action":"BUY","confidence":1.5e3}', "[1,2,3]",
])
def test_parse_invalid_raises(bad):
    with pytest.raises(Exception):
        parse_decision(bad)


def test_reasoning_truncated():
    d = AIDecision(action="HOLD", confidence=0.1, reasoning="x" * 5000)
    assert len(d.reasoning) == 500


# ---------------------------------------------------------------- HTTP behaviour
async def test_success_openai_style():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json=openai_body(json.dumps(GOOD)))

    client = make_client(handler)
    res = await client.decide(SNAPSHOT)
    await client.aclose()
    assert res.ok and res.attempts == 1 and res.decision.action is Action.BUY
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0]["role"] == "system"
    assert json.loads(seen["body"]["messages"][1]["content"]) == SNAPSHOT
    assert client.consecutive_failures == 0


async def test_retry_on_5xx_then_success():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=openai_body(json.dumps(GOOD)))

    client = make_client(handler)
    res = await client.decide(SNAPSHOT)
    assert res.ok and res.attempts == 3 and calls["n"] == 3


async def test_all_retries_fail_falls_back_to_hold():
    client = make_client(lambda req: httpx.Response(500))
    res = await client.decide(SNAPSHOT)
    assert not res.ok and res.decision.action is Action.HOLD and res.decision.confidence == 0
    assert res.attempts == 3 and client.consecutive_failures == 1
    await client.decide(SNAPSHOT)
    assert client.consecutive_failures == 2


async def test_4xx_no_retry():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, text="bad key")

    res = await make_client(handler).decide(SNAPSHOT)
    assert calls["n"] == 1 and res.decision.action is Action.HOLD and "401" in res.error


async def test_timeout_falls_back_to_hold():
    def handler(req):
        raise httpx.ReadTimeout("timeout", request=req)

    res = await make_client(handler, max_retries=2).decide(SNAPSHOT)
    assert not res.ok and res.decision.action is Action.HOLD and "Timeout" in res.error


async def test_invalid_output_retried_then_hold():
    res = await make_client(lambda req: httpx.Response(200, json=openai_body("I think buy!"))).decide(SNAPSHOT)
    assert not res.ok and res.decision.action is Action.HOLD and res.raw_text == "I think buy!"


async def test_bad_response_shape_hold():
    res = await make_client(lambda req: httpx.Response(200, json={"foo": 1})).decide(SNAPSHOT)
    assert res.decision.action is Action.HOLD


async def test_success_after_consecutive_failure_resets_counter():
    state = {"fail": True}

    def handler(req):
        return httpx.Response(500) if state["fail"] else httpx.Response(200, json=openai_body(json.dumps(GOOD)))

    client = make_client(handler, max_retries=1)
    await client.decide(SNAPSHOT)
    assert client.consecutive_failures == 1
    state["fail"] = False
    await client.decide(SNAPSHOT)
    assert client.consecutive_failures == 0


async def test_custom_style():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"decision": GOOD, "meta": {"latency": 1}})

    res = await make_client(handler, style="custom").decide(SNAPSHOT)
    assert res.ok and res.decision.action is Action.BUY
    assert seen["body"]["snapshot"] == SNAPSHOT and "system" in seen["body"]


async def test_json_mode_off():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=openai_body(json.dumps(GOOD)))

    await make_client(handler, json_mode=False).decide(SNAPSHOT)
    assert "response_format" not in seen["body"]
