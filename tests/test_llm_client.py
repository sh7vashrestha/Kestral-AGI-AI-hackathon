import base64
import json

import httpx
import pytest

from llm_client import (LLMClient, LLMError, LLMInvalidResponse, LLMTimeout, LLMUnavailable,
                        extract_json, image_message)
from models import MatchResult

GOOD = {"is_human": True, "matches": False, "match_confidence": 0.2, "reason": "no"}


def _resp(content="", reasoning=None, status=200):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return httpx.Response(status, json={"choices": [{"message": msg}]})


def _client(handler, **kw):
    kw.setdefault("retry_backoff_s", 0.0)
    return LLMClient("http://x:8080/v1", transport=httpx.MockTransport(handler),
                     async_transport=httpx.MockTransport(handler), **kw)


def test_chat_json_sends_json_schema_and_parses():
    seen = []

    def h(req):
        seen.append(json.loads(req.content))
        return _resp(json.dumps(GOOD))

    r = _client(h).chat_json([{"role": "user", "content": "hi"}], MatchResult, max_tokens=99)
    assert r == MatchResult(**GOOD)
    body = seen[0]
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == MatchResult.model_json_schema()
    assert body["max_tokens"] == 99 and body["stream"] is False


def test_fallback_to_json_object_on_400():
    fmts = []

    def h(req):
        fmt = json.loads(req.content)["response_format"]
        fmts.append(fmt["type"])
        if fmt["type"] == "json_schema":
            return httpx.Response(400, json={"error": "bad"})
        assert fmt["schema"]
        return _resp(json.dumps(GOOD))

    assert _client(h).chat_json([], MatchResult).is_human is True
    assert fmts == ["json_schema", "json_object"]


def test_400_on_both_raises_llmerror():
    c = _client(lambda r: httpx.Response(400, text="nope"))
    with pytest.raises(LLMError, match="400"):
        c.chat_json([], MatchResult)


def test_reasoning_content_fallback():
    c = _client(lambda r: _resp("", reasoning="thinking... " + json.dumps(GOOD)))
    assert c.chat_json([], MatchResult).matches is False


def test_empty_reply_raises_after_retries():
    calls = []

    def h(req):
        calls.append(1)
        return _resp("", reasoning="")

    with pytest.raises(LLMInvalidResponse):
        _client(h, retries=2).chat_json([], MatchResult)
    assert len(calls) == 3


def test_validation_failure_retries_then_succeeds():
    replies = iter([{"is_human": "maybe"}, GOOD])
    c = _client(lambda r: _resp(json.dumps(next(replies))), retries=2)
    assert c.chat_json([], MatchResult).reason == "no"


def test_validation_range_enforced():
    bad = dict(GOOD, match_confidence=1.7)
    with pytest.raises(LLMInvalidResponse):
        _client(lambda r: _resp(json.dumps(bad)), retries=0).chat_json([], MatchResult)


def test_timeout_retries_then_llmtimeout():
    calls = []

    def h(req):
        calls.append(1)
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeout):
        _client(h, retries=2).chat_json([], MatchResult)
    assert len(calls) == 3


def test_connect_error_is_unavailable():
    def h(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailable):
        _client(h, retries=1).chat([])


def test_503_retried_and_recovers():
    codes = iter([503, 200])
    c = _client(lambda r: _resp(json.dumps(GOOD)) if next(codes) == 200 else httpx.Response(503), retries=1)
    assert c.chat_json([], MatchResult).is_human


def test_503_exhausted_is_unavailable():
    with pytest.raises(LLMUnavailable):
        _client(lambda r: httpx.Response(503, text="loading"), retries=1).chat([])


def test_500_fatal_status_404():
    with pytest.raises(LLMError):
        _client(lambda r: httpx.Response(404, text="x"), retries=0).chat([])


def test_health_url_and_status():
    urls = []

    def h(req):
        urls.append(str(req.url))
        return httpx.Response(200, json={"status": "ok"})

    assert _client(h).health() is True
    assert urls == ["http://x:8080/health"]
    assert _client(lambda r: httpx.Response(503)).health() is False

    def boom(req):
        raise httpx.ConnectError("no")

    assert _client(boom).health() is False


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 1} hope it helps') == {"a": 1}
    assert extract_json('draft {"a": 1} final {"a": 2, "b": {"c": 3}}') == {"a": 2, "b": {"c": 3}}
    with pytest.raises(LLMInvalidResponse):
        extract_json("no json here")


def test_image_message():
    m = image_message(b"\xff\xd8abc", "describe")
    assert m["role"] == "user"
    url = m["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8abc"
    with pytest.raises(ValueError):
        image_message(b"", "x")


@pytest.mark.asyncio
async def test_async_chat_json_and_health():
    c = _client(lambda r: _resp(json.dumps(GOOD)) if r.method == "POST" else httpx.Response(200))
    assert (await c.achat_json([], MatchResult)).is_human
    assert await c.ahealth() is True


@pytest.mark.asyncio
async def test_async_timeout():
    def h(req):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeout):
        await _client(h, retries=1).achat_json([], MatchResult)
