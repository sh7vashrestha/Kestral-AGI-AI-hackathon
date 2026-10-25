"""Thin sync + async client for llama.cpp ``llama-server`` (OpenAI-compatible API).

Shared by the planner (text) and the matcher (image + text).

    client = LLMClient()
    result = client.chat_json(messages, MatchResult, max_tokens=200)
    result = await client.achat_json(messages, MatchResult)
    msgs = [text_message("system", "..."), image_message(jpeg_bytes, "Does this match?")]

Structured output: ``response_format = json_schema`` (strict) first; if the server answers
HTTP 400 we fall back to ``{"type": "json_object", "schema": ...}``. The reply is always
validated against the given pydantic model.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings, get_settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY_STATUS = {429, 500, 502, 503, 504}


# ---------- exceptions ----------------------------------------------------

class LLMError(Exception):
    """Base class for every LLM client failure."""


class LLMTimeout(LLMError):
    """The server did not answer within the timeout (after retries)."""


class LLMUnavailable(LLMError):
    """Cannot connect / server is not ready (after retries)."""


class LLMInvalidResponse(LLMError):
    """The server answered but the reply is empty, not JSON, or fails validation."""


class _BadRequest(Exception):
    """Internal: HTTP 400 (used to trigger the json_object fallback)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


# ---------- message helpers -------------------------------------------------

def text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


def image_message(jpeg: bytes, text: str, role: str = "user", mime: str = "image/jpeg") -> dict[str, Any]:
    """OpenAI-style multimodal message: one text part + one base64 data-URI image part."""
    if not jpeg:
        raise ValueError("image_message: empty image bytes")
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {
        "role": role,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }


# ---------- response parsing --------------------------------------------------

def extract_text(data: dict[str, Any]) -> str:
    """Return assistant text from a chat completion; falls back to ``reasoning_content``."""
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMInvalidResponse(f"unexpected response shape: {str(data)[:200]}") from exc
    content = msg.get("content")
    if isinstance(content, list):  # list of parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if isinstance(content, str) and content.strip():
        return content
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        log.debug("empty content, using reasoning_content")
        return reasoning
    raise LLMInvalidResponse("model returned empty content")


def extract_json(text: str) -> Any:
    """Parse JSON from model text. Tolerates code fences and prose around the object;
    if several objects appear (e.g. reasoning text) the last top-level one wins."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    found: Any = None
    i = 0
    while True:
        i = s.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            i += 1
            continue
        found = obj
        i = end
    if found is None:
        raise LLMInvalidResponse(f"no JSON object in reply: {text[:200]!r}")
    return found


def parse_model(text: str, model_cls: type[T]) -> T:
    obj = extract_json(text)
    try:
        return model_cls.model_validate(obj)
    except ValidationError as exc:
        raise LLMInvalidResponse(f"reply does not match {model_cls.__name__}: {exc.errors()[:3]}") from exc


# ---------- client ------------------------------------------------------------

class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        retries: Optional[int] = None,
        model: str = "local",
        retry_backoff_s: float = 0.5,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        s = settings or get_settings()
        self.base_url = (base_url or s.llm_base_url).rstrip("/")
        self.timeout_s = s.llm_timeout_s if timeout_s is None else timeout_s
        self.retries = s.llm_retries if retries is None else retries
        self.max_tokens = s.llm_max_tokens
        self.temperature = s.llm_temperature
        self.model = model
        self.retry_backoff_s = retry_backoff_s
        self._transport = transport
        self._async_transport = async_transport

    # -- urls ----
    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def health_url(self) -> str:
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return f"{root.rstrip('/')}/health"

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout_s, connect=min(5.0, self.timeout_s))

    # -- payload ----
    def _payload(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[dict[str, Any]],
        max_tokens: Optional[int],
        temperature: Optional[float],
        extra: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        p: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if response_format is not None:
            p["response_format"] = response_format
        if extra:
            p.update(extra)
        return p

    @staticmethod
    def response_formats(model_cls: type[BaseModel]) -> list[dict[str, Any]]:
        """Structured-output modes in order of preference."""
        schema = model_cls.model_json_schema()
        return [
            {"type": "json_schema",
             "json_schema": {"name": model_cls.__name__, "schema": schema, "strict": True}},
            {"type": "json_object", "schema": schema},
        ]

    # -- error mapping (shared by sync/async) ----
    def _handle_status(self, r: httpx.Response) -> Optional[LLMError]:
        """None -> ok. LLMError -> retryable failure. Raises for fatal ones."""
        if r.status_code < 400:
            return None
        detail = r.text[:300]
        if r.status_code == 400:
            raise _BadRequest(detail)
        if r.status_code in _RETRY_STATUS:
            cls = LLMUnavailable if r.status_code == 503 else LLMError
            return cls(f"HTTP {r.status_code}: {detail}")
        raise LLMError(f"HTTP {r.status_code}: {detail}")

    def _json_body(self, r: httpx.Response) -> dict[str, Any]:
        try:
            body = r.json()
        except ValueError as exc:
            raise LLMInvalidResponse(f"non-JSON HTTP body: {r.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise LLMInvalidResponse("unexpected response body")
        return body

    # -- health ----
    def health(self) -> bool:
        try:
            with httpx.Client(timeout=httpx.Timeout(3.0), transport=self._transport) as c:
                r = c.get(self.health_url)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def ahealth(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0), transport=self._async_transport) as c:
                r = await c.get(self.health_url)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # -- raw post with retries ----
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: LLMError = LLMError("no attempt made")
        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(self.retry_backoff_s * 2 ** (attempt - 1))
            try:
                with httpx.Client(timeout=self._timeout(), transport=self._transport) as c:
                    r = c.post(self.chat_url, json=payload)
            except httpx.TimeoutException as exc:
                last = LLMTimeout(f"LLM timed out after {self.timeout_s}s")
                last.__cause__ = exc
            except httpx.TransportError as exc:
                last = LLMUnavailable(f"cannot reach llama-server at {self.base_url}: {exc}")
                last.__cause__ = exc
            else:
                err = self._handle_status(r)
                if err is None:
                    return self._json_body(r)
                last = err
            log.warning("LLM attempt %d/%d failed: %s", attempt + 1, self.retries + 1, last)
        raise last

    async def _apost(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: LLMError = LLMError("no attempt made")
        for attempt in range(self.retries + 1):
            if attempt:
                await asyncio.sleep(self.retry_backoff_s * 2 ** (attempt - 1))
            try:
                async with httpx.AsyncClient(timeout=self._timeout(), transport=self._async_transport) as c:
                    r = await c.post(self.chat_url, json=payload)
            except httpx.TimeoutException as exc:
                last = LLMTimeout(f"LLM timed out after {self.timeout_s}s")
                last.__cause__ = exc
            except httpx.TransportError as exc:
                last = LLMUnavailable(f"cannot reach llama-server at {self.base_url}: {exc}")
                last.__cause__ = exc
            else:
                err = self._handle_status(r)
                if err is None:
                    return self._json_body(r)
                last = err
            log.warning("LLM attempt %d/%d failed: %s", attempt + 1, self.retries + 1, last)
        raise last

    # -- public: plain chat ----
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        try:
            data = self._post(self._payload(messages, None, max_tokens, temperature, extra))
        except _BadRequest as exc:
            raise LLMError(f"HTTP 400: {exc.detail}") from exc
        return extract_text(data)

    async def achat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> str:
        try:
            data = await self._apost(self._payload(messages, None, max_tokens, temperature, extra))
        except _BadRequest as exc:
            raise LLMError(f"HTTP 400: {exc.detail}") from exc
        return extract_text(data)

    # -- public: structured chat ----
    def chat_json(
        self,
        messages: list[dict[str, Any]],
        model_cls: type[T],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> T:
        """Chat and return a validated ``model_cls`` instance.

        Raises LLMTimeout / LLMUnavailable / LLMInvalidResponse / LLMError."""
        formats = self.response_formats(model_cls)
        last: LLMError = LLMError("no attempt made")
        for fi, fmt in enumerate(formats):
            for attempt in range(self.retries + 1):
                try:
                    data = self._post(self._payload(messages, fmt, max_tokens, temperature, extra))
                    return parse_model(extract_text(data), model_cls)
                except _BadRequest as exc:
                    last = LLMError(f"HTTP 400 with {fmt['type']}: {exc.detail}")
                    log.warning("%s; %s", last, "falling back" if fi + 1 < len(formats) else "giving up")
                    break  # next format
                except LLMInvalidResponse as exc:
                    last = exc
                    log.warning("invalid LLM reply (attempt %d): %s", attempt + 1, exc)
                    if attempt < self.retries:
                        time.sleep(self.retry_backoff_s)
                    continue
            else:
                raise last  # invalid replies exhausted retries
        raise last

    async def achat_json(
        self,
        messages: list[dict[str, Any]],
        model_cls: type[T],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> T:
        formats = self.response_formats(model_cls)
        last: LLMError = LLMError("no attempt made")
        for fi, fmt in enumerate(formats):
            for attempt in range(self.retries + 1):
                try:
                    data = await self._apost(self._payload(messages, fmt, max_tokens, temperature, extra))
                    return parse_model(extract_text(data), model_cls)
                except _BadRequest as exc:
                    last = LLMError(f"HTTP 400 with {fmt['type']}: {exc.detail}")
                    log.warning("%s; %s", last, "falling back" if fi + 1 < len(formats) else "giving up")
                    break
                except LLMInvalidResponse as exc:
                    last = exc
                    log.warning("invalid LLM reply (attempt %d): %s", attempt + 1, exc)
                    if attempt < self.retries:
                        await asyncio.sleep(self.retry_backoff_s)
                    continue
            else:
                raise last
        raise last
