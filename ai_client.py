"""Client bất đồng bộ gọi AI endpoint và chuẩn hoá quyết định giao dịch.

Hỗ trợ 2 kiểu endpoint (`AI_API_STYLE`):
- ``openai``: OpenAI-compatible ``/v1/chat/completions`` (OpenAI, OpenRouter, vLLM, Ollama, LM Studio...).
- ``custom``: POST ``{"system": ..., "snapshot": {...}}`` và nhận lại JSON quyết định
  (ở top-level hoặc trong khoá ``"decision"``).

Mọi lỗi (timeout, HTTP lỗi, JSON sai, schema sai) đều fallback về ``HOLD`` an toàn.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# Hết hạn mức / quá tải: thử lại ngay cùng model hầu như vô ích -> chuyển sang model dự phòng
SWITCH_MODEL_STATUS = {429, 503}


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class AIDecision(BaseModel):
    """Schema cố định mà AI phải trả về."""

    model_config = ConfigDict(extra="ignore")

    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    reasoning: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _norm_action(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_confidence(cls, v: Any) -> Any:
        # Chấp nhận "0.8", 80 (phần trăm) -> 0.8
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        return f / 100.0 if 1.0 < f <= 100.0 else f

    @field_validator("stop_loss", "take_profit", mode="before")
    @classmethod
    def _norm_price(cls, v: Any) -> Any:
        if v in (None, "", 0, "0", "null", "None"):
            return None
        return v

    @field_validator("reasoning", mode="before")
    @classmethod
    def _norm_reasoning(cls, v: Any) -> str:
        return ("" if v is None else str(v)).strip()[:500]

    @classmethod
    def hold(cls, reason: str) -> "AIDecision":
        return cls(action=Action.HOLD, confidence=0.0, reasoning=reason)


@dataclass
class AIResult:
    decision: AIDecision
    ok: bool
    attempts: int
    latency_ms: int
    raw_text: str | None = None
    error: str | None = None
    request: dict | None = None

    @property
    def is_fallback(self) -> bool:
        return not self.ok


SYSTEM_PROMPT_TEMPLATE = """You are a disciplined, risk-averse trading decision engine for a MetaTrader 5 bot.
You receive a JSON market snapshot (multi-timeframe indicators, current bid/ask/spread, open positions
on the symbol, account context and risk rules). Decide ONE action for the given symbol.

Respond with ONLY one JSON object, no markdown, no extra text, exactly this schema:
{{"action": "BUY" | "SELL" | "HOLD" | "CLOSE",
  "confidence": number between 0 and 1,
  "stop_loss": absolute price or null,
  "take_profit": absolute price or null,
  "reasoning": "short explanation, max 300 characters, in {language}"}}

Rules:
- BUY/SELL MUST include stop_loss and take_profit as absolute prices (not pips), using the symbol's digits.
- BUY: stop_loss < current ask < take_profit. SELL: take_profit < current bid < stop_loss.
- Base stop_loss on market structure (swing high/low) or ATR; reward:risk must be >= {min_rr}.
- Stop distance from price must be at least {min_sl_atr} x ATR and at most {max_sl_atr} x ATR of the main timeframe.
- If risk_rules.max_affordable_sl_distance is present, the account is small: the stop distance from entry
  MUST NOT exceed it. If no sensible structural stop fits within it, answer HOLD.
- CLOSE only when there is an open position on this symbol AND the thesis is clearly invalidated.
- If an open position already exists on the symbol, prefer HOLD unless CLOSE is justified.
- If signals conflict, spread is wide, or data is insufficient: HOLD with low confidence.
- Never invent data that is not in the snapshot. Confidence must reflect real conviction.
- Write "reasoning" in natural {language} with full, correct diacritics/accents
  (for Vietnamese: "Xu hướng giảm mạnh", NEVER "Xu huong giam manh"). Do not wrap the JSON in markdown.
"""


def build_system_prompt(language: str, min_rr: float, min_sl_atr: float, max_sl_atr: float) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        language=language, min_rr=min_rr, min_sl_atr=min_sl_atr, max_sl_atr=max_sl_atr or "unlimited"
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Tách object JSON từ text của LLM (chịu được code fence và chữ thừa xung quanh)."""
    if not text or not text.strip():
        raise ValueError("empty response")
    candidate = text.strip()
    m = _FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response") from None
        obj = json.loads(candidate[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


def parse_decision(text: str) -> AIDecision:
    """Parse + validate text thành AIDecision. Raise ValueError/ValidationError nếu sai."""
    obj = extract_json(text)
    if "decision" in obj and isinstance(obj["decision"], dict):
        obj = obj["decision"]
    return AIDecision.model_validate(obj)


class AIClient:
    """Gọi AI endpoint với timeout, retry + exponential backoff (có jitter)."""

    def __init__(
        self,
        url: str,
        api_key: str = "",
        model: str = "",
        style: str = "openai",
        timeout: float = 30.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        json_mode: bool = True,
        system_prompt: str | None = None,
        backoff_base: float = 1.0,
        backoff_max: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.url = url
        self.model = model
        self.models = [model] + [m for m in (fallback_models or []) if m and m != model]
        self.last_model = model
        self.style = style
        self.max_retries = max_retries
        self.temperature = temperature
        self.json_mode = json_mode
        self.system_prompt = system_prompt or build_system_prompt("English", 1.2, 0.3, 5.0)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.consecutive_failures = 0
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)), headers=headers, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ payload
    def _build_payload(self, snapshot: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        model = model or self.model
        if self.style == "custom":
            return {"model": model, "system": self.system_prompt, "snapshot": snapshot}
        payload: dict[str, Any] = {
            "model": model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract_text(self, data: Any) -> str:
        if self.style == "custom":
            return json.dumps(data) if isinstance(data, dict) else str(data)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"unexpected OpenAI response shape: {str(data)[:200]}") from exc

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_max)
        delay = self.backoff_base * (2 ** (attempt - 1))
        return min(delay, self.backoff_max) * (0.8 + random.random() * 0.4)

    # ------------------------------------------------------------------ main
    async def decide(self, snapshot: dict[str, Any]) -> AIResult:
        """Gửi snapshot, trả về AIResult. KHÔNG bao giờ raise – lỗi => HOLD fallback.

        Mỗi model được thử tối đa ``max_retries`` lần; gặp HTTP 429/503 (hết quota/quá tải)
        thì chuyển ngay sang model dự phòng kế tiếp (``AI_FALLBACK_MODELS``).
        """
        started = time.perf_counter()
        last_error = "unknown error"
        raw_text: str | None = None
        attempt = 0
        payload: dict[str, Any] = {}

        for mi, model in enumerate(self.models):
            payload = self._build_payload(snapshot, model)
            for try_no in range(1, self.max_retries + 1):
                attempt += 1
                retry_after: float | None = None
                switch = False
                tag = f"{model} {try_no}/{self.max_retries}"
                try:
                    resp = await self._client.post(self.url, json=payload)
                    if resp.status_code in RETRYABLE_STATUS:
                        ra = resp.headers.get("retry-after")
                        retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
                        switch = resp.status_code in SWITCH_MODEL_STATUS and mi + 1 < len(self.models)
                        raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    if resp.status_code >= 400:
                        # 4xx khác (401, 400, 404...) – lỗi cấu hình: thử model khác nếu có, không retry
                        last_error = f"[{model}] HTTP {resp.status_code}: {resp.text[:300]}"
                        log.warning("AI call %s failed: %s", tag, last_error)
                        break
                    raw_text = self._extract_text(resp.json())
                    decision = parse_decision(raw_text)
                    self.consecutive_failures = 0
                    self.last_model = model
                    return AIResult(
                        decision=decision, ok=True, attempts=attempt,
                        latency_ms=int((time.perf_counter() - started) * 1000), raw_text=raw_text, request=payload,
                    )
                except httpx.TimeoutException as exc:
                    last_error = f"[{model}] {type(exc).__name__}: AI không trả lời trong {self.timeout:.0f}s"
                    log.warning("AI call %s failed: %s", tag, last_error)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    last_error = f"[{model}] {type(exc).__name__}: {exc}"
                    log.warning("AI call %s failed: %s", tag, last_error)
                except (ValueError, ValidationError) as exc:
                    # JSON/Schema sai – thử lại (LLM có tính ngẫu nhiên)
                    last_error = f"[{model}] invalid AI output: {exc}".replace("\n", " ")[:400]
                    log.warning("AI output invalid (%s): %s", tag, last_error)
                if switch:
                    log.info("AI model %s hết quota/quá tải -> chuyển sang %s", model, self.models[mi + 1])
                    break
                if try_no < self.max_retries:
                    await asyncio.sleep(self._backoff(try_no, retry_after))

        self.consecutive_failures += 1
        return AIResult(
            decision=AIDecision.hold(f"AI fallback: {last_error}"[:500]),
            ok=False, attempts=attempt, latency_ms=int((time.perf_counter() - started) * 1000),
            raw_text=raw_text, error=last_error, request=payload,
        )
