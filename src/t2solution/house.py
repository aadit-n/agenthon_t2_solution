"""Bounded House transport. Diagnostics never include credentials or response bodies."""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .legacy_text import _extract_json, _url

MAX_CALLS = 25
MAX_OUTPUT_TOKENS = 4000


def log(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), file=sys.stderr, flush=True)


@dataclass
class Reply:
    value: dict[str, Any] | None
    reason: str
    retryable: bool = False
    retry_delay: float = 0.0


class HouseClient:
    def __init__(self, endpoint: str, token: str, model: str, seconds: float = 600) -> None:
        self.endpoint, self.token, self.model = endpoint, token, model
        self.deadline = time.monotonic() + seconds
        self.calls = 0
        self.blocked = False
        self.attempts: list[dict[str, Any]] = []

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def _record(self, phase: str, attempt: int, tokens: int, reason: str, **extra: Any) -> None:
        row = {"phase": phase, "attempt": attempt, "max_tokens": tokens, "reason": reason, **extra}
        self.attempts.append(row)
        log("house_call", call=self.calls, **row)

    def request(
        self, system: str, payload: dict[str, Any], tokens: int, *, phase: str,
        attempt: int, reserve_seconds: float = 0,
    ) -> Reply:
        if not 1 <= tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError("House output token budget exceeds the permitted range")
        if self.blocked:
            return Reply(None, "route_blocked")
        if self.calls >= MAX_CALLS:
            return Reply(None, "call_budget")
        available = self.remaining_seconds - reserve_seconds
        if available < 1:
            return Reply(None, "time_budget")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "temperature": 0, "seed": 0, "max_tokens": tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _url(self.endpoint), data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        # Conservatively count every attempted POST, including rejected or timed-out requests.
        self.calls += 1
        try:
            with urllib.request.urlopen(request, timeout=min(90.0, available)) as response:
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                self._record(phase, attempt, tokens, "oversized_response")
                return Reply(None, "oversized_response")
            result = json.loads(raw.decode("utf-8"))
            choice = result["choices"][0]
            finish = choice.get("finish_reason")
            safe_finish = finish if finish in (None, "stop", "length", "content_filter") else "other"
            extra: dict[str, Any] = {"finish_reason": safe_finish}
            usage = result.get("usage", {})
            count = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if type(count) is int and 0 <= count <= 1_000_000:
                extra["completion_tokens"] = count
            if finish == "length":
                self._record(phase, attempt, tokens, "truncated", **extra)
                return Reply(None, "truncated", True)
            if finish not in (None, "stop"):
                self._record(phase, attempt, tokens, "unfinished_response", **extra)
                return Reply(None, "unfinished_response")
            content = choice["message"]["content"]
            value = _extract_json(content) if isinstance(content, str) else None
            reason = "json_received" if value is not None else "invalid_json"
            self._record(phase, attempt, tokens, reason, **extra)
            return Reply(value, reason, value is None)
        except urllib.error.HTTPError as error:
            status = error.code
            if status in (401, 403, 404):
                self.blocked = True
            retryable = status == 429 or 500 <= status < 600
            delay = 1.0
            try:
                advertised = float(error.headers.get("Retry-After", "1"))
                if math.isfinite(advertised):
                    delay = min(3.0, max(0.0, advertised))
            except (AttributeError, TypeError, ValueError):
                pass
            self._record(phase, attempt, tokens, "http_error", http_status=status)
            return Reply(None, "http_error", retryable, delay if retryable else 0)
        except (OSError, TimeoutError):
            self._record(phase, attempt, tokens, "connection_error")
            return Reply(None, "connection_error", True, 1)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            self._record(phase, attempt, tokens, "malformed_envelope")
            return Reply(None, "malformed_envelope", True)
