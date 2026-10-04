"""One-call, budget-safe text-to-scenario reasoning for the organizer House model."""

from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .unit import Unit


@dataclass
class Scenario:
    name: str
    probability: float
    drift_sd: dict[str, float] = field(default_factory=dict)
    vol_mult: dict[str, float] = field(default_factory=dict)
    tail_tilt: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class TextView:
    scenarios: list[Scenario]
    used: bool = False
    model: str | None = None
    call_count: int = 0
    note: str = "text disabled or unavailable; numerical forecast retained"

    @classmethod
    def neutral(cls) -> "TextView":
        return cls([Scenario("numerical baseline", 1.0)])


def _read_documents(text_dir: pathlib.Path, limit: int = 12, total_chars: int = 56_000) -> list[dict[str, str]]:
    if not text_dir.is_dir():
        return []
    metadata: dict[str, dict[str, Any]] = {}
    index = text_dir / "corpus_index.json"
    if index.is_file():
        try:
            raw = json.loads(index.read_text(encoding="utf-8"))
            rows = raw.get("documents", raw) if isinstance(raw, dict) else raw
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        key = str(row.get("path") or row.get("filename") or row.get("doc_id") or "")
                        metadata[pathlib.Path(key).name] = row
        except (OSError, json.JSONDecodeError):
            pass
    candidates: list[tuple[str, pathlib.Path, dict[str, Any]]] = []
    for path in text_dir.glob("*.txt"):
        meta = metadata.get(path.name, {})
        stamp = str(meta.get("timestamp") or meta.get("date") or path.stem)
        candidates.append((stamp, path, meta))
    candidates.sort(key=lambda row: row[0], reverse=True)
    docs: list[dict[str, str]] = []
    remaining = total_chars
    for _, path, meta in candidates[:limit]:
        if remaining <= 0:
            break
        try:
            body = path.read_text(encoding="utf-8", errors="replace")[: min(8_000, remaining)]
        except OSError:
            continue
        doc_id = str(meta.get("doc_id") or meta.get("id") or path.stem)
        timestamp = str(meta.get("timestamp") or meta.get("date") or "unknown")
        docs.append({"doc_id": doc_id, "timestamp": timestamp, "text": body})
        remaining -= len(body)
    return docs


def _url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def _extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _parse(raw: dict[str, Any], unit: Unit) -> list[Scenario]:
    family_shrink = {"T2-F1": 0.22, "T2-F2": 0.50, "T2-F3": 0.42, "T2-F4": 0.45}.get(unit.family, 0.3)
    scenarios: list[Scenario] = []
    rows = raw.get("scenarios", [])
    if not isinstance(rows, list):
        return []
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        probability = float(np.clip(_number(row.get("probability"), 0.0), 0.0, 1.0))
        drift_raw = row.get("drift_sd", {}) if isinstance(row.get("drift_sd"), dict) else {}
        vol_raw = row.get("vol_mult", {}) if isinstance(row.get("vol_mult"), dict) else {}
        tail_raw = row.get("tail_tilt", {}) if isinstance(row.get("tail_tilt"), dict) else {}
        drift = {a: float(np.clip(family_shrink * _number(drift_raw.get(a)), -0.55, 0.55)) for a in unit.assets}
        # Inferred text may widen, never narrow. Magnitudes are deliberately shrunk.
        vol = {a: float(np.clip(1.0 + 0.45 * (max(1.0, _number(vol_raw.get(a), 1.0)) - 1.0), 1.0, 1.7)) for a in unit.assets}
        tilt = {a: float(np.clip(_number(tail_raw.get(a)), -1.0, 1.0)) for a in unit.assets}
        evidence = [str(x)[:120] for x in row.get("evidence", []) if isinstance(x, str)][:6]
        scenarios.append(Scenario(
            name=str(row.get("name", "text scenario"))[:60], probability=probability,
            drift_sd=drift, vol_mult=vol, tail_tilt=tilt, evidence=evidence,
            explanation=str(row.get("explanation", ""))[:300],
        ))
    total = sum(x.probability for x in scenarios)
    if total <= 0:
        return []
    for scenario in scenarios:
        scenario.probability /= total
    return scenarios


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def infer_text_view(unit: Unit, asof: str, numeric_context: dict[str, Any]) -> TextView:
    if os.environ.get("T2_DISABLE_TEXT", "").lower() in {"1", "true", "yes", "on"}:
        view = TextView.neutral()
        view.note = "text disabled by T2_DISABLE_TEXT"
        return view
    endpoint, token, model = os.environ.get("MODEL_ENDPOINT"), os.environ.get("MODEL_TOKEN"), os.environ.get("MODEL_NAME")
    if not endpoint or not token or not model:
        return TextView.neutral()
    docs = _read_documents(unit.text_dir)
    if not docs:
        view = TextView.neutral()
        view.note = "no readable text documents"
        return view

    system = """You are a cautious probabilistic macro forecaster. Use only the supplied pre-cutoff
documents and numeric context. Do not use remembered future outcomes. Produce 1-3 mutually exclusive
scenarios. Text should change a numerical forecast only when dated evidence is relevant and corroborated.
For each target, drift_sd is a centre shift in one-shortest-horizon standard deviations and must be
between -1 and 1. vol_mult is 1 to 2.5; never narrow uncertainty from inferred tone. tail_tilt is -1
for downside asymmetry, +1 for upside asymmetry, or 0. Scenario probabilities must sum to one.
For multi-asset cards, express one coherent economic scenario across all assets. Cite only supplied doc_ids.
Return one JSON object and no prose."""
    user = json.dumps({
        "task": {"unit_id": unit.unit_id, "family": unit.family, "asof": asof,
                 "assets": unit.assets, "horizons": unit.horizons,
                 "target_type": unit.target_type, "value_unit": unit.value_unit},
        "numeric_context": numeric_context,
        "documents": docs,
        "schema": {"scenarios": [{"name": "string", "probability": 0.5,
                    "drift_sd": {a: 0.0 for a in unit.assets},
                    "vol_mult": {a: 1.0 for a in unit.assets},
                    "tail_tilt": {a: 0.0 for a in unit.assets},
                    "evidence": ["doc_id"], "explanation": "short evidence-based reason"}]},
    }, ensure_ascii=False)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        _url(endpoint), data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        scenarios = _parse(parsed or {}, unit)
        if not scenarios:
            view = TextView.neutral()
            view.model, view.call_count, view.note = model, 1, "House response did not contain valid scenarios"
            return view
        return TextView(scenarios, used=True, model=model, call_count=1, note="one-call House scenario analysis")
    except (OSError, KeyError, ValueError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
        view = TextView.neutral()
        view.model, view.call_count, view.note = model, 1, f"House call failed safely: {type(exc).__name__}"
        return view

