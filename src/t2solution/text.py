"""Budget-safe document extraction and scenario synthesis for the organizer House model."""

from __future__ import annotations

import json
import os
import pathlib
import sys
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


def _read_documents(text_dir: pathlib.Path, limit: int = 10, total_chars: int = 70_000) -> list[dict[str, str]]:
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
                        for field_name in ("file", "path", "filename"):
                            key = str(row.get(field_name) or "")
                            if key:
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
            body = path.read_text(encoding="utf-8", errors="replace")[: min(9_000, remaining)]
        except OSError:
            continue
        doc_id = str(meta.get("doc_id") or meta.get("id") or path.stem)
        timestamp = str(meta.get("timestamp") or meta.get("date") or "unknown")
        docs.append({
            "doc_id": doc_id,
            "timestamp": timestamp,
            "source": str(meta.get("source") or "unknown"),
            "doc_type": str(meta.get("doc_type") or "unknown"),
            "text": body,
        })
        remaining -= len(body)
    return docs


def _url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
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


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _house_json(
    endpoint: str,
    token: str,
    model: str,
    system: str,
    payload: dict[str, Any],
    *,
    max_tokens: int,
    timeout: float = 120,
    thinking: bool = False,
) -> dict[str, Any] | None:
    if not 1 <= max_tokens <= 4000:
        raise ValueError('House output request must be within 1..4000 tokens')
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _url(endpoint), data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        choice = result['choices'][0]
        parsed = _extract_json(str(choice['message']['content']))
        if parsed is None:
            reason = choice.get('finish_reason')
            reason = reason if reason in {'stop', 'length', 'content_filter'} else 'unknown'
            print(f'House invalid JSON; finish_reason={reason}; response ignored', file=sys.stderr)
        return parsed
    except urllib.error.HTTPError as exc:
        print(f'House HTTP {exc.code}; response ignored', file=sys.stderr)
        return None
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        print('House response unavailable or malformed; response ignored', file=sys.stderr)
        return None


def _signal_card(raw: dict[str, Any] | None, doc: dict[str, str], unit: Unit) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    per_asset_raw = raw.get("per_asset", {})
    if not isinstance(per_asset_raw, dict):
        return None
    per_asset: dict[str, dict[str, float]] = {}
    for asset in unit.assets:
        values = per_asset_raw.get(asset, {})
        if not isinstance(values, dict):
            values = {}
        per_asset[asset] = {
            "direction": float(np.clip(_number(values.get("direction")), -1.0, 1.0)),
            "uncertainty": float(np.clip(_number(values.get("uncertainty")), 0.0, 1.0)),
            "tail_tilt": float(np.clip(_number(values.get("tail_tilt")), -1.0, 1.0)),
        }
    return {
        "doc_id": doc["doc_id"],
        "timestamp": doc["timestamp"],
        "relevance": float(np.clip(_number(raw.get("relevance")), 0.0, 1.0)),
        "confidence": float(np.clip(_number(raw.get("confidence"), 0.5), 0.0, 1.0)),
        "event": bool(raw.get("event", False)),
        "established": bool(raw.get("established", False)),
        "summary": str(raw.get("summary", ""))[:300],
        "per_asset": per_asset,
    }


def _parse(raw: dict[str, Any], unit: Unit, cards: list[dict[str, Any]]) -> list[Scenario]:
    family_shrink = {"T2-F1": 0.18, "T2-F2": 0.38, "T2-F3": 0.32, "T2-F4": 0.35}.get(unit.family, 0.25)
    scenarios: list[Scenario] = []
    rows = raw.get("scenarios", [])
    if not isinstance(rows, list):
        return []
    valid_evidence = {str(card["doc_id"]) for card in cards}
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        probability = float(np.clip(_number(row.get("probability"), 0.0), 0.0, 1.0))
        drift_raw = row.get("drift_sd", {}) if isinstance(row.get("drift_sd"), dict) else {}
        vol_raw = row.get("vol_mult", {}) if isinstance(row.get("vol_mult"), dict) else {}
        tail_raw = row.get("tail_tilt", {}) if isinstance(row.get("tail_tilt"), dict) else {}
        drift = {
            asset: float(np.clip(family_shrink * _number(drift_raw.get(asset)), -0.45, 0.45))
            for asset in unit.assets
        }
        vol = {
            asset: float(np.clip(1.0 + 0.35 * (max(1.0, _number(vol_raw.get(asset), 1.0)) - 1.0), 1.0, 1.45))
            for asset in unit.assets
        }
        tilt = {asset: float(np.clip(_number(tail_raw.get(asset)), -1.0, 1.0)) for asset in unit.assets}
        evidence = [
            str(item)[:120] for item in row.get("evidence", [])
            if isinstance(item, str) and str(item) in valid_evidence
        ][:6]
        scenarios.append(Scenario(
            name=str(row.get("name", "text scenario"))[:60], probability=probability,
            drift_sd=drift, vol_mult=vol, tail_tilt=tilt, evidence=evidence,
            explanation=str(row.get("explanation", ""))[:300],
        ))
    total = sum(item.probability for item in scenarios)
    if total <= 0:
        return []
    for scenario in scenarios:
        scenario.probability /= total

    # Deterministic evidence guardrails: a centre move requires corroboration. A genuine binary
    # event may keep multiple branches, but its probability-weighted centre remains neutral when
    # the documents do not agree on direction.
    for asset in unit.assets:
        weighted: list[tuple[float, float, float, bool]] = []
        for card in cards:
            values = card["per_asset"][asset]
            weight = float(card["relevance"]) * float(card["confidence"])
            weighted.append((float(values["direction"]), float(values["uncertainty"]), weight, bool(card["event"])))
        denom = sum(weight for _, _, weight, _ in weighted) or 1.0
        implied = sum(direction * weight for direction, _, weight, _ in weighted) / denom
        uncertainty = sum(value * weight for _, value, weight, _ in weighted) / denom
        agreeing = max(
            sum(1 for direction, _, weight, _ in weighted if weight >= 0.20 and direction >= 0.20),
            sum(1 for direction, _, weight, _ in weighted if weight >= 0.20 and direction <= -0.20),
        )
        event = any(is_event and weight >= 0.20 for _, _, weight, is_event in weighted)
        mixture = sum(s.probability * s.drift_sd[asset] for s in scenarios)
        if agreeing < 2 or abs(implied) < 0.15:
            for scenario in scenarios:
                scenario.drift_sd[asset] -= mixture
        else:
            target = float(np.clip(mixture, -0.40 * abs(implied), 0.40 * abs(implied)))
            if np.sign(target) != np.sign(implied):
                target = 0.0
            for scenario in scenarios:
                scenario.drift_sd[asset] = float(np.clip(scenario.drift_sd[asset] + target - mixture, -0.45, 0.45))
        if uncertainty < 0.25 and not event:
            for scenario in scenarios:
                scenario.vol_mult[asset] = 1.0
                scenario.tail_tilt[asset] = 0.0
        elif not event:
            for scenario in scenarios:
                scenario.vol_mult[asset] = min(scenario.vol_mult[asset], 1.20)
                scenario.tail_tilt[asset] *= 0.5
    return scenarios


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

    task = {
        "unit_id": unit.unit_id, "family": unit.family, "asof": asof,
        "title": str(unit.card.get("task", {}).get("title", ""))[:240],
        "description": str(unit.card.get("metadata", {}).get("description", ""))[:500],
        "assets": unit.assets, "horizons": unit.horizons,
        "target_type": unit.target_type, "value_unit": unit.value_unit,
    }
    extraction_system = """Use only the supplied pre-cutoff document. Never use remembered outcomes.
Extract its ex-ante forecasting signal in the target's native quote direction. Relevance and confidence
are 0..1. For every asset: direction is -1..1, uncertainty is 0..1, tail_tilt is -1..1. Mark event only
for a discrete policy, political, release, peg, crisis, or other branching catalyst inside the forecast
frame. Mark established only for facts already known by the document timestamp. Return JSON only."""
    cards: list[dict[str, Any]] = []
    calls = 0
    for doc in docs:
        raw = _house_json(endpoint, token, model, extraction_system, {
            "task": task,
            "document": doc,
            "schema": {
                "relevance": 0.0, "confidence": 0.0, "event": False, "established": False,
                "summary": "short",
                "per_asset": {asset: {"direction": 0.0, "uncertainty": 0.0, "tail_tilt": 0.0}
                              for asset in unit.assets},
            },
        }, max_tokens=550)
        calls += 1
        card = _signal_card(raw, doc, unit)
        if card is not None:
            cards.append(card)

    synthesis_system = """You are a cautious probabilistic macro forecaster. Use only the supplied
pre-cutoff signal cards and numeric context; never use remembered outcomes. Produce 1-3 mutually
exclusive scenarios. Move a centre only when at least two relevant cards corroborate direction.
Represent a genuine binary event with branches rather than an overconfident single mean. drift_sd is
a centre shift in shortest-horizon standard deviations (-1..1). vol_mult is 1..2.0 and inferred text
must never narrow it below 1. tail_tilt is -1..1. Probabilities sum to one. For multi-asset targets,
each scenario must be economically coherent. Cite only supplied doc_ids. Return JSON only."""
    synthesis = _house_json(endpoint, token, model, synthesis_system, {
        "task": task,
        "numeric_context": numeric_context,
        "signal_cards": cards,
        "schema": {"scenarios": [{
            "name": "string", "probability": 0.5,
            "drift_sd": {asset: 0.0 for asset in unit.assets},
            "vol_mult": {asset: 1.0 for asset in unit.assets},
            "tail_tilt": {asset: 0.0 for asset in unit.assets},
            "evidence": ["doc_id"], "explanation": "short evidence-based reason",
        }]},
    }, max_tokens=1800)
    calls += 1
    scenarios = _parse(synthesis or {}, unit, cards)
    if not scenarios:
        view = TextView.neutral()
        view.model, view.call_count = model, calls
        view.note = f"House extraction produced {len(cards)}/{len(docs)} signal cards but no valid synthesis"
        return view
    return TextView(
        scenarios, used=True, model=model, call_count=calls,
        note=f"two-stage House analysis: {len(cards)}/{len(docs)} documents extracted, then synthesized",
    )

