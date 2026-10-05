"""Cutoff-safe, deduplicated documents and explicit target semantics for House."""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

import pandas as pd

from .monthly import monthly_steps
from .unit import Unit, target_series


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def excerpt(body: str, limit: int) -> str:
    if len(body) <= limit:
        return body
    marker = "\n[... omitted text ...]\n"
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return body[:head] + marker + body[-tail:]


def read_documents(
    text_dir: pathlib.Path, limit: int = 10, total_chars: int = 70_000, *,
    asof: str | None = None, audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    index = text_dir / "corpus_index.json"
    if index.is_file():
        try:
            raw = json.loads(index.read_text(encoding="utf-8"))
            rows = raw.get("documents", []) if isinstance(raw, dict) else raw
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        for field in ("file", "path", "filename"):
                            if row.get(field):
                                metadata[pathlib.Path(str(row[field])).name] = row
            else:
                excluded["invalid_index"] += 1
        except (OSError, ValueError):
            excluded["invalid_index"] += 1
    cutoff = pd.Timestamp(asof, tz="UTC") + pd.Timedelta(days=1) if asof else None
    candidates = []
    for path in text_dir.glob("*.txt"):
        meta = metadata.get(path.name, {})
        stamp = meta.get("timestamp") or meta.get("date")
        try:
            if not isinstance(stamp, str) or not stamp:
                raise ValueError("missing timestamp")
            date = pd.Timestamp(stamp)
            if pd.isna(date):
                raise ValueError("invalid timestamp")
            date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
        except (ValueError, TypeError, OverflowError):
            excluded["undated"] += 1
            continue
        if cutoff is not None and date >= cutoff:
            excluded["after_cutoff"] += 1
            continue
        candidates.append((date, path, meta))
    candidates.sort(key=lambda row: (row[0], row[1].name), reverse=True)
    docs = []
    remaining = total_chars
    seen_text: set[str] = set()
    seen_ids: set[str] = set()
    for _, path, meta in candidates:
        if len(docs) >= limit or remaining < 200:
            break
        try:
            if path.is_symlink() or path.resolve().parent != text_dir.resolve():
                excluded["unsafe_path"] += 1
                continue
            if path.stat().st_size > 4 * 1024 * 1024:
                excluded["oversized_document"] += 1
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            excluded["unreadable"] += 1
            continue
        normalized = normalize(body)
        if not normalized:
            excluded["empty"] += 1
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        doc_id = str(meta.get("doc_id") or meta.get("id") or path.stem)
        if digest in seen_text or doc_id in seen_ids:
            excluded["duplicate"] += 1
            continue
        seen_text.add(digest)
        seen_ids.add(doc_id)
        selected = excerpt(body, min(9_000, remaining))
        docs.append({
            "doc_id": doc_id, "timestamp": str(meta.get("timestamp") or meta.get("date")),
            "source": str(meta.get("source") or "unknown"),
            "doc_type": str(meta.get("doc_type") or "unknown"),
            "text": selected, "excerpted": len(selected) < len(body), "_full_text": body,
        })
        remaining -= len(selected)
    if audit is not None:
        audit.update(documents_available=len(candidates), documents_selected=len(docs),
                     document_exclusions=dict(excluded), document_characters=total_chars - remaining)
    return docs


def task_context(unit: Unit, asof: str) -> dict[str, Any]:
    last_dates, panels = {}, {}
    for asset in unit.assets:
        series, panel = target_series(unit, asset, asof)
        last_dates[asset], panels[asset] = series.index[-1].strftime("%Y-%m-%d"), panel
    monthly = unit.target_frequency.lower() == "monthly"
    leads = monthly_steps(unit.assets, unit.horizons, last_dates, unit.card, unit.spec) if monthly else {}
    usd_quotes = {"DEXUSEU", "DEXUSUK", "DEXUSAL", "DEXUSNZ"}
    foreign_quotes = {"DEXJPUS", "DEXSZUS", "DEXCAUS", "DEXSDUS", "DEXNOUS", "DEXDNUS",
                      "DEXCHUS", "DEXINUS", "DEXBZUS"}
    g10_usd = {"EUR", "GBP", "AUD", "NZD"}
    g10_foreign = {"JPY", "CHF", "CAD", "SEK", "NOK", "DKK"}
    asset_context = {}
    for asset in unit.assets:
        declared = unit.card.get("panels", {}).get(panels[asset], {})
        codes = declared.get("series_codes", {}) if isinstance(declared, dict) else {}
        code = codes.get(asset) if isinstance(codes, dict) else None
        if code is None and isinstance(declared, dict):
            ids, series_codes = declared.get("asset_ids", []), declared.get("series", [])
            if asset in ids and len(ids) == len(series_codes):
                code = series_codes[ids.index(asset)]
        usd = code in usd_quotes or (panels[asset] == "g10_fx_daily" and asset in g10_usd)
        foreign = code in foreign_quotes or (panels[asset] == "g10_fx_daily" and asset in g10_foreign)
        quote, positive = unit.value_unit, "target value increases in its declared native unit"
        if usd:
            quote, positive = "USD per unit of " + asset, asset + " strengthens against USD"
        elif foreign:
            quote, positive = asset + " units per USD", asset + " weakens against USD"
        horizon_leads = {}
        for horizon in unit.horizons:
            steps = leads[(asset, horizon)] if monthly else horizon
            details: dict[str, Any] = {"steps_from_last_observation": steps,
                                       "step_unit": "calendar_month" if monthly else "business_day"}
            if monthly:
                details["observation_period"] = str(pd.Period(last_dates[asset][:7], freq="M") + steps)
            horizon_leads[str(horizon)] = details
        asset_context[asset] = {"native_quote": quote, "positive_direction_means": positive,
                                "last_observation_date": last_dates[asset], "horizon_leads": horizon_leads}
    return {
        "unit_id": unit.unit_id, "family": unit.family, "asof": asof,
        "title": str(unit.card.get("task", {}).get("title", ""))[:240],
        "description": str(unit.card.get("metadata", {}).get("description", ""))[:500],
        "assets": unit.assets, "horizons": unit.horizons, "asset_context": asset_context,
        "target_type": unit.target_type, "target_frequency": unit.target_frequency,
        "value_unit": unit.value_unit,
        "horizon_note": "Preserve authored horizon labels. Use resolved leads; monthly observation periods are not publication dates.",
    }
