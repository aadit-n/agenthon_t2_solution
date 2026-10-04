"""Task-card and input-panel loading with strict cutoff handling."""

from __future__ import annotations

import json
import pathlib
import tomllib
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class Unit:
    root: pathlib.Path
    card: dict[str, Any]
    spec: dict[str, Any]
    panels: dict[str, pd.DataFrame]
    text_dir: pathlib.Path
    unit_id: str
    family: str
    assets: list[str]
    horizons: list[int]
    target_type: str
    target_frequency: str
    value_unit: str
    n_draws_min: int


def _find_root(panels_path: pathlib.Path, explicit_card: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    if explicit_card is not None:
        return explicit_card.parent, explicit_card
    candidates = [panels_path / "card.toml", panels_path.parent / "card.toml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent, candidate
    raise FileNotFoundError("card.toml not found beside --panels or in its parent")


def _read_panels(root: pathlib.Path, panels_path: pathlib.Path, asof: str) -> dict[str, pd.DataFrame]:
    files = sorted(panels_path.glob("*.parquet")) if panels_path.is_dir() else []
    if not files and panels_path != root:
        files = sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet panels found under {panels_path} or {root}")
    result: dict[str, pd.DataFrame] = {}
    for path in files:
        frame = pd.read_parquet(path)
        required = {"date", "asset", "value"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "asset", "value"])
        frame = frame[frame["date"] <= pd.Timestamp(asof)]
        if frame.empty:
            continue
        panel_id = str(frame["panel_id"].iloc[0]) if "panel_id" in frame.columns else path.stem
        result[panel_id] = frame.sort_values(["date", "asset"])
    if not result:
        raise ValueError("no usable panel rows at or before --asof")
    return result


def load_unit(
    panels_path: pathlib.Path,
    text_path: pathlib.Path,
    asof: str,
    explicit_card: pathlib.Path | None = None,
) -> Unit:
    root, card_path = _find_root(panels_path, explicit_card)
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    spec_path = root / "forecast_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {}
    target = card.get("targets", {})
    spec_target = spec.get("targets", spec)

    assets = list(target.get("asset_ids") or spec_target.get("asset_ids") or [])
    horizons = [int(x) for x in (target.get("horizons") or spec_target.get("horizons") or [])]
    if not assets or not horizons:
        raise ValueError("card does not declare non-empty target assets and horizons")
    if len(set(assets)) != len(assets) or len(set(horizons)) != len(horizons):
        raise ValueError("target assets and horizons must be unique")

    task = card.get("task", {})
    metadata = card.get("metadata", {})
    return Unit(
        root=root,
        card=card,
        spec=spec,
        panels=_read_panels(root, panels_path, asof),
        text_dir=text_path,
        unit_id=str(task.get("id", root.name)),
        family=str(metadata.get("category", "T2-F1")),
        assets=[str(x) for x in assets],
        horizons=horizons,
        target_type=str(target.get("target_type") or spec_target.get("target_type") or "level"),
        target_frequency=str(target.get("target_frequency") or metadata.get("target_frequency") or "daily"),
        value_unit=str(target.get("value_unit") or spec_target.get("value_unit") or "unknown"),
        n_draws_min=int(target.get("n_draws_min") or spec_target.get("n_draws_min") or 200),
    )


def target_series(unit: Unit, asset: str, asof: str) -> tuple[pd.Series, str]:
    matches: list[tuple[pd.Series, str]] = []
    declared = unit.card.get("panels", {})
    preferred = [
        str(panel_id) for panel_id, metadata in declared.items()
        if isinstance(metadata, dict) and asset in [str(value) for value in metadata.get("asset_ids", [])]
    ]
    order = preferred + [panel_id for panel_id in unit.panels if panel_id not in preferred]
    for panel_id in order:
        frame = unit.panels.get(panel_id)
        if frame is None:
            continue
        sub = frame[(frame["asset"].astype(str) == asset) & (frame["date"] <= pd.Timestamp(asof))]
        if sub.empty:
            continue
        series = sub.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["value"].astype(float)
        matches.append((series, panel_id))
    if not matches:
        raise KeyError(f"target asset {asset!r} is absent from supplied panels")
    return matches[0] if preferred else max(matches, key=lambda item: len(item[0]))

