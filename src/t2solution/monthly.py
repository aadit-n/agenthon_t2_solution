"""Resolve explicit monthly observation periods while preserving authored horizon keys."""

from __future__ import annotations

from datetime import date
from typing import Any


def _month(value: str, date_label: bool = False) -> int:
    parsed = date.fromisoformat(value if date_label else value + "-01")
    return parsed.year * 12 + parsed.month - 1


def monthly_steps(
    assets: list[str], horizons: list[int], last_dates: dict[str, str], card: dict[str, Any], spec: dict[str, Any]
) -> dict[tuple[str, int], int]:
    periods: dict[tuple[str, int], int] = {}
    for source in (card, spec):
        targets = source.get("targets", source if "horizons" in source else {})
        declared_assets = targets.get("asset_ids", assets)
        declared_horizons = targets.get("horizons", horizons)
        for field, is_date in (("observation_periods", False), ("target_dates", True)):
            values = targets.get(field)
            if not values:
                continue
            if len(values) != len(declared_horizons):
                raise ValueError("monthly periods do not align with horizon keys")
            for horizon, value in zip(declared_horizons, values, strict=True):
                for asset in declared_assets:
                    periods[(str(asset), int(horizon))] = _month(str(value), is_date)
        for row in source.get("questions", []):
            if "observation_period" in row:
                periods[(str(row["asset"]), int(row["horizon"]))] = _month(str(row["observation_period"]))
            elif "target_date" in row:
                periods[(str(row["asset"]), int(row["horizon"]))] = _month(str(row["target_date"]), True)
    result: dict[tuple[str, int], int] = {}
    for asset in assets:
        start = _month(last_dates[asset][:10], True)
        for horizon in horizons:
            key = (asset, horizon)
            if key not in periods:
                raise ValueError("monthly target lacks an explicit observation-period mapping")
            result[key] = periods[key] - start
            if result[key] <= 0:
                raise ValueError("monthly target period must follow the last observation")
    return result
