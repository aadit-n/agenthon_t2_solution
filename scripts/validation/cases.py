"""Historical cases use only each source card's supplied, pre-cutoff numeric history.

Training panels end at the pseudo-origin. Labels stay outside training and end before the
real card cutoff. These are retrospective history proxies, not new official card answers.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from t2solution.monthly import monthly_steps
from t2solution.unit import Unit, target_series

SPLITS = {"tune": ("2000-01-01", "2012-12-31"),
          "holdout_1": ("2013-01-01", "2017-12-31"),
          "holdout_2": ("2018-01-01", "2024-12-31")}


@dataclass
class Window:
    trial: Unit
    actual: np.ndarray
    target_dates: list[str]
    label_start: str
    label_end: str


def step_counts(unit: Unit) -> np.ndarray:
    if unit.target_frequency.lower() != "monthly":
        return np.tile(np.array(unit.horizons, dtype=int), (len(unit.assets), 1))
    cutoff = unit.card["provenance"]["data_cutoff"]
    last = {a: str(target_series(unit, a, cutoff)[0].index[-1].date()) for a in unit.assets}
    mapping = monthly_steps(unit.assets, unit.horizons, last, unit.card, unit.spec)
    return np.array([[mapping[a, h] for h in unit.horizons] for a in unit.assets], dtype=int)


def split_for(origin: str, label_end: str) -> str | None:
    for name, (start, end) in SPLITS.items():
        if start <= origin <= label_end <= end:
            return name
    return None


def histories(unit: Unit) -> dict[str, pd.Series]:
    return {a: target_series(unit, a, unit.card["provenance"]["data_cutoff"])[0] for a in unit.assets}


def describe_window(unit: Unit, origin: pd.Timestamp, counts: np.ndarray, raw: dict[str, pd.Series], min_train: int):
    if counts.shape != (len(unit.assets), len(unit.horizons)) or (counts <= 0).any():
        raise ValueError("invalid_step_grid")
    monthly = unit.target_frequency.lower() == "monthly"
    actual, targets, starts, ends = [], [], [], []
    for i, asset in enumerate(unit.assets):
        series = raw[asset]
        cut = int(series.index.searchsorted(origin, side="right"))
        if cut < min_train:
            raise ValueError("insufficient_training")
        length = int(counts[i].max())
        future = series.iloc[cut:cut + length]
        if len(future) < length:
            raise ValueError("insufficient_future")
        dates = series.index[cut - 1:cut + length]
        gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
        if (gaps > (50 if monthly else 10)).any() or (gaps <= 0).any():
            raise ValueError("future_gap")
        if monthly and (np.diff(dates.to_period("M").asi8) != 1).any():
            raise ValueError("future_gap")
        values = future.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (unit.target_type == "log_return" and (values <= -1).any()):
            raise ValueError("invalid_future_values")
        starts.append(str(future.index[0].date()))
        ends.append(str(future.index[-1].date()))
        for steps in counts[i]:
            actual.append(float(np.log1p(values[:steps]).sum()) if unit.target_type == "log_return"
                          else float(values[steps - 1]))
            targets.append(str(future.index[steps - 1].date()))
    return np.array(actual), targets, min(starts), max(ends)


def build_window(unit: Unit, origin: pd.Timestamp, counts: np.ndarray, *, min_train: int) -> Window:
    actual, dates, start, end = describe_window(unit, origin, counts, histories(unit), min_train)
    asof = str(origin.date())
    panels = {name: frame.loc[frame.date <= origin].copy(deep=True) for name, frame in unit.panels.items()}
    card = copy.deepcopy(unit.card)
    # Do not expose a later event's title, prose, question text, or original root/text paths.
    task_id = "historical-" + asof
    card["task"] = {"id": task_id, "track": "forecasting"}
    card["metadata"] = {"category": unit.family, "target_frequency": unit.target_frequency}
    card["provenance"] = {"data_cutoff": asof}
    card.pop("text", None)
    card.pop("questions", None)
    for info in card.get("panels", {}).values():
        if isinstance(info, dict):
            info.pop("notes", None)
            info["end_date"] = asof
    target = {"asset_ids": unit.assets.copy(), "horizons": unit.horizons.copy(),
              "target_type": unit.target_type, "target_frequency": unit.target_frequency,
              "value_unit": unit.value_unit, "n_draws_min": unit.n_draws_min}
    card["targets"] = copy.deepcopy(target)
    spec: dict[str, Any] = {"targets": copy.deepcopy(target), "questions": []}
    if unit.target_frequency.lower() == "monthly":
        for i, asset in enumerate(unit.assets):
            past = histories(unit)[asset].loc[:origin]
            month = past.index[-1].to_period("M")
            for j, horizon in enumerate(unit.horizons):
                spec["questions"].append({"asset": asset, "horizon": horizon,
                                          "observation_period": str(month + int(counts[i, j]))})
    unavailable = unit.root / ".validation-unavailable"
    trial = replace(unit, root=unavailable, text_dir=unavailable / "text", unit_id=task_id,
                    panels=panels, card=card, spec=spec)
    return Window(trial, actual, dates, start, end)


def choose_origins(unit: Unit, counts: np.ndarray, *, per_split: int, embargo_days: int = 5):
    if per_split < 1 or embargo_days < 0:
        raise ValueError("invalid_origin_settings")
    raw = histories(unit)
    common = next(iter(raw.values())).index
    for series in raw.values():
        common = common.intersection(series.index)
    monthly = unit.target_frequency.lower() == "monthly"
    minimum = 60 if monthly else 300
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    excluded: Counter[str] = Counter()
    last_end = None
    # This chooses windows by calendars/missingness only, never by the label's magnitude.
    for origin in common:
        if last_end is not None and origin <= last_end + pd.offsets.BDay(embargo_days):
            continue
        try:
            _, target_dates, start, end = describe_window(unit, origin, counts, raw, minimum)
        except ValueError as error:
            excluded[str(error)] += 1
            continue
        split = split_for(str(origin.date()), end)
        if split is None:
            excluded["crosses_split_boundary"] += 1
            continue
        selected[split].append({"origin": str(origin.date()), "label_start": start,
                                "label_end": end, "target_dates": target_dates, "split": split})
        last_end = pd.Timestamp(end)
    result = []
    for options in selected.values():
        if options:
            indexes = np.linspace(0, len(options) - 1, min(per_split, len(options)), dtype=int)
            result.extend(options[i] for i in indexes)
    return result, dict(excluded)
