"""Leakage-safe marginal backtest using only pseudo-cutoffs inside each supplied history.

This intentionally does not reconstruct public-card realized outcomes. It evaluates historical
forecast origins that precede the card cutoff, giving a calibration signal that is safer to tune on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd

from t2solution.model import _historical_width, _rolling_drift, prepare
from t2solution.monthly import monthly_steps
from t2solution.unit import load_unit


def crps(draws: np.ndarray, realized: float) -> float:
    values = np.sort(np.asarray(draws, dtype=float))
    n = len(values)
    first = float(np.mean(np.abs(values - realized)))
    weights = 2 * np.arange(1, n + 1) - n - 1
    pair = float(2.0 * np.dot(weights, values) / (n * n))
    return first - 0.5 * pair


def evaluate_steps(
    steps: pd.Series, horizon: int, *, seed: int, monthly: bool, transform: str, family: str
) -> list[dict[str, float]]:
    x = steps.reset_index(drop=True).astype(float)
    minimum = max(100, 4 * horizon)
    if len(x) < minimum + horizon:
        return []
    available = np.arange(minimum, len(x) - horizon + 1)
    origins = available[np.linspace(0, len(available) - 1, min(32, len(available)), dtype=int)]
    rows: list[dict[str, float]] = []
    for sequence, origin in enumerate(origins):
        train = x.iloc[: int(origin)]
        realized = float(x.iloc[int(origin) : int(origin) + horizon].sum())
        long = float(train.std(ddof=1))
        floor = {"T2-F1": 1.00, "T2-F2": 0.95, "T2-F3": 0.90, "T2-F4": 1.15}.get(family, 0.80)
        recent = max(float(train.tail(60 if not monthly else 18).std(ddof=1)), floor * long)
        width = _historical_width(train, [horizon], monthly, long)
        if transform == "cumlog":
            width = float(np.clip(0.82 * width, 0.75, 1.25))
        candidate_drift, _ = _rolling_drift(train, horizon)
        drift = (0.25 * candidate_drift if transform == "cumlog" else
                 0.40 * candidate_drift if transform == "log" and monthly else 0.0)
        sd = math.sqrt(horizon) * math.sqrt(0.55 * recent**2 + 0.45 * long**2)
        rng = np.random.default_rng(seed + sequence)
        normal = rng.standard_normal(1200)
        df = {"T2-F2": 10.0, "T2-F3": 10.0, "T2-F4": 5.0}.get(family)
        innovation = normal if df is None else normal * np.sqrt((df - 2.0) / rng.chisquare(df, 1200))
        model_draws = drift * horizon + innovation * sd * width
        baseline_draws = normal * sd
        lo, hi = np.quantile(model_draws, [0.05, 0.95])
        rows.append({
            "crps": crps(model_draws, realized),
            "baseline_crps": crps(baseline_draws, realized),
            "in90": float(lo <= realized <= hi),
            "below05": float(realized < lo),
            "above95": float(realized > hi),
            "width": width,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("backtest_runs/results.csv"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    units = sorted(p for p in (args.repo / "units").iterdir() if (p / "forecast_spec.json").is_file())
    if args.limit:
        units = units[: args.limit]
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, str]] = set()
    os.environ["T2_DISABLE_TEXT"] = "1"
    for unit_dir in units:
        import tomllib
        card = tomllib.loads((unit_dir / "card.toml").read_text(encoding="utf-8"))
        asof = str(card.get("provenance", {}).get("data_cutoff"))
        unit = load_unit(unit_dir, unit_dir / "text", asof)
        states = prepare(unit, asof)
        monthly = unit.target_frequency.lower() == "monthly"
        grid = None
        if monthly:
            grid = monthly_steps(unit.assets, unit.horizons, {s.asset: s.last_date for s in states}, unit.card, unit.spec)
        for state in states:
            for authored_horizon in unit.horizons:
                horizon = grid[(state.asset, authored_horizon)] if grid else authored_horizon
                key = (state.panel, state.asset, int(horizon), unit.target_type)
                if key in seen:
                    continue
                seen.add(key)
                stable_seed = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], "big")
                for row in evaluate_steps(state.steps, int(horizon), seed=stable_seed, monthly=monthly,
                                          transform=state.transform, family=unit.family):
                    records.append({"unit": unit.unit_id, "family": unit.family, "asset": state.asset,
                                    "panel": state.panel, "target_type": unit.target_type,
                                    "horizon_key": authored_horizon, "steps": horizon, **row})
    if not records:
        raise SystemExit("no eligible pseudo-cutoffs")
    result = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = {
        "pseudo_forecasts": int(len(result)),
        "mean_crps_ratio": float(result.crps.sum() / result.baseline_crps.sum()),
        "in90": float(result.in90.mean()),
        "below05": float(result.below05.mean()),
        "above95": float(result.above95.mean()),
        "mean_width": float(result.width.mean()),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
