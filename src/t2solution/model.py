"""Rolling-selected statistical ensemble and coherent joint path simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .monthly import monthly_steps
from .text import TextView
from .unit import Unit, target_series


@dataclass
class AssetState:
    asset: str
    panel: str
    transform: str
    anchor: float
    last_date: str
    steps: pd.Series
    standardized: pd.Series
    vol_recent: float
    vol_long: float
    drift_step: float
    width_calibration: float
    model_weights: dict[str, float] = field(default_factory=dict)


def _gap_safe_diff(series: pd.Series, monthly: bool) -> pd.Series:
    values = series.diff()
    dates = pd.Series(series.index, index=series.index)
    gaps = pd.to_datetime(dates).diff().dt.days
    limit = 50 if monthly else 10
    return values.where(gaps <= limit).dropna()


def _candidate_drifts(x: np.ndarray, origin: int) -> np.ndarray:
    hist = x[:origin]
    return np.array([
        0.0,
        float(np.mean(hist[-20:])),
        float(np.mean(hist[-63:])),
        0.35 * float(np.mean(hist)),
    ])


def _rolling_drift(steps: pd.Series, horizon: int) -> tuple[float, dict[str, float]]:
    x = steps.to_numpy(dtype=float)
    names = ["zero", "fast", "medium", "long-shrunk"]
    if len(x) < max(80, 3 * horizon):
        return 0.0, dict(zip(names, [0.7, 0.15, 0.1, 0.05], strict=True))
    origins = np.linspace(max(63, len(x) - 500), len(x) - horizon - 1, num=min(80, len(x) - horizon - 63), dtype=int)
    losses = np.zeros(4)
    scale = float(np.std(x)) * math.sqrt(horizon) + 1e-12
    for origin in origins:
        actual = float(np.sum(x[origin : origin + horizon]))
        prediction = _candidate_drifts(x, int(origin)) * horizon
        losses += np.minimum(np.abs(actual - prediction) / scale, 4.0)
    losses /= max(len(origins), 1)
    relative = losses - losses.min()
    weights = np.exp(-3.0 * relative)
    weights /= weights.sum()
    current = _candidate_drifts(x, len(x))
    drift = float(np.dot(weights, current))
    cap = 0.15 * float(np.std(x))
    return float(np.clip(drift, -cap, cap)), dict(zip(names, weights.tolist(), strict=True))


def _historical_width(steps: pd.Series, horizons: list[int], monthly: bool, step_vol: float) -> float:
    """Correct sqrt-time scaling using only historical multi-step forecast errors."""
    if monthly:
        return 1.0
    ratios: list[float] = []
    for horizon in horizons:
        h = int(horizon)
        if h < 2 or len(steps) < max(80, 4 * h):
            continue
        realized = steps.rolling(h).sum().dropna()
        # Non-overlapping endpoints reduce the false confidence caused by overlapping windows.
        realized = realized.iloc[h - 1 :: h]
        if len(realized) < 12:
            continue
        ratio = float(realized.std(ddof=1)) / (step_vol * math.sqrt(h) + 1e-12)
        if np.isfinite(ratio):
            ratios.append(ratio)
    if not ratios:
        return 1.06
    # A small finite-sample correction targets the observed systematic 90% under-coverage of
    # plug-in volatility estimates while remaining far below indiscriminate stress widening.
    return float(np.clip(1.06 * np.median(ratios), 0.85, 1.50))


def _prepare(unit: Unit, asset: str, asof: str) -> AssetState:
    series, panel = target_series(unit, asset, asof)
    monthly = unit.target_frequency.lower() == "monthly"
    if unit.target_type == "log_return":
        if (series <= -1).any():
            raise ValueError(f"{asset}: simple returns must be greater than -1")
        transform, anchor = "cumlog", 0.0
        steps = pd.Series(np.log1p(series.to_numpy(dtype=float)), index=series.index)
    elif "rate" in panel.lower() or (series <= 0).any():
        transform, anchor = "diff", float(series.iloc[-1])
        steps = _gap_safe_diff(series, monthly)
    else:
        transform, anchor = "log", float(series.iloc[-1])
        steps = _gap_safe_diff(np.log(series), monthly)
    if len(steps) < 12:
        raise ValueError(f"{asset}: insufficient usable history")
    long = float(steps.std(ddof=1))
    recent = float(steps.tail(60 if not monthly else 18).std(ddof=1))
    recent = recent if np.isfinite(recent) and recent > 0 else long
    long = long if np.isfinite(long) and long > 0 else max(abs(float(steps.mean())), 1e-6)
    # Avoid calm-before-event collapse while retaining recent-regime information. Event and
    # cross-asset families receive a firmer floor; F1 remains primarily data-driven.
    floor_by_family = {"T2-F1": 1.00, "T2-F2": 0.95, "T2-F3": 0.90, "T2-F4": 1.15}
    vol_recent = max(recent, floor_by_family.get(unit.family, 0.80) * long)
    lagged = steps.pow(2).ewm(halflife=20 if not monthly else 6, min_periods=5).mean().pow(0.5).shift(1)
    lagged = lagged.bfill().clip(lower=0.25 * long)
    standardized = (steps / lagged).replace([np.inf, -np.inf], np.nan).dropna()
    candidate_drift, weights = _rolling_drift(steps, max(1, min(unit.horizons)))
    # Daily yields and FX are extremely hard to directionally time and pseudo-cutoff tests show
    # that small sample trends overfit. Keep them driftless. Retain a strongly-shrunk drift only
    # for cumulative factor returns and genuinely monthly positive level series.
    if transform == "cumlog":
        drift = 0.25 * candidate_drift
    elif transform == "log" and monthly:
        drift = 0.40 * candidate_drift
    else:
        drift = 0.0
    width = _historical_width(steps, unit.horizons, monthly, long)
    if transform == "cumlog":
        # Factor-return histories already contain episodic large moves; the generic calibration
        # correction otherwise double-counts their tail width.
        width = float(np.clip(0.82 * width, 0.75, 1.25))
    return AssetState(asset, panel, transform, anchor, str(series.index[-1].date()), steps,
                      standardized, vol_recent, long, drift, width, weights)


def _nearest_correlation(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = 0.85 * value + 0.15 * np.eye(len(value))
    eigenvalues, eigenvectors = np.linalg.eigh((value + value.T) / 2)
    rebuilt = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-5, None)) @ eigenvectors.T
    sd = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(sd, sd)


def _factor_basis(correlation: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(correlation)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    cumulative = np.cumsum(np.clip(values, 0, None)) / max(float(np.clip(values, 0, None).sum()), 1e-12)
    k = max(1, min(len(values), int(np.searchsorted(cumulative, 0.88)) + 1))
    return vectors[:, :k]


def _project_text(view: TextView, assets: list[str], basis: np.ndarray, family: str) -> None:
    if len(assets) <= 1:
        return
    for scenario in view.scenarios:
        vector = np.array([scenario.drift_sd.get(asset, 0.0) for asset in assets])
        projected = basis @ (basis.T @ vector)
        # F3 must remain inside empirical factor space; other families retain a little idiosyncratic signal.
        blend = 1.0 if family == "T2-F3" else 0.65
        coherent = blend * projected + (1.0 - blend) * vector
        scenario.drift_sd = {asset: float(np.clip(coherent[i], -0.55, 0.55)) for i, asset in enumerate(assets)}


def numeric_context(states: list[AssetState]) -> dict[str, Any]:
    return {state.asset: {
        "last_observation": round(state.anchor, 7),
        "last_date": state.last_date,
        "recent_step_vol": round(state.vol_recent, 7),
        "long_step_vol": round(state.vol_long, 7),
        "numeric_drift_step": round(state.drift_step, 8),
        "historical_width_calibration": round(state.width_calibration, 4),
        "selected_models": {k: round(v, 3) for k, v in state.model_weights.items()},
    } for state in states}


def prepare(unit: Unit, asof: str) -> list[AssetState]:
    return [_prepare(unit, asset, asof) for asset in unit.assets]


def simulate(
    unit: Unit, asof: str, states: list[AssetState], text_view: TextView, n_draws: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    monthly = unit.target_frequency.lower() == "monthly"
    if monthly:
        explicit = monthly_steps(unit.assets, unit.horizons,
                                 {s.asset: s.last_date for s in states}, unit.card, unit.spec)
        step_grid = np.array([[explicit[(a, h)] for h in unit.horizons] for a in unit.assets], dtype=int)
    else:
        step_grid = np.array([[int(h) for h in unit.horizons] for _ in unit.assets], dtype=int)
    max_steps = int(step_grid.max())

    aligned = pd.concat([s.standardized.rename(s.asset) for s in states], axis=1).dropna()
    if len(states) == 1:
        correlation = np.ones((1, 1))
    elif len(aligned) >= 30:
        correlation = _nearest_correlation(aligned.corr().to_numpy(dtype=float))
    else:
        correlation = np.eye(len(states))
    basis = _factor_basis(correlation)
    _project_text(text_view, unit.assets, basis, unit.family)

    chol = np.linalg.cholesky(correlation + 1e-8 * np.eye(len(states)))
    # Tail thickness is family-aware. Empirical residuals supply non-Gaussian shape on every card;
    # the parametric component is Gaussian for continuation cards and progressively heavier for
    # regime/tail cards.
    z = rng.standard_normal((n_draws, max_steps, len(states))) @ chol.T
    df_by_family = {"T2-F2": 10.0, "T2-F3": 10.0, "T2-F4": 5.0}
    tail_df = df_by_family.get(unit.family)
    if tail_df is not None:
        t_scale = np.sqrt((tail_df - 2.0) / rng.chisquare(tail_df, size=(n_draws, 1, 1)))
        z *= t_scale

    # Add empirical serial shape by block-resampling joint standardized residual rows where possible.
    if len(aligned) >= 40:
        pool = aligned.to_numpy(dtype=float)
        empirical = np.empty_like(z)
        block = 6 if not monthly else 3
        for draw in range(n_draws):
            offset = 0
            while offset < max_steps:
                start = int(rng.integers(0, len(pool)))
                take = min(block, max_steps - offset)
                empirical[draw, offset : offset + take] = pool[(start + np.arange(take)) % len(pool)]
                offset += take
        z = 0.72 * empirical + 0.28 * z
        z /= np.std(z, axis=(0, 1), keepdims=True).clip(min=1e-6)

    probs = np.array([scenario.probability for scenario in text_view.scenarios], dtype=float)
    probs = probs / probs.sum()
    chosen = rng.choice(len(text_view.scenarios), size=n_draws, p=probs)
    shortest = int(step_grid.min())
    paths = np.empty_like(z)
    for i, state in enumerate(states):
        progress = np.arange(1, max_steps + 1, dtype=float)
        vol_path = np.sqrt(state.vol_long**2 + (state.vol_recent**2 - state.vol_long**2) * np.exp(-progress / (60 if not monthly else 6)))
        paths[:, :, i] = z[:, :, i] * vol_path[None, :] * state.width_calibration + state.drift_step
    base = paths.copy()
    for index, scenario in enumerate(text_view.scenarios):
        selected = chosen == index
        if not selected.any():
            continue
        for i, state in enumerate(states):
            multiplier = scenario.vol_mult.get(state.asset, 1.0)
            horizon_sd = state.vol_recent * math.sqrt(max(shortest, 1))
            drift_step = scenario.drift_sd.get(state.asset, 0.0) * horizon_sd / max(shortest, 1)
            paths[selected, :, i] = base[selected, :, i] * multiplier + drift_step
            tilt = scenario.tail_tilt.get(state.asset, 0.0)
            if abs(tilt) > 0.05:
                shock = np.abs(z[selected, :, i]) * state.vol_recent
                rare = rng.random((selected.sum(), max_steps)) < 0.035 * abs(tilt)
                paths[selected, :, i] += rare * shock * np.sign(tilt) * 1.5

    cumulative = np.cumsum(paths, axis=1)
    output = np.empty((n_draws, len(states), len(unit.horizons)), dtype=float)
    for i, state in enumerate(states):
        for j, _ in enumerate(unit.horizons):
            change = cumulative[:, step_grid[i, j] - 1, i]
            if state.transform == "log":
                output[:, i, j] = state.anchor * np.exp(change)
            elif state.transform == "diff":
                output[:, i, j] = state.anchor + change
            else:
                output[:, i, j] = change
    diagnostics = {
        "aligned_rows": int(len(aligned)),
        "correlation": correlation.tolist(),
        "factor_dimension": int(basis.shape[1]),
        "parametric_tail_df": tail_df,
        "step_grid": step_grid.tolist(),
        "text_used": text_view.used,
        "text_note": text_view.note,
        "house_calls": text_view.call_count,
        "model": text_view.model,
        "scenarios": [scenario.__dict__ for scenario in text_view.scenarios],
        "assets": numeric_context(states),
    }
    return output, diagnostics
