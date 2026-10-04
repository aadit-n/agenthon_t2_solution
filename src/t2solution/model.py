"""Rolling-selected statistical ensemble and coherent joint path simulator."""

from __future__ import annotations

import math
import os
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
    baseline_steps: pd.Series
    standardized: pd.Series
    vol_recent: float
    vol_long: float
    drift_step: float
    width_calibration: float
    level_factor: float = 1.0
    proxy_asset: str | None = None
    model_weights: dict[str, float] = field(default_factory=dict)


def _gap_safe_diff(series: pd.Series, monthly: bool) -> pd.Series:
    values = series.diff()
    dates = pd.Series(series.index, index=series.index)
    gaps = pd.to_datetime(dates).diff().dt.days
    limit = 50 if monthly else 10
    return values.where(gaps <= limit).dropna()


def _gap_safe_rows(series: pd.Series, monthly: bool) -> pd.Series:
    """Keep per-row returns only when their preceding observation interval is usable."""
    dates = pd.Series(series.index, index=series.index)
    gaps = pd.to_datetime(dates).diff().dt.days
    limit = 50 if monthly else 10
    return series.where(gaps <= limit).dropna()


def _baseline_steps(series: pd.Series, target_type: str, monthly: bool) -> pd.Series:
    """Reproduce the published M0 trailing-window step definition."""
    trailing = series.tail(300)
    if target_type == "log_return":
        return _gap_safe_rows(trailing, monthly)
    return _gap_safe_diff(trailing, monthly)


def _fx_proxy_steps(unit: Unit, panel: str, asset: str, asof: str, monthly: bool) -> tuple[pd.Series, str] | None:
    """Return a typical same-panel FX innovation series for a managed/pegged target."""
    frame = unit.panels.get(panel)
    if monthly or frame is None or "fx" not in panel.lower():
        return None
    candidates: list[tuple[float, str, pd.Series]] = []
    for other in frame["asset"].astype(str).unique():
        if other == asset:
            continue
        sub = frame[(frame["asset"].astype(str) == other) & (frame["date"] <= pd.Timestamp(asof))]
        series = sub.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["value"].astype(float)
        if len(series) < 100 or (series <= 0).any():
            continue
        diff = _gap_safe_diff(series, False)
        vol = float(diff.std(ddof=1))
        if np.isfinite(vol) and vol > 0:
            candidates.append((vol, other, diff))
    if not candidates:
        return None
    median = float(np.median([row[0] for row in candidates]))
    _, name, steps = min(candidates, key=lambda row: abs(row[0] - median))
    return steps, name


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
    baseline_steps = _baseline_steps(series, unit.target_type, monthly)
    if unit.target_type == "log_return":
        if float(series.abs().median()) >= 0.2:
            raise ValueError(f"{asset}: log_return panel does not look like per-step returns")
        transform, anchor = "cumulative_return", 0.0
        steps = _gap_safe_rows(series, monthly)
    else:
        # The official M0 denominator walks level changes, including FX. Keeping the defensive
        # component in that same space avoids losing to the denominator because of a transform.
        transform, anchor = "level", float(series.iloc[-1])
        steps = _gap_safe_diff(series, monthly)
    if len(steps) < 12:
        raise ValueError(f"{asset}: insufficient usable history")
    proxy_asset: str | None = None
    own_long = float(steps.std(ddof=1))
    proxy = _fx_proxy_steps(unit, panel, asset, asof, monthly)
    if proxy is not None:
        proxy_steps, proxy_name = proxy
        proxy_vol = float(proxy_steps.std(ddof=1))
        if np.isfinite(own_long) and own_long < 0.25 * proxy_vol:
            # Preserve the target anchor and M0 steps. Only the regime-aware innovation pool is
            # borrowed, at a conservative half-panel scale, for peg-break risk.
            steps = proxy_steps * (0.50 * proxy_vol / max(float(proxy_steps.std(ddof=1)), 1e-12))
            proxy_asset = proxy_name
    long = float(steps.std(ddof=1))
    ewma = steps.pow(2).ewm(halflife=20 if not monthly else 6, min_periods=5).mean().pow(0.5)
    recent = float(ewma.iloc[-1])
    recent = recent if np.isfinite(recent) and recent > 0 else long
    long = long if np.isfinite(long) and long > 0 else max(abs(float(steps.mean())), 1e-6)
    level_factor = 1.0
    if transform == "level" and "percent_per_annum" in unit.value_unit.lower():
        mean_level = float(series.abs().mean())
        if mean_level > 0 and abs(anchor) > 0:
            level_factor = min(1.0, math.sqrt(abs(anchor) / mean_level))
            long *= level_factor
    # The validation roster is event-selected. A calm latest window is therefore weak evidence
    # of a calm forecast window, especially for F2/F4. This floor is applied only to the
    # regime-aware component; the explicit M0 hedge remains untouched.
    floor_by_family = {"T2-F1": 0.90, "T2-F2": 1.15, "T2-F3": 1.05, "T2-F4": 1.35}
    vol_recent = max(recent, floor_by_family.get(unit.family, 1.00) * long)
    lagged = steps.pow(2).ewm(halflife=20 if not monthly else 6, min_periods=5).mean().pow(0.5).shift(1)
    lagged = lagged.bfill().clip(lower=0.25 * long)
    standardized = (steps / lagged).replace([np.inf, -np.inf], np.nan).dropna()
    candidate_drift, weights = _rolling_drift(steps, max(1, min(unit.horizons)))
    # Daily yields and FX are extremely hard to directionally time and pseudo-cutoff tests show
    # that small sample trends overfit. Keep them driftless. Retain a strongly-shrunk drift only
    # for cumulative factor returns and genuinely monthly positive level series.
    if transform == "cumulative_return":
        drift = 0.15 * candidate_drift
    elif transform == "level" and monthly:
        drift = 0.35 * candidate_drift
    else:
        drift = 0.0
    width = _historical_width(steps, unit.horizons, monthly, long)
    if transform == "cumulative_return":
        width = float(np.clip(0.90 * width, 0.80, 1.20))
    return AssetState(
        asset, panel, transform, anchor, str(series.index[-1].date()), steps, baseline_steps,
        standardized, vol_recent, long, drift, width, level_factor, proxy_asset, weights
    )


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
        "level_volatility_factor": round(state.level_factor, 4),
        "innovation_proxy": state.proxy_asset,
        "selected_models": {k: round(v, 3) for k, v in state.model_weights.items()},
    } for state in states}


def prepare(unit: Unit, asof: str) -> list[AssetState]:
    return [_prepare(unit, asset, asof) for asset in unit.assets]


def _stationary_indices(rng: np.random.Generator, rows: int, length: int, mean_block: float) -> np.ndarray:
    indexes = np.empty(length, dtype=np.int64)
    offset = 0
    while offset < length:
        start = int(rng.integers(rows))
        block = int(rng.geometric(1.0 / mean_block))
        take = min(block, length - offset)
        indexes[offset : offset + take] = (start + np.arange(take)) % rows
        offset += take
    return indexes


def _normalise_pool(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float, copy=True)
    values -= values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=1, keepdims=True)
    return values / np.where(scale > 1e-12, scale, 1.0)


def _safe_covariance(frame: pd.DataFrame, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(frame) >= 12:
        mean = frame.mean().to_numpy(dtype=float)
        raw = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=1)
        cov = np.atleast_2d(raw).astype(float)
    else:
        mean = np.zeros(len(fallback), dtype=float)
        cov = np.diag(np.square(fallback))
    cov = np.nan_to_num((cov + cov.T) / 2, nan=0.0, posinf=0.0, neginf=0.0)
    values, vectors = np.linalg.eigh(cov)
    floor = max(float(np.max(values)), 1e-12) * 1e-8
    cov = vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T
    return mean, cov


def simulate(
    unit: Unit, asof: str, states: list[AssetState], text_view: TextView, n_draws: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """M0-anchored mixture with an event-aware empirical component and cautious text overlay."""
    rng = np.random.default_rng(seed)
    monthly = unit.target_frequency.lower() == "monthly"
    if monthly:
        explicit = monthly_steps(
            unit.assets, unit.horizons, {s.asset: s.last_date for s in states}, unit.card, unit.spec
        )
        step_grid = np.array([[explicit[(a, h)] for h in unit.horizons] for a in unit.assets], dtype=int)
    else:
        step_grid = np.array([[int(h) for h in unit.horizons] for _ in unit.assets], dtype=int)
    max_steps = int(step_grid.max())
    n_assets = len(states)

    # Defensive component: the published M0 distribution (trailing-300 additive Gaussian walk).
    baseline_frame = pd.concat([s.baseline_steps.rename(s.asset) for s in states], axis=1).dropna()
    baseline_scale = np.array([
        max(float(s.baseline_steps.std(ddof=1)), 1e-9) for s in states
    ])
    baseline_mean, baseline_cov = _safe_covariance(baseline_frame, baseline_scale)
    baseline_chol = np.linalg.cholesky(baseline_cov + 1e-12 * np.eye(n_assets))
    baseline_shocks = rng.standard_normal((n_draws, max_steps, n_assets)) @ baseline_chol.T
    baseline_paths = baseline_shocks + baseline_mean[None, None, :]

    # Alpha component: joint stationary bootstrap of volatility-standardised shocks.
    aligned = pd.concat([s.standardized.rename(s.asset) for s in states], axis=1).dropna()
    if n_assets == 1:
        correlation = np.ones((1, 1))
    elif len(aligned) >= 30:
        correlation = _nearest_correlation(aligned.corr().to_numpy(dtype=float))
    else:
        correlation = np.eye(n_assets)
    basis = _factor_basis(correlation)
    _project_text(text_view, unit.assets, basis, unit.family)

    custom_z = np.empty((n_draws, max_steps, n_assets), dtype=float)
    stress_default = {"T2-F1": 0.02, "T2-F2": 0.10, "T2-F3": 0.07, "T2-F4": 0.18}.get(unit.family, 0.05)
    stress_paths = np.zeros(n_draws, dtype=bool)
    if len(aligned) >= 40:
        full_pool = _normalise_pool(aligned)
        recent_rows = 36 if monthly else 252
        recent_pool = _normalise_pool(aligned.tail(min(recent_rows, len(aligned))))
        energy = np.sqrt(np.mean(np.square(full_pool), axis=1))
        stress_pool = full_pool[energy >= np.quantile(energy, 0.85)]
        stress_ok = len(stress_pool) >= 12
        stress_paths = rng.random(n_draws) < (stress_default if stress_ok else 0.0)
        block = 3.0 if monthly else 7.0
        for draw in range(n_draws):
            if stress_paths[draw]:
                pool = stress_pool
            else:
                pool = recent_pool if rng.random() < 0.60 else full_pool
            custom_z[draw] = pool[_stationary_indices(rng, len(pool), max_steps, block)]
    else:
        chol = np.linalg.cholesky(correlation + 1e-8 * np.eye(n_assets))
        custom_z = rng.standard_normal(custom_z.shape) @ chol.T

    # Only the empirical component receives volatility-state and vol-of-vol adjustments.
    custom_paths = np.empty_like(custom_z)
    progress = np.arange(1, max_steps + 1, dtype=float)
    vol_factor = np.exp(0.18 * rng.standard_normal(n_draws) - 0.5 * 0.18**2)
    for i, state in enumerate(states):
        tau = 6.0 if monthly else 60.0
        variance = state.vol_long**2 + (state.vol_recent**2 - state.vol_long**2) * np.exp(-progress / tau)
        calibration = float(np.clip(state.width_calibration, 0.85, 1.25))
        custom_paths[:, :, i] = (
            custom_z[:, :, i]
            * np.sqrt(np.clip(variance, 1e-18, None))[None, :]
            * calibration
            * vol_factor[:, None]
            + state.drift_step
        )
    # A modest common scale mixture gives event cards continuous tail support without allowing
    # it to dominate the M0 anchor.
    tail_df = 7.0 if unit.family == "T2-F4" else None
    if tail_df is not None:
        tail_scale = np.sqrt((tail_df - 2.0) / rng.chisquare(tail_df, size=n_draws))
        custom_paths *= tail_scale[:, None, None]

    default_m0_share = {"T2-F1": 0.80, "T2-F2": 0.65, "T2-F3": 0.70, "T2-F4": 0.55}.get(unit.family, 0.70)
    try:
        m0_share = float(os.environ.get("T2_M0_WEIGHT", default_m0_share))
    except ValueError:
        m0_share = default_m0_share
    m0_share = float(np.clip(m0_share, 0.25, 0.95))
    use_m0 = rng.random(n_draws) < m0_share
    paths = custom_paths.copy()
    paths[use_m0] = baseline_paths[use_m0]
    path_centres = np.tile(np.array([s.drift_step for s in states]), (n_draws, 1))
    path_centres[use_m0] = baseline_mean

    # Evidence-gated scenario mixture. Scale innovations around their own component centre, then
    # add a horizon-standardised text shift. Tail tilt is a rare path-level shock, not a repeated
    # daily jump.
    probabilities = np.array([s.probability for s in text_view.scenarios], dtype=float)
    probabilities /= probabilities.sum()
    chosen = rng.choice(len(text_view.scenarios), size=n_draws, p=probabilities)
    shortest = int(step_grid.min())
    base = paths.copy()
    for scenario_index, scenario in enumerate(text_view.scenarios):
        selected = chosen == scenario_index
        if not selected.any():
            continue
        selected_rows = np.flatnonzero(selected)
        for i, state in enumerate(states):
            multiplier = float(scenario.vol_mult.get(state.asset, 1.0))
            centre = path_centres[selected, i][:, None]
            horizon_sd = max(state.vol_recent, baseline_scale[i]) * math.sqrt(max(shortest, 1))
            drift_step = float(scenario.drift_sd.get(state.asset, 0.0)) * horizon_sd / max(shortest, 1)
            paths[selected, :, i] = centre + (base[selected, :, i] - centre) * multiplier + drift_step
            tilt = float(scenario.tail_tilt.get(state.asset, 0.0))
            rare = rng.random(len(selected_rows)) < 0.035 * abs(tilt)
            for row in selected_rows[rare]:
                when = int(rng.integers(max_steps))
                magnitude = horizon_sd * float(np.clip(abs(rng.normal(0.9, 0.25)), 0.4, 1.5))
                paths[row, when, i] += math.copysign(magnitude, tilt)

    cumulative = np.cumsum(paths, axis=1)
    output = np.empty((n_draws, n_assets, len(unit.horizons)), dtype=float)
    for i, state in enumerate(states):
        for j, _ in enumerate(unit.horizons):
            change = cumulative[:, step_grid[i, j] - 1, i]
            output[:, i, j] = state.anchor + change if state.transform == "level" else change
    diagnostics = {
        "aligned_rows": int(len(aligned)),
        "baseline_aligned_rows": int(len(baseline_frame)),
        "correlation": correlation.tolist(),
        "factor_dimension": int(basis.shape[1]),
        "parametric_tail_df": tail_df,
        "m0_share": m0_share,
        "stress_probability": stress_default,
        "stress_draw_share": float(stress_paths.mean()),
        "step_grid": step_grid.tolist(),
        "text_used": text_view.used,
        "text_note": text_view.note,
        "house_calls": text_view.call_count,
        "model": text_view.model,
        "scenarios": [scenario.__dict__ for scenario in text_view.scenarios],
        "assets": numeric_context(states),
    }
    return output, diagnostics
