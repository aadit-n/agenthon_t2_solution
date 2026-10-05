"""Small, independently implemented research extensions to the scored v0.2 engine.

Corrected target semantics are separate from optional drift/ECC experiments. All
fitting uses this unit's pre-cutoff inputs; no fitted weights or answers are shipped.
ECC inspiration: Schefzik et al. (2013), https://arxiv.org/abs/1302.7149.
Quantile pooling: Berrisch & Ziel (2021), https://arxiv.org/abs/2102.00968.
These adaptations are not literal implementations or claims of the papers' results.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from . import model
from .monthly import monthly_steps
from .text import TextView
from .unit import Unit


PROFILES = ('v2', 'corrected', 'drift', 'ecc', 'recent', 'combined')


def prepare(unit: Unit, asof: str, profile: str = 'corrected') -> list[model.AssetState]:
    if profile not in PROFILES:
        raise ValueError(f'unknown candidate profile: {profile}')
    if profile == 'v2':
        return model.prepare(unit, asof)
    panels = {}
    for name, original in unit.panels.items():
        frame = original[original.date <= pd.Timestamp(asof)].copy()
        if unit.target_type == 'log_return':
            mask = frame.asset.astype(str).isin(unit.assets)
            values = frame.loc[mask, 'value'].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= -1).any():
                raise ValueError('cumulative log-return targets require finite simple returns greater than -1')
            frame.loc[mask, 'value'] = np.log1p(values)
        panels[name] = frame
    trial = replace(unit, panels=panels)
    states = model.prepare(trial, asof)
    if unit.target_frequency.lower() == 'monthly':
        grid = monthly_steps(unit.assets, unit.horizons, {s.asset: s.last_date for s in states}, unit.card, unit.spec)
        for state in states:
            h = min(grid[(state.asset, key)] for key in unit.horizons)
            _, weights = model._rolling_drift(state.steps, h)
            # The legacy helper caps DAILY price drift at 0.15 innovation SD. Monthly
            # index increments can have a genuine mean larger than that SD; retain the
            # selected estimates and the existing 0.35 shrink, not the daily-price cap.
            estimates = model._candidate_drifts(state.steps.to_numpy(), len(state.steps))
            state.drift_step = 0.35 * float(np.dot(list(weights.values()), estimates))
            state.model_weights = weights
    if profile == 'recent':
        # Limit only the empirical component, not the trailing-300 numerical hedge.
        # Keep sparse transfer histories intact; a large gap is never manufactured into a step.
        for state in states:
            if unit.target_frequency.lower() != 'monthly' and len(state.steps) > 504:
                state.steps = state.steps.tail(504)
                state.standardized = state.standardized.tail(504)
                state.vol_long = float(state.steps.std(ddof=1)) * state.level_factor
                floor = {'T2-F1': 0.90, 'T2-F2': 1.15, 'T2-F3': 1.05, 'T2-F4': 1.35}.get(unit.family, 1.0)
                recent = float(state.steps.pow(2).ewm(halflife=20).mean().iloc[-1] ** 0.5)
                state.vol_recent = max(recent, floor * state.vol_long)
    return states


def quantile_blend(source: np.ndarray, proposal: np.ndarray, weight: float) -> np.ndarray:
    """Pool marginal quantiles, then restore the incumbent empirical copula exactly."""
    if source.shape != proposal.shape or not 0 <= weight <= 1:
        raise ValueError('quantile pools require equal shapes and a weight in [0,1]')
    if weight == 0:
        return source.copy()
    if not np.isfinite(source).all() or not np.isfinite(proposal).all():
        raise ValueError('nonfinite quantile pool')
    order = np.argsort(source, axis=0, kind='stable')
    pooled = (1 - weight) * np.sort(source, axis=0) + weight * np.sort(proposal, axis=0)
    result = np.empty_like(source)
    np.put_along_axis(result, order, pooled, axis=0)
    return result


def _error_pool(state: model.AssetState, h: int, monthly: bool) -> np.ndarray:
    """Completed, nonoverlapping h-step errors; the conditioning window is always earlier."""
    series = state.steps
    x = series.to_numpy(dtype=float)
    minimum = 36 if monthly else 100
    errors = []
    for origin in list(range(len(x) - h, minimum, -h))[:48]:
        future = series.iloc[origin:origin + h]
        # A filtered step series can still have a discontinuity. Do not bridge it.
        span = series.index[origin - 1:origin + h]
        gaps = np.diff(span.values).astype('timedelta64[D]').astype(float)
        if (gaps > (50 if monthly else 10)).any():
            continue
        hist = x[max(0, origin - 300):origin]
        sd = float(np.std(hist, ddof=1)) * math.sqrt(h)
        if not np.isfinite(sd) or sd <= 1e-10:
            continue
        drift = float(np.mean(hist)) if monthly or state.transform == 'cumulative_return' else 0.0
        errors.append((float(future.sum()) - h * drift) / sd)
    return np.asarray(errors)


def postprocess(unit: Unit, states: list[model.AssetState], samples: np.ndarray,
                diagnostics: dict, profile: str, seed: int) -> tuple[np.ndarray, dict]:
    result = samples.copy()
    details = {'profile': profile, 'copula_preserved': True, 'cells': {}}
    grid = np.asarray(diagnostics['step_grid'])
    baseline = pd.concat([s.baseline_steps.rename(s.asset) for s in states], axis=1).dropna()
    means = baseline.mean().to_numpy() if len(baseline) else np.zeros(len(states))
    monthly = unit.target_frequency.lower() == 'monthly'
    rng = np.random.default_rng(seed + 719)
    for i, state in enumerate(states):
        for j, horizon in enumerate(unit.horizons):
            cell = result[:, i, j]
            sd = max(float(np.std(cell, ddof=1)), 1e-10)
            shift = 0.0
            if profile in {'drift', 'combined'} and not monthly and state.transform == 'level':
                # Shrink only a small part of noisy historical drift; do not invent a new trend.
                shift = float(np.clip(-0.5 * diagnostics['m0_share'] * means[i] * grid[i, j], -0.25 * sd, 0.25 * sd))
                cell = cell + shift
            pool = _error_pool(state, int(grid[i, j]), monthly) if profile in {'ecc', 'combined'} else np.array([])
            if len(pool) >= 12:
                # A small quantile pool retains 85% of the incumbent distribution and all
                # cross-asset/horizon ranks. Historical errors cannot remove its tail support.
                reference_sd = max(float(state.baseline_steps.std(ddof=1)) * math.sqrt(grid[i, j]), 1e-10)
                proposal = np.median(cell) + rng.choice(pool, len(cell), replace=True) * reference_sd
                pooled = quantile_blend(cell[:, None, None], proposal[:, None, None], 0.15).reshape(-1)
                # Additional trust region: no cell moves more than a quarter of an incumbent SD.
                cell = np.clip(pooled, cell - 0.25 * sd, cell + 0.25 * sd)
            result[:, i, j] = cell
            details['cells'][f'{state.asset}:{horizon}'] = {'drift_shift': shift, 'completed_errors': len(pool)}
    diagnostics = {**diagnostics, 'research': details}
    return result, diagnostics


def simulate(unit: Unit, asof: str, states: list[model.AssetState], view: TextView,
             n_draws: int, seed: int, profile: str = 'corrected') -> tuple[np.ndarray, dict]:
    samples, diagnostics = model.simulate(unit, asof, states, copy.deepcopy(view), n_draws, seed)
    return postprocess(unit, states, samples, diagnostics, profile, seed)
