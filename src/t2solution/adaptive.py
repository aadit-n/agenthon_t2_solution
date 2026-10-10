"""Cutoff-local probabilistic experts; no fitted artifact or future-answer lookup.

Each expert is refitted on the supplied history. Completed, non-overlapping historical
forecasts determine shared mixture weights, so a draw uses one expert across the whole
asset/horizon grid. The learner's analytic Gaussian CRPS/pinball objective is a training
surrogate, not a reimplementation of the organizer's normalized competition scorer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from .unit import Unit, target_series

NAMES = ("trailing_drift", "driftless", "conditional_vol", "ridge_context", "empirical_horizons")
PRIOR = np.array([.25, .20, .20, .20, .15])
TAIL_LEVELS = np.array([.01, .05, .95, .99])
TAIL_Z = np.array([NormalDist().inv_cdf(float(q)) for q in TAIL_LEVELS])
# A bounded, in-memory cache avoids refitting the same cutoff for every simulation seed.
# The key binds actual input values/dates and semantics, not a unit id or answer file.
_CACHE: dict[str, tuple[list["Expert"], np.ndarray, dict]] = {}


@dataclass
class Expert:
    name: str
    mean: np.ndarray
    covariance: np.ndarray
    pool: np.ndarray | None = None


def _psd(cov: np.ndarray) -> np.ndarray:
    cov = np.atleast_2d(cov)
    cov = (cov + cov.T) * .5
    eig, basis = np.linalg.eigh(cov)
    floor = max(float(np.max(eig)), 1e-14) * 1e-9
    return (basis * np.maximum(eig, floor)) @ basis.T


def _ew_variance(steps: np.ndarray, halflife: float) -> np.ndarray:
    decay = math.exp(math.log(.5) / halflife)
    result = np.empty_like(steps)
    state = np.square(steps[0])
    for i, row in enumerate(steps):
        state = decay * state + (1.0 - decay) * np.square(row)
        result[i] = state
    return result


def _features(raw: np.ndarray, steps: np.ndarray, vol: np.ndarray, rates: bool) -> np.ndarray:
    """Features at each origin use only observations at or before that origin."""
    cumulative = np.vstack([np.zeros(steps.shape[1]), np.cumsum(steps, axis=0)])
    n, a = raw.shape
    features = np.zeros((n, a, 5 if rates else 4))
    features[:, :, 0] = 1.0
    for j, lag in enumerate((21, 63, 252), 1):
        begin = np.maximum(np.arange(n) - lag, 0)
        denominator = np.maximum(vol * math.sqrt(lag), 1e-12)
        features[:, :, j] = (cumulative - cumulative[begin]) / denominator
    if rates:
        rolling = pd.DataFrame(raw).rolling(756, min_periods=20).mean().to_numpy()
        features[:, :, 4] = np.nan_to_num((raw - rolling) / np.maximum(vol * math.sqrt(252), 1e-12))
    return np.clip(features, -5, 5)


def _experts(raw: np.ndarray, horizons: np.ndarray, returns: bool, rates: bool) -> list[Expert]:
    # Panel returns are decimal simple returns; the requested target is cumulative log.
    steps = np.log1p(raw[1:]) if returns else np.diff(raw, axis=0)
    n_assets, n_h = raw.shape[1], len(horizons)
    recent = steps[-299:]
    covariance = np.atleast_2d(np.cov(recent, rowvar=False, ddof=1))
    sd = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
    mean_step = recent.mean(axis=0)
    anchor = np.zeros(n_assets) if returns else raw[-1]
    anchors = np.repeat(anchor, n_h)
    hflat = np.tile(horizons, n_assets)
    asset_index = np.repeat(np.arange(n_assets), n_h)
    brownian = covariance[asset_index[:, None], asset_index[None, :]] * np.minimum(hflat[:, None], hflat[None, :])
    brownian = _psd(brownian)
    m0 = Expert(NAMES[0], anchors + np.repeat(mean_step, n_h) * hflat, brownian)
    zero = Expert(NAMES[1], anchors, brownian)

    fast = _ew_variance(steps, 21)
    slow = _ew_variance(steps, 252)
    # The origin volatility estimate is strictly lag-safe for future labels.
    v_origin = np.vstack([np.sqrt(np.maximum(slow[0], 1e-18)),
                          np.sqrt(np.maximum(.65 * fast + .35 * slow, 1e-18))])
    current_var = .65 * fast[-1] + .35 * slow[-1]
    long_var = slow[-1]
    times = np.arange(1, int(horizons.max()) + 1)
    variances = long_var[:, None] + (current_var - long_var)[:, None] * np.exp(-times[None, :] / 63)
    integrated = np.cumsum(variances, axis=1)
    corr = covariance / np.outer(sd, sd)
    corr = .9 * corr + .1 * np.eye(n_assets)
    small = np.minimum(hflat[:, None], hflat[None, :]).astype(int) - 1
    # Integrated cross-covariance shares innovations up to the earlier horizon.
    cov_time = np.cumsum(np.sqrt(variances[:, None, :] * variances[None, :, :]), axis=2)
    conditional_cov = _psd(corr[asset_index[:, None], asset_index[None, :]] *
                          cov_time[asset_index[:, None], asset_index[None, :], small])
    conditional = Expert(NAMES[2], anchors, conditional_cov)

    features = _features(raw, steps, v_origin, rates)
    cumulative = np.vstack([np.zeros(n_assets), np.cumsum(steps, axis=0)])
    # Complete joint episodes end inside this prefix. All horizons use the same episode.
    origins = np.arange(max(252, len(raw) - 2500), len(raw) - int(horizons.max()),
                        max(5, int(horizons.min()) // 4), dtype=int)
    if len(origins) < 16:
        return [m0, zero, conditional,
                Expert(NAMES[3], anchors.copy(), conditional_cov.copy()),
                Expert(NAMES[4], anchors.copy(), conditional_cov.copy())]
    changes = np.stack([cumulative[origins + h] - cumulative[origins] for h in horizons], axis=2)
    norm = v_origin[origins, :, None] * np.sqrt(horizons)[None, None, :]
    responses = changes / np.maximum(norm, 1e-12)
    response_flat = responses.reshape(len(origins), -1)
    current_scale = (np.sqrt(integrated[:, horizons - 1])).reshape(-1)
    predicted = np.zeros((n_assets, n_h))
    residual = np.empty_like(responses)
    for a in range(n_assets):
        x = features[origins, a, :]
        # Shrink direction strongly; the intercept is also penalized. No external coefficients.
        penalty = 20.0 * np.eye(x.shape[1])
        beta = np.linalg.solve(x.T @ x + penalty, x.T @ responses[:, a, :])
        predicted[a] = np.clip(features[-1, a] @ beta, -1.25, 1.25)
        residual[:, a, :] = responses[:, a, :] - x @ beta
    # Empirical multihorizon variance captures non-sqrt-time scaling and dependence.
    empirical_cov = np.atleast_2d(np.cov(response_flat, rowvar=False, ddof=1))
    empirical_cov = .85 * empirical_cov + .15 * np.eye(len(anchors))
    empirical_cov = _psd(empirical_cov * np.outer(current_scale, current_scale))
    residual_cov = np.atleast_2d(np.cov(residual.reshape(len(origins), -1), rowvar=False, ddof=1))
    # Finite-sample protection: fit residuals are not honest out-of-sample residuals.
    residual_cov = 1.12 * (.7 * residual_cov + .3 * np.eye(len(anchors)))
    residual_cov = _psd(residual_cov * np.outer(current_scale, current_scale))
    ridge = Expert(NAMES[3], anchors + predicted.reshape(-1) * current_scale, residual_cov)
    centered = response_flat - response_flat.mean(axis=0, keepdims=True)
    empirical = Expert(NAMES[4], anchors.copy(), empirical_cov, centered * current_scale)
    return [m0, zero, conditional, ridge, empirical]


def _training_loss(expert: Expert, actual: np.ndarray, scale: np.ndarray) -> float:
    """Analytic Gaussian CRPS plus tail pinball; input-only scale, no realized divisor."""
    sd = np.sqrt(np.maximum(np.diag(expert.covariance), 1e-18))
    z = (actual - expert.mean) / sd
    cdf = .5 * (1 + np.array([math.erf(float(v) / math.sqrt(2)) for v in z]))
    phi = np.exp(-.5 * z**2) / math.sqrt(2 * math.pi)
    crps = sd * (z * (2 * cdf - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    quantiles = expert.mean[None, :] + TAIL_Z[:, None] * sd[None, :]
    errors = actual[None, :] - quantiles
    pinball = np.maximum(TAIL_LEVELS[:, None] * errors, (TAIL_LEVELS[:, None] - 1) * errors)
    tail_expected = float(np.mean(np.exp(-.5 * TAIL_Z**2) / math.sqrt(2 * math.pi)))
    return float(.7 * np.mean(crps / (scale / math.sqrt(math.pi))) +
                 .3 * np.mean(pinball / (scale[None, :] * tail_expected)))


def fit(unit: Unit, asof: str) -> tuple[list[Expert], np.ndarray, dict] | None:
    if unit.target_frequency.lower() == "monthly":
        return None
    series = [target_series(unit, a, asof)[0].rename(a) for a in unit.assets]
    history = pd.concat(series, axis=1).dropna()
    if len(history) < 300 or max(unit.horizons) <= 0:
        return None
    gaps = history.index.to_series().diff().dt.days.to_numpy()
    raw = history.to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("adaptive_invalid_history")
    returns = unit.target_type == "log_return"
    if returns and np.any(raw <= -1):
        raise ValueError("adaptive_invalid_simple_returns")
    # Never bridge an absent segment as one trading observation.
    if np.any(gaps[1:] > 10):
        return None
    rates = not returns and "percent_per_annum" in unit.value_unit.lower()
    horizons = np.array(unit.horizons, dtype=int)
    key = hashlib.sha256(raw.tobytes() + history.index.asi8.tobytes() + horizons.tobytes() +
                         repr((returns, rates, unit.assets)).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    stride = int(horizons.max()) + 5
    origins = np.arange(max(300, len(raw) - 2500), len(raw) - int(horizons.max()), stride)
    if len(origins) < 4:
        return None
    losses = []
    for cut in origins[-20:]:
        experts = _experts(raw[:cut], horizons, returns, rates)
        if returns:
            actual = np.stack([np.log1p(raw[cut:cut + h]).sum(axis=0) for h in horizons], axis=1).reshape(-1)
        else:
            actual = np.stack([raw[cut + h - 1] for h in horizons], axis=1).reshape(-1)
        scale = np.sqrt(np.maximum(np.diag(experts[0].covariance), 1e-18))
        losses.append([_training_loss(expert, actual, scale) for expert in experts])
    losses = np.asarray(losses)
    # Recent completed episodes carry more weight, without making overlapping labels look IID.
    recency = np.exp(np.linspace(-math.log(2), 0, len(losses)))
    average = np.average(np.minimum(losses, 8), weights=recency, axis=0)
    evidence = len(losses) / (len(losses) + 8.0)
    likelihood = PRIOR * np.exp(-12.0 * evidence * (average - average.min()))
    learned = likelihood / likelihood.sum()
    weights = evidence * learned + (1 - evidence) * PRIOR
    experts = _experts(raw, horizons, returns, rates)
    diagnostics = {"weights": dict(zip(NAMES, weights.tolist(), strict=True)),
                   "training_surrogate_loss": dict(zip(NAMES, average.tolist(), strict=True)),
                   "completed_nonoverlapping_origins": int(len(losses)),
                   "first_training_origin": str(history.index[int(origins[-20]) - 1].date()) if len(origins) >= 20
                                            else str(history.index[int(origins[0]) - 1].date()),
                   "last_training_origin": str(history.index[int(origins[-1]) - 1].date()),
                   "last_training_label": str(history.index[int(origins[-1] + horizons.max() - 1)].date()),
                   "history_end": str(history.index[-1].date()),
                   "native_log_returns": returns,
                   "input_fingerprint": key}
    result = experts, weights, diagnostics
    if len(_CACHE) >= 24:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = result
    return result


def sample(fitted, draws: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    experts, weights, diagnostics = fitted
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(experts), size=draws, p=weights)
    width = len(experts[0].mean)
    output, centers = np.empty((draws, width)), np.empty((draws, width))
    for index, expert in enumerate(experts):
        selected = chosen == index
        count = int(selected.sum())
        if not count:
            continue
        chol = np.linalg.cholesky(_psd(expert.covariance))
        if expert.pool is not None:
            innovations = expert.pool[rng.integers(len(expert.pool), size=count)]
            # Continuous support; no scenario/outcome table is packaged or read.
            innovations = math.sqrt(.95) * innovations + math.sqrt(.05) * (rng.standard_normal((count, width)) @ chol.T)
        else:
            innovations = rng.standard_normal((count, width)) @ chol.T
        output[selected] = expert.mean + innovations
        centers[selected] = expert.mean
    return output, centers, {**diagnostics, "realized_expert_shares": dict(zip(
        NAMES, np.bincount(chosen, minlength=len(experts)).astype(float) / draws, strict=True))}
