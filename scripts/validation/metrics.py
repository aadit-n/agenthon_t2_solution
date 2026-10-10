"""Use organizer scoring functions; compute only the documented input-only M0 yardstick.

The Gaussian distribution and expected-error divisors reproduce the published M0 procedure
at the pinned Track 2 revision (docs/M0-BASELINE.md, sections 3 and 5). No outcome is used
to build a divisor. This is a historical proxy, not a sealed official reference artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gamma, hyp1f1, ndtr
from scipy.stats import norm
from qfbench2_track_forecasting.scoring import _composite

from t2solution.unit import Unit

CAP = 8.0
LEVELS = (.01, .05, .95, .99)


@dataclass
class GaussianReference:
    mean: np.ndarray
    covariance: np.ndarray
    cholesky: np.ndarray

    def sample(self, draws: int, seed: int) -> np.ndarray:
        return self.mean + np.random.default_rng(seed).standard_normal((draws, len(self.mean))) @ self.cholesky.T


def gaussian_reference(trial: Unit, counts: np.ndarray) -> GaussianReference:
    steps, anchors = [], []
    asof = trial.card["provenance"]["data_cutoff"]
    for asset in trial.assets:
        # M0 selects the first supplied panel containing the asset, in file/load order.
        for frame in trial.panels.values():
            rows = frame.loc[(frame.asset.astype(str) == asset) & (frame.date <= pd.Timestamp(asof))]
            if not rows.empty:
                rows = rows.sort_values("date")
                if rows.date.duplicated().any():
                    raise ValueError("reference_duplicate_dates")
                series = rows.set_index("date")["value"].astype(float).tail(300)
                break
        else:
            raise ValueError("reference_missing_asset")
        gaps = series.index.to_series().diff().dt.total_seconds() / 86400
        threshold = max(10 * float(gaps.median()), 5)
        if trial.target_type == "log_return":
            if float(series.abs().median()) >= .2:
                raise ValueError("reference_invalid_returns")
            changes, anchor = series, 0.0
        else:
            changes, anchor = series.diff(), float(series.iloc[-1])
        steps.append(changes.where(gaps <= threshold).rename(asset))
        anchors.append(anchor)
    aligned = pd.concat(steps, axis=1).dropna()
    if len(aligned) < 12:
        raise ValueError("reference_insufficient_aligned_history")
    mu = aligned.mean().to_numpy(dtype=float)
    cov = np.atleast_2d(np.cov(aligned.to_numpy(dtype=float), rowvar=False, ddof=1))
    indexes = np.repeat(np.arange(len(trial.assets)), len(trial.horizons))
    grid = counts.reshape(-1)
    mean = np.repeat(anchors, len(trial.horizons)) + mu[indexes] * grid
    covariance = cov[np.ix_(indexes, indexes)] * np.minimum.outer(grid, grid)
    covariance = (covariance + covariance.T) / 2 + 1.1e-9 * np.eye(len(grid))
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("reference_nonfinite")
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        covariance = np.diag(np.diag(covariance))
        root = np.linalg.cholesky(covariance)
    return GaussianReference(mean, covariance, root)


def expected_scales(mean: np.ndarray, covariance: np.ndarray, levels=LEVELS) -> dict[str, float]:
    mean, covariance = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    if covariance.shape != (len(mean), len(mean)) or not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("invalid_reference_distribution")
    sd = np.sqrt(np.diag(covariance))
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("invalid_reference_variance")
    if not levels or not all(0 < value < 1 for value in levels):
        raise ValueError("invalid_tail_levels")
    joint = 0.0
    for i in range(len(mean)):
        for j in range(i):
            variance = float(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j])
            if variance <= 0:
                continue
            s, delta = np.sqrt(variance), float(mean[i] - mean[j])
            z = delta / s
            absolute = s * np.sqrt(2 / np.pi) * np.exp(-.5 * z*z) + delta * (1 - 2 * ndtr(-z))
            half = np.sqrt(s) * 2**.25 * gamma(.75) / np.sqrt(np.pi) * hyp1f1(-.25, .5, -.5 * z*z)
            pair = absolute - half**2
            if not np.isfinite(pair) or pair < -1e-10 * max(absolute, 1.0):
                raise ValueError("unstable_reference_joint_scale")
            joint += 2 * max(float(pair), 0.0)  # The public variogram sums ordered pairs.
    values = {"marginal": float(sd.mean() / np.sqrt(np.pi)),
              "joint": joint if joint > 0 else 1.0,
              "tail": float(sd.mean() * norm.pdf(norm.ppf(levels)).mean())}
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError("invalid_reference_scale")
    return values


def score_samples(samples: np.ndarray, actual: np.ndarray, unit: Unit, scales: dict[str, float]):
    samples = np.asarray(samples, dtype=float)
    cells = len(unit.assets) * len(unit.horizons)
    if samples.ndim == 3 and samples.shape[1:] != (len(unit.assets), len(unit.horizons)):
        raise ValueError("invalid_sample_grid")
    if samples.ndim not in (2, 3) or samples.shape[0] < max(200, unit.n_draws_min):
        raise ValueError("invalid_sample_count")
    flat = samples.reshape(len(samples), -1)
    if flat.shape[1] != cells or actual.shape != (cells,) or not np.isfinite(flat).all() or not np.isfinite(actual).all():
        raise ValueError("invalid_sample_values")
    if not all(np.isfinite(scales[name]) and scales[name] > 0 for name in ("marginal", "joint", "tail")):
        raise ValueError("invalid_reference_scale")
    params = unit.card.get("scoring", {}).get("params", {})
    joint, tail = params.get("joint", "variogram"), params.get("tail_metric", "pinball")
    if joint != "variogram" or tail != "pinball":
        raise ValueError("unsupported_validation_metric")
    declared = params.get("weights", {"marginal": .5, "joint": .3, "tail": .2})
    weights = tuple(float(declared[name]) for name in ("marginal", "joint", "tail"))
    if not all(np.isfinite(w) and w >= 0 for w in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("invalid_component_weights")
    if cells == 1:
        live = weights[0] + weights[2]
        if live <= 0:
            raise ValueError("invalid_single_cell_weights")
        weights = (weights[0] / live, 0.0, weights[2] / live)
    result = _composite(flat, actual, weights=weights, tail_levels=tuple(params.get("tail_levels", LEVELS)),
                        joint=joint, tail_metric=tail, ref_scale=scales)
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite_component")
    lo, hi = np.quantile(flat, [.05, .95], axis=0)
    return {**{name: float(result[name]) for name in ("marginal", "joint", "tail")},
            **{name + "_normalized": float(result[name] / scales[name]) for name in ("marginal", "joint", "tail")},
            "unclipped_loss": float(result["composite"]), "local_loss": float(np.clip(result["composite"], 0, CAP)),
            "clip_applied": bool(result["composite"] < 0 or result["composite"] > CAP),
            "coverage90": float(np.mean((lo <= actual) & (actual <= hi))),
            "below05": float(np.mean(actual < lo)), "above95": float(np.mean(actual > hi)),
            "joint_weight": weights[1]}
