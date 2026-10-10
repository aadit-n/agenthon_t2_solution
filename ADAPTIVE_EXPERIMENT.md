# Adaptive numerical ensemble — research, not a release

Main's scored v0.5 predictor is preserved. Work is on `experiment/adaptive-ensemble`.
The normal CLI and `model.simulate()` still run the incumbent exactly. The experimental
engine requires `--numeric-engine adaptive-experimental` or `scripts/validate_adaptive.py`.
Do not build/upload this branch until the acceptance criteria below have been met.
No registry write or CodaBench submission is part of this experiment.

## Hypothesis

The incumbent puts 55–80% of numerical paths in a trailing-window M0 distribution,
with fixed weights by family. That cannot adapt when a task's historical forecasts
show that drift extrapolation, current volatility or sqrt-time scaling are poor.

The candidate refits five non-neural experts on each supplied input prefix:

- Trailing-window drift and Brownian covariance, in native target units.
- Driftless Brownian forecasts.
- Fast/slow conditional volatility with horizon-dependent mean reversion of variance.
- Ridge-regression context forecasts using lagged momentum and, for rates, level deviation.
- Empirical complete multihorizon episodes, preserving dependence and non-Gaussian shape.

Completed non-overlapping historical forecasts determine shared expert weights using an
analytic Gaussian CRPS/tail-pinball training surrogate. This is **not** the organizer's
competition scorer or an implementation of the papers' BOA algorithm. The actual offline
comparison still uses the unchanged pinned organizer scorer, joint and tail terms included.
Weights are shrunk toward a diversified prior; selection has at most 20 historical origins.
Each output draw selects one expert across the entire asset/horizon grid. 20% of paths
retain the incumbent numerical distribution as a stress hedge. Short, monthly or gapped
histories use the incumbent exactly.

The v0.5 text reader, House transport, evidence requirements and numerical signal caps
are unchanged. For eligible daily tasks the existing scenario adjustments are applied to
the new joint distribution. We cannot measure real House behavior locally: no development
endpoint is available. Synthetic tests check that text adjustments still reach forecasts.

## Information boundary

The learner reads only in-memory supplied panels at or before the invocation cutoff.
All inner fitting and selection labels end within that same supplied prefix. No outcomes
from a public card's future target window are reconstructed. Return experts use log1p of
decimal simple returns, as required for cumulative log-return targets. Cache keys include
input values, dates and semantics; they are not task-id/answer lookup tables. The cache is
bounded and exists in memory only. No fitted artifact, global calibration table or training
panel is packaged.

The supplied-vintage limitation of retrospective validation still applies. Rows dated
before a pseudo-origin do not independently prove historical publication availability.

## Predeclared release bar

The user does not want to spend a submission for a tiny improvement. Therefore:

1. Use only the 264-case tuning split for development; 2,000 draws and seeds 0/19/73.
2. A candidate must reduce tuning group-balanced composite loss by **at least 10%**
   before opening either holdout. No tuning family may regress by more than .01 loss units.
3. Freeze one candidate before accessing holdouts. Both holdout losses must improve by
   **at least 10%**, and the existing validation promotion gate must also pass.
4. No engine failures; numerical/schema/preflight checks must pass. Verify runtime and
   reproducibility. Do not modify the validation harness to accommodate a candidate.
5. If the bar is missed, retain the incumbent. Do not call a failed candidate an improvement,
   do not build its submission ZIP, and do not consume a CodaBench attempt.

10% is an experimental acceptance threshold, not a prediction or a guarantee about the
leaderboard. Event-selected text-conditioned tasks differ from these historical proxies.
Do not repeatedly tune against the same two holdouts after opening them.

## Sources and eligibility

- [Organizer artifact policy](https://github.com/Agenthon-2026/track2-forecasting-public/blob/main/docs/ARTIFACT-POLICY.md):
  non-neural numerical fitting is allowed subject to per-task cutoffs and provenance.
- [V'yugin and Trunov, probabilistic expert aggregation](https://arxiv.org/abs/2109.14309):
  supports the general research direction, not competition performance claims.
- [Berrisch and Ziel, multivariate probabilistic CRPS learning](https://arxiv.org/abs/2303.10019):
  motivates scoring-led forecast combination. Our shared whole-grid mixture is simpler
  than their quantile-based method and preserves joint draws explicitly.

Pure runtime fitting code is not a bundled pretrained model. No additional language model,
neural checkpoint or external market dataset has been introduced. The House disclosure
remains unchanged. Update release provenance only if this experiment is actually promoted.

## Completed v1 experiment: rejected, do not submit

Full tuning comparison: 264 historical cases, 66 groups, 2,000 draws, seeds 0/19/73,
zero failures. House was neutral. Runtime was 312.188 seconds.

| Family / overall | Incumbent local loss | Adaptive local loss | Relative reduction |
|---|---:|---:|---:|
| F1 | 1.412676 | 1.383721 | 2.05% |
| F2 | 1.234984 | 1.260261 | -2.05% (worse) |
| F3 | 1.038730 | 1.039382 | -0.06% (worse) |
| F4 | 1.180700 | 1.103390 | 6.55% |
| Group-balanced overall | 1.199504 | 1.180587 | 1.58% |

This misses both the 10% tuning threshold and the F2 regression guard. Neither holdout
was opened. No candidate ZIP, image push or official submission was made. Passing 86
automated tests and the exemplar's g0–g3 checks established mechanical correctness only.

Evidence is in ignored `backtest_runs/validation/adaptive-v1-tune` and
`adaptive-v1-tune-comparison`; source/input/scorer hashes are in the completed manifest.
That run selected the experimental engine as the branch default during development.
After rejection, the engine became explicitly opt-in. The run manifest therefore binds
the earlier model source, not the present incumbent-default adapter; do not claim a
present-source fingerprint match. The numerical expert code/parameters are unchanged.

After restoring the incumbent default, 87 automated tests passed. Seven representative
tuning cases (21 seed/case forecasts) reproduced every stored candidate metric bit-exactly
through the current opt-in engine. Three long-history public inputs also reproduced the
main-branch numerical draws bit-exactly through the default engine. Neither check used
holdout results, real House calls or a submission.

To reproduce the same experiment through the unchanged harness with the current adapter:

```powershell
..\.venv_t2_validation_v26\Scripts\python.exe scripts/validate_adaptive.py `
  --plan backtest_runs\validation\full-plan-2026-10-10\plan.json `
  --splits tune --out backtest_runs\validation\adaptive-v1-tune-reproduction
```

Use `scripts/compare_validation.py --splits tune` against the original incumbent run.
Do not use this experiment as a reason to spend a CodaBench attempt.
