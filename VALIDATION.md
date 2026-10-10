# Real-engine validation

This is a development tool, not a new forecasting submission. Branch `v0.6-validation`
starts at the scored v0.5 baseline, `ff8ede4`. The forecasting package, Dockerfile,
submission descriptor and packaging environment are unchanged. No new ZIP is needed.

The tool runs the actual `model.prepare()` and `model.simulate()` functions. Each forecast
uses one simulator call at the requested draw count, rather than concatenating 200-draw
batches or replacing the engine with a Gaussian approximation. House is not called.
Text is deliberately neutral; this does **not** measure the quality of House reasoning.

## Setup and tests

The isolated `.venv_t2_validation_v26` environment was created for this work. From this
repository in PowerShell:

```powershell
# Only needed on a machine where the new validation environment has not been set up:
..\.venv_t2_validation\Scripts\python.exe -m venv ..\.venv_t2_validation_v26
..\.venv_t2_validation_v26\Scripts\python.exe -m pip install -r requirements-validation.txt

# New validation tests, separate from the submission's dependency set:
..\.venv_t2_validation_v26\Scripts\python.exe -m unittest discover -s validation_tests -v

# Existing forecasting/text tests in their unchanged environment:
$env:PYTHONPATH = "$PWD\src"
..\.venv_t2_validation\Scripts\python.exe -m unittest discover -s tests -v
```

The scorer is pinned to Track 2 commit `30c8019d997f930eab9ba569d089d12298d86d8c`.
The shared toolkit is v2.6.0, commit `bd01548e34d21fd660d88fd06157078fb25ece4e`.
NumPy/pandas/PyArrow retain the submission's versions. Dependencies, metric source
checksums and engine/harness source checksums are recorded with each run. A wrong
environment fails explicitly; it is not a skipped-green validation result.

The validation environment also installs the organizer's reference `forecast` executable.
Do not use that bare command to test our agent. Use the validation scripts here, or invoke
our source CLI in the original environment as shown in the README.

Do not install these requirements into `.venv_t2_validation`, overwrite the existing
image, or change the descriptor just to run this tool. The old packaging script still
pins toolkit 2.4.4; review current submission instructions before a future release.

## Historical data and frozen cases

The tool never reconstructs public-card target-window answers. It samples earlier
windows wholly inside each source card's own supplied, pre-cutoff history.

At every pseudo-origin:

- All supplied panels are copied and truncated at that origin, including non-target panels.
- Later task titles, event descriptions and original questions are removed from the trial.
- Labels are separate from the trial panels and must end by the real source card's cutoff.
- Level labels are historical levels; return labels are `sum(log1p(daily_simple_return))`.
- Monthly observation periods are shifted using resolved leads, preserving authored labels.
- Gapped or incomplete target windows are excluded before either engine runs.
- Windows do not overlap within a group, with a five-business-day embargo after labels.

The numerical engine is trusted code, not an arbitrary-code sandbox. It reads the trial's
in-memory panels, not source files. Audit information flow again if a future engine starts
reading files or other metadata directly.

Task-shape duplicates are grouped using family, assets, horizons, panel identities, target
semantics and scoring parameters; the latest supplied source cutoff is chosen per group.
Aliases are recorded. This prevents repeated copies of a history from multiplying evidence.
The resulting group-balanced average is **not** the official equal-weight card roster.

Folds are global calendar periods, not a per-series random split:

| Role | Origin and complete label window must be inside |
|---|---|
| Tuning | 2000–2012 |
| Holdout 1 | 2013–2017 |
| Holdout 2 | 2018–2024 |

Windows crossing fold boundaries are purged. A short/early source can lack a holdout;
missing-fold coverage is reported, not manufactured from another card's later data.

Monthly macro histories are excluded by default because their pseudo-origin first-release
vintages have not been reconstructed. `--include-monthly` enables a separate
`retrospective_monthly` diagnostic tier, never part of the primary promotion gate. Even
daily/factor histories do not prove historical publication availability: factor libraries
may lag or revise values. These remain retrospective history proxies, not live trading tests.

## Scoring and normalization

All forecast scoring is imported from the pinned organizer implementation: fair marginal
CRPS, joint variogram of order 1/2, and tail pinball loss. Single-cell weights are
redistributed over marginal and tail components, matching the published rule.

The October 2026 published normalization differs from the archived research runner:
divisors are M0's **expected** component errors, computed from forecast inputs only,
not M0's error against the observed label. The published cap is now 8.0 rather than 4.0.
See [M0's published procedure](https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/docs/M0-BASELINE.md).

The validation-only Gaussian reference and analytical divisors implement sections 3 and 5
of that procedure: trailing 300 rows, gap-safe date alignment, joint path covariance,
Gaussian expected CRPS/pinball and noncentral Gaussian variogram moments. SciPy evaluates
the documented normal/hypergeometric functions. The labels are not arguments to the
divisor calculation. Monte Carlo tests independently check the joint expectation.

M0 diagnostic draws use the requested draw count and seed for these historical proxies;
they are not the official card's fixed 500-draw artifact. M0's observed local score is
**not forced to 1.0**: 1.0 is its expected error under its own distribution, not its error
on every possible historical outcome.

Each whole composite is clipped to `[0, 8]`; individual components are not separately
clipped. An agent failure stays in the denominator at 8.0. A defective reference or
changed input/scoring snapshot aborts the run instead of quietly removing cases.

The current published procedure is pinned, but that does not establish which bundle a
particular CodaBench run deployed. Do not compare these positive local losses directly
with the aggregate leaderboard number, or mix them with archived outcome-ratio reports.

## Run and compare

Every output must be a **new** directory under `backtest_runs`; existing runs are never
overwritten. Artifacts are ignored by Git and not copied into the submission image.

Freeze a complete plan without generating forecasts:

```powershell
..\.venv_t2_validation_v26\Scripts\python.exe scripts/validate_engine.py `
  --plan-only --out backtest_runs\validation\new-plan
```

The initial complete baseline audit created this plan:
`backtest_runs/validation/full-plan-2026-10-10/plan.json`. It contains 606 cases from
66 deduplicated groups, covering 264 tuning, 205 first-holdout and 137 second-holdout
windows. Four monthly source cards were explicitly excluded.

Reuse the exact plan for every comparison. Example for a future numerical candidate:

```powershell
# Default split is tune; start here. Default: 2,000 draws, seeds 0/19/73.
..\.venv_t2_validation_v26\Scripts\python.exe scripts/validate_engine.py `
  --plan backtest_runs\validation\full-plan-2026-10-10\plan.json `
  --out backtest_runs\validation\candidate-tune

..\.venv_t2_validation_v26\Scripts\python.exe scripts/compare_validation.py `
  --baseline backtest_runs\validation\incumbent-2026-10-10 `
  --candidate backtest_runs\validation\candidate-tune --splits tune `
  --out backtest_runs\validation\candidate-tune-comparison
```

Once a candidate is frozen, run both held-out periods explicitly:

```powershell
..\.venv_t2_validation_v26\Scripts\python.exe scripts/validate_engine.py `
  --plan backtest_runs\validation\full-plan-2026-10-10\plan.json `
  --splits all --out backtest_runs\validation\candidate-confirmation

..\.venv_t2_validation_v26\Scripts\python.exe scripts/compare_validation.py `
  --baseline backtest_runs\validation\incumbent-2026-10-10 `
  --candidate backtest_runs\validation\candidate-confirmation `
  --out backtest_runs\validation\candidate-confirmation-comparison
```

`--groups-per-family 1 --origins-per-split 1 --draws 400 --seeds 0,19` gives a small,
stratified pipeline check when creating a new plan. It is not promotion evidence.

## Read the artifacts

- `plan.json`: frozen calendars, source hashes, scales, aliases and exclusions.
- `manifest.json`: completion state, expected row count, exact pins and source fingerprints.
- `results.parquet`: per-case/per-seed agent and reference components, clipping and failures.
- `results.partial.jsonl`: progress/recovery audit; incomplete runs are not comparable.
- `summary.json` / `summary.md`: fold, family, population and seed diagnostics.
- `comparison.json`: paired deltas and a calendar-block sensitivity interval. Negative is better.

Seeds are averaged within historical cases. Groups, not individual asset cells or currency
units, have equal weight. The joint diagnostic includes structural zeros on single-cell
grids; use the grid/family breakdown and effective weights when interpreting it.

Comparison refuses missing/duplicate cases, modified results, incomplete runs, changed
data, different draws/seeds and different scorer/harness versions. If the harness changes,
re-run the baseline rather than overriding those checks.

The conservative local evidence gate requires improvement on both holdouts, an upper
calendar-block interval below zero, no family regression larger than 0.01 local-loss units,
no engine failures, at least two groups per family in each holdout, and at least three seeds.
It uses two-year time blocks shared across instruments, not a seed-by-seed IID bootstrap.
The interval is a sensitivity diagnostic: shared markets and model selection mean it is
not a certified independent confidence interval or a guarantee of leaderboard improvement.

Do not repeatedly tune on holdouts. Do not bake a globally learned parameter table from
later historical windows into earlier-cutoff tasks. Future fitting/calibration must respect
each actual task cutoff, with provenance and disclosures where required.

Admissibility tests still matter and remain separate from predictive diagnostics. No public
card answer files, live market data, House cached responses, fitted artifacts or new
model requests are introduced here. Local runs spend no CodaBench submissions.

## Verification on 2026-10-10

- 32 validation tests and the 47 existing forecasting/text tests passed.
- Source/exemplar preflight passed; production source and configuration have no diff from main.
- The full incumbent audit finished in about ten minutes: 606 windows, three seeds, 2,000
  draws per forecast, 3,636 agent/reference metric rows and zero engine failures.
- A separate monthly diagnostic run completed 48 windows / 288 rows, with zero failures.
  Its retrospective-vintage limitation remains; it is not promoted into the primary tier.
- Eleven representative windows were repeated: all 66 metric rows were bit-exact.
- Comparing the full incumbent run with itself returned delta 0 and interval `[0, 0]`,
  correctly refusing to call it an improvement.

The current incumbent report is
`backtest_runs/validation/incumbent-2026-10-10/summary.md`. It is a numerical baseline
for future paired experiments, not evidence that the scored forecast has improved.
