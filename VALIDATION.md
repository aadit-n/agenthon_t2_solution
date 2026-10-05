# Validation results - 5 October 2026

## Decision

The user subsequently submitted v0.4 and reported **-0.9744**, compared with
**-0.9738** for v0.2. Higher is better: v0.4 regressed by **0.0006** and is not a
scored improvement. Keep v0.2 as the best scored release. The source currently
defaults to `corrected` with legacy text; that source default identifies the v0.4
candidate, not the competition incumbent.

The monthly repair has a repeatable local benefit, but its small aggregate local
gain did not transfer to the reported CodaBench score. The score difference is
small; without repeated runs or uncertainty estimates its significance is unknown.
It must not be dismissed as noise or advertised as an improvement.

The most promising statistical experiment is `recent`. It improved the current
pinball-based local composite in both runs, but the second holdout gain shrank to
0.81%, some families deteriorated, and the legacy coverage-tail composite worsened
in that holdout. Keep it optional pending stronger evidence. `drift`, `ecc` and
`combined` also stay optional. The new House trajectory mode has no live quality
validation because authorized development access is unavailable.

Both official scores above are user reported; no component scores or runtime logs
are available. The user generated `submissions/v0.4-development.zip` after the
initial research. The ZIP references a different immutable image from v0.2 and
declares the same approved House snapshot. Existing v0.2 files remain preserved.

## Completed engineering checks

- **30 unit/regression tests passed**, including proper log-return targets,
  future-row isolation, input immutability, actual monthly leads, joint-rank
  preservation, scoring normalization and exclusions for missing calendar windows.
- The House tests cover eligible timestamps, duplicate evidence, verified quotes,
  required mechanisms, unsupported responses, audit disagreement, audit timeout,
  whole-draw preservation, neutral fallback, output caps and secret-safe errors.
  They use mocks and synthetic data; they do not measure House forecasting skill.
- **103/103 public cards passed organizer g0-g3 gates** with `corrected`, legacy
  text disabled and 2,000 draws, using fresh Python processes. This is admissibility
  validation with no official outcomes supplied.
- **103/103 cards also passed with `recent`**, 2,000 draws and text disabled, using
  the faster in-process gate runner. This second run does not check process isolation.
- The end-to-end `trajectory` CLI fallback passed the exemplar's organizer gates
  with 2,000 draws and zero House calls. The reader/auditor wiring is mock-tested.
- The numerical `v2` rollback and the unchanged daily-level `corrected` branch
  preserve incumbent samples at the same seed/draw count in regression tests.
- `git diff --check` passed.

Docker's Linux engine was unavailable during the initial research build check.
The user subsequently built the submitted image. Local inspection of the exact
image referenced by the v0.4 ZIP confirmed Linux/amd64 and no baked-in
`T2_DISABLE_TEXT`, `T2_PROFILE` or `T2_TEXT_MODE` environment override. This rules
out carrying the local text-disable flag into that image; it does not prove actual
House-call success on CodaBench. Forecast dependencies are unchanged: NumPy 2.1.3,
pandas 2.2.3, PyArrow 18.1.0. Source validation used Python 3.13.3 and toolkit 2.4.4.

## Historical quality experiment

The benchmark uses only supplied histories before each public card's cutoff.
Each earlier pseudo-origin hides all subsequent rows from the forecasting engine;
its outcome is then taken from the remaining pre-card-cutoff history. Public-card
future outcomes, organizer reference scales and competitor answers are never read.

We deduplicated asset/horizon/family configurations to **70 groups**, taking the
latest supplied history for each. Up to eight nonoverlapping forecast windows are
used per group, with the earlier half marked development and the later half
holdout. These labels describe chronology within each group; they do not establish
independence across related assets, families or panel vintages.

The same actual numerical simulator, seeds and outcomes are used for matched
profiles. Draws are generated in independent 200-draw batches to bound memory.
This preserves the stochastic construction but differs from one 2,000-draw CLI
invocation. The local Gaussian control follows the published M0 construction;
its finite-sample losses supply local normalization denominators. They are not
the sealed competition denominators.

Organizer implementations compute marginal CRPS, joint variogram and tail loss.
The whole weighted composite is clipped to [0,4], with single-cell weights
renormalized. `local_loss` below uses pinball tails. **Lower is better. None of
these numbers is an official or predicted leaderboard score.**

### Screening: 400 draws, seed 31, origin offset 0

The first simulation started before the calendar-gap exclusion was added. All
saved origins were subsequently checked against the input calendars, and four
invalid transfer windows were removed identically from every profile. The table
uses the corrected `research-screen-filtered.csv`, with **554 pseudo-tasks**:
278 development and 276 holdout. The original output is retained for audit.

| Profile | Development loss | Holdout loss |
|---|---:|---:|
| v2 incumbent | 1.11664 | 1.07716 |
| corrected default | 1.11592 | 1.07331 |
| recent | 1.07508 | 1.03686 |
| drift | 1.14894 | 1.08628 |
| ecc | 1.10866 | 1.05930 |
| combined | 1.13458 | 1.06431 |

`recent` reduced holdout loss by **3.74%** relative to v2. It won on the mean loss
of 39/70 groups. Its aggregate gain therefore does not imply improvement everywhere.
The default's holdout reduction was **0.36%**.

### Robustness run: 800 draws, seed 73, origin offset 0.5

This run shifted each origin by half the forecast lead and doubled draws. It
contains **557 pseudo-tasks**: 278 development and 279 holdout. No gapped window
was eligible. Shifted windows can overlap the screening windows, so this is a
robustness check rather than an independent untouched dataset.

| Profile | Development loss | Holdout loss | Holdout legacy-tail loss |
|---|---:|---:|---:|
| v2 incumbent | 1.11905 | 1.04685 | 0.96932 |
| corrected default | 1.11720 | 1.04507 | 0.96876 |
| recent | 1.06193 | 1.03836 | 0.97894 |

`recent` reduced pinball-composite holdout loss by **0.81%**, but its legacy-tail
loss increased by about **0.99%**. Its holdout F1 and F3 losses also increased.
The default's aggregate holdout reduction was **0.17%**.

For monthly levels specifically, the default reduced holdout loss from **1.09227
to 1.04791**, about **4.06%**. Factor-return results slightly worsened despite the
required unit correction. Daily-level default samples are unchanged. Reporting
only the favorable monthly or screening result would overstate overall progress.

The M0 control has local normalized loss 1 by construction and beats these agents
on several aggregate diagnostics. That is useful negative evidence: adding an
empirical component is not automatically beneficial without text. This does not
establish that a text-blind M0 submission would beat the scored incumbent.

## Reproduce and inspect

Run from `agenthon_t2_solution` using the existing validation environment:

```powershell
..\.venv_t2_validation\Scripts\python.exe -B scripts\research_backtest.py `
  --draws 400 --origins 8 --seed 31 --out backtest_runs\screen-rerun.csv

..\.venv_t2_validation\Scripts\python.exe -B scripts\research_backtest.py `
  --profiles v2,corrected,recent --draws 800 --origins 8 --offset 0.5 --seed 73 `
  --out backtest_runs\confirm-rerun.csv
```

The current script excludes calendar gaps before simulation, so the first command
does not need the post-run repair used for the initial screening. The retained
screening was repaired with:

```powershell
..\.venv_t2_validation\Scripts\python.exe -B scripts\research_backtest.py `
  --rescore backtest_runs\research-screen.partial.csv `
  --out backtest_runs\research-screen-filtered.csv
```

Evidence files are under Docker-ignored `backtest_runs/`:

- `research-screen-filtered.csv`, `.summary.csv`, `.families.csv`,
  `.populations.csv`, `.paired.csv`, `.exclusions.json`.
- `research-confirm.csv` and corresponding summaries, plus
  `research-confirm.settings.json`.

The files are local generated diagnostics and are not bundled in the agent image.
The original approximate `rolling_backtest.py` is retained for history; use
`research_backtest.py` for comparisons of this implementation.

## Limits that matter for the release decision

All quality comparisons use neutral text. Monthly histories may contain revised
vintages. Related configurations share data. Local reference losses, event
selection and deployed tail-metric revisions differ from the private evaluation.
These limits prevent a reliable conversion of the measured changes into a
CodaBench score or a guarantee of improvement.

Preserve the scored v0.2 release. The source default provides verified correctness
repairs; optional experiments provide concrete candidates for further testing.
The evidence from this round is insufficient to promise a large leaderboard gain.
