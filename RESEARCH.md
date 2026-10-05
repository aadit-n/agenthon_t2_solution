# Research and implementation decisions — 5 October 2026

## Executive summary

The scored v0.2 is the incumbent (-0.9738, user reported). The reverted v0.3 scored
-0.9887 and is not used as the foundation. This round separates correctness repairs
from statistical experiments and a new, opt-in House conditioning method. After
the initial research, the user submitted the default v0.4 candidate and reported
-0.9744: a regression of 0.0006 from v0.2. Optional experiments have no official
quality score. See `VALIDATION.md` for the updated release decision.

## Official sources and concrete defects

- [Current target transform](https://github.com/Agenthon-2026/track2-forecasting-public/blob/2299dab3d31f5e93c10df087af86ffb0b7bfe938/qfbench2_track_forecasting/targets.py):
  daily factor panel rows are simple returns; a cumulative log-return target sums
  `log(1+r)`. Both v0.2 and the earlier local backtest summed raw rows instead. This
  omitted volatility drag. The new `corrected` profile fixes the forecasting inputs,
  and the actual-engine benchmark fixes its historical outcomes. The published M0
  construction separately specifies raw row innovations; do not confuse its
  normalization distribution with the target definition.
- [Monthly horizon guide](https://github.com/Agenthon-2026/track2-forecasting-public/blob/2299dab3d31f5e93c10df087af86ffb0b7bfe938/docs/MONTHLY-HORIZONS.md):
  authored business-day keys are not month counts. The old simulator mapped them
  correctly, but its *drift selector* still used the authored integer. On monthly
  targets this could declare insufficient history and return zero drift. It also
  applied a daily-price drift cap to real monthly index trends. `corrected` repairs
  the selector while retaining the original 0.35 shrink and ensemble shares.
- [Artifact policy](https://github.com/Agenthon-2026/track2-forecasting-public/blob/main/docs/ARTIFACT-POLICY.md):
  numerical/statistical code is permitted; extra pretrained neural checkpoints need
  approval, and external/self-hosted language models are not permitted. No model
  checkpoints, datasets, fitted cross-unit weights or cached forecasts are added.
- [House guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md):
  the runtime alias and approved snapshot remain organizer-controlled. Thinking is
  an experimental option, not a replacement model; output requests stay below 4,000.
- [M0 specification](https://github.com/Agenthon-2026/track2-forecasting-public/blob/main/docs/M0-BASELINE.md):
  there are fidelity and deployed-scorer caveats. The benchmark therefore reports
  both pinball and legacy coverage-tail diagnostics rather than assuming which
  private service revision produced an aggregate score.

## Papers reviewed and how they influenced the code

| Primary source | Relevant idea | Decision |
|---|---|---|
| [Schefzik, Thorarinsdottir & Gneiting (2013), Ensemble Copula Coupling](https://arxiv.org/abs/1302.7149) | Calibrate marginals without discarding joint ranks | Independent ECC-inspired quantile postprocessor; opt-in `ecc` profile |
| [Berrisch & Ziel (2021), CRPS Learning](https://arxiv.org/abs/2102.00968) | Combine forecasts in quantile space and judge them with proper scores | Small quantile-pooling experiment; not a reproduction of their BOA algorithm |
| [Gneiting & Ranjan, Combining Predictive Distributions](https://arxiv.org/abs/1106.1638) | A mixture is not automatically calibrated | Keep separate marginal, joint and tail checks; do not assume an ensemble wins |
| [Gneiting et al. (2005), Minimum-CRPS Ensemble Model Output Statistics](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml) | Match calibration and sharpness using the evaluated score | Organizer metric implementations drive the new benchmark; no point-error-only selection |
| [Hyndman & Athanasopoulos, rolling-origin validation](https://otexts.com/fpp3/tscv.html) | Train only before each test origin; evaluate the required lead | Pre-cutoff, nonoverlapping historical windows and chronological splits |
| [Hyndman & Athanasopoulos, damped trends](https://otexts.com/fpp3/holt.html) | Persistent monthly trends differ from daily-price drift | Repair month-count handling; do not impose a global daily drift rule on monthly levels |
| [Williams et al., Context is Key (ICML 2025)](https://arxiv.org/abs/2410.18959) | Text can contain essential context, but numerical-only validation is insufficient | Keep text experiments isolated; do not claim live House uplift from mock tests |
| [Nguyen et al. (2026), LLM as Forecasting Planner](https://arxiv.org/abs/2607.24892) | Restrict LLM influence to numerically proposed trajectories | Optional bounded whole-draw reweighting; no neural backbone, tree search, or invented medians |
| [Direct Prompting for Forecast Correction / CorDP](https://openreview.net/pdf?id=dkjHHFJkVI) | Correct an existing forecast instead of regenerating it | Reviewed, but unconstrained direct median correction is not adopted |

These papers motivate experiments. Their reported results are not evidence that
our adaptations improve Agenthon. No paper implementation is copied.

## Public Agenthon repositories inspected

Discovery used GitHub's public repository search and source trees, then public
architecture/provenance documents and selected method descriptions. Leaderboard
positions were not verified. Repository benchmark claims below are self-reported,
not reproduced, and may use different populations, vintages or scoring revisions.

- [Yif-design/Agenthon2026](https://github.com/Yif-design/Agenthon2026): the already
  cloned reference; stationary bootstrap, regime mixing and document extraction.
  No implementation is imported from it.
- [aicreator27/agenthon-2026-agents](https://github.com/aicreator27/agenthon-2026-agents):
  public calibration decisions report benefits from a shorter numeric history and
  warn that daily-level results did not transfer to monthly/return populations.
  This supports testing *separate* recent-history and population-specific variants,
  not copying their parameters or accepting their scores as ours.
- [Felix772/Agenthon_2026 — forecast agent](https://github.com/Felix772/Agenthon_2026/tree/main/forecast-agent):
  useful provenance and negative evidence: their online selector did not establish
  conclusive improvement. Admissibility results are explicitly not quality scores.
- [muanch18/agenthon-t2-OffTheTape](https://github.com/muanch18/agenthon-t2-OffTheTape):
  PCA joint forecasting, causal historical harness and text-state separation.
- [lucaluo925/agenthon-t2](https://github.com/lucaluo925/agenthon-t2):
  conservative joint Gaussian baseline; its source describes rejected drift,
  covariance and text variants. This is useful negative evidence, not a winning recipe.
- [yuang-lee/agenthon-track2-cordp-v1](https://github.com/yuang-lee/agenthon-track2-cordp-v1):
  direct native-unit median correction. Its provenance notes local 30B experiments
  do not establish performance with the official 120B House model. We do not adopt
  unbounded corrections, static calendars of unclear availability, or cached answers.
- [JavierAmo13/agenthon2026-forecasting](https://github.com/JavierAmo13/agenthon2026-forecasting):
  runtime non-neural models and causal feature/residual estimation. Heavier learned
  models remain future experiments, not an untested default.

Empty, unrelated and non-Track-2 repositories were excluded. Outcome/provenance
audit repositories, answer files and scored-card future target windows were not
used. No competitor source was downloaded into or executed by this agent's image.

## Implemented profiles

- `v2`: original numerical/text behavior for a reproducible source rollback.
  Metadata version and error logging differ; the scored immutable v0.2 image/ZIP
  remains the authoritative rollback.
- `corrected` (default): only factor log-return semantics and monthly drift selection
  repairs. Daily level samples remain bit-identical at the same seed/draw count.
- `drift`: experiment shrinking a small part of noisy daily-level baseline drift;
  the centre change is capped at 0.25 incumbent standard deviations.
- `ecc`: experiment pooling 15% empirical historical-error quantiles with incumbent
  quantiles, restoring joint ranks and bounding per-draw changes.
- `recent`: experiment limiting the empirical component to 504 eligible daily steps;
  leaves the trailing-300 baseline component and sparse transfer histories intact.
- `combined`: drift and ECC experiments together, not a recommended default.

No candidate profile is silently selected using cross-card labels or a stored
best-parameter table. Runtime historical error pools read only the current unit's
inputs available by its cutoff. Tests do not read sealed scales or outcomes.

## Optional text mode: trajectory selection

`--text-mode trajectory` replaces the legacy text overlay with a new experiment:

1. Generate the complete numerical joint draw library.
2. Form up to seven deterministic runtime clusters and show their prior probabilities
   and native-unit endpoint medians to House.
3. Read only timestamp-eligible, deduplicated supplied documents. Keep relevant
   paragraphs and document conclusions within fixed excerpt budgets.
4. House proposes relative cluster weights between 0.75 and 1.25. Non-neutral
   changes require verbatim quotes in the supplied excerpts.
5. A skeptical second pass can neutralize or weaken changes, never amplify them.
6. Require effective weight sample-size ratio at least 0.95 and an expected mean
   movement at most 0.15 incumbent SD in every cell, then resample **whole** draws.

No new forecast value is created. Neutral, unsupported, incomplete, conflicting,
or failed responses preserve the numerical forecast. There are at most two calls,
each requesting at most 3,600 output tokens, with 90-second request timeouts.
`--thinking` is available only for this experimental mode and is off by default.

Live House credentials are not configured locally. Fault-injection and mock tests
prove safeguards and wiring, not predictive uplift. The scored legacy text method
remains default; its prompts, caps and call allocation are unchanged.

## Validation and remaining uncertainty

See `VALIDATION.md` for completed results. There is no proof of first place or
guaranteed official improvement. In particular:

- All historical quality comparisons use neutral text. They cannot validate either
  the legacy House pipeline or the trajectory experiment.
- Related grids reuse market history; samples are not statistically independent.
- Monthly panels can be revised vintages; historical pseudo-origins are not certified
  point-in-time first-release experiments.
- M0 controls and denominators are locally generated for pseudo-cards, not the sealed
  per-card reference scales. Deployment revisions can also differ.
- Numeric correctness repairs can still move a competition score in either direction.

The completed 70-group screen and shifted-origin robustness run gave the `recent`
experiment 3.74% and 0.81% lower aggregate pinball-composite holdout loss than v0.2.
Its legacy-tail metric worsened in the second holdout and some families regressed,
so it is not promoted to the default. The default's monthly repair helped locally,
but its aggregate gains were only 0.36% and 0.17%. These are neutral-text historical
results, not leaderboard predictions.

Keep the existing -0.9738 result. Do not spend a submission merely to repeat it,
and do not promote an optional experiment on paper reputation or smoke tests alone.
