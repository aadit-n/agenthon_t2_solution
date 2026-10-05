# v0.5: text reliability experiment

This candidate starts from the scored v0.2 commit `6bd7ba2`, on branch
`v0.5-text-reliability`. It does not include the v0.3/v0.4 numerical experiments.
`model.py`, `monthly.py`, and `unit.py` are unchanged. The original House pipeline is
preserved in `legacy_text.py`; its family shrinkage, corroboration thresholds, volatility
limits, and factor projection remain the adjustment rules for this candidate.

## What changed

- Token budgets scale with the number of target assets. In particular, the ten-currency
  extraction budget is 1,730 tokens rather than the v0.2 fixed 550. Synthesis starts at
  3,350 for that grid. Repairs receive a larger budget, capped at 4,000 per request.
- Every signal and scenario must explicitly cover every asset. Numeric strings, string
  booleans, missing values, NaNs, out-of-range values, invalid probabilities, and unknown
  evidence IDs are rejected instead of being silently filled or coerced.
- `finish_reason=length` is rejected even if the remaining response parses as JSON.
  Incomplete/invalid JSON and transient network/API failures get at most one retry.
  Authentication/route failures stop the unit's House work immediately.
- Nonzero signals require short quotes checked against both the supplied excerpt and
  original document. Nonneutral scenarios require accepted document citations. These
  checks establish textual support, not that the model interpreted the evidence correctly.
- Only indexed, dated documents at or before the cutoff are eligible. Identical normalized
  documents and repeated IDs cannot count as separate corroboration. Long documents use
  head-and-tail excerpts, still within the v0.2 9,000/document and 70,000/unit character limits.
- Prompts explicitly distinguish USD-per-currency from currency-per-USD quotes, and monthly
  horizon labels from actual observation-period leads. Output horizon labels are unchanged.
- Existing `forecast_meta.json`, rationale, and stderr diagnostics report fallback reasons,
  valid cards, call counts and response failures. No additional output file is emitted.
  Credentials, endpoint URLs and raw responses are not included in transport diagnostics.

## Resource limits and fallback

Ten documents, at most two attempts each, plus at most two synthesis attempts: 22 attempted
requests in the maximum normal path, below the published 25-call allowance. A separate hard
25-call guard protects the client. Each request permits at most 4,000 output tokens. House
work has a 600-second soft deadline, with 90 seconds reserved for synthesis and individual
request timeouts capped at 90 seconds. Retries count as calls. Slow or invalid House work
can therefore cause numerical fallback rather than an unlimited retry loop.

Text is **enabled by default**. Official runs use the injected `MODEL_ENDPOINT`, `MODEL_TOKEN`
and `MODEL_NAME`; no alternate API, model or credential is introduced. Organizer proxy settings
are preserved. `T2_DISABLE_TEXT=1` explicitly disables text for offline numerical checks; it is
not set in the Dockerfile. Missing development access likewise causes numerical fallback.

The default is `--text-mode reliable`. `--text-mode legacy` or `T2_TEXT_MODE=legacy` invokes
the preserved v0.2 pipeline for controlled local comparisons. Host environment variables are
not automatically baked into a container or carried into an official run.

## Validation and its limits

Run from this repository in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
..\.venv_t2_validation\Scripts\python.exe scripts/preflight.py --repo ..\track2-forecasting-public
..\.venv_t2_validation\Scripts\python.exe scripts/audit_text_reliability.py `
  --repo ..\track2-forecasting-public --out backtest_runs\v0.5-text-audit.json
```

Fault-injection tests use mocked HTTP responses, never the real House endpoint. They exercise
successful application, malformed/partial/truncated output, quote validation, bounded retries,
authentication failures, resource caps, and exact neutral-sample parity with v0.2.

Local validation on 2026-10-05:

- 47 unit/fault-injection tests passed, including a mocked House-to-CLI output run.
- All 103 public cards passed the organizer's four admissibility gates with text disabled.
- The local Linux/amd64 candidate passed Docker/exemplar preflight.
- Six representative tasks produced bit-exact sample arrays against the actual v0.2 image
  with text disabled, 2,000 draws and seed 0 (60,000 compared forecast cells). This covers
  daily rates/FX, monthly CPI, factor returns and all four task families. Both images ran
  with a read-only root filesystem and network disabled.

To reproduce the image comparison:

```powershell
..\.venv_t2_validation\Scripts\python.exe scripts/compare_text_baseline.py `
  --repo ..\track2-forecasting-public --out backtest_runs\v0.5-image-parity.json
```

The public-corpus audit covers 103 cards and selects 666 documents, with no missing-date,
future-date or exact-duplicate exclusions in the supplied corpora. It selects the same document
sets as v0.2; 40 cards have ordering differences from deterministic tie-breaking. Thus these
corpus guards prevent future input failures, but are not evidence of a score improvement on
the current corpus. Quote-direction fixes, complete output handling and larger token budgets
are the material text-path changes being tested.

Neither these tests nor public admissibility gates measure the official predictive score.
There is no authorized House development access, and this version has not been scored on
CodaBench. Stricter validation or new prompts could improve or worsen forecasts. No better
score, rank or win is guaranteed. Preserve v0.2 as the incumbent and treat v0.5 as one focused
candidate, not a proven replacement. No submission is spent by local checks.

## Release

The local test image is `t2-solution:v0.5-test`. Build/preflight:

```powershell
docker buildx build --platform linux/amd64 --provenance=false --sbom=false --load `
  -t t2-solution:v0.5-test .
..\.venv_t2_validation\Scripts\python.exe scripts/preflight.py `
  --repo ..\track2-forecasting-public --image t2-solution:v0.5-test
```

Building a local image does not publish or submit it. If you choose to test this candidate on
CodaBench, follow the README's publish/package procedure, using a **new v0.5 tag and its new
registry digest**, never v0.2's digest. The model declaration stays the same approved House
snapshot. The ZIP contains the descriptor and team claim, not the code or local tests.
