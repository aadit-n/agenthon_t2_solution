# agenthon_t2_solution

## Executive summary

This is a clean-room Agenthon 2026 Track 2 forecasting agent. Version 0.2 uses the organizers'
published M0 distribution as a defensive ensemble component, then adds an independently designed
event-aware stationary bootstrap, correlated stress paths, volatility uncertainty, and a cautious
document-extraction/synthesis layer. Text suggestions are corroboration-gated, projected into the
empirical joint-factor space, and shrunk by card family.

The image implements the required command:

```bash
forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet
```

Local smoke run from the organizer repository's parent directory:

```bash
pip install -e ./agenthon_t2_solution
forecast \
  --panels ./track2-forecasting-public/units/t2-EXAMPLE-ust-curve-1m \
  --text ./track2-forecasting-public/units/t2-EXAMPLE-ust-curve-1m/text \
  --asof 2024-06-28 \
  --out ./agenthon_t2_solution/out/forecast.parquet
```

Set `T2_DISABLE_TEXT=1` for a deterministic text-ablation run. Official House-model access uses
`MODEL_ENDPOINT`, `MODEL_TOKEN`, and `MODEL_NAME`. The implementation makes one short extraction
request per selected document plus one synthesis request (at most 11 requests per unit, below the
published allowance), and disables model thinking for reliable structured JSON.

## Install and test

From this directory in PowerShell:

```powershell
..\.venv\Scripts\python.exe -m pip install -e .
..\.venv\Scripts\python.exe .\scripts\preflight.py `
  --repo ..\track2-forecasting-public
```

Run the complete 103-card source regression with `--full`. To check a locally built image too:

```powershell
docker buildx build --platform linux/amd64 --load -t t2-solution:dev .
..\.venv\Scripts\python.exe .\scripts\preflight.py `
  --repo ..\track2-forecasting-public --image t2-solution:dev
```

## Leakage-safe calibration

The rolling backtest creates pseudo-cutoffs inside each panel's pre-as-of history. It does not
reconstruct or consult the practice-card target window:

```powershell
$env:PYTHONPATH = "$PWD\src"
..\.venv\Scripts\python.exe .\scripts\rolling_backtest.py `
  --repo ..\track2-forecasting-public `
  --out .\backtest_runs\results.csv
```

The summary reports aggregate CRPS relative to a driftless baseline and 90% coverage. It is a
development diagnostic, not the official normalized leaderboard score.

## Build and package a Development submission

Build and push an anonymously pullable Linux/amd64 image. Use an immutable digest, never a tag, in
the descriptor. Example for GHCR:

```powershell
docker buildx build --platform linux/amd64 --provenance=false --sbom=false `
  -t ghcr.io/<owner>/t2-solution:v0.2 --push .
docker buildx imagetools inspect ghcr.io/<owner>/t2-solution:v0.2
```

Install the pinned submission toolkit:

```powershell
..\.venv\Scripts\python.exe -m pip install `
  "qfbench2-common[data] @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.4#subdirectory=common"
```

Then package. The Team Key is requested through a hidden prompt and never stored:

```powershell
..\.venv\Scripts\python.exe .\scripts\package_submission.py `
  --team-number <N> `
  --registry ghcr.io `
  --repository <owner>/t2-solution `
  --digest sha256:<64-lowercase-hex> `
  --phase dev `
  --out .\submissions\v0.2.zip
```

The packer pins toolkit 2.4.4, discloses the approved House model snapshot, validates the twelve-key
descriptor, binds it to the Development phase, creates a v2 team claim, and verifies that the ZIP
contains exactly `submission.json` and `team-claim.json`. Never paste the Team Key into chat or put
it on a command line.

