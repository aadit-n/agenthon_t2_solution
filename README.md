# agenthon_t2_solution

## Executive summary

Version 0.4 is a research candidate based on the scored v0.2 (-0.9738, user reported),
not the reverted v0.3. It repairs cumulative log-return inputs and monthly drift
selection, while retaining the incumbent bootstrap, numerical ensemble shares and
legacy House text pipeline. Optional statistical and text experiments are isolated
behind flags. No first-place result or guaranteed official improvement is claimed.

Subsequent CodaBench result, user reported: **v0.4 -0.9744**, versus **v0.2 -0.9738**.
v0.2 remains the best scored release. The current source default reproduces the
v0.4 candidate; it should not be mistaken for an improved competition incumbent.

Read [RESEARCH.md](RESEARCH.md) for the papers, public repositories and design
decisions, [VALIDATION.md](VALIDATION.md) for measured results and limitations, and
[ARTIFACT_PROVENANCE.md](ARTIFACT_PROVENANCE.md) for cutoff and artifact provenance.
The existing scored image and ZIP are not overwritten by this work.

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
$env:PYTHONPATH = "$PWD\src"
..\.venv_t2_validation\Scripts\python.exe -B -m unittest discover -s tests -v
..\.venv_t2_validation\Scripts\python.exe -B .\scripts\preflight.py `
  --repo ..\track2-forecasting-public
```

Run the complete 103-card source regression with `--full`. To check a locally built image too:

```powershell
docker buildx build --platform linux/amd64 --load -t t2-solution:v0.4-test .
..\.venv_t2_validation\Scripts\python.exe -B .\scripts\preflight.py `
  --repo ..\track2-forecasting-public --image t2-solution:v0.4-test
```

The validation environment already contains the dependencies and pinned organizer
toolkit. Forecast inference itself needs only the three packages in `pyproject.toml`.
Source gate checks verify admissibility, not forecasting quality. Docker Desktop
must be running before the separate container check; source tests do not prove that
a newly built container runs correctly.

For a faster source check of another profile, reuse the Python imports across cards:

```powershell
..\.venv_t2_validation\Scripts\python.exe -B .\scripts\run_practice.py `
  --repo ..\track2-forecasting-public --profile recent --in-process `
  --out .\practice_runs\my_recent_check
```

Use a fresh output directory each time. This option calls the same forecasting and
organizer gate functions, but does not test isolation between Python processes.

## Profiles and rollback

The default is `--profile corrected --text-mode legacy`. It changes
only the target/monthly repairs; the daily-level numerical draws remain unchanged.
The latest promotion decision is recorded in `VALIDATION.md`.

- `--profile v2`: original numerical behavior; use `--text-mode legacy` for the
  source rollback. The previously scored immutable image is the authoritative rollback.
- `--profile recent`: limit the empirical daily component to 504 eligible steps;
  leave the trailing-300 baseline and short transfer histories intact.
- `--profile drift`, `ecc`, or `combined`: bounded drift/quantile-calibration
  experiments, not automatically selected winners.
- `--text-mode trajectory`: experimental two-pass House reweighting of complete
  existing numerical draws. Requires eligible quotations, a skeptical audit and
  sample-size/mean-movement bounds. It is **not** the default and has no live House
  quality validation. `--thinking` applies only to this mode.

Environment equivalents are `T2_PROFILE` and `T2_TEXT_MODE`. `T2_DISABLE_TEXT=1`
disables either text pipeline. Never point the agent at an unauthorized alternative
model endpoint. No local House credentials are available for this research round.

## Historical quality comparison, without using a submission

The new harness runs the **actual simulator**, rather than the approximate older
`rolling_backtest.py`. It uses completed, nonoverlapping windows inside each input
panel's pre-card-cutoff history, imports organizer scoring functions, converts factor
outcomes to log-return units and excludes unavailable calendar gaps identically for
every candidate. It does not read official card outcomes or sealed reference scales.

```powershell
$env:PYTHONPATH = "$PWD\src"
..\.venv_t2_validation\Scripts\python.exe -B .\scripts\research_backtest.py `
  --repo ..\track2-forecasting-public `
  --profiles v2,corrected,recent `
  --draws 800 --origins 8 --offset 0.5 --seed 73 `
  --out .\backtest_runs\research-confirm.csv
```

Outputs include per-window losses, split/family/population summaries, paired
differences, run settings and exclusions. Lower loss is better. These are **local
neutral-text diagnostics, not a CodaBench score estimate**: historical populations,
monthly vintages, reference denominators, Monte Carlo batches and the private text
responses differ. Never negate a local loss and advertise it as a leaderboard score.

## Build and package a Development submission

Build and push an anonymously pullable Linux/amd64 image. Use an immutable digest, never a tag, in
the descriptor. Example for GHCR:

```powershell
docker buildx build --platform linux/amd64 --provenance=false --sbom=false `
  -t ghcr.io/<owner>/t2-solution:v0.4 --push .
docker buildx imagetools inspect ghcr.io/<owner>/t2-solution:v0.4
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
  --out .\submissions\v0.4.zip
```

The packer pins toolkit 2.4.4, discloses the approved House model snapshot, validates the twelve-key
descriptor, binds it to the Development phase, creates a v2 team claim, and verifies that the ZIP
contains exactly `submission.json` and `team-claim.json`. Never paste the Team Key into chat or put
it on a command line.

These are instructions for a later deliberate release, not actions performed by
this research task. Do not reuse the old image digest for changed source. Check
the organizer's pinned submission contract again before spending a submission.

