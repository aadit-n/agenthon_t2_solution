# Artifact provenance

## Executive summary

Version 0.4 packages participant-authored numerical code, prompts and runtime
statistical procedures. It contains no new neural checkpoint, fitted cross-unit
parameter file, dataset, calendar, retrieval index, cached reply or unit answer.
The only language-model access is the organizer's injected House route. See
`RESEARCH.md` for methodological sources and their limits.

## Source and dependencies

- Incumbent source: local solution commit `6bd7ba2` (v0.2); scored result -0.9738 is
  user reported, not independently verified.
- New code: independent implementation in `candidate.py`, `conditioning.py`, tests
  and `scripts/research_backtest.py`. Numerical libraries remain NumPy 2.1.3,
  pandas 2.2.3 and PyArrow 18.1.0. No dependency upgrade is needed for forecasting.
- Validation: organizer Track 2 checkout `2299dab3d31f5e93c10df087af86ffb0b7bfe938`,
  common toolkit 2.4.4, Python 3.13.3 in `.venv_t2_validation`.
- House: `nvidia/nemotron-3-super-120b-a12b`, revision/version `rl-030326-fp8`,
  training cutoff `unpublished`, access `api`; requests use injected `MODEL_NAME`.
- Research: public papers and public repository architecture/provenance material
  accessed 5 October 2026. No third-party source implementation is bundled.

## Cutoff and fitting

Panel rows are filtered at the unit cutoff before any conversion, fitting,
covariance calculation, error pooling or clustering. Monthly horizons use explicit
observation periods. Statistical models and optional draw-cluster centroids are
computed in memory from the current unit, never loaded as packaged fitted artifacts.
Optional text mode reads only the supplied corpus, uses its publication timestamps,
verifies exact quotes and does not fetch additional evidence at inference time.

Historical benchmark outputs remain under Docker-ignored `backtest_runs/`. The
benchmark creates earlier pseudo-cutoffs inside supplied pre-card-cutoff histories,
never reads official realized files or reference scales, and never supplies its
future pseudo-outcomes to an inference function. Related windows and revised
monthly inputs limit what those quality diagnostics establish.

## Image and submission scope

`Dockerfile` copies only `pyproject.toml` and `src/`. Existing `.dockerignore` excludes
submissions, local outputs, experiments, tests and scripts. Pure procedural code
and on-input statistical estimates are documented here; no fictitious checkpoint
is added to `models[]`. The approved House disclosure stays unchanged. Any future
bundled learned model would need actual revision/cutoff disclosure and provenance.

No public registry upload or CodaBench submission is made by this research work.
The scored v0.2 image and submission ZIP are not overwritten.
