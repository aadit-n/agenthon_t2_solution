# Artifact provenance: v0.5 text reliability

This image contains forecasting source code and its pinned Python dependencies. It does not
contain fitted weights, learned calibration tables, cached House outputs, unit-specific future
outcomes, public practice panels, or another team's implementation.

The only model access remains the organizer-injected House API. Its approved snapshot disclosure
in `scripts/package_submission.py` is unchanged. No additional external models are requested.

At inference, the agent reads only supplied numeric panels, task metadata and the supplied text
corpus. Text eligibility uses corpus publication timestamps and the invocation cutoff; numerical
loading and forecasting semantics are unchanged from v0.2. Quote validation uses supplied text
only. The document's `_full_text` field stays local and is not transmitted with the excerpt.

Offline tests use synthetic/mock responses or supplied pre-cutoff data. No withheld realizations
are reconstructed. Public gate checks are admissibility checks, not predictive scores. Audit and
validation outputs are ignored by Git and are not copied into the Docker image.
