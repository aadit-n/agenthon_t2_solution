"""Run the experimental engine through the unchanged validation/scoring harness.

This explicitly selects the adaptive engine and records the adapter's identity. The normal
forecast command and validate_engine.py still run the scored incumbent. Tuning is the default;
do not open holdouts for a candidate that missed the predeclared tuning acceptance bar.
"""

from __future__ import annotations

import pathlib

import validate_engine
from validation import runner

runner.model.simulate = runner.model.simulate_adaptive
_incumbent_identity = validate_engine.runtime_identity


def experimental_identity() -> dict:
    identity = _incumbent_identity()
    identity["experiment_adapter"] = {
        "engine_entrypoint": "t2solution.model.simulate_adaptive",
        "source_sha256": runner.file_hash(pathlib.Path(__file__)),
    }
    return identity


validate_engine.runtime_identity = experimental_identity

if __name__ == "__main__":
    raise SystemExit(validate_engine.main())
