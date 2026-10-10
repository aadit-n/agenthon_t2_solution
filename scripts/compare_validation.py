"""Compare two complete runs on exactly the same frozen cases, scorer, draws and seeds."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from validation.reporting import compare_frames
from validation.runner import file_hash, json_write, safe_output


def load_run(folder: pathlib.Path):
    meta = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if meta.get("state") != "complete" or meta.get("rows") != meta.get("expected_rows"):
        raise ValueError("Refusing incomplete or partial validation run")
    if file_hash(folder / "results.parquet") != meta.get("results_sha256"):
        raise ValueError("Validation results were modified after completion")
    frame = pd.read_parquet(folder / "results.parquet")
    if len(frame) != meta["rows"]:
        raise ValueError("Validation result count differs from completed manifest")
    if frame.duplicated(["case_id", "seed", "engine"]).any() or set(frame.engine) != {"agent", "m0"}:
        raise ValueError("Invalid engine rows in completed result")
    if not frame.groupby(["case_id", "seed"]).engine.nunique().eq(2).all():
        raise ValueError("Missing agent/reference row in completed result")
    return meta, frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--splits", choices=("tune", "holdout_1", "holdout_2", "all"), default="all")
    args = parser.parse_args()
    out = safe_output(args.out)
    baseline, before = load_run(args.baseline)
    candidate, after = load_run(args.candidate)
    for key in ("format_version", "plan_sha256", "draws", "seeds", "text_mode"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("Runs differ in required paired setting: " + key)
    for key in ("metrics", "harness_files"):
        if baseline["identity"][key] != candidate["identity"][key]:
            raise ValueError("Runs differ in scorer or validation harness; rerun the baseline")
    if args.splits != "all":
        before, after = before[before.split == args.splits], after[after.split == args.splits]
    reference_keys = ["case_id", "seed"]
    a = before[before.engine == "m0"].set_index(reference_keys).sort_index()
    b = after[after.engine == "m0"].set_index(reference_keys).sort_index()
    if not a.index.equals(b.index) or not np.allclose(a.local_loss, b.local_loss, rtol=1e-10, atol=1e-12):
        raise ValueError("Reference results changed on paired cases")
    report = compare_frames(before, after, bootstrap=args.bootstrap, seed=args.seed)
    report["baseline_engine_files"] = baseline["identity"]["engine_files"]
    report["candidate_engine_files"] = candidate["identity"]["engine_files"]
    out.mkdir(parents=True)
    json_write(out / "comparison.json", report)
    print(json.dumps({key: report[key] for key in ("overall_delta", "holdout_block_interval", "promotion_gate_passed")}, indent=2))
    print("Local evidence only; this does not predict or guarantee the CodaBench score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
