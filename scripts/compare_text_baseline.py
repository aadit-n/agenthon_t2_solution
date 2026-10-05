"""Compare offline forecasts from two local images. Never pushes, submits or calls House."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile
import tomllib

import numpy as np
import pandas as pd

UNITS = ["t2-EXAMPLE-ust-curve-1m", "t2-F1-cpi-glidepath-2023",
         "t2-F2-whatever-it-takes-2012", "t2-F3-dollar-squeeze-2020",
         "t2-F4-factor-stress-2008", "t2-F4-cpi-friday-2022"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", default="t2-solution:v0.2")
    parser.add_argument("--candidate", default="t2-solution:v0.5-test")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="t2-image-parity-") as temp:
        for name in UNITS:
            root = (args.repo / "units" / name).resolve()
            card = tomllib.loads((root / "card.toml").read_text(encoding="utf-8"))
            asof = str(card["provenance"]["data_cutoff"])
            frames = []
            for role, image in (("baseline", args.baseline), ("candidate", args.candidate)):
                output = pathlib.Path(temp) / name / role
                output.mkdir(parents=True)
                result = subprocess.run([
                    "docker", "run", "--rm", "--network=none", "--read-only",
                    "--tmpfs", "/tmp:rw,nosuid,size=64m", "-e", "T2_DISABLE_TEXT=1",
                    "-e", "QFBENCH_SEED=0", "-v", f"{root}:/input:ro",
                    "-v", f"{output}:/output", image, "forecast", "--panels", "/input/panels",
                    "--text", "/input/text", "--asof", asof, "--out", "/output/forecast.parquet",
                    "--seed", "0", "--n-draws", "2000",
                ], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f"{name}/{role} failed: {result.stderr}")
                frames.append(pd.read_parquet(output / "forecast.parquet"))
            before, after = frames
            identical_grid = before[["draw", "asset", "horizon"]].equals(after[["draw", "asset", "horizon"]])
            old, new = before["value"].to_numpy(), after["value"].to_numpy()
            exact = identical_grid and np.array_equal(old, new)
            row = {"unit_id": name, "cells": len(before), "grid_identical": identical_grid,
                   "samples_bit_exact": bool(exact), "max_abs_difference": float(np.max(np.abs(old - new)))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {"baseline": args.baseline, "candidate": args.candidate, "units": rows,
              "all_bit_exact": all(row["samples_bit_exact"] for row in rows), "house_calls": 0}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["all_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
