"""Competition `forecast` command."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

from . import __version__
from .model import numeric_context, prepare, simulate
from .rationale import render
from .text import infer_text_view
from .unit import load_unit


def _seed(value: int | None) -> int:
    if value is not None:
        return value
    raw = os.environ.get("QFBENCH_SEED", "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def _write_outputs(
    output: pathlib.Path, unit, asof: str, samples: np.ndarray, diagnostics: dict, seed: int, elapsed: float
) -> None:
    if not np.isfinite(samples).all():
        raise ValueError("forecast contains NaN or infinite values")
    n_draws, n_assets, n_horizons = samples.shape
    if n_draws < max(200, unit.n_draws_min):
        raise ValueError("forecast does not meet n_draws minimum")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "draw": np.repeat(np.arange(n_draws, dtype=np.int32), n_assets * n_horizons),
        "asset": pd.array(np.tile(np.repeat(np.array(unit.assets, dtype=object), n_horizons), n_draws), dtype="string"),
        "horizon": np.tile(np.array(unit.horizons, dtype=np.int32), n_draws * n_assets),
        "value": samples.reshape(-1).astype(np.float64),
    })
    if frame.duplicated(["draw", "asset", "horizon"]).any():
        raise ValueError("duplicate forecast grid cells")
    frame.to_parquet(output, index=False)
    meta = {
        "unit_id": unit.unit_id,
        "asof": asof,
        "asset_ids": unit.assets,
        "horizons": unit.horizons,
        "representation": "samples",
        "n_draws": int(n_draws),
        "agent": {"name": "t2-solution", "version": __version__, "seed": seed,
                  "elapsed_seconds": round(elapsed, 3), "house_calls": diagnostics["house_calls"],
                  "text_mode": diagnostics["text_mode"]},
    }
    if diagnostics.get("text_reliability"):
        meta["agent"]["text_diagnostics"] = {
            key: value for key, value in diagnostics["text_reliability"].items() if key != "attempts"
        }
    (output.parent / "forecast_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (output.parent / "forecast_rationale.md").write_text(
        render(unit, asof, n_draws, diagnostics), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecast", description="Agenthon 2026 Track 2 solution")
    parser.add_argument("--panels", type=pathlib.Path, required=True)
    parser.add_argument("--text", type=pathlib.Path, required=True)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--card", type=pathlib.Path)
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--text-mode", choices=("reliable", "legacy"),
                        default=os.environ.get("T2_TEXT_MODE", "reliable"))
    args = parser.parse_args(argv)

    started = time.time()
    unit = load_unit(args.panels, args.text, args.asof, args.card)
    n_draws = min(20_000, max(200, unit.n_draws_min, args.n_draws))
    seed = _seed(args.seed)
    states = prepare(unit, args.asof)
    view = infer_text_view(unit, args.asof, numeric_context(states), mode=args.text_mode)
    samples, diagnostics = simulate(unit, args.asof, states, view, n_draws, seed)
    diagnostics["text_mode"] = args.text_mode
    diagnostics["text_reliability"] = getattr(view, "reliability", {})
    _write_outputs(args.out, unit, args.asof, samples, diagnostics, seed, time.time() - started)
    print(
        f"wrote {args.out}, forecast_meta.json and forecast_rationale.md; "
        f"{len(unit.assets)} assets x {len(unit.horizons)} horizons x {n_draws} draws; "
        f"text={view.used}, House calls={view.call_count}, elapsed={time.time() - started:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

