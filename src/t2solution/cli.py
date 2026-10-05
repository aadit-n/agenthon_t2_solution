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
from . import candidate, conditioning
from .model import numeric_context
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
                  "elapsed_seconds": round(elapsed, 3), "house_calls": diagnostics["house_calls"]},
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
    parser.add_argument('--profile', choices=candidate.PROFILES, default=os.environ.get('T2_PROFILE', 'corrected'))
    parser.add_argument('--text-mode', choices=('legacy', 'trajectory'), default=os.environ.get('T2_TEXT_MODE', 'legacy'))
    parser.add_argument('--thinking', action='store_true', help='experimental thinking for trajectory reads only')
    args = parser.parse_args(argv)

    started = time.time()
    unit = load_unit(args.panels, args.text, args.asof, args.card)
    n_draws = min(20_000, max(200, unit.n_draws_min, args.n_draws))
    seed = _seed(args.seed)
    states = candidate.prepare(unit, args.asof, args.profile)
    if args.text_mode == 'legacy':
        # Keep the scored text prompts, 11-call allocation and adjustment caps unchanged.
        view = infer_text_view(unit, args.asof, numeric_context(states))
    else:
        from .text import TextView
        view = TextView.neutral()
    samples, diagnostics = candidate.simulate(unit, args.asof, states, view, n_draws, seed, args.profile)
    if args.text_mode == 'trajectory':
        context = numeric_context(states)
        for i, state in enumerate(states):
            context[state.asset]['forecast'] = {
                str(h): {'steps': diagnostics['step_grid'][i][j],
                         'q05': float(np.quantile(samples[:, i, j], 0.05)),
                         'median': float(np.median(samples[:, i, j])),
                         'q95': float(np.quantile(samples[:, i, j], 0.95))}
                for j, h in enumerate(unit.horizons)}
            if 'fx' in state.panel.lower() and state.asset in {'EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CHF', 'CAD', 'SEK', 'NOK', 'DKK', 'CNY', 'INR', 'BRL'}:
                usd_base = state.asset not in {'EUR', 'GBP', 'AUD', 'NZD'}
                context[state.asset]['native_quote'] = f'{state.asset} per USD' if usd_base else f'USD per {state.asset}'
                context[state.asset]['native_quote_rises_when'] = f'{state.asset} weakens' if usd_base else f'{state.asset} strengthens'
        samples, details = conditioning.condition(unit, args.asof, samples, context, seed, thinking=args.thinking)
        diagnostics.update(conditioning=details, house_calls=details['house_calls'],
                           text_used=details['text_used'], text_note=details['note'])
    _write_outputs(args.out, unit, args.asof, samples, diagnostics, seed, time.time() - started)
    print(
        f"wrote {args.out}, forecast_meta.json and forecast_rationale.md; "
        f"{len(unit.assets)} assets x {len(unit.horizons)} horizons x {n_draws} draws; "
        f"profile={args.profile}, text_mode={args.text_mode}, text={diagnostics['text_used']}, "
        f"House calls={diagnostics['house_calls']}, elapsed={time.time() - started:.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

