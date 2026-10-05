"""Run the agent and organizer gate checks over public practice units."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("practice_runs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--profile", default=None, help="explicit numerical profile; otherwise use the agent default")
    parser.add_argument("--in-process", action="store_true", help="reuse Python imports for faster source gate checks; not a process-isolation test")
    args = parser.parse_args()
    if args.in_process:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
        sys.path.insert(0, str(args.repo.resolve()))
        from t2solution.cli import main as forecast_main
        from qfbench2_track_forecasting.scoring import _main as score_main
    units = sorted(p for p in (args.repo / "units").iterdir() if (p / "forecast_spec.json").is_file())
    if args.limit:
        units = units[: args.limit]
    os.environ["T2_DISABLE_TEXT"] = "1"
    failures: list[str] = []
    for index, unit in enumerate(units, 1):
        destination = args.out / unit.name
        if destination.exists():
            raise SystemExit(f'refusing to overwrite an existing run: {destination}; choose a fresh --out folder')
        destination.mkdir(parents=True)
        cmd = [sys.executable, "-m", "t2solution.cli", "--panels", str(unit), "--text", str(unit / "text"),
               "--asof", _asof(unit / "card.toml"), "--out", str(destination / "forecast.parquet")]
        if args.profile:
            cmd.extend(['--profile', args.profile])
        if args.in_process:
            result = _capture(forecast_main, cmd[3:])
        else:
            result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode:
            failures.append(f"{unit.name}: forecast: {result.stderr.strip()}")
        else:
            score_args = ['score', '--card', str(unit / 'card.toml'),
                          '--forecast', str(destination / 'forecast.parquet')]
            score = (_capture(score_main, score_args) if args.in_process else
                     subprocess.run([sys.executable, str(args.repo / 'scoring' / 'scoring.py'), *score_args],
                                    text=True, capture_output=True))
            if score.returncode or '"admissible": true' not in score.stdout.lower():
                failures.append(f"{unit.name}: gates: {score.stdout.strip()} {score.stderr.strip()}")
        print(f"[{index}/{len(units)}] {unit.name}: {'FAIL' if failures and failures[-1].startswith(unit.name) else 'ok'}", flush=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


def _capture(function, arguments):
    out, error = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(error):
            code = function(arguments)
    except Exception as exc:
        return subprocess.CompletedProcess(arguments, 1, out.getvalue(), str(exc))
    return subprocess.CompletedProcess(arguments, code, out.getvalue(), error.getvalue())


def _asof(card: pathlib.Path) -> str:
    import tomllib
    data = tomllib.loads(card.read_text(encoding="utf-8"))
    panels = data.get("panels", {})
    for key, value in panels.items():
        if key != "panel_ids" and isinstance(value, dict) and value.get("end_date"):
            return str(value["end_date"])
    return str(data.get("provenance", {}).get("data_cutoff"))


if __name__ == "__main__":
    raise SystemExit(main())
