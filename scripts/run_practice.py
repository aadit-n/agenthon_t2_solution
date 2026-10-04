"""Run the agent and organizer gate checks over public practice units."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("practice_runs"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    units = sorted(p for p in (args.repo / "units").iterdir() if (p / "forecast_spec.json").is_file())
    if args.limit:
        units = units[: args.limit]
    os.environ["T2_DISABLE_TEXT"] = "1"
    failures: list[str] = []
    for index, unit in enumerate(units, 1):
        destination = args.out / unit.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        cmd = [sys.executable, "-m", "t2solution.cli", "--panels", str(unit), "--text", str(unit / "text"),
               "--asof", _asof(unit / "card.toml"), "--out", str(destination / "forecast.parquet")]
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode:
            failures.append(f"{unit.name}: forecast: {result.stderr.strip()}")
        else:
            score = subprocess.run([sys.executable, str(args.repo / "scoring" / "scoring.py"), "score",
                                    "--card", str(unit / "card.toml"),
                                    "--forecast", str(destination / "forecast.parquet")],
                                   text=True, capture_output=True)
            if score.returncode or '"admissible": true' not in score.stdout.lower():
                failures.append(f"{unit.name}: gates: {score.stdout.strip()} {score.stderr.strip()}")
        print(f"[{index}/{len(units)}] {unit.name}: {'FAIL' if failures and failures[-1].startswith(unit.name) else 'ok'}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


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
