"""Fail-fast local preflight for source, outputs, organizer gates, and an optional Docker image."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def assert_output_tree(folder: pathlib.Path) -> None:
    expected = {"forecast.parquet", "forecast_meta.json", "forecast_rationale.md"}
    actual = {p.name for p in folder.iterdir() if p.is_file()}
    if actual != expected:
        raise SystemExit(f"output tree mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    total = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    if total > 64 * 1024 * 1024:
        raise SystemExit(f"output tree is {total} bytes, above 64 MiB")
    meta = json.loads((folder / "forecast_meta.json").read_text(encoding="utf-8"))
    if meta.get("representation") != "samples" or int(meta.get("n_draws", 0)) < 200:
        raise SystemExit("forecast_meta.json does not describe an admissible sample forecast")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=ROOT.parent / "track2-forecasting-public")
    parser.add_argument("--image", help="optional local Docker image tag to inspect and execute")
    parser.add_argument("--full", action="store_true", help="also execute the 103-card source regression")
    args = parser.parse_args()
    repo = args.repo.resolve()
    unit = repo / "units" / "t2-EXAMPLE-ust-curve-1m"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["T2_DISABLE_TEXT"] = "1"

    run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], env=env)
    with tempfile.TemporaryDirectory(prefix="t2-preflight-") as temp:
        output = pathlib.Path(temp)
        run([sys.executable, "-m", "t2solution.cli", "--panels", str(unit), "--text", str(unit / "text"),
             "--asof", "2024-06-28", "--out", str(output / "forecast.parquet")], env=env)
        assert_output_tree(output)
        score = run([sys.executable, str(repo / "scoring" / "scoring.py"), "score",
                     "--card", str(unit / "card.toml"), "--forecast", str(output / "forecast.parquet")], env=env)
        if '"admissible": true' not in score.stdout.lower():
            raise SystemExit("organizer scorer did not mark the exemplar admissible")

    if args.full:
        with tempfile.TemporaryDirectory(prefix="t2-all-cards-") as temp:
            run([sys.executable, str(ROOT / "scripts" / "run_practice.py"), "--repo", str(repo),
                 "--out", temp], env=env)

    if args.image:
        inspected = run(["docker", "image", "inspect", args.image, "--format",
                         "{{json .Os}} {{json .Architecture}} {{json .Config.Labels}}"])
        line = inspected.stdout.strip()
        if '"linux"' not in line or '"amd64"' not in line:
            raise SystemExit("Docker image must be linux/amd64")
        if 'qfbench2.interface_version' not in line or '\"2.0\"' not in line:
            raise SystemExit("Docker image is missing qfbench2.interface_version=2.0")
        run(["docker", "run", "--rm", "--network=none", args.image, "forecast", "--help"])
        with tempfile.TemporaryDirectory(prefix="t2-docker-") as temp:
            docker_output = pathlib.Path(temp)
            run(["docker", "run", "--rm", "--network=none", "-e", "T2_DISABLE_TEXT=1",
                 "-v", f"{unit}:/input:ro", "-v", f"{docker_output}:/output", args.image,
                 "forecast", "--panels", "/input/panels", "--text", "/input/text",
                 "--asof", "2024-06-28", "--out", "/output/forecast.parquet"])
            assert_output_tree(docker_output)
            score = run([sys.executable, str(repo / "scoring" / "scoring.py"), "score",
                         "--card", str(unit / "card.toml"),
                         "--forecast", str(docker_output / "forecast.parquet")], env=env)
            if '"admissible": true' not in score.stdout.lower():
                raise SystemExit("Docker exemplar failed organizer gates")

    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

