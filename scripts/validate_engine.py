"""Freeze and run actual-engine historical validation without House or CodaBench submissions."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Match the container's BLAS limits before numerical libraries are imported.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validation.runner import execute, freeze_plan, json_write, runtime_identity, safe_output, verify_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=ROOT.parent / "track2-forecasting-public")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="new directory below backtest_runs")
    parser.add_argument("--plan", type=pathlib.Path, help="reuse an incumbent's frozen plan.json for paired comparisons")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--origins-per-split", type=int, default=4)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seeds", default="0,19,73")
    parser.add_argument("--groups-per-family", type=int, help="stratified quick run; full scope by default")
    parser.add_argument("--include-monthly", action="store_true", help="separate retrospective-vintage diagnostic tier only")
    parser.add_argument("--splits", choices=("tune", "holdout_1", "holdout_2", "all"), default="tune",
                        help="tune by default; open holdouts only for a frozen candidate or initial baseline audit")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.draws < 200 or len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        parser.error("require draws >=200 and at least two distinct nonnegative seeds")
    if args.origins_per_split < 1 or args.embargo_days < 0 or (args.groups_per_family is not None and args.groups_per_family < 1):
        parser.error("origin/group counts must be positive and embargo nonnegative")
    identity = runtime_identity()
    out, repo = safe_output(args.out), args.repo.resolve()
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
    else:
        plan = freeze_plan(repo, per_split=args.origins_per_split, embargo_days=args.embargo_days,
                           include_monthly=args.include_monthly, groups_per_family=args.groups_per_family)
    verify_plan(plan, repo)
    out.mkdir(parents=True)
    json_write(out / "plan.json", plan)
    if args.plan_only:
        print(f"PLAN ONLY: froze {len(plan['cases'])} cases in {out}; no forecasts or predictive scores", flush=True)
        return 0
    active = ["tune", "holdout_1", "holdout_2"] if args.splits == "all" else [args.splits]
    report = execute(plan, repo, out, draws=args.draws, seeds=seeds, identity=identity, active_splits=active)
    print(json.dumps({"cases": report["historical_cases"], "rows": report["rows"], "failures": report["failures"],
                      "official_score": False, "text_mode": "neutral", "output": str(out)}, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
