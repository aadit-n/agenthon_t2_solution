"""Freeze historical cases, run the unchanged numerical engine, and retain every case.

Development-only: no inference-time code, cached forecasts, official references, or House
calls are added to the submission. A failed agent case is charged the local cap, not dropped.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import time
import tomllib

import numpy as np
import pandas as pd

from t2solution import model
from t2solution.text import TextView
from t2solution.unit import load_unit
from .cases import SPLITS, build_window, choose_origins, step_counts
from .metrics import CAP, LEVELS, expected_scales, gaussian_reference, score_samples
from .reporting import markdown, summarize

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRACK_COMMIT = "30c8019d997f930eab9ba569d089d12298d86d8c"
COMMON_COMMIT = "bd01548e34d21fd660d88fd06157078fb25ece4e"
FORMAT_VERSION = 1


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_write(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runtime_identity() -> dict:
    wanted = {"numpy": "2.1.3", "pandas": "2.2.3", "pyarrow": "18.1.0", "scipy": "1.16.3",
              "qfbench2-common": "2.6.0", "qfbench2-track-forecasting": "3.1.0"}
    packages = {name: importlib.metadata.version(name) for name in wanted}
    if packages != wanted:
        raise ValueError("Validation requires the isolated pins in requirements-validation.txt, not the submission environment")
    for name, commit in (("qfbench2-common", COMMON_COMMIT), ("qfbench2-track-forecasting", TRACK_COMMIT)):
        direct = json.loads(importlib.metadata.distribution(name).read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != commit:
            raise ValueError("Validation scorer must be installed at the immutable required commits")
    import qfbench2_common.scoring.crps as common_crps
    import qfbench2_track_forecasting.scoring as scoring
    import qfbench2_track_forecasting.tail as tail
    metrics_identity = {"packages": packages, "common_commit": COMMON_COMMIT, "track_commit": TRACK_COMMIT,
                        "common_crps_sha256": file_hash(pathlib.Path(common_crps.__file__)),
                        "track_scoring_sha256": file_hash(pathlib.Path(scoring.__file__)),
                        "track_tail_sha256": file_hash(pathlib.Path(tail.__file__)),
                        "normalizer_sha256": file_hash(pathlib.Path(__file__).with_name("metrics.py")),
                        "score_cap": CAP, "normalization": "input_only_expected_gaussian_M0_error"}
    return {"python": platform.python_version(), "platform": platform.platform(), "metrics": metrics_identity,
            "engine_files": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "src/t2solution").glob("*.py"))},
            "harness_files": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(
                list((ROOT / "scripts/validation").glob("*.py"))
                + [ROOT / "scripts/validate_engine.py", ROOT / "scripts/compare_validation.py"])}}


@contextlib.contextmanager
def neutral_environment():
    names = [name for name in os.environ if name.startswith("T2_") or name.startswith("MODEL_")]
    previous = {name: os.environ[name] for name in names}
    try:
        for name in names:
            os.environ.pop(name)
        os.environ["T2_DISABLE_TEXT"] = "1"
        yield
    finally:
        for name in list(os.environ):
            if name.startswith("T2_") or name.startswith("MODEL_"):
                os.environ.pop(name)
        os.environ.update(previous)


def run_agent(trial, asof: str, draws: int, seed: int) -> np.ndarray:
    with neutral_environment():
        trial = copy.deepcopy(trial)
        states = model.prepare(trial, asof)
        # One call at the actual draw count. Batching into 200s changes seeded path generation.
        samples, diagnostics = model.simulate(trial, asof, states, TextView.neutral(), draws, seed)
    if diagnostics["house_calls"] or diagnostics["text_used"]:
        raise ValueError("validation_must_be_neutral")
    return samples


def safe_output(out: pathlib.Path) -> pathlib.Path:
    out = out.resolve()
    allowed = (ROOT / "backtest_runs").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Validation output must be a new subdirectory of backtest_runs")
    if out.exists():
        raise ValueError("Refusing to overwrite existing validation output")
    return out


def input_files(root: pathlib.Path) -> dict:
    files = [root / "card.toml", root / "forecast_spec.json"]
    files += sorted(root.glob("*.parquet")) + sorted((root / "panels").glob("*.parquet"))
    return {p.relative_to(root).as_posix(): file_hash(p) for p in files if p.is_file()}


def freeze_plan(repo: pathlib.Path, *, per_split: int, embargo_days: int, include_monthly: bool,
                groups_per_family: int | None = None) -> dict:
    groups, exclusions = {}, []
    for path in sorted((repo / "units").glob("*/card.toml")):
        if not (path.parent / "forecast_spec.json").is_file():
            continue
        card = tomllib.loads(path.read_text(encoding="utf-8"))
        target, family = card["targets"], card["metadata"]["category"]
        frequency = target.get("target_frequency", card["metadata"].get("target_frequency", "daily"))
        if frequency == "monthly" and not include_monthly:
            exclusions.append({"unit": path.parent.name, "reason": "monthly_first_release_vintage_unavailable"})
            continue
        counts = None
        if frequency == "monthly":
            loaded = load_unit(path.parent, path.parent / "text", card["provenance"]["data_cutoff"])
            counts = step_counts(loaded).tolist()
        shape = {"family": family, "assets": target["asset_ids"], "horizons": target["horizons"],
                 "target_type": target.get("target_type", "level"), "frequency": frequency,
                 "value_unit": target.get("value_unit", "unknown"), "monthly_counts": counts,
                 "panels": card.get("panels", {}).get("panel_ids", []),
                 "scoring_params": card.get("scoring", {}).get("params", {})}
        key = digest(shape)
        entry = {"unit": path.parent.name, "cutoff": str(card["provenance"]["data_cutoff"]), "shape": shape}
        if key not in groups:
            groups[key] = {**entry, "aliases": [entry["unit"]]}
        else:
            aliases = groups[key]["aliases"] + [entry["unit"]]
            if (entry["cutoff"], entry["unit"]) > (groups[key]["cutoff"], groups[key]["unit"]):
                groups[key] = {**entry, "aliases": aliases}
            else:
                groups[key]["aliases"] = aliases
    ordered = sorted(groups.items(), key=lambda row: (row[1]["shape"]["family"], row[0]))
    if groups_per_family is not None:
        keep, used = [], {}
        for key, entry in ordered:
            family = entry["shape"]["family"]
            if used.get(family, 0) < groups_per_family:
                keep.append((key, entry))
                used[family] = used.get(family, 0) + 1
        ordered = keep
    cases, sources = [], {}
    for number, (key, entry) in enumerate(ordered, 1):
        root = repo / "units" / entry["unit"]
        unit = load_unit(root, root / "text", entry["cutoff"])
        counts = step_counts(unit)
        choices, counts_excluded = choose_origins(unit, counts, per_split=per_split, embargo_days=embargo_days)
        exclusions.append({"unit": unit.unit_id, "group_id": key, "calendar_exclusions": counts_excluded,
                           "missing_splits": sorted(set(SPLITS) - {row["split"] for row in choices})})
        sources[unit.unit_id] = {**entry, "files": input_files(root)}
        for choice in choices:
            origin = pd.Timestamp(choice["origin"])
            window = build_window(unit, origin, counts, min_train=60 if unit.target_frequency == "monthly" else 300)
            try:
                ref = gaussian_reference(window.trial, counts)
                levels = tuple(unit.card.get("scoring", {}).get("params", {}).get("tail_levels", LEVELS))
                scales = expected_scales(ref.mean, ref.covariance, levels)
            except ValueError as error:
                exclusions.append({"unit": unit.unit_id, "origin": choice["origin"], "reason": str(error)})
                continue
            case = {**choice, "unit": unit.unit_id, "group_id": key, "family": unit.family,
                    "target_type": unit.target_type, "frequency": unit.target_frequency,
                    "tier": "retrospective_monthly" if unit.target_frequency == "monthly" else "history_proxy",
                    "assets": unit.assets, "horizons": unit.horizons, "counts": counts.tolist(), "scales": scales}
            case["case_id"] = digest({"group": key, "origin": choice["origin"]})
            cases.append(case)
        print(f"plan [{number}/{len(ordered)}] {unit.unit_id}: {len(choices)} calendar cases", flush=True)
    if not cases:
        raise ValueError("No eligible historical cases; check input histories and chosen scope")
    body = {"format_version": FORMAT_VERSION, "official_score": False, "sources": sources,
            "cases": cases, "exclusions": exclusions, "splits": SPLITS,
            "settings": {"per_split": per_split, "embargo_business_days": embargo_days,
                         "include_monthly": include_monthly, "groups_per_family": groups_per_family},
            "tier_note": "Dated historical rows do not prove first-release availability. Factors may have reporting lags and revisions."}
    return {**body, "plan_sha256": digest(body)}


def verify_plan(plan: dict, repo: pathlib.Path) -> None:
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("format_version") != FORMAT_VERSION or digest(body) != plan.get("plan_sha256"):
        raise ValueError("Invalid or modified historical validation plan")
    for name, entry in plan["sources"].items():
        if pathlib.Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid source unit path")
        if input_files(repo / "units" / name) != entry["files"]:
            raise ValueError("Input snapshot changed; do not compare this run against the old plan")
    ids = [case["case_id"] for case in plan["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate historical case")


def execute(plan: dict, repo: pathlib.Path, out: pathlib.Path, *, draws: int, seeds: list[int], identity: dict,
            active_splits: list[str] | None = None) -> dict:
    active_splits = active_splits or list(SPLITS)
    cases = [case for case in plan["cases"] if case["split"] in active_splits]
    if not cases:
        raise ValueError("No cases in the requested split")
    manifest = {"format_version": FORMAT_VERSION, "state": "running", "official_score": False,
                "plan_sha256": plan["plan_sha256"], "draws": draws, "seeds": seeds, "text_mode": "neutral",
                "active_splits": active_splits, "identity": identity, "expected_rows": len(cases) * len(seeds) * 2}
    json_write(out / "manifest.json", manifest)
    rows, cache_unit, cached = [], None, None
    started = time.monotonic()
    with (out / "results.partial.jsonl").open("x", encoding="utf-8") as partial:
        for number, case in enumerate(cases, 1):
            if case["unit"] != cache_unit:
                root = repo / "units" / case["unit"]
                cached = load_unit(root, root / "text", plan["sources"][case["unit"]]["cutoff"])
                cache_unit = case["unit"]
            counts = np.array(case["counts"], dtype=int)
            window = build_window(cached, pd.Timestamp(case["origin"]), counts,
                                  min_train=60 if case["frequency"] == "monthly" else 300)
            if window.target_dates != case["target_dates"] or window.label_end != case["label_end"]:
                raise ValueError("Historical labels no longer match the frozen calendar")
            reference = gaussian_reference(window.trial, counts)
            levels = tuple(window.trial.card.get("scoring", {}).get("params", {}).get("tail_levels", LEVELS))
            scales = expected_scales(reference.mean, reference.covariance, levels)
            if not all(np.isclose(scales[name], case["scales"][name], rtol=1e-10, atol=0) for name in scales):
                raise ValueError("Historical normalization changed; freeze a new plan and baseline")
            prefix = {key: case[key] for key in ("case_id", "group_id", "unit", "origin", "label_end", "split",
                                                "family", "target_type", "frequency", "tier")}
            for seed in seeds:
                rng_seed = (int(case["case_id"][:8], 16) + seed) % 2**32
                m0 = reference.sample(draws, rng_seed)
                row = {**prefix, "seed": seed, "rng_seed": rng_seed, "engine": "m0", "status": "ok",
                       **score_samples(m0, window.actual, window.trial, scales)}
                rows.append(row)
                partial.write(json.dumps(row, allow_nan=False) + "\n")
                try:
                    samples = run_agent(window.trial, case["origin"], draws, rng_seed)
                    result = score_samples(samples, window.actual, window.trial, scales)
                    row = {**prefix, "seed": seed, "rng_seed": rng_seed, "engine": "agent", "status": "ok", **result}
                except Exception as error:
                    # A defective agent cannot benefit from shrinking the comparison denominator.
                    row = {**prefix, "seed": seed, "rng_seed": rng_seed, "engine": "agent", "status": "engine_failed",
                           "error_type": type(error).__name__, "local_loss": CAP, "unclipped_loss": CAP,
                           "clip_applied": False, **{key: None for key in ("marginal_normalized", "joint_normalized", "tail_normalized", "coverage90")}}
                rows.append(row)
                partial.write(json.dumps(row, allow_nan=False) + "\n")
            partial.flush()
            print(f"run [{number}/{len(cases)}] {case['family']} {case['unit']} {case['origin']} {case['split']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "results.parquet", index=False)
    report = summarize(frame)
    json_write(out / "summary.json", report)
    (out / "summary.md").write_text(markdown(report), encoding="utf-8")
    manifest.update(state="complete", elapsed_seconds=round(time.monotonic() - started, 3),
                    rows=len(rows), engine_failed_rows=report["failures"],
                    results_sha256=file_hash(out / "results.parquet"))
    json_write(out / "manifest.json", manifest)
    return report
