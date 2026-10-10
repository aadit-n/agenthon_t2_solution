"""Group-balanced diagnostics and paired, calendar-block sensitivity intervals.

Random seeds are averaged within each historical case, not counted as independent cases.
Shared instruments and periods remain correlated: bootstrap all groups together by time block.
These reports are local proxies and cannot certify leaderboard accuracy or a win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["local_loss", "unclipped_loss", "marginal_normalized", "joint_normalized", "tail_normalized", "coverage90"]
KEYS = ["case_id", "seed", "engine"]


def case_means(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("empty_or_duplicate_results")
    if not np.isfinite(frame.local_loss).all():
        raise ValueError("missing_case_score")
    group = ["engine", "tier", "split", "family", "group_id", "case_id", "origin", "frequency", "target_type"]
    return frame.groupby(group, sort=True, dropna=False)[METRICS].mean().reset_index()


def balanced(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    # Equal weight to each deduplicated task-shape group, not raw currency units or seed rows.
    by_group = frame.groupby(keys + ["group_id"], dropna=False)[METRICS].mean().reset_index()
    means = by_group.groupby(keys, dropna=False)[METRICS].mean().reset_index()
    counts = frame.groupby(keys, dropna=False).agg(historical_cases=("case_id", "nunique"), groups=("group_id", "nunique")).reset_index()
    means = means.merge(counts, on=keys, validate="one_to_one")
    return means.astype(object).where(pd.notna(means), None).to_dict("records")


def summarize(frame: pd.DataFrame) -> dict:
    cases = case_means(frame)
    return {"official_score": False, "text_mode": "neutral", "rows": len(frame),
            "historical_cases": int(frame.case_id.nunique()), "failures": int((frame.status != "ok").sum()),
            "overall": balanced(cases, ["engine", "tier"]),
            "splits": balanced(cases, ["engine", "tier", "split"]),
            "families": balanced(cases, ["engine", "tier", "split", "family"]),
            "populations": balanced(cases, ["engine", "tier", "split", "target_type", "frequency"]),
            "seed_sensitivity": balanced(frame, ["engine", "tier", "seed"]),
            "clipped_rows": int(frame.clip_applied.sum()),
            "warnings": ["Historical shapes are not the official roster or event-selected outcome distribution.",
                         "No live House predictive test is performed; text is neutral.",
                         "Dated panel rows do not establish first-release availability or exclude later revisions.",
                         "Do not use holdout feedback repeatedly to choose hyperparameters."]}


def block_interval(paired: pd.DataFrame, *, bootstrap: int, seed: int):
    if bootstrap < 100:
        raise ValueError("bootstrap_requires_at_least_100")
    hold = paired[paired.split.str.startswith("holdout")].copy()
    hold["block"] = pd.to_datetime(hold.origin).dt.year // 2 * 2
    blocks = sorted(hold.block.unique())
    if len(blocks) < 4:
        return {"blocks": len(blocks), "low": None, "high": None, "reason": "insufficient_time_blocks"}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        sampled = rng.choice(blocks, len(blocks), replace=True)
        weights = pd.Series(sampled).value_counts()
        multiplicity = hold.block.map(weights).fillna(0).to_numpy(dtype=float)
        weighted = hold.assign(weight=multiplicity, delta_weighted=hold.delta * multiplicity)
        sums = weighted.groupby("group_id")[["weight", "delta_weighted"]].sum()
        estimates.append(float((sums.delta_weighted / sums.weight.replace(0, np.nan)).mean()))
    low, high = np.quantile(estimates, [.025, .975])
    return {"blocks": len(blocks), "low": float(low), "high": float(high),
            "reason": "calendar_2_year_block_sensitivity_not_independent_unit_ci"}


def compare_frames(base: pd.DataFrame, candidate: pd.DataFrame, *, bootstrap: int, seed: int) -> dict:
    for frame in (base, candidate):
        if frame.empty or frame.duplicated(KEYS).any():
            raise ValueError("empty_or_duplicate_results")
    before = base[base.engine == "agent"].set_index(["case_id", "seed"])
    after = candidate[candidate.engine == "agent"].set_index(["case_id", "seed"])
    if before.empty or not before.index.sort_values().equals(after.index.sort_values()):
        raise ValueError("comparison_case_or_seed_mismatch")
    after = after.reindex(before.index)
    metadata = ["group_id", "split", "family", "origin", "label_end", "frequency", "target_type", "tier"]
    if not before[metadata].equals(after[metadata]):
        raise ValueError("comparison_case_metadata_mismatch")
    if not np.isfinite(before.local_loss).all() or not np.isfinite(after.local_loss).all():
        raise ValueError("comparison_nonfinite_scores")
    paired = before[metadata].copy()
    paired["delta"] = after.local_loss - before.local_loss
    paired = paired.reset_index().groupby(["case_id"] + metadata, dropna=False).delta.mean().reset_index()
    # A positive delta is worse. Never include retrospective monthly diagnostics in promotion.
    primary = paired[paired.tier == "history_proxy"]
    overall = float(primary.groupby("group_id").delta.mean().mean()) if len(primary) else None
    folds, families = [], []
    for fold, subset in primary.groupby("split"):
        folds.append({"split": fold, "delta": float(subset.groupby("group_id").delta.mean().mean()),
                      "cases": len(subset), "groups": int(subset.group_id.nunique())})
    for (fold, family), subset in primary.groupby(["split", "family"]):
        families.append({"split": fold, "family": family,
                         "delta": float(subset.groupby("group_id").delta.mean().mean()), "cases": len(subset)})
    interval = block_interval(primary, bootstrap=bootstrap, seed=seed)
    held = {row["split"]: row["delta"] for row in folds if row["split"].startswith("holdout")}
    failures = int((after.status != "ok").sum())
    family_safe = all(row["delta"] <= .01 for row in families if row["split"].startswith("holdout"))
    coverage_ok = all(
        primary[(primary.split == fold) & (primary.family == family)].group_id.nunique() >= 2
        for fold in ("holdout_1", "holdout_2") for family in ("T2-F1", "T2-F2", "T2-F3", "T2-F4")
    )
    seeds_ok = before.index.get_level_values("seed").nunique() >= 3
    gate = (set(held) == {"holdout_1", "holdout_2"} and all(value < 0 for value in held.values())
            and interval["high"] is not None and interval["high"] < 0 and family_safe and failures == 0
            and coverage_ok and seeds_ok)
    return {"official_score": False, "overall_delta": overall, "splits": folds, "families": families,
            "holdout_block_interval": interval, "historical_cases": len(paired), "candidate_failed_rows": failures,
            "promotion_gate_passed": bool(gate),
            "coverage_gate_passed": coverage_ok, "seed_gate_passed": seeds_ok,
            "gate_definition": "Both holdouts improve; block interval upper < 0; no family worsens > .01; no failures; >=2 groups/family/holdout; >=3 seeds.",
            "warnings": ["A passed local gate is not a guarantee of an official score improvement.",
                         "Intervals are sensitivity diagnostics on overlapping instruments, not certified IID confidence.",
                         "Monthly retrospective-vintage cases are excluded from the primary gate."]}


def markdown(report: dict) -> str:
    lines = ["# Real-engine historical validation", "", "Local historical proxy, **not an official score**. Text is neutral.", "",
             "Random seeds are averaged within cases; task-shape groups have equal weight.", "",
             "| Engine | Tier | Split | Cases | Groups | Local loss | Marginal | Joint | Tail | 90% coverage |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["splits"]:
        vals = [f"{row[key]:.5f}" if row[key] is not None else "n/a"
                for key in ("local_loss", "marginal_normalized", "joint_normalized", "tail_normalized", "coverage90")]
        lines.append(f"| {row['engine']} | {row['tier']} | {row['split']} | {row['historical_cases']} | {row['groups']} | " + " | ".join(vals) + " |")
    lines.extend(["", f"Failed engine rows: {report['failures']}; clipped rows: {report['clipped_rows']}.", "", "## Limits", ""])
    lines.extend("- " + warning for warning in report["warnings"])
    return "\n".join(lines) + "\n"
