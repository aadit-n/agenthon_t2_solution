"""Human-readable audit artifact for each forecast."""

from __future__ import annotations

from typing import Any

from .unit import Unit


def render(unit: Unit, asof: str, n_draws: int, diagnostics: dict[str, Any]) -> str:
    lines = [
        f"# Forecast rationale: {unit.unit_id}", "",
        "## Executive summary", "",
        f"This forecast contains {n_draws} joint Monte Carlo paths using information available through "
        f"{asof}. A {diagnostics.get('m0_share', 0):.0%} defensive component follows the published M0 "
        "Gaussian-walk distribution; the remaining paths use a correlated event-aware stationary "
        "bootstrap with stochastic volatility.", "",
        "## Numerical anchors", "",
        "| Asset | Last observation | Recent step volatility | Long-run step volatility | Width calibration | Numerical drift/step |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for asset, values in diagnostics["assets"].items():
        lines.append(
            f"| {asset} | {values['last_observation']:.7g} | {values['recent_step_vol']:.5g} | "
            f"{values['long_step_vol']:.5g} | {values['historical_width_calibration']:.3f} | "
            f"{values['numeric_drift_step']:.5g} |"
        )
    lines.extend(["", "## Joint structure", "",
                  f"The joint residual alignment used {diagnostics['aligned_rows']} common observations and "
                  f"{diagnostics['factor_dimension']} empirical factors. All horizons are read from the same paths. "
                  f"The empirical component assigned {diagnostics.get('stress_draw_share', 0):.1%} of draws "
                  "to historically stressed innovations.", "",
                  "## Text-conditioned scenarios", "",
                  f"Text used: **{diagnostics['text_used']}**. {diagnostics['text_note']}", "",
                  "| Scenario | Probability | Evidence | Explanation |", "|---|---:|---|---|"])
    for scenario in diagnostics["scenarios"]:
        evidence = ", ".join(scenario.get("evidence", [])) or "none"
        explanation = scenario.get("explanation") or "No text adjustment; numerical forecast retained."
        lines.append(f"| {scenario['name']} | {scenario['probability']:.1%} | {evidence} | {explanation} |")
    lines.extend(["", "## Guardrails", "",
                  "Documents are extracted independently before synthesis. Centre shifts require corroboration, "
                  "text drift is family-shrunk and projected into empirical factor space, and inferred text cannot "
                  "narrow uncertainty. The model is asked for ex-ante scenarios rather than future values; unavailable "
                  "model access falls back to the M0-anchored numerical forecast.", ""])
    return "\n".join(lines)

