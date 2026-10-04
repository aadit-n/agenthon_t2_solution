"""Human-readable audit artifact for each forecast."""

from __future__ import annotations

from typing import Any

from .unit import Unit


def render(unit: Unit, asof: str, n_draws: int, diagnostics: dict[str, Any]) -> str:
    lines = [
        f"# Forecast rationale: {unit.unit_id}", "",
        "## Executive summary", "",
        f"This forecast contains {n_draws} joint Monte Carlo paths using information available through "
        f"{asof}. The numerical centre is selected by rolling historical validation. Innovations use "
        "correlated empirical residuals blended with multivariate heavy-tailed shocks.", "",
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
                  f"{diagnostics['factor_dimension']} empirical factors. All horizons are read from the same paths.", "",
                  "## Text-conditioned scenarios", "",
                  f"Text used: **{diagnostics['text_used']}**. {diagnostics['text_note']}", "",
                  "| Scenario | Probability | Evidence | Explanation |", "|---|---:|---|---|"])
    for scenario in diagnostics["scenarios"]:
        evidence = ", ".join(scenario.get("evidence", [])) or "none"
        explanation = scenario.get("explanation") or "No text adjustment; numerical forecast retained."
        lines.append(f"| {scenario['name']} | {scenario['probability']:.1%} | {evidence} | {explanation} |")
    lines.extend(["", "## Guardrails", "",
                  "Text drift is family-shrunk and projected into the empirical factor space for joint cards. "
                  "Inferred text cannot narrow uncertainty. The House model is asked for scenarios, not future "
                  "values, and unavailable model access falls back to the numerical forecast.", ""])
    return "\n".join(lines)

