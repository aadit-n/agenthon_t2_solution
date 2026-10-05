"""Human-readable audit artifact for each forecast."""

from __future__ import annotations

from typing import Any

from .unit import Unit


def render(unit: Unit, asof: str, n_draws: int, diagnostics: dict[str, Any]) -> str:
    lines = [
        f"# Forecast rationale: {unit.unit_id}", "",
        "## Executive summary", "",
        f"This forecast contains {n_draws} joint Monte Carlo paths using information available through "
        f"{asof}. A {diagnostics.get('m0_share', 0):.0%} defensive component uses the baseline "
        "Gaussian-walk construction; the remaining paths use a correlated event-aware stationary "
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
    research = diagnostics.get('research', {})
    lines.extend(['', '## Research profile', '',
                  f"Profile: {research.get('profile', 'v2')}. Cumulative log-return candidates use log(1+r); "
                  'monthly trend fitting uses explicit month counts. The v2 rollback retains its original semantics.'])
    if 'conditioning' in diagnostics:
        details = diagnostics['conditioning']
        lines.extend(['', '## Experimental trajectory selection', '',
                      f"Applied: {details.get('text_used', False)}; House calls: {details.get('house_calls', 0)}; "
                      f"effective sample-size ratio: {details.get('ess_ratio', 1):.3f}.",
                      'House may only reweight existing complete joint draws; it cannot invent forecast values.',
                      f"Details: {details}"])
    if 'conditioning' in diagnostics:
        guardrails = ('Only dated, eligible document excerpts are supplied to House. Weight changes require '
                      'verified quotations and a skeptical audit. Mean movement and effective sample size are '
                      'bounded, and complete original joint draws are selected without inventing values. '
                      'Unavailable or failed model access retains the numerical forecast.')
    else:
        guardrails = ('Documents are extracted independently before synthesis. Centre shifts require '
                      'corroboration, text drift is family-shrunk and projected into empirical factor space, '
                      'and inferred text cannot narrow uncertainty. The model is asked for ex-ante scenarios '
                      'rather than future values; unavailable model access retains the numerical forecast.')
    lines.extend(['', '## Guardrails', '', guardrails, ''])
    return "\n".join(lines)

