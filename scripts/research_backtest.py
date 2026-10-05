"""Actual-engine, cutoff-only comparison; never reads public-card target-window outcomes.

Reports both per-pseudo-card normalized/clipped losses and group-mean normalized
diagnostics, using organizer scoring functions. These are NOT leaderboard scores.
Text is neutral; current-vintage monthly histories are not first-release backtests.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import pathlib
import sys
import tomllib
from dataclasses import replace

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from t2solution import candidate, model
from t2solution.monthly import monthly_steps
from t2solution.text import TextView
from t2solution.unit import load_unit, target_series


def counts_for(unit, cutoff):
    if unit.target_frequency.lower() != 'monthly':
        return np.tile(unit.horizons, (len(unit.assets), 1))
    last = {a: str(target_series(unit, a, cutoff)[0].index[-1].date()) for a in unit.assets}
    mapping = monthly_steps(unit.assets, unit.horizons, last, unit.card, unit.spec)
    return np.array([[mapping[(a, h)] for h in unit.horizons] for a in unit.assets])


def pseudo_task(unit, origin, counts):
    panels = {name: frame[frame.date <= origin].copy() for name, frame in unit.panels.items()}
    card, spec = copy.deepcopy(unit.card), copy.deepcopy(unit.spec)
    if unit.target_frequency.lower() == 'monthly':
        for source in (card, spec):
            target = source.get('targets', source)
            target.pop('observation_periods', None)
            target.pop('target_dates', None)
            source['questions'] = []
        for i, a in enumerate(unit.assets):
            past = target_series(unit, a, str(origin.date()))[0]
            start = past.index[-1].to_period('M')
            for j, h in enumerate(unit.horizons):
                spec['questions'].append({'asset': a, 'horizon': h,
                                           'observation_period': str(start + int(counts[i, j]))})
    actual = []
    for i, a in enumerate(unit.assets):
        full = target_series(unit, a, '9999-01-01')[0]
        future = full[full.index > origin]
        for j in range(len(unit.horizons)):
            h = int(counts[i, j])
            if len(future) < h:
                raise ValueError('pseudo-target exceeds supplied pre-card-cutoff history')
            span = full[full.index <= origin].index[-1:].append(future.index[:h])
            gaps = np.diff(span.values).astype('timedelta64[D]').astype(float)
            if (gaps > (50 if unit.target_frequency.lower() == 'monthly' else 10)).any():
                raise ValueError('pseudo-window spans an unavailable gap')
            value = float(np.log1p(future.iloc[:h]).sum()) if unit.target_type == 'log_return' else float(future.iloc[h - 1])
            actual.append(value)
    return replace(unit, panels=panels, card=card, spec=spec), np.array(actual)


def reference(trial, cutoff, counts, draws, seed):
    # Published M0 uses raw return rows, distinct from the target's log1p semantics.
    states = model.prepare(trial, cutoff)
    frame = pd.concat([s.baseline_steps.rename(s.asset) for s in states], axis=1).dropna()
    mu, cov = model._safe_covariance(frame, np.array([s.baseline_steps.std() for s in states]))
    indices = np.repeat(np.arange(len(states)), len(trial.horizons))
    steps = counts.reshape(-1)
    covariance = cov[np.ix_(indices, indices)] * np.minimum.outer(steps, steps)
    vals, vecs = np.linalg.eigh(covariance)
    root = vecs @ np.diag(np.sqrt(np.maximum(vals, 0)))
    samples = np.random.default_rng(seed).standard_normal((draws, len(steps))) @ root.T
    return samples + np.repeat([s.anchor for s in states], len(trial.horizons)) + mu[indices] * steps


def components(samples, actual):
    from qfbench2_common.scoring import crps
    from qfbench2_track_forecasting.tail import tail_coverage, tail_pinball
    flat = samples.reshape(len(samples), -1)
    lo, hi = np.quantile(flat, [0.05, 0.95], axis=0)
    return {'marginal': crps.crps_marginal(flat, actual),
            'joint': crps.variogram_score(flat, actual, p=0.5) if len(actual) > 1 else 0.0,
            'pinball': tail_pinball(flat, actual, (0.01, 0.05, 0.95, 0.99)),
            'coverage_tail': tail_coverage(flat, actual, (0.01, 0.05, 0.95, 0.99)),
            'coverage90': float(np.mean((lo <= actual) & (actual <= hi)))}


def summarize(frame):
    keys = ['group', 'origin']
    reference = frame[frame.profile == 'm0'].set_index(keys)
    joined = frame.join(reference[['marginal', 'joint', 'pinball', 'coverage_tail']].add_prefix('ref_'), on=keys)
    for metric in ('marginal', 'joint', 'pinball', 'coverage_tail'):
        joined[metric + '_ratio'] = joined[metric] / joined['ref_' + metric].clip(lower=1e-12)
    multi = joined.ref_joint > 0
    for tail, label in [('pinball', 'local_loss'), ('coverage_tail', 'legacy_tail_loss')]:
        # Official clipping applies to the WHOLE composite, not separately to its terms.
        value = np.where(multi, 0.5 * joined.marginal_ratio + 0.3 * joined.joint_ratio + 0.2 * joined[tail + '_ratio'],
                         5 / 7 * joined.marginal_ratio + 2 / 7 * joined[tail + '_ratio'])
        joined[label] = np.clip(value, 0, 4)
    group_ref = frame[frame.profile == 'm0'].groupby(['group', 'split'])[['marginal', 'joint', 'pinball']].mean().add_prefix('group_ref_')
    joined = joined.join(group_ref, on=['group', 'split'])
    m = joined.marginal / joined.group_ref_marginal.clip(lower=1e-12)
    j = joined.joint / joined.group_ref_joint.clip(lower=1e-12)
    t = joined.pinball / joined.group_ref_pinball.clip(lower=1e-12)
    joined['group_loss'] = np.where(multi, 0.5 * m + 0.3 * j + 0.2 * t, 5 / 7 * m + 2 / 7 * t)
    return joined


def save_results(frame, out, excluded):
    # Discard any previously calculated ratios before recomputing denominators.
    columns = ['group', 'unit', 'family', 'origin', 'target_type', 'frequency', 'split',
               'profile', 'marginal', 'joint', 'pinball', 'coverage_tail', 'coverage90']
    result = summarize(frame[columns])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    metrics = ['local_loss', 'group_loss', 'legacy_tail_loss', 'marginal_ratio',
               'joint_ratio', 'pinball_ratio', 'coverage90']
    summary = result.groupby(['split', 'profile'])[metrics].mean()
    summary.to_csv(out.with_suffix('.summary.csv'))
    result.groupby(['split', 'family', 'profile'])[metrics].mean().to_csv(out.with_suffix('.families.csv'))
    result.groupby(['split', 'target_type', 'frequency', 'profile'])[metrics].mean().to_csv(out.with_suffix('.populations.csv'))
    out.with_suffix('.exclusions.json').write_text(json.dumps(excluded, indent=2) + '\n', encoding='utf-8')
    paired = result.pivot(index=['group', 'unit', 'origin', 'split'], columns='profile', values='local_loss')
    differences = paired.drop(columns=['v2']).subtract(paired['v2'], axis=0).reset_index()
    differences.groupby('split').mean(numeric_only=True).drop(columns=['group']).to_csv(out.with_suffix('.paired.csv'))
    print(summary.round(5).to_string())
    print(f'Saved {len(result)} forecasts; excluded {len(excluded)} gapped windows equally for every profile. Neutral text; local losses are NOT leaderboard scores.')


def rescore_existing(repo, source, out):
    """Validate every saved origin against the supplied panel calendar, identically for all profiles."""
    frame = pd.read_csv(source)
    invalid, excluded = set(), []
    for unit_id, subset in frame.groupby('unit'):
        path = repo / 'units' / unit_id
        card = tomllib.loads((path / 'card.toml').read_text(encoding='utf-8'))
        cutoff = str(card['provenance']['data_cutoff'])
        unit = load_unit(path, path / 'text', cutoff)
        counts = counts_for(unit, cutoff)
        for origin in subset.origin.unique():
            try:
                pseudo_task(unit, pd.Timestamp(origin), counts)
            except ValueError as exc:
                if str(exc) != 'pseudo-window spans an unavailable gap':
                    raise
                invalid.add((unit_id, origin))
                excluded.append({'unit': unit_id, 'origin': origin, 'reason': str(exc)})
    keep = [(u, o) not in invalid for u, o in zip(frame.unit, frame.origin)]
    save_results(frame.loc[keep], out, excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=pathlib.Path, default=ROOT.parent / 'track2-forecasting-public')
    parser.add_argument('--out', type=pathlib.Path, default=ROOT / 'backtest_runs/research-screen.csv')
    parser.add_argument('--profiles', default='v2,corrected,drift,ecc,recent,combined')
    parser.add_argument('--draws', type=int, default=600)
    parser.add_argument('--origins', type=int, default=8)
    parser.add_argument('--offset', type=float, default=0)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rescore', type=pathlib.Path, help='validate calendars and resummarize a saved run; no new simulations')
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    profiles = args.profiles.split(',')
    if args.rescore:
        if args.rescore.resolve() == args.out.resolve():
            raise SystemExit('rescore output must differ from the original result file')
        rescore_existing(args.repo, args.rescore, args.out)
        return 0
    if not profiles or 'v2' not in profiles or any(p not in candidate.PROFILES for p in profiles):
        raise SystemExit('profiles must include v2 and contain only the documented candidate profiles')
    if args.draws < 200 or args.origins < 4 or not 0 <= args.offset < 1:
        raise SystemExit('require draws >= 200, origins >= 4 and offset in [0,1)')
    groups = {}
    for path in sorted((args.repo / 'units').glob('*/card.toml')):
        if not (path.parent / 'forecast_spec.json').is_file():
            continue
        card = tomllib.loads(path.read_text(encoding='utf-8'))
        cutoff = str(card['provenance']['data_cutoff'])
        unit = load_unit(path.parent, path.parent / 'text', cutoff)
        counts = counts_for(unit, cutoff)
        key = repr((unit.family, tuple(unit.assets), tuple(counts.reshape(-1)), unit.target_type))
        if key not in groups or cutoff > groups[key][1]:
            groups[key] = (path.parent, cutoff)
    selected = sorted(groups.items())[:args.limit] if args.limit else sorted(groups.items())
    rows, excluded = [], []
    for group, (key, (path, cutoff)) in enumerate(selected):
        unit = load_unit(path, path / 'text', cutoff)
        raw = [target_series(unit, a, cutoff)[0] for a in unit.assets]
        counts = counts_for(unit, cutoff)
        max_h = int(counts.max())
        dates = raw[0].index
        for hist in raw[1:]:
            dates = dates.intersection(hist.index)
        eligible = dates[60 if unit.target_frequency.lower() == 'monthly' else 300:-max_h]
        positions = list(range(len(eligible) - 1 - int(max_h * args.offset), -1, -max_h))[:args.origins]
        origins = [eligible[p] for p in reversed(positions)]
        if len(origins) < 4:
            continue
        print(f'[{group + 1}/{len(selected)}] {unit.unit_id}: {len(origins)} origins', flush=True)
        for number, origin in enumerate(origins):
            try:
                trial, actual = pseudo_task(unit, origin, counts)
            except ValueError as exc:
                if str(exc) != 'pseudo-window spans an unavailable gap':
                    raise
                excluded.append({'unit': unit.unit_id, 'origin': str(origin.date()), 'reason': str(exc)})
                continue
            asof = str(origin.date())
            seed = (int.from_bytes(hashlib.sha256((key + asof).encode()).digest()[:4], 'big') + args.seed) % (2**32 - 10000)
            prefix = {'group': group, 'unit': unit.unit_id, 'family': unit.family, 'origin': asof,
                      'target_type': unit.target_type, 'frequency': unit.target_frequency,
                      'split': 'development' if number < len(origins) // 2 else 'holdout'}
            rows.append({**prefix, 'profile': 'm0', **components(reference(trial, asof, counts, args.draws, seed), actual)})
            cache = {}
            for profile in profiles:
                # Postprocessing experiments share an identical numerical prior.
                base = profile if profile in {'v2', 'recent'} else 'corrected'
                if base not in cache:
                    states = candidate.prepare(trial, asof, base)
                    batches = []
                    for start in range(0, args.draws, 200):
                        samples, diag = model.simulate(trial, asof, states, TextView.neutral(), min(200, args.draws - start), seed + start)
                        batches.append(samples)
                    cache[base] = (states, np.concatenate(batches), diag)
                    del batches
                states, prior, diag = cache[base]
                samples, _ = candidate.postprocess(trial, states, prior, diag, profile, seed)
                rows.append({**prefix, 'profile': profile, **components(samples, actual)})
                del samples
            del cache
            gc.collect()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out.with_suffix('.partial.csv'), index=False)
    if not rows:
        raise SystemExit('no eligible historical tasks')
    save_results(pd.DataFrame(rows), args.out, excluded)
    args.out.with_suffix('.settings.json').write_text(json.dumps({
        'profiles': profiles, 'draws': args.draws, 'origins': args.origins,
        'offset': args.offset, 'seed': args.seed, 'groups_considered': len(selected),
        'neutral_text': True, 'minimum_training_rows_daily': 300,
        'minimum_training_rows_monthly': 60, 'max_calendar_gap_daily_days': 10,
        'max_calendar_gap_monthly_days': 50, 'official_score': False,
    }, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
