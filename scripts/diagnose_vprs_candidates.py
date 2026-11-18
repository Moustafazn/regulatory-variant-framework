"""Read-only test of candidate fixes for the rough-soft (VPRS) layer.

    nice -n 19 venv/bin/python scripts/diagnose_vprs_candidates.py

Compares three granulations (model features, measurements, both) on the cohort main.py evaluates, with Decima
set to zero. It reuses scripts/diagnose_vprs.py helpers and writes only output/diagnostics/vprs_candidates.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diagnose_vprs as base  # noqa: E402
from regulatory_model.evaluation import grouped_folds, locus_groups  # noqa: E402
from regulatory_model.model import MechanisticFusion, VariablePrecisionRoughSoft  # noqa: E402
from regulatory_model.pipeline import (  # noqa: E402
    FrameworkOptions, _enabled_feature_blocks, _step2c_load_tracks, build_cohort,
    fit_fold_head, load_config, predict_fold_heads,
)

BETAS = (0.5, 0.55, 0.6, 0.65)
SPACES = ('measurement', 'features', 'features+measurement', 'jurkat_activity', 'jurkat+measurement')

# Written into the output before anything runs; the gate for touching model.py is fixed here, not after the fact.
PRE_REGISTRATION = {
    'granules': {
        'measurement': 'source-SE baseline: [log source SE, |source z|, shrunk source]',
        'features': 'current VPRS granulation: the fused model feature matrix',
        'features+measurement': 'current VPRS granulation concatenated with the source-SE baseline',
        'jurkat_activity': ('Jurkat activity granules: [max(A_log2FC, B_log2FC), max(A_logPadj_BF, B_logPadj_BF), '
                            'log1p(A_Ctrl_Mean + B_Ctrl_Mean)] from sheet 9, all source-context columns'),
        'jurkat+measurement': 'Jurkat activity granules concatenated with the source-SE baseline',
    },
    'comparison': ('Equal-locus cross-entropy gain over the class prior and error-flagging AUROC, '
                   'compared against both the current VPRS (features) and the source-SE baseline (measurement).'),
    'pass_criterion': ('A Jurkat granulation passes only if, in the held-out confirmation half of loci, its paired '
                       'improvement over BOTH features and measurement has a locus-bootstrap 95% CI excluding zero. '
                       'model.py is changed only if this passes; otherwise the null result stands.'),
    'known_problems': {
        'circularity': ('Class labels are computed from the source and target SEs, so the measurement granulation '
                        'is close to circular. Noise-adjusted metrics are reported alongside the raw ones.'),
        'prior_information': ('Neighbourhood information over the class prior was negative in every feature space '
                              'tested and accepted-row rates were under 1% at beta 0.55; this run reports whether '
                              'Jurkat activity changes either.'),
        'activity_not_transfer': ('Jurkat activity predicts T-cell activity with AUROC 0.774 [0.760, 0.788]. That is '
                                  'activity, not context transfer; the granule test must not conflate them.'),
    },
}


# Load the sheet-9 Jurkat source-context columns keyed by variant_id; returns None when the workbook is absent.
def jurkat_activity_columns(workbook_path, variant_ids):
    import openpyxl
    path = Path(workbook_path)
    if not path.is_file():
        return None
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook['9. jurkat MPRA results'].values
        sheet = pd.DataFrame(rows, columns=[str(c) for c in next(rows)])
    finally:
        workbook.close()
    needed = ['A_log2FC', 'B_log2FC', 'A_logPadj_BF', 'B_logPadj_BF', 'A_Ctrl_Mean', 'B_Ctrl_Mean']
    if not set(needed) <= set(sheet.columns):
        return None
    sheet = sheet.drop_duplicates('SNP19').set_index('SNP19')
    for column in needed:
        sheet[column] = pd.to_numeric(sheet[column], errors='coerce')
    matched = sheet.reindex(pd.Index([str(v) for v in variant_ids]))
    activity = matched[['A_log2FC', 'B_log2FC']].max(axis=1).to_numpy(float)
    significance = matched[['A_logPadj_BF', 'B_logPadj_BF']].max(axis=1).to_numpy(float)
    depth = np.log1p(matched.A_Ctrl_Mean.to_numpy(float) + matched.B_Ctrl_Mean.to_numpy(float))
    matrix = np.column_stack([activity, significance, depth])
    if not np.isfinite(matrix).all():
        column_means = np.nanmean(matrix, axis=0)
        matrix = np.where(np.isfinite(matrix), matrix, column_means)
    return matrix


# Paired locus-bootstrap CI for the mean difference of a per-row quantity between two granulations.
def paired_locus_difference(values_a, values_b, groups, draws, seed):
    delta = np.asarray(values_a, float) - np.asarray(values_b, float)
    per_group = pd.Series(delta).groupby(np.asarray(groups)).mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = per_group[rng.integers(len(per_group), size=(draws, len(per_group)))].mean(axis=1)
    return {'point_estimate': float(per_group.mean()),
            'ci95': [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            'loci': int(len(per_group)),
            'excludes_zero': bool(np.quantile(samples, 0.025) > 0 or np.quantile(samples, 0.975) < 0)}


# Build [log source SE, |source z|, shrunk source]; the shrinkage variance is fitted on the fit rows only.
def measurement_matrix(frame, fit_rows):
    source = frame.source_effect.to_numpy(float)
    se = frame.source_se.to_numpy(float)
    tau2 = max(float(np.var(source[fit_rows]) - np.mean(se[fit_rows] ** 2)), 1e-6)
    shrunk = source * tau2 / (tau2 + se ** 2)
    return np.column_stack([np.log(se), np.abs(source) / se, shrunk])


# Evaluate every granulation and selector on one dataset with grouped folds.
def run(config, frame, avi, tracks, draws, jurkat=None):
    frame = frame.reset_index(drop=True).copy()
    frame['split_group'] = locus_groups(frame, gap_bp=int(config['gap_bp']))
    n = len(frame)
    decima = {state: np.zeros((n, 1), dtype=np.float32) for state in config['cell_states']}
    blocks = _enabled_feature_blocks(config, tracks is not None)
    fused = MechanisticFusion(config, enabled_blocks=blocks).transform(avi, decima, frame, track_scores=tracks)
    target = frame.target_effect.to_numpy(float)
    classes = VariablePrecisionRoughSoft(config).n_classes
    pred = np.full(n, np.nan)
    own = np.full((n, classes), np.nan)
    spaces = [s for s in SPACES if jurkat is not None or not s.startswith('jurkat')]
    support = {space: np.full((n, classes), np.nan) for space in spaces}
    gains = {space: [] for space in spaces}
    row_gain = {space: np.full(n, np.nan) for space in spaces}
    for fit, held in grouped_folds(frame, folds=int(config['folds']), seed=int(config['seed'])):
        head = fit_fold_head(config, fused[fit], frame.iloc[fit])
        pred[held] = predict_fold_heads(head, fused[held], frame.iloc[held])['target']
        measured = measurement_matrix(frame, fit)
        matrices = {'measurement': measured, 'features': fused, 'features+measurement': np.hstack([fused, measured])}
        if jurkat is not None:
            matrices['jurkat_activity'] = jurkat
            matrices['jurkat+measurement'] = np.hstack([jurkat, measured])
        for space, matrix in matrices.items():
            rough = VariablePrecisionRoughSoft(config).fit(matrix[fit], frame.iloc[fit])
            out = rough.predict(matrix[held])
            soft = rough._class_support(frame.iloc[held])
            support[space][held] = out['support']
            own[held] = soft
            prior = np.tile(rough.memberships.mean(axis=0), (len(held), 1))
            gains[space].append(base.soft_cross_entropy(soft, prior) - base.soft_cross_entropy(soft, out['support']))
            per_row_prior = -(soft * np.log(np.clip(prior, 1e-6, 1.0))).sum(axis=1)
            per_row_space = -(soft * np.log(np.clip(out['support'], 1e-6, 1.0))).sum(axis=1)
            row_gain[space][held] = per_row_prior - per_row_space
    groups = frame.split_group.to_numpy()
    source_se, target_se = frame.source_se.to_numpy(float), frame.target_se.to_numpy(float)
    error = (target - pred) ** 2
    outcomes = {'raw_error': error, 'noise_adjusted_error': error / np.clip(target_se ** 2, 1e-6, None)}
    result = {
        'rows': n, 'loci': int(len(set(groups))),
        'spearman_source_se_vs_target_se': float(pd.Series(source_se).corr(pd.Series(target_se), method='spearman')),
        'oracle_share_own_max_ge_beta': {str(b): float((own.max(axis=1) >= b).mean()) for b in BETAS},
        'spaces': [], 'selectors': []}
    for space in spaces:
        row = {'space': space, 'cross_entropy_gain': float(np.mean(gains[space]))}
        for beta in BETAS:
            accepted, boundary, _ = base.partition(support[space], beta)
            row[f'accepted@{beta}'], row[f'boundary@{beta}'] = float(accepted.mean()), float(boundary.mean())
        result['spaces'].append(row)
    scores = {'features_low_support': 1.0 - support['features'].max(axis=1),
              'measurement_low_support': 1.0 - support['measurement'].max(axis=1),
              'source_se': source_se, 'target_se (oracle, not deployable)': target_se,
              'random': np.random.default_rng(int(config['seed'])).random(n)}
    if jurkat is not None:
        scores['jurkat_low_support'] = 1.0 - support['jurkat_activity'].max(axis=1)
        scores['jurkat_activity_max'] = -jurkat[:, 0]
    for label, score in scores.items():
        row = {'selector': label}
        for outcome, values in outcomes.items():
            point, low, high = base.flag_auroc(score, values, groups, draws, int(config['seed']))
            row[f'{outcome}_auroc'], row[f'{outcome}_ci'] = round(point, 4), f'[{low:.3f}, {high:.3f}]'
        result['selectors'].append(row)
    if jurkat is not None:
        labels = np.unique(groups).copy()
        np.random.default_rng(int(config['seed'])).shuffle(labels)
        discovery = np.isin(groups, labels[:len(labels) // 2])
        gate = {'discovery_loci': int(len(labels) // 2),
                'confirmation_loci': int(len(labels) - len(labels) // 2), 'comparisons': []}
        for candidate in ('jurkat_activity', 'jurkat+measurement'):
            for reference in ('features', 'measurement'):
                entry = {'candidate': candidate, 'reference': reference}
                for half, mask in (('discovery', discovery), ('confirmation', ~discovery)):
                    entry[half] = paired_locus_difference(
                        row_gain[candidate][mask], row_gain[reference][mask],
                        groups[mask], draws, int(config['seed']))
                gate['comparisons'].append(entry)
        beats = {}
        for reference in ('features', 'measurement'):
            relevant = [c for c in gate['comparisons'] if c['reference'] == reference]
            beats[reference] = {c['candidate']: bool(c['confirmation']['excludes_zero']
                                                     and c['confirmation']['point_estimate'] > 0)
                                for c in relevant}
        gate['passes'] = {candidate: bool(beats['features'].get(candidate) and beats['measurement'].get(candidate))
                          for candidate in ('jurkat_activity', 'jurkat+measurement')}
        gate['gate_met'] = bool(any(gate['passes'].values()))
        gate['decision'] = ('model.py may be changed: a Jurkat granulation beats both references in the '
                            'confirmation half' if gate['gate_met'] else
                            'NULL RESULT: model.py stays unchanged; no Jurkat granulation beats both references '
                            'in the confirmation half with a CI excluding zero')
        result['jurkat_gate'] = gate
    return result


# Print one dataset's tables.
def show(name, result):
    print(f'\n##### {name}: {result["rows"]} rows, {result["loci"]} loci')
    print(f'Spearman(source SE, target SE) = {result["spearman_source_se_vs_target_se"]:.3f}')
    print('Share of rows whose OWN (oracle) max class membership reaches beta:', result['oracle_share_own_max_ge_beta'])
    print('\nGranulation: information over the class prior (>0 = informative) and partition shares')
    print(pd.DataFrame(result['spaces']).round(4).to_string(index=False))
    print('\nSelectors for flagging the worst 20% of the direct head\'s errors (0.5 = random)')
    print(pd.DataFrame(result['selectors']).to_string(index=False))
    gate = result.get('jurkat_gate')
    if gate:
        print('\nPre-registered Jurkat granule gate (paired per-locus information gain)')
        rows = [{'candidate': c['candidate'], 'vs': c['reference'],
                 'discovery': round(c['discovery']['point_estimate'], 5),
                 'confirmation': round(c['confirmation']['point_estimate'], 5),
                 'confirmation_ci': f"[{c['confirmation']['ci95'][0]:.5f}, {c['confirmation']['ci95'][1]:.5f}]",
                 'excludes_zero': c['confirmation']['excludes_zero']} for c in gate['comparisons']]
        print(pd.DataFrame(rows).to_string(index=False))
        print(f'  => {gate["decision"]}')


# Build the cohort once, test development and Ho, and save the JSON.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--siraj', type=int, default=1500)
    parser.add_argument('--ho', type=int, default=6000)
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'diagnostics' / 'vprs_inputs')
    parser.add_argument('--workbook', type=Path,
                        default=ROOT / 'data' / 'regulatory_audit' / 'ho_supp.xlsx')
    parser.add_argument('--result', type=Path, default=ROOT / 'output' / 'diagnostics' / 'vprs_candidates.json')
    args = parser.parse_args(argv)
    config = load_config()
    options = FrameworkOptions(output=args.output, sample_per_dataset=(('siraj', args.siraj), ('ho', args.ho)))
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / 'candidates_log.jsonl'
        frames, avi, *_ = build_cohort(options, config, log)
        tracks, _, _ = _step2c_load_tracks(options, config, frames, log)
    results = {'pre_registration': PRE_REGISTRATION}
    for name in ('development', 'ho'):
        frame = frames[name]
        jurkat = None
        if name == 'ho':
            jurkat = jurkat_activity_columns(args.workbook, frame.variant_id.astype(str))
            if jurkat is None:
                print('  Ho workbook unavailable; Jurkat granules skipped', flush=True)
        else:
            # Siraj has no equivalent source-activity columns: its prepared frame carries only Log2Skew/Skew_SE.
            results['siraj_source_activity'] = {
                'available': False,
                'reason': ('The Siraj prepared frame exposes only Log2Skew and Skew_SE per context. The archive DE '
                           'tables carry no per-allele activity (A_log2FC / B_log2FC) or activity-significance '
                           'columns equivalent to Ho sheet 9, so the Jurkat-activity granule test cannot be run '
                           'on Siraj.')}
        results[name] = run(config, frame, avi[name], tracks[name], args.draws, jurkat=jurkat)
        show('siraj (development)' if name == 'development' else name, results[name])
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=1, default=float))
    print(f'\nSaved {args.result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
