"""Read-only diagnosis of the rough-soft (VPRS) layer on the cohort the pipeline evaluates.

    nice -n 19 venv/bin/python scripts/diagnose_vprs.py

It builds the same cohort as main.py (same code path), sets Decima scores to zero (no Decima or API calls),
and writes only under output/diagnostics/. It does not touch output/experiments.
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
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from regulatory_model.evaluation import group_rmse, grouped_folds, locus_groups  # noqa: E402
from regulatory_model.model import MechanisticFusion, VariablePrecisionRoughSoft, fold_baselines  # noqa: E402
from regulatory_model.pipeline import (  # noqa: E402
    FrameworkOptions, _enabled_feature_blocks, _step2c_load_tracks, build_cohort,
    fit_fold_head, load_config, predict_fold_heads,
)

BETAS = (0.55, 0.65, 0.75)


# Mean cross-entropy of a predicted class distribution against soft (Monte Carlo) class memberships.
def soft_cross_entropy(soft, predicted):
    return float(-(soft * np.log(np.clip(predicted, 1e-6, 1.0))).sum(axis=1).mean())


# Split rows into accepted, boundary and "negative for every class" exactly as the pipeline defines them.
def partition(support, beta):
    lower = support >= beta
    upper = support > (1.0 - beta)
    accepted = lower.any(axis=1)
    boundary = ((upper.sum(axis=1) - lower.sum(axis=1)) > 0) & (support.max(axis=1) < beta)
    return accepted, boundary, ~upper.any(axis=1)


# Yield locus-bootstrap index arrays so AUROC intervals respect the grouped design.
def locus_bootstrap(groups, draws, seed):
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    for _ in range(draws):
        yield np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])


# AUROC of a score for flagging the worst 20% of squared errors, with a locus-bootstrap 95% interval.
def flag_auroc(score, error, groups, draws, seed):
    flag = error >= np.quantile(error, 0.8)
    point = float(roc_auc_score(flag, score))
    values = []
    for index in locus_bootstrap(groups, draws, seed):
        if 0 < flag[index].sum() < len(index):
            values.append(roc_auc_score(flag[index], score[index]))
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


# Share of total squared error that falls inside the top-q rows ranked by a score.
def captured_share(score, error, q):
    k = max(1, int(np.ceil(q * len(score))))
    return float(error[np.argsort(-score, kind='stable')[:k]].sum() / error.sum())


# Run every check on one dataset and return printable tables.
def diagnose(name, config, frame, avi, tracks, draws):
    frame = frame.reset_index(drop=True).copy()
    frame['split_group'] = locus_groups(frame, gap_bp=int(config['gap_bp']))
    n = len(frame)
    decima = {state: np.zeros((n, 1), dtype=np.float32) for state in config['cell_states']}
    blocks = _enabled_feature_blocks(config, tracks is not None)
    spaces = {'all': blocks}
    for label, subset in (('avi', ('avi',)), ('tracks', ('tracks',)), ('mechanistic_dhs', ('mechanistic', 'dhs'))):
        if all(b in blocks for b in subset) and (label != 'mechanistic_dhs' or name == 'ho'):
            spaces[label] = subset
    fused = {label: MechanisticFusion(config, enabled_blocks=b).transform(avi, decima, frame, track_scores=tracks)
             for label, b in spaces.items()}
    target = frame.target_effect.to_numpy(float)
    source = frame.source_effect.to_numpy(float)
    source_se = frame.source_se.to_numpy(float)
    classes = VariablePrecisionRoughSoft(config).n_classes
    pred, learned, dist = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    support, own_soft = np.full((n, classes), np.nan), np.full((n, classes), np.nan)
    prior, baselines = np.zeros(classes), {}
    ce = {label: {'knn': [], 'prior': []} for label in fused}
    folds = list(grouped_folds(frame, folds=int(config['folds']), seed=int(config['seed'])))
    for fit, held in folds:
        head = fit_fold_head(config, fused['all'][fit], frame.iloc[fit])
        pred[held] = predict_fold_heads(head, fused['all'][held], frame.iloc[held])['target']
        fit_pred = predict_fold_heads(head, fused['all'][fit], frame.iloc[fit])['target']
        features = lambda rows: np.column_stack([fused['all'][rows], source_se[rows], np.abs(source[rows])])  # noqa: E731
        scaler = StandardScaler().fit(features(fit))
        error_model = Ridge(alpha=1000.0).fit(np.nan_to_num(scaler.transform(features(fit))),
                                              np.log((target[fit] - fit_pred) ** 2 + 1e-3))
        learned[held] = error_model.predict(np.nan_to_num(scaler.transform(features(held))))
        for label, matrix in fused.items():
            rough = VariablePrecisionRoughSoft(config).fit(matrix[fit], frame.iloc[fit])
            out, own = rough.predict(matrix[held]), rough._class_support(frame.iloc[held])
            fit_prior = rough.memberships.mean(axis=0)
            ce[label]['knn'].append(soft_cross_entropy(own, out['support']))
            ce[label]['prior'].append(soft_cross_entropy(own, np.tile(fit_prior, (len(held), 1))))
            if label == 'all':
                support[held], dist[held], own_soft[held] = out['support'], out['distance'], own
                prior += fit_prior / len(folds)
        for key, values in fold_baselines(frame.iloc[fit], frame.iloc[held]).items():
            baselines.setdefault(key, np.full(n, np.nan))[held] = values
    groups = frame.split_group.to_numpy()
    error = (target - pred) ** 2
    result = {'rows': n, 'loci': int(len(set(groups))), 'class_prior': prior.round(3).tolist()}

    audit = []
    for beta in BETAS:
        accepted, boundary, negative_all = partition(support, beta)
        changed = support[:, 1:].sum(axis=1)
        binary = np.maximum(changed, 1.0 - changed) >= beta
        audit.append({'beta': beta, 'accepted': accepted.mean(), 'boundary': boundary.mean(),
                      'negative_for_every_class': negative_all.mean(), 'binary_2class_accepted': binary.mean()})
    result['partition'] = audit
    result['support'] = {'own_membership_max_mean': float(own_soft.max(axis=1).mean()),
                         'smoothed_support_max_mean': float(support.max(axis=1).mean()),
                         'share_smoothed_max_ge_0.55': float((support.max(axis=1) >= 0.55).mean())}
    result['information_gain_over_prior'] = [
        {'space': label, 'columns': int(fused[label].shape[1]),
         'cross_entropy_gain': float(np.mean(v['prior']) - np.mean(v['knn']))} for label, v in ce.items()]

    fallbacks = {'source_copy': source, **baselines}
    rmse = [{'prediction': 'ai_direct_head', 'equal_locus_rmse': group_rmse(frame, pred)[0]}]
    rmse += [{'prediction': k, 'equal_locus_rmse': group_rmse(frame, v)[0]} for k, v in fallbacks.items()]
    for beta in (0.55, 0.65):
        accepted = partition(support, beta)[0]
        for k, v in fallbacks.items():
            rmse.append({'prediction': f'rough_soft(beta={beta}, fallback={k})',
                         'equal_locus_rmse': group_rmse(frame, np.where(accepted, pred, v))[0]})
    result['rmse'] = rmse

    scores = {'vprs_low_support': 1.0 - support.max(axis=1),
              'vprs_boundary': partition(support, float(config['rough']['beta']))[1].astype(float),
              'knn_distance': dist, 'source_se': source_se, 'learned_error': learned,
              'random': np.random.default_rng(int(config['seed'])).random(n)}
    selectors = []
    for label, score in scores.items():
        point, low, high = flag_auroc(score, error, groups, draws, int(config['seed']))
        selectors.append({'selector': label, 'auroc_worst20pct': point, 'ci_low': low, 'ci_high': high,
                          'error_in_top10pct': captured_share(score, error, 0.10),
                          'error_in_top25pct': captured_share(score, error, 0.25)})
    result['selectors'] = selectors
    return result


# Print one dataset's tables.
def show(name, result):
    print(f'\n##### {name}: {result["rows"]} rows, {result["loci"]} loci; class prior {result["class_prior"]}')
    for title, key in (('Partition (share of rows)', 'partition'), ('Support', None),
                       ('Neighbourhood information over the class prior (>0 = informative)', 'information_gain_over_prior'),
                       ('Equal-locus RMSE', 'rmse'), ('Error-flagging selectors', 'selectors')):
        print(f'\n{title}')
        table = result[key] if key else [result['support']]
        print(pd.DataFrame(table).round(4).to_string(index=False))


# Build the cohort once, diagnose development and Ho, and save the JSON.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--siraj', type=int, default=1500)
    parser.add_argument('--ho', type=int, default=6000)
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'diagnostics' / 'vprs_inputs')
    parser.add_argument('--result', type=Path, default=ROOT / 'output' / 'diagnostics' / 'vprs_diagnosis.json')
    args = parser.parse_args(argv)
    config = load_config()
    options = FrameworkOptions(output=args.output, sample_per_dataset=(('siraj', args.siraj), ('ho', args.ho)))
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / 'diagnose_log.jsonl'
        frames, avi, *_ = build_cohort(options, config, log)
        tracks, _, _ = _step2c_load_tracks(options, config, frames, log)
    results = {}
    for name in ('development', 'ho'):
        results[name] = diagnose(name, config, frames[name], avi[name], tracks[name], args.draws)
        show('siraj (development)' if name == 'development' else name, results[name])
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=1, default=float))
    print(f'\nSaved {args.result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
