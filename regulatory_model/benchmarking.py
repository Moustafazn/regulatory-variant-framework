"""Explicit endpoints, training-only discretisation, and paired benchmark summaries."""
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin


class QuantileDiscretizer(TransformerMixin, BaseEstimator):
    """Training quantiles preserve ties; missing observations have their own -1 category."""
    def __init__(self, bins=3):
        self.bins = bins

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.bins < 2 or X.ndim != 2:
            raise ValueError('Need a matrix and at least two bins.')
        self.n_features_in_ = X.shape[1]
        self.edges_ = []
        for column in X.T:
            values = column[np.isfinite(column)]
            self.edges_.append(np.unique(np.quantile(values, np.arange(1, self.bins)/self.bins))
                               if len(values) else np.array([]))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError('Feature shape changed since fit.')
        out = np.column_stack([np.searchsorted(edge, X[:, j], side='right')
                               for j, edge in enumerate(self.edges_)])
        out[~np.isfinite(X)] = -1
        return out


def locus_weights(groups):
    _, inverse, sizes = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    return 1.0 / sizes[inverse]


def regression_summary(y, prediction, groups):
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    error = y-p
    if y.shape != p.shape or not np.isfinite(error).all():
        raise ValueError('Finite aligned predictions required.')
    weights = locus_weights(groups)
    mean = np.average(y, weights=weights)
    mse = np.average(error**2, weights=weights)
    variance = np.average((y-mean)**2, weights=weights)
    return {'equal_locus_mse': float(mse), 'equal_locus_rmse': float(np.sqrt(mse)),
            'equal_locus_mae': float(np.average(abs(error), weights=weights)),
            'equal_locus_r2': float(1-mse/variance) if variance else None,
            'row_rmse': float(np.sqrt(np.mean(error**2))),
            'row_mae': float(np.mean(abs(error))), 'rows': len(y), 'loci': len(np.unique(groups))}


def paired_mse_gain(y, baseline, candidate, groups, draws=2000, seed=42):
    """Positive values favour candidate. CI conditions on already fitted predictions."""
    losses = pd.DataFrame({'group': groups, 'base': (np.asarray(y)-baseline)**2,
                           'candidate': (np.asarray(y)-candidate)**2}).groupby('group').mean()
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(losses), size=(draws,len(losses)))
    base, other = losses.base.to_numpy(), losses.candidate.to_numpy()
    samples = (base-other)[indices].mean(axis=1)
    bootstrap_base = base[indices].mean(axis=1)
    valid = bootstrap_base > 0
    relative = 1-other[indices].mean(axis=1)[valid]/bootstrap_base[valid]
    return {'mse_reduction': float(np.mean(base-other)),
            'ci95': np.quantile(samples,[.025,.975]).tolist(),
            'relative_reduction': float(1-other.mean()/base.mean()) if base.mean()>0 else None,
            'relative_ci95': np.quantile(relative,[.025,.975]).tolist() if len(relative) else None,
            'relative_defined_bootstrap_draws': int(valid.sum()),
            'loci': len(losses), 'bootstrap_draws': draws,
            'inference': 'paired locus bootstrap; conditional on fitted predictions'}


def deterministic_order(scores, ids):
    if len(scores) != len(ids) or not np.isfinite(scores).all():
        raise ValueError('Finite aligned routing scores required.')
    tie = np.array([int(hashlib.sha256(str(x).encode()).hexdigest()[:12],16) for x in ids])
    return np.lexsort((tie,-np.asarray(scores)))


def review_mask(scores, ids, budget):
    """A review budget is a fraction of rows, with outcome-independent tie breaking."""
    if not 0 <= budget < 1 or not len(ids):
        raise ValueError('Nonempty candidates and review budget in [0, 1) required.')
    mask = np.zeros(len(ids), dtype=bool)
    mask[deterministic_order(scores, ids)[:int(np.ceil(budget * len(ids)))]] = True
    if mask.all():
        raise ValueError('The requested budget leaves no retained observations.')
    return mask


def selective_summary(y, prediction, groups, reviewed):
    """Conditional risk under the original equal-locus sampling distribution.

    Original locus weights stay fixed when rows are deferred. No expert correction
    is imputed. Row coverage and weighted coverage are both reported.
    """
    error = (np.asarray(y, float) - np.asarray(prediction, float)) ** 2
    kept = ~np.asarray(reviewed, bool)
    w = locus_weights(groups)
    if not kept.any() or not np.isfinite(error).all():
        raise ValueError('Finite errors and at least one retained observation required.')
    risk = float(np.average(error[kept], weights=w[kept]))
    return {'retained_mse': risk, 'retained_rmse': float(np.sqrt(risk)),
            'row_coverage': float(kept.mean()), 'weighted_coverage': float(w[kept].sum()/w.sum()),
            'reviewed_rows': int((~kept).sum()), 'retained_loci': int(len(np.unique(np.asarray(groups)[kept]))),
            'observed_loss_captured_fraction': float(np.sum(w[~kept]*error[~kept])/np.sum(w*error))
                if np.sum(w*error) > 0 else None}


def paired_selective_gain(y, prediction, groups, baseline_reviewed, candidate_reviewed,
                          draws=2000, seed=42):
    """Paired locus resampling of ratios; includes entirely deferred loci."""
    error = (np.asarray(y, float)-np.asarray(prediction, float))**2
    a, b = ~np.asarray(baseline_reviewed, bool), ~np.asarray(candidate_reviewed, bool)
    per = pd.DataFrame({'g': groups, 'a_loss': error*a, 'a_mass': a.astype(float),
                        'b_loss': error*b, 'b_mass': b.astype(float)}).groupby('g').mean()
    if not a.any() or not b.any():
        raise ValueError('Both policies must retain observations.')
    point = per.mean()
    base, other = point.a_loss/point.a_mass, point.b_loss/point.b_mass
    rng = np.random.default_rng(seed)
    sample = per.to_numpy()[rng.integers(len(per), size=(draws, len(per)))].mean(axis=1)
    valid = (sample[:, 1] > 0) & (sample[:, 3] > 0)
    sample = sample[valid]
    delta = sample[:, 0]/sample[:, 1] - sample[:, 2]/sample[:, 3]
    return {'mse_reduction': float(base-other), 'relative_reduction': float(1-other/base) if base > 0 else None,
            'ci95': np.quantile(delta, [.025,.975]).tolist() if len(delta) else [None, None],
            'bootstrap_draws': draws, 'valid_bootstrap_draws': int(valid.sum()), 'loci': len(per),
            'inference': 'Paired locus bootstrap conditional on fitted models, selected policies and fixed review masks.'}


def review_cards(frame, prediction, scores, selector, budget, context_note):
    """Outcome-blind cards: only an explicit source-side field whitelist is exported."""
    mask = review_mask(scores, frame.example_id, budget)
    cards = []
    for i, row in enumerate(frame.itertuples(index=False)):
        cards.append({'example_id': str(row.example_id), 'variant_id': str(row.variant_id),
            'chrom': str(row.chrom), 'position': int(row.position), 'ref': str(row.ref), 'alt': str(row.alt),
            'genome_build': str(row.genome_build), 'source_context': str(row.source_context),
            'target_context': str(row.target_context), 'source_effect': float(row.source_effect),
            'source_se': float(row.source_se), 'predicted_target_effect': float(prediction[i]),
            'predicted_context_transfer_delta': float(prediction[i]-row.source_effect),
            'routing_score': float(scores[i]), 'selector': selector,
            'status': 'review_candidate' if mask[i] else 'retained_estimate',
            'reason': f'Within the top {budget:.0%} by {selector}.' if mask[i] else 'Outside the fixed review budget.',
            'context_limitation': context_note,
            'adjudication': 'not_performed', 'expert_corrected_effect': None,
            'interpretation': 'A prioritization cue, not proof that this variant is wrong or that review will resolve it.'})
    return cards


def transfer_evidence(frame, margin=.25, confidence=.95):
    """Gaussian error sensitivity, not independently measured truth or an AUROC ceiling."""
    delta = (frame.target_effect-frame.source_effect).to_numpy(float)
    variance = (frame.target_se**2+frame.source_se**2-2*frame.effect_covariance).to_numpy(float)
    if (variance < -1e-10).any():
        raise ValueError('Invalid covariance gives negative transfer error variance.')
    se = np.sqrt(np.maximum(variance,1e-12))
    p_changed = norm.cdf((-margin-delta)/se)+norm.sf((margin-delta)/se)
    return pd.DataFrame({'observed_delta':delta,'delta_se':se,
                         'point_changed':abs(delta)>margin,'gaussian_p_changed':p_changed,
                         'evidence_status':np.where(p_changed>=confidence,'changed',
                                           np.where(p_changed<=1-confidence,'equivalent','unresolved'))})


def conformal_quantile(scores, alpha=.1):
    """Finite-sample order statistic; +infinity if calibration cannot support alpha."""
    scores = np.sort(np.asarray(scores,float))
    if not 0 < alpha < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite calibration scores and alpha in (0,1) required.')
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(scores[k-1]) if k<=len(scores) else float('inf')
