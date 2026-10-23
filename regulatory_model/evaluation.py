"""Leakage-aware splits, experimental labels, locus-grouped metrics and four-method evaluation."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

CLASSES = ['maintained', 'weakened', 'strengthened', 'reversed']


def effect_classes(source, target, margin):
    # Assign operational point classes; non-significance is not equivalence.
    source, target = np.broadcast_arrays(np.asarray(source), np.asarray(target))
    changed = np.abs(target - source) > margin
    reverse = (source * target < 0) & (np.minimum(np.abs(source), np.abs(target)) > margin)
    label = np.where(np.abs(target) > np.abs(source), 2, 1)
    return np.where(changed, np.where(reverse, 3, label), 0).astype(int)


def experimental_labels(frame, margin, confidence=0.8, seed=42, draws=256):
    # Propagate reported source/target errors and covariance into experimental class support.
    rng = np.random.default_rng(seed)
    source = frame.source_effect.to_numpy()[:, None]
    target = frame.target_effect.to_numpy()[:, None]
    source_se = frame.source_se.to_numpy()[:, None]
    target_se = frame.target_se.to_numpy()[:, None]
    rho = np.clip(frame.effect_covariance.to_numpy()[:, None] / (source_se * target_se), -1, 1)
    z1, z2 = rng.normal(size=(2, len(frame), draws))
    labels = effect_classes(source + source_se * z1,
                            target + target_se * (rho * z1 + np.sqrt(1 - rho ** 2) * z2), margin)
    support = np.column_stack([(labels == i).mean(axis=1) for i in range(len(CLASSES))])
    point = effect_classes(frame.source_effect, frame.target_effect, margin)
    reliability = support[np.arange(len(frame)), point]
    return point, reliability >= confidence, support


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return int(item)

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def locus_groups(frame, gap_bp=250_000):
    # Join nearby variants and shared identifiers into indivisible leakage groups.
    frame = frame.reset_index(drop=True)
    links = UnionFind(len(frame))
    for _, part in frame.groupby(['genome_build', 'chrom'], sort=True):
        order = part.sort_values('position').index.to_numpy()
        for a, b in zip(order[:-1], order[1:]):
            if frame.at[b, 'position'] - frame.at[a, 'position'] <= gap_bp:
                links.union(a, b)
    for column in ['variant_id', 'rsid', 'lead_group']:
        if column not in frame.columns:
            continue
        for _, part in frame.dropna(subset=[column]).groupby(['genome_build', column]):
            indices = part.index.to_numpy()
            for item in indices[1:]:
                links.union(indices[0], item)
    return np.array([f'locus_{links.find(i):07d}' for i in range(len(frame))])


def grouped_folds(frame, folds, seed, group_column='split_group'):
    # Yield (fit, held) index arrays using locus-grouped K-fold.
    groups = frame[group_column].to_numpy()
    for fit, held in GroupKFold(n_splits=folds).split(frame, groups=groups):
        yield fit, held


def inner_split(fit_indices, groups, fraction, seed):
    # Split a fitting fold once more into an inner fit / inner validation subset by locus.
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    inner_fit, inner_valid = next(splitter.split(fit_indices, groups=groups[fit_indices]))
    return fit_indices[inner_fit], fit_indices[inner_valid]


def group_rmse(frame, prediction):
    # Equal-weight-per-locus RMSE on real target effects.
    errors = (frame.target_effect.to_numpy() - np.asarray(prediction)) ** 2
    per_group = pd.Series(errors).groupby(frame.split_group.to_numpy()).mean().to_numpy()
    return float(np.sqrt(per_group.mean())), per_group


def group_rmse_bootstrap(per_group_mse, draws=300, seed=42):
    # Non-parametric bootstrap over loci, returning (low, high) 95% CI.
    rng = np.random.default_rng(seed)
    samples = [np.sqrt(rng.choice(per_group_mse, len(per_group_mse), replace=True).mean()) for _ in range(draws)]
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def selective_curve(frame, prediction, scores, coverage_grid):
    # Risk vs retained coverage, with deterministic hash-based tie-breaking.
    prediction = np.asarray(prediction)
    errors = (frame.target_effect.to_numpy() - prediction) ** 2
    tie = np.array([int(hashlib.sha256(str(x).encode()).hexdigest()[:12], 16) for x in frame.example_id])
    rows = []
    for coverage in coverage_grid:
        n = max(1, int(np.ceil(len(frame) * coverage)))
        order = np.lexsort((tie, -np.asarray(scores)))
        kept = order[:n]
        rmse = float(np.sqrt(errors[kept].mean()))
        rows.append({'requested_coverage': coverage, 'actual_coverage': n / len(frame),
                     'rows': int(n), 'rmse': rmse})
    return rows


def paired_locus_bootstrap(frame, a, b, draws=1000, seed=42):
    # Paired-locus bootstrap over the difference of equal-weight MSEs.
    delta = (frame.target_effect.to_numpy() - np.asarray(a)) ** 2 - (frame.target_effect.to_numpy() - np.asarray(b)) ** 2
    per_group = pd.Series(delta).groupby(frame.split_group.to_numpy()).mean().to_numpy()
    rng = np.random.default_rng(seed)
    samples = per_group[rng.integers(len(per_group), size=(draws, len(per_group)))].mean(axis=1)
    return {'point_estimate_mse_difference': float(per_group.mean()),
            'ci95': [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            'loci': int(len(per_group))}
