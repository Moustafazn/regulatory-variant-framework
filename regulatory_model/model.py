"""Ziarko variable-precision rough-soft set (box 3) and mechanistic feature fusion.

The AI predictor (prediction stage of the framework) combines local AlphaGenome AVI
annotations with Decima scores. This module contains the
uncertainty wrapper (``VariablePrecisionRoughSoft``), a Ridge-based scalar head that
converts fused representations into a scalar effect prediction, and ``MechanisticFusion``
which concatenates the per-variant representation used by both the head and the VPRS.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .evaluation import CLASSES, effect_classes


class VariablePrecisionRoughSoft:
    """Ziarko 1993 variable-precision rough-set on learned kNN neighbourhoods."""

    def __init__(self, settings):
        self.cfg = settings['rough']
        self.margin = float(settings['margin'])
        self.n_classes = len(CLASSES)
        self.draws = int(settings['rough'].get('membership_draws', 256))
        self.seed = int(settings.get('seed', 42))

    def fit(self, representations, frame):
        representations = np.asarray(representations, dtype=float)
        self.scaler = StandardScaler().fit(representations)
        n_neighbors = int(min(self.cfg['neighbors'], len(frame)))
        self.neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(
            self.scaler.transform(representations))
        self.memberships = self._class_support(frame)
        return self

    def _class_support(self, frame):
        rng = np.random.default_rng(self.seed)
        n = len(frame)
        source = frame.source_effect.to_numpy(dtype=float)[:, None]
        target = frame.target_effect.to_numpy(dtype=float)[:, None]
        source_se = np.clip(frame.source_se.to_numpy(dtype=float)[:, None], 1e-6, None)
        target_se = np.clip(frame.target_se.to_numpy(dtype=float)[:, None], 1e-6, None)
        if 'effect_covariance' in frame.columns:
            cov = frame.effect_covariance.to_numpy(dtype=float)[:, None]
        else:
            cov = np.zeros_like(source)
        rho = np.clip(cov / (source_se * target_se), -1.0, 1.0)
        z1, z2 = rng.normal(size=(2, n, self.draws))
        sampled_source = source + source_se * z1
        sampled_target = target + target_se * (rho * z1 + np.sqrt(1.0 - rho ** 2) * z2)
        labels = effect_classes(sampled_source, sampled_target, self.margin)
        return np.column_stack([(labels == c).mean(axis=1) for c in range(self.n_classes)])

    def predict(self, representations):
        # Return per-class {support, lower, upper, mean neighbour distance}.
        representations = np.asarray(representations, dtype=float)
        distances, indices = self.neighbors.kneighbors(self.scaler.transform(representations))
        values = self.memberships[indices]
        temperature = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)
        weights = np.exp(-distances / temperature)
        support = (values * weights[..., None]).sum(1) / weights.sum(1)[:, None]
        beta = float(self.cfg['beta'])
        lower = (support >= beta).astype(float)
        upper = (support > (1.0 - beta)).astype(float)
        return {'support': support, 'lower': lower, 'upper': upper,
                'distance': distances.mean(1), 'neighbor_indices': indices}


class FeatureBlock:
    """biologically-meaningful block of the fused representation."""

    def __init__(self, name, column_names, builder, required_inputs=()):
        self.name = name
        self.column_names = tuple(column_names)
        self._builder = builder
        self.required_inputs = tuple(required_inputs)

    def build(self, frame, inputs):
        missing = [key for key in self.required_inputs if inputs.get(key) is None]
        if missing:
            raise KeyError(f'Feature block {self.name!r} requires inputs {missing!r}.')
        matrix = np.asarray(self._builder(frame, inputs), dtype=float)
        if matrix.shape != (len(frame), len(self.column_names)):
            raise ValueError(
                f'Feature block {self.name!r} produced shape {matrix.shape}; '
                f'expected ({len(frame)}, {len(self.column_names)}).'
            )
        return matrix


def _avi_block(settings):
    features = tuple(settings['fusion']['avi_features'])
    columns = [f'avi:{feature}' for feature in features]

    def build(frame, inputs):
        avi = inputs['avi']
        return np.column_stack([
            np.asarray(avi[feature], dtype=float).reshape(len(frame))
            for feature in features
        ])

    return FeatureBlock('avi', columns, build, required_inputs=('avi',))


def _tracks_block(settings):
    # Tracks may cover only a subset of cell_states, because cached NPZs are keyed by the states present when downloaded.
    cell_states = tuple(settings['fusion'].get('track_cell_states')
                        or settings['cell_states'].keys())
    modalities = tuple(settings['fusion'].get('alphagenome_track_modalities', ()))
    columns = [f'tracks:{state}:{modality}' for state in cell_states for modality in modalities]

    def build(frame, inputs):
        tracks = inputs['tracks']
        parts = []
        for state in cell_states:
            vec = np.asarray(tracks[state], dtype=float).reshape(len(frame), -1)
            if vec.shape[1] != len(modalities):
                raise ValueError(
                    f'tracks[{state!r}] has {vec.shape[1]} modalities; '
                    f'expected {len(modalities)} ({modalities}).'
                )
            parts.append(vec)
        return np.concatenate(parts, axis=1)

    return FeatureBlock('tracks', columns, build, required_inputs=('tracks',))


def _decima_block(settings):
    cell_states = tuple(settings['cell_states'].keys())
    columns = [f'decima:{state}' for state in cell_states]

    def build(frame, inputs):
        decima = inputs['decima']
        return np.column_stack([
            np.asarray(decima[state], dtype=float).reshape(len(frame))
            for state in cell_states
        ])

    return FeatureBlock('decima', columns, build, required_inputs=('decima',))


def _mechanistic_block(settings):
    fields = tuple(settings['fusion']['ho_mechanistic_fields'])
    columns = [f'ho:{field}' for field in fields]

    def build(frame, _inputs):
        return frame.reindex(columns=fields).to_numpy(dtype=float, na_value=0.0)

    return FeatureBlock('mechanistic', columns, build)


def _dhs_block(settings):
    fields = tuple(settings['fusion']['dhs_fields'])
    columns = [f'dhs:{field}' for field in fields]

    def build(frame, _inputs):
        return frame.reindex(columns=fields).to_numpy(dtype=float, na_value=0.0)

    return FeatureBlock('dhs', columns, build)


FEATURE_BLOCK_FACTORIES = {
    'avi': _avi_block,
    'tracks': _tracks_block,
    'decima': _decima_block,
    'mechanistic': _mechanistic_block,
    'dhs': _dhs_block,
}


class MechanisticFusion:
    """Build the fused representation from an ordered registry of feature blocks."""

    def __init__(self, settings, enabled_blocks=None):
        order = tuple(settings['fusion'].get('feature_blocks',
                                             ('avi', 'tracks', 'decima', 'mechanistic', 'dhs')))
        enabled = set(enabled_blocks) if enabled_blocks is not None else set(order)
        unknown = enabled - set(FEATURE_BLOCK_FACTORIES)
        if unknown:
            raise ValueError(f'Unknown feature blocks requested: {sorted(unknown)}')
        self.blocks = [FEATURE_BLOCK_FACTORIES[name](settings) for name in order if name in enabled]
        self.column_names = tuple(column for block in self.blocks for column in block.column_names)

    def transform(self, avi_scores, decima_scores, frame, track_scores=None):
        inputs = {'avi': avi_scores, 'decima': decima_scores, 'tracks': track_scores}
        parts = [block.build(frame, inputs) for block in self.blocks]
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(frame), 0))


def source_prior(source, source_se):
    """Mean and variance of the true source effect, estimated from the rows given."""
    source = np.asarray(source, dtype=float)
    se = np.asarray(source_se, dtype=float)
    return float(source.mean()), float(max(source.var() - np.mean(se ** 2), 1e-6))


def shrink_source(source, source_se, prior_mean, prior_var):
    """Empirical-Bayes posterior mean of the true source effect given its measurement error."""
    se2 = np.clip(np.asarray(source_se, dtype=float), 1e-6, None) ** 2
    weight = prior_var / (prior_var + se2)
    return prior_mean + weight * (np.asarray(source, dtype=float) - prior_mean)


class RidgeHead:
    """Standardised Ridge heads; the 'target' head is the primary-T prediction and also takes the shrunk source effect."""

    HEADS = ('context_transfer', 'source', 'target')
    SOURCE_FEATURE = 'source:shrunk'

    def __init__(self, settings):
        head = settings['head']
        self.alpha = float(head['ridge_alpha'])
        self.alpha_grid = tuple(sorted(float(a) for a in head.get('ridge_alpha_grid', ())))
        self.alpha_cv_folds = int(head.get('alpha_cv_folds', 3))
        self.use_source_feature = bool(head.get('shrink_source_feature', True))
        # Half the Ho rows carry 5.5% of target variance; weighting by 1/target_se^2 stops them dominating the fit.
        self.precision_weight = bool(head.get('precision_weight_target', False))
        self.seed = int(settings.get('seed', 42))

    def _weights(self, frame):
        """Fit-row weights: inverse target measurement variance, normalised to mean 1, or None when disabled."""
        if not self.precision_weight or 'target_se' not in frame.columns:
            return None
        se = np.clip(frame.target_se.to_numpy(dtype=float), 1e-3, None)
        weight = 1.0 / se ** 2
        mean = float(weight.mean())
        return weight / mean if mean > 0 else None

    def fit(self, representations, frame):
        """Fit every head on the given rows; the prior and each alpha are chosen from these rows only."""
        x = np.asarray(representations, dtype=float)
        source = frame.source_effect.to_numpy(dtype=float)
        target = frame.target_effect.to_numpy(dtype=float)
        source_se = frame.source_se.to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(x)
        self.prior_mean, self.prior_var = source_prior(source, source_se)
        self.mean_source_se = float(np.sqrt(np.mean(source_se ** 2)))
        shrunk = shrink_source(source, source_se, self.prior_mean, self.prior_var)
        self.shrunk_mean = float(shrunk.mean())
        self.shrunk_std = float(shrunk.std()) or 1.0
        z = self.scaler.transform(x)
        designs = {'context_transfer': z, 'source': z, 'target': self._design(z, shrunk)}
        y = {'context_transfer': target - source, 'source': source, 'target': target}
        groups = frame['split_group'].to_numpy() if 'split_group' in frame.columns else None
        weights = self._weights(frame)
        self.alphas = {h: self._select_alpha(designs[h], y[h], groups, weights) for h in self.HEADS}
        # Only the target head is weighted; the source and delta heads keep their own error structure.
        self.models = {h: Ridge(alpha=self.alphas[h], random_state=0).fit(
            designs[h], y[h], sample_weight=weights if h == 'target' else None) for h in self.HEADS}
        return self

    def _design(self, z, shrunk):
        """Append the standardised shrunk source effect to the standardised fused features."""
        if not self.use_source_feature:
            return z
        return np.column_stack([z, (shrunk - self.shrunk_mean) / self.shrunk_std])

    def _select_alpha(self, design, y, groups, weights=None):
        """Pick alpha from the grid by inner grouped cross-validation on the fit rows, scoring on the same weights."""
        if len(self.alpha_grid) < 2:
            return self.alpha_grid[0] if self.alpha_grid else self.alpha
        folds = min(self.alpha_cv_folds, len(np.unique(groups)) if groups is not None else len(y))
        if folds < 2:
            return self.alpha
        splitter = (GroupKFold(n_splits=folds).split(design, y, groups) if groups is not None
                    else KFold(folds, shuffle=True, random_state=self.seed).split(design))
        splits = list(splitter)
        errors = []
        for alpha in self.alpha_grid:
            sse = 0.0
            for fit, valid in splits:
                model = Ridge(alpha=alpha, random_state=0).fit(
                    design[fit], y[fit], sample_weight=None if weights is None else weights[fit])
                residual = (y[valid] - model.predict(design[valid])) ** 2
                sse += float(residual.sum() if weights is None else (weights[valid] * residual).sum())
            errors.append(sse)
        return self.alpha_grid[int(np.argmin(errors))]

    def predict(self, representations, source_effect, source_se=None):
        """Predict all heads; a missing source_se falls back to the mean fit-row source error."""
        z = self.scaler.transform(np.asarray(representations, dtype=float))
        source = np.asarray(source_effect, dtype=float)
        se = np.full(len(source), self.mean_source_se) if source_se is None else np.asarray(source_se, dtype=float)
        shrunk = shrink_source(source, se, self.prior_mean, self.prior_var)
        raw = {'context_transfer': self.models['context_transfer'].predict(z),
               'source': self.models['source'].predict(z),
               'target': self.models['target'].predict(self._design(z, shrunk))}
        raw['primary_T_reconstructed'] = source + raw['context_transfer']
        return raw

    def predict_context_transfer(self, representations):
        """Delta prediction from the fused features alone, for candidates with no measured source."""
        z = self.scaler.transform(np.asarray(representations, dtype=float))
        return self.models['context_transfer'].predict(z)

    def target_coefficients(self):
        """Standardised coefficients of the target head: fused columns first, shrunk source last."""
        return self.models['target'].coef_.copy()


BASELINES = ('train_mean', 'source_copy', 'source_slope', 'source_shrunk')


def fold_baselines(frame_fit, frame_held):
    """No-feature target predictions for the held rows, fitted on the fit rows only."""
    s_fit = frame_fit.source_effect.to_numpy(dtype=float)
    t_fit = frame_fit.target_effect.to_numpy(dtype=float)
    se_fit = frame_fit.source_se.to_numpy(dtype=float)
    s_held = frame_held.source_effect.to_numpy(dtype=float)
    se_held = frame_held.source_se.to_numpy(dtype=float)
    mean, var = source_prior(s_fit, se_fit)
    shrunk_fit = shrink_source(s_fit, se_fit, mean, var)
    shrunk_held = shrink_source(s_held, se_held, mean, var)
    return {'train_mean': np.full(len(s_held), t_fit.mean()),
            'source_copy': s_held.copy(),
            'source_slope': np.polyval(np.polyfit(s_fit, t_fit, 1), s_held),
            'source_shrunk': np.polyval(np.polyfit(shrunk_fit, t_fit, 1), shrunk_held)}
