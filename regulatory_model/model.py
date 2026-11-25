"""Source-assisted effect regressors and benchmarked rough-set membership scores."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .evaluation import CLASSES, effect_classes
from .benchmarking import QuantileDiscretizer, locus_weights

def validate_beta(value):
    """beta = 1 - canonical error: 0.5 < beta <= 1 preserves lower <= upper, including ties."""
    beta = float(value)
    if not 0.5 < beta <= 1.0:
        raise ValueError(
            f'rough.beta must satisfy 0.5 < beta <= 1 (got {beta}). beta is stored as '
            f'1 - canonical_ziarko_error, so the canonical error is 1 - beta. At or below 0.5 the '
            f'predicates lower = support >= beta and upper = support > 1 - beta admit rows '
            f'into the lower approximation that are absent from the upper one.')
    return beta


class VariablePrecisionRoughSoft:
    """Variable-precision lower/upper approximations of distance-weighted class support."""

    def __init__(self, settings):
        self.cfg = settings['rough']
        self.margin = float(settings['margin'])
        self.n_classes = len(CLASSES)
        self.draws = int(settings['rough'].get('membership_draws', 256))
        self.seed = int(settings.get('seed', 42))
        self.beta = validate_beta(self.cfg['beta'])
        self.membership = str(self.cfg.get('membership', 'crisp')).lower()
        if self.membership not in ('crisp', 'soft'):
            raise ValueError(
                f"rough.membership must be 'crisp' or 'soft', got {self.membership!r}.")

    def fit(self, representations, frame):
        representations = np.asarray(representations, dtype=float)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(representations)
        representations = self.imputer.transform(representations)
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
        soft = np.column_stack([(labels == c).mean(axis=1) for c in range(self.n_classes)])
        if self.membership == 'soft':
            return soft
        crisp = np.zeros_like(soft)
        crisp[np.arange(len(soft)), soft.argmax(axis=1)] = 1.0
        return crisp

    def predict(self, representations):
        # Return per-class {support, lower, upper, mean neighbour distance}.
        representations = np.asarray(representations, dtype=float)
        representations = self.imputer.transform(representations)
        distances, indices = self.neighbors.kneighbors(self.scaler.transform(representations))
        values = self.memberships[indices]
        temperature = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)
        weights = np.exp(-distances / temperature)
        support = (values * weights[..., None]).sum(1) / weights.sum(1)[:, None]
        beta = self.beta
        lower = (support >= beta).astype(float)
        upper = (support > (1.0 - beta)).astype(float)
        return {'support': support, 'lower': lower, 'upper': upper,
                'distance': distances.mean(1), 'neighbor_indices': indices}


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


class EffectRegressor:
    """Fit preprocessing and source shrinkage exclusively on training observations."""
    def __init__(self, name='source_shrunk', sequence=(), context=(), seed=20261122):
        self.name, self.sequence, self.context, self.seed = name, list(sequence), list(context), seed

    def design(self, frame):
        source = frame.source_effect.to_numpy(float)
        se = frame.source_se.to_numpy(float)
        if not np.isfinite(source).all() or not np.isfinite(se).all() or (se <= 0).any():
            raise ValueError('Finite source effects and positive source SE required.')
        x = np.column_stack([source, np.log(se), shrink_source(source, se, *self.prior)])
        family = self.name.split(':')[0]
        if family in ('source', 'source_calibrated', 'source_shrunk', 'source_copy', 'train_mean'):
            return x
        columns = self.sequence + (self.context if family == 'source_sequence_context' else [])
        h = frame[columns].to_numpy(float)
        return h if family == 'sequence' else np.column_stack([x, h])

    def fit(self, frame):
        self.prior = source_prior(frame.source_effect, frame.source_se)
        y = frame.target_effect.to_numpy(float)
        w = locus_weights(frame.split_group); w /= w.mean()
        self.mean = float(np.average(y, weights=w))
        if self.name in ('train_mean', 'source_copy'):
            return self
        x = self.design(frame)
        if self.name in ('source_calibrated', 'source_shrunk'):
            self.column = 0 if self.name == 'source_calibrated' else 2
            self.model = LinearRegression().fit(x[:, [self.column]], y, sample_weight=w)
            return self
        kind = self.name.split(':')[1]
        if kind == 'ridge':
            self.model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                                       StandardScaler(), Ridge(alpha=10))
            self.model.fit(x, y, ridge__sample_weight=w)
        elif kind in ('gbm', 'binned_gbm'):
            model = HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_leaf_nodes=7,
                min_samples_leaf=30, l2_regularization=10, early_stopping=False, random_state=self.seed)
            if kind == 'binned_gbm':
                self.model = make_pipeline(QuantileDiscretizer(3), model)
                self.model.fit(x, y, histgradientboostingregressor__sample_weight=w)
            else:
                self.model = model.fit(x, y, sample_weight=w)
        else:
            raise ValueError(f'Unknown estimator: {kind}')
        return self

    def predict(self, frame):
        if not hasattr(self, 'prior'):
            raise RuntimeError('Fit the effect regressor before prediction.')
        if self.name == 'source_copy':
            return frame.source_effect.to_numpy(float)
        if self.name == 'train_mean':
            return np.full(len(frame), self.mean)
        x = self.design(frame)
        if self.name in ('source_calibrated', 'source_shrunk'):
            x = x[:, [self.column]]
        return self.model.predict(x)
