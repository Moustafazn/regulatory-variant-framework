"""Ziarko variable-precision rough-soft set (box 3) and mechanistic feature fusion.

The AI predictor (prediction stage of the framework) combines local AlphaGenome AVI
annotations with Decima scores. This module contains the
uncertainty wrapper (``VariablePrecisionRoughSoft``), a Ridge-based scalar head that
converts fused representations into a scalar effect prediction, and ``MechanisticFusion``
which concatenates the per-variant representation used by both the head and the VPRS.
"""
import numpy as np
from sklearn.linear_model import Ridge
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
    cell_states = tuple(settings['cell_states'].keys())
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


class RidgeHead:
    """Ridge heads on the fused representation. Primary endpoint is the context-transfer delta
    (target_effect - source_effect); source and target heads are kept as diagnostics only."""

    HEADS = ('context_transfer', 'source', 'target')

    def __init__(self, settings):
        self.alpha = float(settings['head']['ridge_alpha'])

    def fit(self, representations, frame):
        source = frame.source_effect.to_numpy(dtype=float)
        target = frame.target_effect.to_numpy(dtype=float)
        y = {'context_transfer': target - source, 'source': source, 'target': target}
        self.models = {h: Ridge(alpha=self.alpha, random_state=0).fit(representations, y[h])
                       for h in self.HEADS}
        return self

    def predict(self, representations, source_effect):
        raw = {h: self.models[h].predict(representations) for h in self.HEADS}
        raw['primary_T_reconstructed'] = np.asarray(source_effect, dtype=float) + raw['context_transfer']
        return raw
