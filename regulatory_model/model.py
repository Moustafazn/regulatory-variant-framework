"""Ziarko variable-precision rough-soft set (box 3) and mechanistic feature fusion.

The AI predictor (box 2 of the Neurojico figure) is the frozen AlphaGenome + Decima
score pipeline implemented in ``training.py``. This module only contains the
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


class MechanisticFusion:
    """Concatenate AlphaGenome track deltas, Decima gene deltas, Ho mechanistic predictors, DHS."""

    def __init__(self, settings):
        f = settings['fusion']
        self.alphagenome_outputs = tuple(f['alphagenome_track_outputs'])
        self.mechanistic_fields = tuple(f['ho_mechanistic_fields'])
        self.dhs_fields = tuple(f['dhs_fields'])
        self.cell_states = tuple(settings['cell_states'].keys())

    def transform(self, alphagenome_scores, decima_scores, frame):
        # Stack: [AlphaGenome deltas per cell_state x output] + [Decima deltas per cell_state] + [mechanistic] + [DHS].
        blocks = []
        column_names = []
        for cell_state in self.cell_states:
            for output in self.alphagenome_outputs:
                key = f'{cell_state}:{output}'
                vec = np.asarray(alphagenome_scores.get(key), dtype=float).reshape(len(frame), -1)
                blocks.append(vec)
                column_names.extend([f'ag:{key}:{i}' for i in range(vec.shape[1])])
        for cell_state in self.cell_states:
            vec = np.asarray(decima_scores.get(cell_state), dtype=float).reshape(len(frame), -1)
            blocks.append(vec)
            column_names.extend([f'decima:{cell_state}:{i}' for i in range(vec.shape[1])])
        mech = frame.reindex(columns=self.mechanistic_fields).to_numpy(dtype=float, na_value=0.0)
        blocks.append(mech)
        column_names.extend([f'ho:{c}' for c in self.mechanistic_fields])
        dhs = frame.reindex(columns=self.dhs_fields).to_numpy(dtype=float, na_value=0.0)
        blocks.append(dhs)
        column_names.extend([f'dhs:{c}' for c in self.dhs_fields])
        matrix = np.concatenate(blocks, axis=1)
        self.column_names = tuple(column_names)
        return matrix


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
