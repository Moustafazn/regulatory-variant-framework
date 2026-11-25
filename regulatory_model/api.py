"""In-memory source-assisted estimates; target measurements are training-only."""
import numpy as np
from .data_prep import validate_frame, MECHANISTIC_FEATURES, BASE_FEATURES, SEQUENCE_FEATURES
from .evaluation import locus_groups
from .model import EffectRegressor


class RegulatoryVariantModel:
    """An explicit estimator interface; benchmark selection occurs in the study pipeline."""
    def __init__(self, name='source_shrunk', *, sequence=(), context=(), seed=20261122):
        allowed = set(BASE_FEATURES+SEQUENCE_FEATURES+MECHANISTIC_FEATURES)
        if not set(sequence) <= allowed or not all(c in ('access_source','access_target','access_delta') for c in context):
            raise ValueError('Predictors must be on the outcome-independent feature whitelist.')
        self.estimator = EffectRegressor(name, sequence, context, seed)
        self.is_fitted = False

    def fit(self, training_data):
        if 'effect_covariance' not in training_data:
            raise ValueError('Declare effect_covariance explicitly, including a zero working assumption.')
        frame = validate_frame(training_data)
        if not frame.genome_build.eq('GRCh38').all():
            raise ValueError('Verified GRCh38 coordinates required.')
        pairs = frame[['source_context','target_context']].drop_duplicates()
        if len(pairs) != 1:
            raise ValueError('Fit one source/target context pair at a time.')
        self.context_pair = tuple(pairs.iloc[0])
        if 'split_group' not in frame:
            frame['split_group'] = locus_groups(frame)
        self.estimator.fit(frame)
        self.is_fitted = True
        return self

    def predict(self, candidates):
        if not self.is_fitted:
            raise RuntimeError('Call fit before prediction.')
        if any(c in candidates for c in ('target_effect','target_se','effect_covariance')) or any(c.startswith('rep_') for c in candidates):
            raise ValueError('Provide unlabeled candidates; target measurements are forbidden.')
        if not candidates.genome_build.eq('GRCh38').all():
            raise ValueError('Verified GRCh38 coordinates required.')
        for col, value in zip(('source_context','target_context'), self.context_pair):
            if not candidates[col].eq(value).all():
                raise ValueError('Candidate contexts differ from fitted contexts.')
        prediction = self.estimator.predict(candidates)
        out = candidates[['example_id']].copy()
        out['predicted_target_effect'] = prediction
        out['predicted_context_transfer_delta'] = prediction-candidates.source_effect.to_numpy(float)
        return out


NeurojicoModel = RegulatoryVariantModel
