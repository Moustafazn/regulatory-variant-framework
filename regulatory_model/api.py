"""In-memory regulatory variant prediction and rough-soft uncertainty interface."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data_prep import validate_frame
from .evaluation import CLASSES
from .model import MechanisticFusion, RidgeHead, VariablePrecisionRoughSoft
from .training import (
    DEFAULT_CONFIG,
    DecimaGeneScorer,
    _avi_features_from_frame,
    load_config,
)


class RegulatoryVariantModel:
    """Fit on measured context pairs and predict unlabeled variants with class support."""

    def __init__(self, *, use_decima: bool = True, beta: float | None = None,
                 config_path=DEFAULT_CONFIG):
        self.settings = load_config(config_path)
        if beta is not None:
            if not 0.0 < beta < 1.0:
                raise ValueError('beta must be between 0 and 1.')
            self.settings = {
                **self.settings,
                'rough': {**self.settings['rough'], 'beta': float(beta)},
            }
        self.use_decima = bool(use_decima)
        default_blocks = tuple(self.settings['fusion'].get(
            'feature_blocks', ('avi', 'tracks', 'decima', 'mechanistic', 'dhs')))
        self._enabled_blocks = tuple(block for block in default_blocks if block != 'tracks')
        self._fusion = MechanisticFusion(self.settings, enabled_blocks=self._enabled_blocks)
        self._head: RidgeHead | None = None
        self._rough: VariablePrecisionRoughSoft | None = None

    @property
    def is_fitted(self) -> bool:
        return self._head is not None and self._rough is not None

    def fit(self, training_data: pd.DataFrame) -> "RegulatoryVariantModel":
        """Fit the prediction head and rough-soft support from labeled pairs."""
        if not isinstance(training_data, pd.DataFrame):
            raise TypeError('training_data must be a pandas DataFrame.')
        if 'effect_covariance' not in training_data.columns:
            raise ValueError(
                'Provide effect_covariance explicitly. Use 0 only when the '
                'source/target measurement errors can reasonably be treated as independent.'
            )
        frame = validate_frame(training_data)
        self._require_grch38(frame)
        pairs = frame[['source_context', 'target_context']].drop_duplicates()
        if len(pairs) != 1:
            raise ValueError(
                'Fit one RegulatoryVariantModel per source/target context pair; '
                f'found {len(pairs)} pairs.'
            )
        self.source_context = str(pairs.iloc[0].source_context)
        self.target_context = str(pairs.iloc[0].target_context)
        self.source_cell_state = self._configured_state(self.source_context)
        self.target_cell_state = self._configured_state(self.target_context)
        if self.source_cell_state == self.target_cell_state:
            raise ValueError('Source and target contexts must map to different configured cell states.')
        alpha, decima = self._score_features(frame)
        representation = self._fusion.transform(alpha, decima, frame)
        self._head = RidgeHead(self.settings).fit(representation, frame)
        self._rough = VariablePrecisionRoughSoft(self.settings).fit(representation, frame)
        return self

    def predict(self, variants: pd.DataFrame) -> pd.DataFrame:
        """Predict context effects from GRCh38 candidate-variant DataFrame rows.

        Required columns are ``chrom``, ``position``, ``genome_build``, ``ref``
        and ``alt``. Add ``source_effect`` to also receive a target-effect estimate.
        """
        if not self.is_fitted:
            raise RuntimeError('Call fit(training_data) before predict(variants).')
        frame = self._prepare_candidates(variants)
        for column, expected in (
            ('source_context', self.source_context),
            ('target_context', self.target_context),
        ):
            if column in frame and not frame[column].astype(str).eq(expected).all():
                raise ValueError(
                    f'Candidate {column} must match the fitted context pair ({expected}).'
                )
        alpha, decima = self._score_features(frame)
        representation = self._fusion.transform(alpha, decima, frame)

        assert self._head is not None and self._rough is not None
        context_delta = self._head.models['context_transfer'].predict(representation)
        rough = self._rough.predict(representation)
        support = rough['support']
        lower = rough['lower']
        upper = rough['upper']
        beta = float(self.settings['rough']['beta'])
        in_boundary = ((upper.sum(axis=1) - lower.sum(axis=1)) > 0) & (
            support.max(axis=1) < beta
        )

        result = pd.DataFrame({
            'example_id': frame['example_id'].to_numpy(),
            'variant_id': frame['variant_id'].to_numpy(),
            'source_context': self.source_context,
            'target_context': self.target_context,
            'predicted_context_transfer_delta': context_delta,
            'in_rough_soft_boundary': in_boundary,
            'mean_neighbor_distance': rough['distance'],
        }, index=frame.index)
        if 'source_effect' in frame:
            source = pd.to_numeric(frame['source_effect'], errors='raise').to_numpy(float)
            if not np.isfinite(source).all():
                raise ValueError('source_effect must contain finite numeric values.')
            result['predicted_target_effect'] = source + context_delta

        for index, class_name in enumerate(CLASSES):
            result[f'support_{class_name}'] = support[:, index]
            result[f'lower_{class_name}'] = lower[:, index]
            result[f'upper_{class_name}'] = upper[:, index]
        return result

    def _score_features(self, frame: pd.DataFrame):
        # The static archive scores are supplied explicitly; only Decima is run here.
        scoring_frame = frame.copy()
        variant_keys = [self._variant_key(row.chrom, row.position, row.ref, row.alt)
                        for row in scoring_frame.itertuples(index=False)]
        scoring_frame['variant_id'] = variant_keys
        scoring_frame['example_id'] = [
            'variant-' + hashlib.sha256(key.encode()).hexdigest() for key in variant_keys
        ]
        alpha = _avi_features_from_frame(scoring_frame)
        if not self.use_decima:
            decima = {
                state: np.zeros((len(frame), 1), dtype=np.float32)
                for state in self.settings['cell_states']
            }
        else:
            decima, _ = DecimaGeneScorer(self.settings).score(scoring_frame)
        return alpha, decima

    @staticmethod
    def _variant_key(chrom, position, ref, alt) -> str:
        chromosome = str(chrom).strip().removeprefix('chr')
        return f'{chromosome}:{int(position)}:{str(ref).upper()}>{str(alt).upper()}'

    def _configured_state(self, context: str) -> str:
        normalize = lambda value: ''.join(ch.lower() for ch in str(value) if ch.isalnum())
        requested = normalize(context)
        matches = []
        for name, state in self.settings['cell_states'].items():
            aliases = [name, state.get('label', ''), *state.get('context_aliases', [])]
            if requested in {normalize(alias) for alias in aliases}:
                matches.append(name)
        if len(matches) != 1:
            available = sorted(self.settings['cell_states'])
            raise ValueError(
                f'Context {context!r} must match one configured cell state or its '
                f'context_aliases. Configured states: {available}.'
            )
        return matches[0]

    @classmethod
    def _prepare_candidates(cls, variants: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(variants, pd.DataFrame):
            raise TypeError('variants must be a pandas DataFrame.')
        required = {'chrom', 'position', 'genome_build', 'ref', 'alt'}
        absent = required - set(variants.columns)
        if absent:
            raise ValueError(f'Missing candidate-variant columns: {sorted(absent)}')
        measured_targets = {'target_effect', 'target_se'} & set(variants.columns)
        if measured_targets:
            raise ValueError(
                'predict() accepts unlabeled variants; remove measured target columns '
                f'{sorted(measured_targets)} and keep them separately for evaluation.'
            )

        frame = variants.copy()
        frame['chrom'] = frame['chrom'].astype(str).str.strip().str.removeprefix('chr')
        frame['position'] = pd.to_numeric(frame['position'], errors='raise')
        frame['ref'] = frame['ref'].astype(str).str.strip().str.upper()
        frame['alt'] = frame['alt'].astype(str).str.strip().str.upper()
        if frame.empty:
            raise ValueError('variants must contain at least one candidate.')
        if frame[['chrom', 'position', 'ref', 'alt']].isna().any().any():
            raise ValueError('Candidate coordinates and alleles cannot be missing.')
        invalid_chrom = ~frame['chrom'].str.fullmatch(r'(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)')
        invalid_position = (frame['position'] <= 0) | (frame['position'] % 1 != 0)
        invalid_allele = ~frame['ref'].str.fullmatch('[ACGT]+') | ~frame['alt'].str.fullmatch('[ACGT]+')
        if invalid_chrom.any() or invalid_position.any() or invalid_allele.any():
            raise ValueError('Candidates need valid GRCh38 chromosome, position, and DNA alleles.')
        if frame['ref'].eq(frame['alt']).any():
            raise ValueError('Reference and alternate alleles must differ.')
        builds = set(frame['genome_build'].dropna().astype(str))
        if builds != {'GRCh38'} or frame['genome_build'].isna().any():
            raise ValueError('Local AVI and Decima candidate coordinates must be GRCh38.')

        variant_keys = [cls._variant_key(row.chrom, row.position, row.ref, row.alt)
                        for row in frame.itertuples(index=False)]
        if 'variant_id' not in frame:
            frame['variant_id'] = variant_keys
        else:
            frame['variant_id'] = frame['variant_id'].fillna(pd.Series(variant_keys, index=frame.index))
        if 'example_id' not in frame:
            frame['example_id'] = [f'candidate-{key}' for key in variant_keys]
        else:
            frame['example_id'] = frame['example_id'].fillna(
                pd.Series([f'candidate-{key}' for key in variant_keys], index=frame.index)
            )
        frame['example_id'] = frame['example_id'].astype(str)
        frame['variant_id'] = frame['variant_id'].astype(str)
        return frame

    @staticmethod
    def _require_grch38(frame: pd.DataFrame) -> None:
        builds = set(frame['genome_build'].astype(str))
        if builds != {'GRCh38'}:
            raise ValueError(
                'Local AVI annotations and Decima predictions use GRCh38 coordinates. '
                f'Lift and verify training variants before fit(); found builds: {sorted(builds)}.'
            )


# Retain existing imports and serialized-model lookup paths.
NeurojicoModel = RegulatoryVariantModel
