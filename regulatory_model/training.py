"""Neurojico pipeline: AlphaGenome + Decima scoring, rough-soft wrapping,
boundary review export and four-method evaluation (AI alone / Rough-soft AI /
Expert alone / Neurojico).

This is the single pipeline. Everything upstream of this module is dataset
preparation (``data_prep.py``). Everything inside this module runs the five
figure stages:
  1. Available evidence  -> fused representation (alphagenome + decima + Ho mechanistic + DHS)
  2. AI predictor        -> Ridge head per cell state
  3. Rough-soft sets     -> VariablePrecisionRoughSoft (lower / upper / boundary)
  4. Expert review       -> boundary cards -> expert_corrections.json
  5. Prediction          -> maintained / changed / unresolved + 4-method evaluation
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter

from .data_prep import (
    PROJECT_ROOT, DHS, MECHANISTIC_FEATURES, OUTCOMES,
    prepare_dataset, merge_ho_mechanistic_features, validate_frame,
)
from .evaluation import (
    CLASSES, effect_classes, experimental_labels, locus_groups,
    group_rmse, group_rmse_bootstrap, selective_curve, paired_locus_bootstrap,
    grouped_folds, inner_split,
)
from .model import VariablePrecisionRoughSoft, MechanisticFusion, RidgeHead
from .reporting import (
    write_json, sha256, sha256_bytes, sha256_directory,
    platform_metadata, prepare_output, log_event,
)

DEFAULT_CONFIG = Path(__file__).parent / 'config.json'


# ---------------------------------------------------------------------------
# Review record schemas.
# ---------------------------------------------------------------------------

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Https = Annotated[str, StringConstraints(pattern=r'^https://.+')]


class ExpertAction:
    accept = 'accept'
    challenge = 'challenge'
    defer = 'defer'


class ExpertCorrection(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    example_id: NonEmpty
    field: NonEmpty  # DHS column, or 'expert_action'
    value: Literal[0, 1, 'accept', 'challenge', 'defer'] | None
    reason: NonEmpty
    evidence_url: Https


CORRECTIONS = TypeAdapter(list[ExpertCorrection])


# ---------------------------------------------------------------------------
# Config and option objects.
# ---------------------------------------------------------------------------


def load_config(path=DEFAULT_CONFIG):
    data = json.loads(Path(path).read_text())
    if 'neurojico' not in data:
        raise ValueError('config.json must contain a neurojico block')
    return data['neurojico']


@dataclass(frozen=True)
class NeurojicoOptions:
    output: Path
    data_root: Path = PROJECT_ROOT / 'data' / 'regulatory_audit'
    config_path: Path = DEFAULT_CONFIG
    expert_corrections: Path | None = None
    skip_decima: bool = False


def default_output(name='neurojico'):
    return PROJECT_ROOT / 'output' / 'experiments' / name


# ---------------------------------------------------------------------------
# AlphaGenome scorer (step 2a).
# ---------------------------------------------------------------------------


class AlphaGenomeVariantScorer:
    """Score variants via the AlphaGenome API, caching every response."""

    def __init__(self, settings, cache_dir):
        self.settings = settings['backbone']
        self.cell_states = settings['cell_states']
        self.outputs = tuple(settings['fusion']['alphagenome_track_outputs'])
        self.verify_ontology = bool(settings.get('verify_ontology_terms_on_first_call', True))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._api_model_version = None
        self._ontology_verified = False

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from alphagenome.data import genome
            from alphagenome.models import dna_client
        except ImportError as exc:
            raise RuntimeError(
                'alphagenome client not installed. Install with `pip install alphagenome`.'
            ) from exc
        key = os.environ.get(self.settings['api_env_var'])
        if not key:
            raise RuntimeError(
                f'Missing API key: set the {self.settings["api_env_var"]} environment variable. '
                'Register for a free non-commercial key at the AlphaGenome portal.'
            )
        self._dna_client = dna_client
        self._genome = genome
        self._client = dna_client.create(key)
        self._api_model_version = str(getattr(self._client, 'model_version', 'unknown'))

    def _verify_ontology_terms(self, prediction):
        # Record which specific CURIE matched which output, hit kind (direct vs proxy), and track counts.
        self.ontology_match_report = {}
        for state_name, state in self.cell_states.items():
            direct = set(state.get('direct_hit_curies', state['ontology_curies']))
            proxy = set(state.get('proxy_hit_curies', []))
            all_requested = set(state['ontology_curies'])
            per_output = {}
            matched_curies_overall = set()
            for output_name in self.outputs:
                try:
                    md = getattr(prediction.reference, output_name.lower()).metadata
                except AttributeError:
                    per_output[output_name] = {'tracks_total': 0, 'matched_curies': [], 'hit_kind': 'none'}
                    continue
                track_curies = set(md['ontology_curie'].astype(str).tolist())
                matched = sorted(all_requested & track_curies)
                matched_curies_overall |= set(matched)
                tracks_for_match = int(md['ontology_curie'].astype(str).isin(matched).sum()) if matched else 0
                hit_kind = ('direct' if any(c in direct for c in matched)
                            else 'proxy' if any(c in proxy for c in matched)
                            else 'none')
                per_output[output_name] = {'tracks_total': int(len(track_curies)),
                                           'matched_curies': matched,
                                           'tracks_matched': tracks_for_match,
                                           'hit_kind': hit_kind}
            self.ontology_match_report[state_name] = {
                'requested_curies': sorted(all_requested),
                'declared_direct_hit_curies': sorted(direct),
                'declared_proxy_hit_curies': sorted(proxy),
                'matched_curies_any_output': sorted(matched_curies_overall),
                'overall_hit_kind': ('direct' if matched_curies_overall & direct
                                     else 'proxy' if matched_curies_overall & proxy
                                     else 'none'),
                'per_output': per_output,
            }
        unmatched = [state for state, rep in self.ontology_match_report.items()
                     if rep['overall_hit_kind'] == 'none']
        if unmatched:
            raise RuntimeError(
                f'No configured ontology_curies matched any AlphaGenome track for cell states: '
                f'{unmatched}. Match report: {self.ontology_match_report}. '
                'Edit config.json.neurojico.cell_states to use CURIEs that AlphaGenome exposes.')

    def _cache_path(self, example_id):
        safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(example_id))
        return self.cache_dir / f'{safe}.npz'

    def _score_one(self, variant_row):
        example_id = str(variant_row.example_id)
        path = self._cache_path(example_id)
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return {k: cached[k] for k in cached.files}
        self._ensure_client()
        chrom = str(variant_row.chrom)
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        variant = self._genome.Variant(
            chromosome=chrom, position=int(variant_row.position),
            reference_bases=str(variant_row.ref), alternate_bases=str(variant_row.alt),
        )
        interval = variant.reference_interval.resize(int(self.settings['sequence_length_bp']))
        requested_outputs = [getattr(self._dna_client.OutputType, name) for name in self.outputs]
        organism = getattr(self._dna_client.Organism, self.settings['organism'])
        prediction = self._client.predict_variant(
            interval=interval, variant=variant,
            ontology_terms=sorted({c for state in self.cell_states.values()
                                   for c in state['ontology_curies']}),
            requested_outputs=requested_outputs, organism=organism,
        )
        if self.verify_ontology and not self._ontology_verified:
            self._verify_ontology_terms(prediction)
            self._ontology_verified = True
        result = {}
        for state_name, state in self.cell_states.items():
            curies = set(state['ontology_curies'])
            for output_name in self.outputs:
                key = f'{state_name}:{output_name}'
                try:
                    ref_output = getattr(prediction.reference, output_name.lower())
                    alt_output = getattr(prediction.alternate, output_name.lower())
                except AttributeError:
                    result[key] = np.zeros(1, dtype=np.float32)
                    continue
                metadata = ref_output.metadata
                if 'ontology_curie' not in metadata.columns:
                    result[key] = np.zeros(1, dtype=np.float32)
                    continue
                mask = metadata['ontology_curie'].isin(curies).to_numpy()
                if not mask.any():
                    result[key] = np.zeros(1, dtype=np.float32)
                    continue
                ref_mean = np.asarray(ref_output.values)[:, mask].mean(axis=0)
                alt_mean = np.asarray(alt_output.values)[:, mask].mean(axis=0)
                delta = (alt_mean - ref_mean).astype(np.float32)
                result[key] = delta.mean(keepdims=True)
        np.savez_compressed(path, **result)
        provenance = {
            'example_id': example_id,
            'variant': {'chrom': chrom, 'position': int(variant_row.position),
                        'ref': str(variant_row.ref), 'alt': str(variant_row.alt)},
            'cache_sha256': sha256(path),
            'api_model_version': self._api_model_version,
            'ontology_curies_requested': sorted({c for state in self.cell_states.values()
                                                 for c in state['ontology_curies']}),
            'requested_outputs': list(self.outputs),
            'sequence_length_bp': int(self.settings['sequence_length_bp']),
        }
        (self.cache_dir / f'{path.stem}.provenance.json').write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + '\n')
        return result

    def score(self, frame):
        rows = [self._score_one(r) for r in frame.itertuples(index=False)]
        keys = sorted({k for r in rows for k in r})
        stacked = {}
        for key in keys:
            widths = [r[key].shape[-1] for r in rows if key in r]
            dim = int(max(widths)) if widths else 1
            matrix = np.zeros((len(frame), dim), dtype=np.float32)
            for i, row in enumerate(rows):
                if key in row:
                    v = np.asarray(row[key], dtype=np.float32).ravel()
                    matrix[i, :v.size] = v
            stacked[key] = matrix
        return stacked, {'api_model_version': self._api_model_version,
                         'variants_scored': int(len(frame)),
                         'cache_dir': str(self.cache_dir),
                         'ontology_match_report': getattr(self, 'ontology_match_report', None)}


# ---------------------------------------------------------------------------
# Decima scorer (step 2b).
# ---------------------------------------------------------------------------


class DecimaGeneScorer:
    """Score variants with the Decima model via its real public API."""

    def __init__(self, settings, cache_dir):
        self.settings = settings['secondary']
        self.cell_states = settings['cell_states']
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._task_query = {
            'primary_T': "cell_type == 'CD4-positive, alpha-beta T cell'",
            'jurkat': "cell_type == 'Jurkat'",
        }

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from decima.hub import load_decima_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'decima package not installed. Install with `pip install decima`.'
            ) from exc
        self._model = load_decima_model(
            model=self.settings.get('model_identifier', 'ensemble'),
            device=self.settings.get('device', 'cpu'))

    def score(self, frame):
        try:
            from decima import predict_variant_effect  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'decima package not installed. Install with `pip install decima`.'
            ) from exc
        self._ensure_model()
        df_variant = pd.DataFrame({
            'chrom': ['chr' + str(c).lstrip('chr') for c in frame.chrom],
            'pos':   frame.position.astype(int).to_numpy(),
            'ref':   frame.ref.astype(str).to_numpy(),
            'alt':   frame.alt.astype(str).to_numpy(),
            'variant_id': frame.variant_id.astype(str).to_numpy(),
        })
        per_state = {}
        for state_name in self.cell_states:
            cache_path = self.cache_dir / f'{state_name}.parquet'
            if not cache_path.exists():
                predict_variant_effect(
                    df_variant,
                    output_pq=str(cache_path),
                    tasks=self._task_query.get(state_name),
                    model=self.settings.get('model_identifier', 'ensemble'),
                    device=self.settings.get('device', 'cpu'))
            predictions = pd.read_parquet(cache_path)
            per_variant_mean = (predictions.groupby('variant_id')['effect']
                                .mean()
                                .reindex(df_variant.variant_id)
                                .fillna(0.0)
                                .to_numpy(dtype=np.float32))
            per_state[state_name] = per_variant_mean.reshape(-1, 1)
        return per_state, {
            'model_identifier': self.settings.get('model_identifier', 'ensemble'),
            'variants_scored': int(len(frame)),
            'task_queries': self._task_query,
        }


# ---------------------------------------------------------------------------
# Expert corrections (step 4).
# ---------------------------------------------------------------------------


def _apply_corrections(frame, corrections_path, fit_indices):
    if not corrections_path:
        return frame.copy(), []
    corrections = CORRECTIONS.validate_json(Path(corrections_path).read_text())
    allowed_dhs = set(DHS)
    allowed_actions = {ExpertAction.accept, ExpertAction.challenge, ExpertAction.defer}
    fit_ids = set(frame.iloc[fit_indices].example_id)
    updated = frame.copy()
    if 'expert_action' not in updated.columns:
        updated['expert_action'] = ''
    applied = []
    for c in corrections:
        if c.example_id not in fit_ids:
            raise ValueError(f'Correction for {c.example_id} is outside the fitting fold')
        if c.field in allowed_dhs:
            if c.value not in (0, 1, None):
                raise ValueError(f'DHS correction value must be 0/1/None, got {c.value!r}')
            updated.loc[updated.example_id == c.example_id, c.field] = (
                np.nan if c.value is None else float(c.value))
        elif c.field == 'expert_action':
            if c.value not in allowed_actions:
                raise ValueError(f'expert_action must be accept/challenge/defer, got {c.value!r}')
            updated.loc[updated.example_id == c.example_id, 'expert_action'] = c.value
        else:
            raise ValueError(f'Correction field {c.field!r} is not DHS or expert_action')
        applied.append(c.model_dump(mode='json'))
    return updated, applied


# ---------------------------------------------------------------------------
# Boundary review export (step 4, no target leakage).
# ---------------------------------------------------------------------------


def _as_scalar(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def _collect_review_cards(frame, rough_result, scores, fused, settings, fold):
    support = rough_result['support']
    lower = rough_result['lower']
    upper = rough_result['upper']
    beta = float(settings['rough']['beta'])
    boundary_mask = ((upper.sum(axis=1) - lower.sum(axis=1)) > 0) & (support.max(axis=1) < beta)
    cards = []
    frame_reset = frame.reset_index(drop=True)
    for i in range(len(frame_reset)):
        if not boundary_mask[i]:
            continue
        row = frame_reset.iloc[i]
        card = {
            'fold': int(fold),
            'example_id': str(row.example_id),
            'variant_id': str(row.variant_id),
            'chrom': str(row.chrom),
            'position': int(row.position),
            'ref': str(row.ref),
            'alt': str(row.alt),
            'dhs': {c: _as_scalar(row.get(c)) for c in DHS},
            'ho_mechanistic': {c: _as_scalar(row.get(c)) for c in MECHANISTIC_FEATURES},
            'ai_predictions': {name: float(scores[name][i]) for name in scores},
            'source_minus_target_contrast': float(
                scores.get('source', np.zeros(len(frame_reset)))[i]
                - scores.get('target', np.zeros(len(frame_reset)))[i]),
            'predicted_context_transfer_delta': float(
                scores.get('context_transfer', np.zeros(len(frame_reset)))[i]),
            'vprs': {
                'support': [float(s) for s in support[i]],
                'lower': [float(l) for l in lower[i]],
                'upper': [float(u) for u in upper[i]],
                'classes': CLASSES,
            },
            'editable_fields': list(DHS) + ['expert_action'],
            'allowed_actions': [ExpertAction.accept, ExpertAction.challenge, ExpertAction.defer],
            'required_schema': ['example_id', 'field', 'value', 'reason', 'evidence_url'],
        }
        cards.append(card)
    return cards


# ---------------------------------------------------------------------------
# Fitting helpers.
# ---------------------------------------------------------------------------


def _fit_rough_soft(settings, fused_fit, frame_fit):
    return VariablePrecisionRoughSoft(settings).fit(fused_fit, frame_fit)


def _group_conformal_radius(residuals, group_ids, alpha):
    maxima = pd.Series(residuals).groupby(group_ids).max().to_numpy()
    rank = int(np.ceil((len(maxima) + 1) * (1 - alpha)))
    return float(np.sort(maxima)[rank - 1]) if rank <= len(maxima) else float('inf')


def _save_fold(output, fold, head, rough):
    joblib.dump({'head': head, 'rough': rough},
                output / f'fold_{fold:02d}.joblib')


def fit_fold_head(settings, fused_fit, frame_fit):
    return RidgeHead(settings).fit(fused_fit, frame_fit)


_SEMANTICS = {
    'keep_ai':            lambda ai, source, const: ai,
    'use_source_effect':  lambda ai, source, const: source,
    'constant':           lambda ai, source, const: np.full_like(ai, const, dtype=float),
}


def _resolve_action_value(action_cfg, ai, source):
    # Resolve one action/rejection config entry into a full-length array of predictions.
    return _SEMANTICS[action_cfg['semantics']](ai, source, action_cfg.get('constant_value'))


_EXPERT_ALONE_RULES = {
    'random':                   lambda ai, frame, seed: np.random.default_rng(int(seed)).random(len(ai)),
    'largest_abs_ai_prediction': lambda ai, frame, seed: np.abs(np.asarray(ai, dtype=float)),
    'largest_source_se':         lambda ai, frame, seed: frame.source_se.to_numpy(dtype=float),
}


def _expert_alone_selection_mask(rule, budget, ai, frame, seed):
    # Pick ``budget`` indices to review under the Expert-alone rule (not VPRS boundary).
    scores = _EXPERT_ALONE_RULES[rule](ai, frame, seed)
    n = len(ai)
    budget = int(min(budget, n))
    mask = np.zeros(n, dtype=bool)
    if budget > 0:
        mask[np.argsort(-scores, kind='stable')[:budget]] = True
    return mask


def _rough_soft_ai_prediction(settings, head_prediction, held_rough, held_frame):
    # Rough-soft AI output: AI prediction where VPRS accepts, else the declared rejected-variant value.
    ai = np.asarray(head_prediction, dtype=float)
    source = held_frame.source_effect.to_numpy(dtype=float)
    accepted = held_rough['lower'].sum(axis=1) >= 1
    rejected_value = _resolve_action_value(
        settings['rough_soft_ai_rejected_variant_value'], ai, source)
    return np.where(accepted, ai, rejected_value)


# ---------------------------------------------------------------------------
# Four-method evaluation (step 5).
# ---------------------------------------------------------------------------


def _safe_r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.var(y))
    if denom <= 0:
        return None
    return float(1.0 - np.mean((y - pred) ** 2) / denom)


def _delta_frame(frame):
    # A copy of ``frame`` whose ``target_effect`` IS the context-transfer delta, for RMSE scoring.
    delta = frame.copy()
    delta['target_effect'] = frame.target_effect.to_numpy(dtype=float) - frame.source_effect.to_numpy(dtype=float)
    return delta


def _method_metrics(frame, prediction, config):
    pred = np.asarray(prediction, dtype=float)
    rmse, per_group = group_rmse(frame, pred)
    lo, hi = group_rmse_bootstrap(per_group, draws=int(config['bootstrap']), seed=int(config['seed']))
    return {'rmse': float(rmse), 'equal_locus_rmse_ci95': [lo, hi],
            'r2': _safe_r2(frame.target_effect.to_numpy(dtype=float), pred)}


def evaluate_four_methods(frame, ai_prediction, rough_prediction, expert_prediction,
                          neurojico_prediction, boundary_mask, expert_alone_mask,
                          review_time_seconds, config, delta_predictions=None):
    """Report every method on both the primary-T and context-transfer-delta endpoints. """
    methods = {
        'ai_alone': {'prediction': ai_prediction, 'reviewed': 0,
                     'description': 'Ridge head prediction, no VPRS and no expert.'},
        'rough_soft_ai': {'prediction': rough_prediction, 'reviewed': 0,
                          'description': 'Ridge head with VPRS lower-bound gating.'},
        'expert_alone': {'prediction': expert_prediction,
                         'reviewed': int(expert_alone_mask.sum()),
                         'description': (
                             'Expert replaces AI on B variants selected by '
                             f'{config["evaluation"]["expert_alone_selection_rule"]!r}; '
                             'AI elsewhere. Matched reviewed-variant budget vs Neurojico.')},
        'neurojico': {'prediction': neurojico_prediction,
                      'reviewed': int(boundary_mask.sum()),
                      'description': 'Expert replaces AI on VPRS boundary variants; AI elsewhere.'},
    }
    delta_frame = _delta_frame(frame)
    results = {}
    for name, data in methods.items():
        row = {
            'description': data['description'],
            'reviewed_variants': data['reviewed'],
            'review_time_seconds_assumed': float(review_time_seconds * data['reviewed']),
            'primary_T_endpoint': _method_metrics(frame, data['prediction'], config),
        }
        if delta_predictions is not None and name in delta_predictions:
            row['context_transfer_endpoint'] = _method_metrics(
                delta_frame, delta_predictions[name], config)
        results[name] = row
    return results


_SELECTOR_SEED = 0  # overwritten per-run by run_neurojico from config['seed']


def _selector_scores(rough_result, head_scores, ensemble_disagreement=None,
                     adaptive_conformal_width=None):
    # Per-variant confidence scores. Higher = more confident = retained first at low coverage.
    support = rough_result['support']
    n = support.shape[0]
    # VPRS top-vs-runner-up margin: how much the strongest class dominates the second-strongest.
    # This is distinct from ordinary_neighbor_support (= max) because it penalises multimodal
    # neighbourhoods where two classes are near-tied, which is the regime VPRS boundary is for.
    sorted_support = np.sort(support, axis=1)
    vprs_margin = sorted_support[:, -1] - sorted_support[:, -2]
    neighbour_max = support.max(axis=1)
    representation = -rough_result['distance']
    conformal = (-np.asarray(adaptive_conformal_width, dtype=float)
                 if adaptive_conformal_width is not None else np.zeros(n))
    disagreement = (-np.asarray(ensemble_disagreement, dtype=float)
                    if ensemble_disagreement is not None else np.zeros(n))
    random_scores = np.random.default_rng(int(_SELECTOR_SEED)).random(n)
    return {
        'vprs_lower_support': vprs_margin,
        'ordinary_neighbor_support': neighbour_max,
        'degu_style_ensemble_disagreement': disagreement,
        'representation_distance': representation,
        'split_conformal': conformal,
        'random': random_scores,
    }


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def _cross_dataset_overlap_audit(frames, gap_bp):
    # Audit coordinate overlap across every pair of frames. Hard-fail if any pair declares
    # incompatible genome builds, so an incompatible-build zero-overlap claim cannot slip through.
    import bisect
    audit = {'datasets': {}, 'pairwise': {}}
    for name, frame in frames.items():
        audit['datasets'][name] = {
            'rows': int(len(frame)),
            'genome_builds_present': sorted(set(frame.genome_build.astype(str))),
            'unique_rsids': int(frame.rsid.nunique()) if 'rsid' in frame.columns else None,
            'chromosomes': sorted(set(frame.chrom.astype(str))),
        }
    names = list(frames)
    incompatible = []
    for i, a_name in enumerate(names):
        a = frames[a_name]
        a_builds = set(a.genome_build.astype(str)) - {'unverified'}
        for b_name in names[i + 1:]:
            b = frames[b_name]
            b_builds = set(b.genome_build.astype(str)) - {'unverified'}
            if a_builds and b_builds and not (a_builds & b_builds):
                incompatible.append((a_name, sorted(a_builds), b_name, sorted(b_builds)))
                continue
            a_coords = set(zip(a.chrom.astype(str), a.position.astype(int)))
            b_coords = set(zip(b.chrom.astype(str), b.position.astype(int)))
            exact = len(a_coords & b_coords)
            windowed = 0
            if a_coords and b_coords:
                b_by_chrom = {}
                for chrom, pos in b_coords:
                    b_by_chrom.setdefault(chrom, []).append(pos)
                for chrom in b_by_chrom:
                    b_by_chrom[chrom].sort()
                for chrom, pos in a_coords:
                    positions = b_by_chrom.get(chrom, [])
                    if positions:
                        j = bisect.bisect_left(positions, pos - gap_bp)
                        k = bisect.bisect_right(positions, pos + gap_bp)
                        if k > j:
                            windowed += 1
            shared_rsids = (set(a.rsid.dropna().astype(str)) & set(b.rsid.dropna().astype(str))
                            if 'rsid' in a.columns and 'rsid' in b.columns else set())
            audit['pairwise'][f'{a_name}__{b_name}'] = {
                'exact_coordinate_matches': int(exact),
                f'variants_within_{gap_bp}bp': int(windowed),
                'shared_rsid_count': int(len(shared_rsids)),
                'genome_builds_compared': {a_name: sorted(a_builds), b_name: sorted(b_builds)},
            }
    if incompatible:
        raise RuntimeError(
            'Incompatible genome builds across datasets; harmonise via lift-over before evaluation: '
            + '; '.join(f'{a}:{ab} vs {b}:{bb}' for a, ab, b, bb in incompatible))
    return audit


def _resolve_held_out(data_root, config, output):
    frames = {}
    for name in config['evaluation']['held_out_datasets']:
        prepared = prepare_dataset(name, data_root, output / 'prepared' / f'{name}.csv.gz')
        frames[name] = pd.read_csv(prepared.dataset, dtype={'chrom': str})
    overlap_path = PROJECT_ROOT / config['evaluation']['overlap_file']
    if not overlap_path.is_file():
        raise FileNotFoundError(f'Required overlap file missing: {overlap_path}')
    shared = {line.strip() for line in overlap_path.read_text().splitlines() if line.strip()}
    if 'bourges' in frames and shared:
        bourges = frames['bourges']
        if 'rsid' in bourges.columns:
            frames['bourges'] = bourges[~bourges.rsid.isin(shared)].reset_index(drop=True)
    return frames


def _prepare_development(data_root, output):
    prepared = prepare_dataset('siraj', data_root, output / 'prepared' / 'siraj.csv.gz')
    return pd.read_csv(prepared.dataset, dtype={'chrom': str})


def _augment_with_mechanistic(frame, data_root):
    try:
        return merge_ho_mechanistic_features(data_root, frame)
    except FileNotFoundError:
        for column in MECHANISTIC_FEATURES + DHS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame


def _locally_adaptive_conformal_width(fused_cal, residuals_cal, fused_held, k):
    # kNN-weighted calibration residual: ranks uncertainty within a fold.
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    k = int(min(k, len(fused_cal)))
    if k <= 0:
        return np.zeros(len(fused_held))
    scaler = StandardScaler().fit(fused_cal)
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(fused_cal))
    distances, indices = nn.kneighbors(scaler.transform(fused_held))
    temperature = np.maximum(np.median(distances, axis=1, keepdims=True), 1e-6)
    weights = np.exp(-distances / temperature)
    return (residuals_cal[indices] * weights).sum(axis=1) / weights.sum(axis=1)


def _bootstrap_ridge_ensemble_disagreement(settings, fused_fit, frame_fit, fused_held,
                                           source_held, n_members, seed):
    # Fit ``n_members`` Ridge heads on bootstrap resamples of the fit fold predicting
    # the single context_transfer endpoint; score each held variant by cross-member std.
    rng = np.random.default_rng(int(seed))
    members = []
    for m in range(int(n_members)):
        resample = rng.integers(0, len(fused_fit), size=len(fused_fit))
        head = fit_fold_head(settings, fused_fit[resample], frame_fit.iloc[resample])
        members.append(head.predict(fused_held, source_held)['context_transfer'])
    return np.std(np.column_stack(members), axis=1)


def _tune_beta_on_development(settings, frame, alphagenome_scores, decima_scores, log_path):
    fusion = MechanisticFusion(settings)
    fused = fusion.transform(alphagenome_scores, decima_scores, frame)
    frame = frame.copy()
    frame['split_group'] = locus_groups(frame, gap_bp=int(settings['gap_bp']))
    best_beta, best_rmse = None, float('inf')
    tuning_rows = []
    for beta in settings['tuning']['beta_grid']:
        tuned_settings = {**settings, 'rough': {**settings['rough'], 'beta': beta}}
        rmses = []
        for fit, held in grouped_folds(frame, folds=int(settings['folds']), seed=int(settings['seed'])):
            head = fit_fold_head(tuned_settings, fused[fit], frame.iloc[fit])
            rough = _fit_rough_soft(tuned_settings, fused[fit], frame.iloc[fit])
            source_held = frame.iloc[held].source_effect.to_numpy(dtype=float)
            held_pred = head.predict(fused[held], source_held)['primary_T_reconstructed']
            held_rough = rough.predict(fused[held])
            fused_pred = _rough_soft_ai_prediction(tuned_settings, held_pred, held_rough, frame.iloc[held])
            rmse, _ = group_rmse(frame.iloc[held], fused_pred)
            rmses.append(rmse)
        mean_rmse = float(np.mean(rmses))
        tuning_rows.append({'beta': beta, 'equal_locus_rmse_mean_across_folds': mean_rmse})
        log_event(log_path, 'beta_tuning_fold', beta=beta, rmse=mean_rmse)
        if mean_rmse < best_rmse:
            best_rmse, best_beta = mean_rmse, beta
    return best_beta, tuning_rows


def _evaluate_held_out(settings, frame, alphagenome_scores, decima_scores,
                       corrections_path, output_folds, log_path):
    fusion = MechanisticFusion(settings)
    fused = fusion.transform(alphagenome_scores, decima_scores, frame)
    frame = frame.copy()
    frame['split_group'] = locus_groups(frame, gap_bp=int(settings['gap_bp']))
    n = len(frame)
    ai_pred = np.full(n, np.nan)
    rough_pred = np.full(n, np.nan)
    boundary_mask = np.zeros(n, dtype=bool)
    head_names = ('context_transfer', 'source', 'target', 'primary_T_reconstructed')
    scores_full = {name: np.full(n, np.nan) for name in head_names}
    rough_support = np.full((n, len(CLASSES)), np.nan)
    rough_lower = np.zeros((n, len(CLASSES)))
    rough_upper = np.zeros((n, len(CLASSES)))
    distance = np.full(n, np.nan)
    delta_pred = np.full(n, np.nan)
    adaptive_conformal_width = np.full(n, np.nan)
    ensemble_disagreement = np.full(n, np.nan)
    all_cards = []
    all_applied = []
    output_folds.mkdir(parents=True, exist_ok=True)
    ub = settings['uncertainty_baselines']
    for fold, (fit, held) in enumerate(grouped_folds(frame, folds=int(settings['folds']), seed=int(settings['seed']))):
        corrected, applied = _apply_corrections(frame, corrections_path, fit)
        all_applied.extend(applied)
        # Split-conformal: carve the fit fold into inner_fit (train) and calibration (compute residuals).
        inner_fit_rel, calibration_rel = inner_split(
            np.arange(len(fit)), corrected.iloc[fit].split_group.to_numpy(),
            float(ub['calibration_fraction_within_fit_fold']),
            int(settings['seed']) + fold)
        inner_fit = fit[inner_fit_rel]
        calibration = fit[calibration_rel]
        head = fit_fold_head(settings, fused[inner_fit], corrected.iloc[inner_fit])
        rough = _fit_rough_soft(settings, fused[fit], corrected.iloc[fit])
        source_held = frame.iloc[held].source_effect.to_numpy(dtype=float)
        held_head = head.predict(fused[held], source_held)
        held_rough = rough.predict(fused[held])
        for name in head_names:
            scores_full[name][held] = held_head[name]
        ai_pred[held] = held_head['primary_T_reconstructed']
        delta_pred[held] = held_head['context_transfer']
        rough_pred[held] = _rough_soft_ai_prediction(
            settings, held_head['primary_T_reconstructed'], held_rough, frame.iloc[held])
        accepted = held_rough['lower'].sum(axis=1) >= 1
        # Real split-conformal: residuals computed on the calibration slice, never on the inner_fit fit set.
        source_cal = corrected.iloc[calibration].source_effect.to_numpy(dtype=float)
        cal_head = head.predict(fused[calibration], source_cal)
        cal_residuals = np.abs(
            cal_head['context_transfer']
            - (corrected.iloc[calibration].target_effect.to_numpy(dtype=float) - source_cal))
        adaptive_conformal_width[held] = _locally_adaptive_conformal_width(
            fused[calibration], cal_residuals, fused[held], int(ub['conformal_knn']))
        # Real ensemble-disagreement baseline: 5 bootstrap-resampled Ridge heads on the SAME context_transfer endpoint.
        ensemble_disagreement[held] = _bootstrap_ridge_ensemble_disagreement(
            settings, fused[fit], corrected.iloc[fit], fused[held], source_held,
            int(ub['degu_ensemble_members']), int(settings['seed']) + fold)
        rough_support[held] = held_rough['support']
        rough_lower[held] = held_rough['lower']
        rough_upper[held] = held_rough['upper']
        distance[held] = held_rough['distance']
        beta = float(settings['rough']['beta'])
        boundary = ((held_rough['upper'].sum(axis=1) - held_rough['lower'].sum(axis=1)) > 0) \
                   & (held_rough['support'].max(axis=1) < beta)
        boundary_mask[held] = boundary
        cards = _collect_review_cards(frame.iloc[held], held_rough,
                                      {state: held_head[state] for state in held_head},
                                      fused[held], settings, fold)
        all_cards.extend(cards)
        _save_fold(output_folds, fold, head, rough)
        log_event(log_path, 'held_out_fold', fold=fold, held=len(held),
                  boundary_cases=int(boundary.sum()), accepted_by_lower=int(accepted.sum()))
    rough_result = {'support': rough_support, 'lower': rough_lower, 'upper': rough_upper,
                    'distance': distance}
    return {
        'ai_prediction': ai_pred, 'rough_prediction': rough_pred,
        'delta_prediction': delta_pred,
        'adaptive_conformal_width': adaptive_conformal_width,
        'ensemble_disagreement': ensemble_disagreement,
        'boundary_mask': boundary_mask, 'head_scores': scores_full,
        'rough_result': rough_result, 'cards': all_cards, 'corrections_applied': all_applied,
        'frame_with_groups': frame,
    }


def _expert_prediction_from_actions(frame, ai_prediction, config):
    # Produce the expert-only prediction by applying each action's declared semantics vectorised.
    ai = np.asarray(ai_prediction, dtype=float)
    source = frame.source_effect.to_numpy(dtype=float)
    actions = frame.get('expert_action', pd.Series([''] * len(frame))).to_numpy()
    # Start by assuming the default (no action recorded) == 'accept' semantics.
    predicted = _resolve_action_value(config['expert_actions']['accept'], ai, source).copy()
    for name, cfg in config['expert_actions'].items():
        rows = (actions == name)
        if rows.any():
            predicted[rows] = _resolve_action_value(cfg, ai, source)[rows]
    return predicted


def _write_audit(output, config, alphagenome_prov, decima_prov, source_paths,
                 wall_time_per_variant):
    audit = {
        'pipeline': 'neurojico v1',
        'platform': platform_metadata(),
        'config_sha256': sha256(DEFAULT_CONFIG),
        'alphagenome': alphagenome_prov,
        'decima': decima_prov,
        'cell_states': config['cell_states'],
        'data_sources_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p)
                                for p in source_paths if p.is_file()},
        'wall_time_per_variant_seconds': wall_time_per_variant,
        'honest_limitations': [
            'CD4+ T cell ontology term is not donor-matched to Ho 2025; Jurkat is a cell-line proxy ontology term.',
            'The rough-soft precision parameter beta is tuned on Siraj (K562 vs HepG2), a cross-cell-type proxy for Ho/Bourges primary T. This keeps Ho/Bourges held out but means beta is optimal for Siraj biology, not Ho biology.',
        ],
        'design_choices_not_limitations': [
            'Foundation models are used inference-only (no fine-tuning) to keep the framework portable across backbones and isolate the contribution of the rough-soft layer.',
            'AlphaGenome is served via API; the frozen reproducibility artefact is the per-variant response cache plus the recorded api_model_version.',
            'Only the Ridge head and VPRS parameters are fit locally.',
        ],
        'future_work': [
            'Prospective expert study to measure real per-card review time (currently declared as 90 s in config.json).',
        ],
    }
    write_json(output / 'audit.json', audit)


def run_neurojico(options):
    global _SELECTOR_SEED
    output = prepare_output(
        options.output,
        protected=[options.data_root, options.config_path],
        preserve=['alphagenome_cache', 'decima_cache', 'prepared'],
    )
    config = load_config(options.config_path)
    _SELECTOR_SEED = int(config['seed'])
    log_path = output / 'run_log.jsonl'
    log_event(log_path, 'neurojico_started', config_sha256=sha256(options.config_path))

    print('[1/6] Preparing datasets', flush=True)
    dev_frame = _prepare_development(options.data_root, output)
    held_frames = _resolve_held_out(options.data_root, config, output)
    for name, frame in list(held_frames.items()):
        held_frames[name] = _augment_with_mechanistic(frame, options.data_root)
    dev_frame = _augment_with_mechanistic(dev_frame, options.data_root)
    overlap_audit = _cross_dataset_overlap_audit({'siraj': dev_frame, **held_frames},
                                                 int(config['gap_bp']))
    for name, frame in {'siraj': dev_frame, **held_frames}.items():
        groups = locus_groups(frame, gap_bp=int(config['gap_bp']))
        overlap_audit['datasets'][name]['independent_loci_after_grouping'] = int(len(set(groups)))
    write_json(output / 'cross_dataset_overlap_audit.json', overlap_audit)

    print('[2/6] Scoring with AlphaGenome API (hard-fail on error)', flush=True)
    alphagenome = AlphaGenomeVariantScorer(config, output / 'alphagenome_cache')
    all_frames = {'development': dev_frame, **held_frames}
    alphagenome_scores = {}
    alphagenome_prov = {}
    t0 = time.monotonic()
    total_rows = 0
    for name, frame in all_frames.items():
        scores, prov = alphagenome.score(frame)
        alphagenome_scores[name] = scores
        alphagenome_prov[name] = prov
        total_rows += len(frame)
    alphagenome_seconds = time.monotonic() - t0
    log_event(log_path, 'alphagenome_scoring_complete', seconds=alphagenome_seconds,
              variants=total_rows)

    print('[3/6] Scoring with Decima (CPU/MPS)', flush=True)
    decima = DecimaGeneScorer(config, output / 'decima_cache')
    decima_scores, decima_prov = {}, {}
    t0 = time.monotonic()
    for name, frame in all_frames.items():
        if options.skip_decima:
            decima_scores[name] = {state: np.zeros((len(frame), 1), dtype=np.float32)
                                   for state in config['cell_states']}
            decima_prov[name] = {'status': 'skipped'}
        else:
            scores, prov = decima.score(frame)
            decima_scores[name] = scores
            decima_prov[name] = prov
    decima_seconds = time.monotonic() - t0
    log_event(log_path, 'decima_scoring_complete', seconds=decima_seconds)

    print('[4/6] Tuning beta on Siraj (development only, no Ho/Bourges label leakage)', flush=True)
    best_beta, tuning_rows = _tune_beta_on_development(
        config, dev_frame, alphagenome_scores['development'], decima_scores['development'], log_path)
    config = {**config, 'rough': {**config['rough'], 'beta': best_beta}}
    write_json(output / 'tuning.json', {'selected_beta': best_beta, 'rows': tuning_rows})
    log_event(log_path, 'beta_selected', beta=best_beta)

    print('[5/6] Held-out evaluation on Ho + Bourges', flush=True)
    held_evaluations = {}
    all_cards = []
    for name, frame in held_frames.items():
        print(f'  [5/6] held-out: {name}', flush=True)
        result = _evaluate_held_out(
            config, frame, alphagenome_scores[name], decima_scores[name],
            options.expert_corrections, output / 'folds' / name, log_path)
        for c in result['cards']:
            c['dataset'] = name
        all_cards.extend(result['cards'])
        ai = result['ai_prediction']
        rough = result['rough_prediction']
        frame_eval = result['frame_with_groups']
        expert_full = _expert_prediction_from_actions(frame_eval, ai, config)
        neurojico = np.where(result['boundary_mask'], expert_full, ai)
        budget = int(result['boundary_mask'].sum())
        expert_alone_mask = _expert_alone_selection_mask(
            config['evaluation']['expert_alone_selection_rule'],
            budget, ai, frame_eval, config['seed'])
        expert_alone = np.where(expert_alone_mask, expert_full, ai)
        source = frame_eval.source_effect.to_numpy(dtype=float)
        delta_predictions = {
            'ai_alone': ai - source,
            'rough_soft_ai': rough - source,
            'expert_alone': expert_alone - source,
            'neurojico': neurojico - source,
        }
        four = evaluate_four_methods(
            frame_eval, ai, rough, expert_alone, neurojico,
            boundary_mask=result['boundary_mask'],
            expert_alone_mask=expert_alone_mask,
            review_time_seconds=float(config['evaluation']['expert_review_seconds_per_card']),
            config=config, delta_predictions=delta_predictions)
        selectors = _selector_scores(
            result['rough_result'], result['head_scores'],
            ensemble_disagreement=result['ensemble_disagreement'],
            adaptive_conformal_width=result['adaptive_conformal_width'])
        curves = {sel: selective_curve(frame_eval, ai, scores,
                                       config['evaluation']['coverage_grid'])
                  for sel, scores in selectors.items()}
        strict_lower_coverage = float(np.mean(result['rough_result']['lower'].sum(axis=1) >= 1))
        held_evaluations[name] = {
            'four_methods': four,
            'selective_curves': curves,
            'strict_lower_coverage': strict_lower_coverage,
            'rows': int(len(frame_eval)),
            'loci': int(frame_eval.split_group.nunique()),
            'boundary_rate': float(result['boundary_mask'].mean()),
            'corrections_applied': result['corrections_applied'],
        }

    write_json(output / 'review_cards.json', all_cards)
    if not options.expert_corrections:
        write_json(output / 'expert_corrections.template.json',
                   [{'example_id': '<example_id>', 'field': 'expert_action',
                     'value': 'accept', 'reason': '<reason>',
                     'evidence_url': 'https://doi.org/<doi>'}])

    print('[6/6] Writing audit.json and evaluation.json', flush=True)
    wall_time_per_variant = {'alphagenome_seconds_per_variant': alphagenome_seconds / max(total_rows, 1),
                             'decima_seconds_per_variant': decima_seconds / max(total_rows, 1)}
    source_paths = [options.data_root / p for p in [
        'ho_supp.xlsx', 'bourges_barcode_counts.txt.gz', 'siraj_MPRA.zip',
        'siraj_tables.zip', 'ho_normalized_counts.txt.gz', 'ho_bourges_shared_rsids.txt']]
    _write_audit(output, config, alphagenome_prov, decima_prov, source_paths,
                 wall_time_per_variant)
    evaluation = {
        'held_out': held_evaluations,
        'selected_beta': best_beta,
        'tuning_rows': tuning_rows,
        'wall_time_per_variant': wall_time_per_variant,
        'figure_methods_order': config['evaluation']['four_methods'],
    }
    write_json(output / 'evaluation.json', evaluation)
    log_event(log_path, 'neurojico_complete')
    return {'output': output, 'evaluation': evaluation}
