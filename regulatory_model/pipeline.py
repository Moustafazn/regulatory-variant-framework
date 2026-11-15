"""Regulatory variant framework orchestration, including the NeuroJICO review arm."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from .env import configure_torch_env_from_config
from .data_prep import DHS, MECHANISTIC_FEATURES, PROJECT_ROOT


# Default config path shipped with the package.
DEFAULT_CONFIG = Path(__file__).parent / 'config.json'

# Defence-in-depth: honour config.secondary.device = "cpu" before anything inside this module imports torch.
configure_torch_env_from_config(DEFAULT_CONFIG)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict, StringConstraints, TypeAdapter  # noqa: E402

from .dataset_cohorts import (  # noqa: E402
    _apply_dataset_sampling, _augment_with_mechanistic, _avi_features_from_frame,
    _cross_dataset_overlap_audit, _dataset_fold_feasibility, _load_avi_dataset,
    _load_tracks_dataset, _prepare_development, _resolve_held_out,
)
from .decima_scorer import DecimaGeneScorer  # noqa: E402
from .evaluation import (  # noqa: E402
    CLASSES, grouped_folds, group_rmse, group_rmse_bootstrap,
    inner_split, locus_groups, selective_curve,
)
from .model import MechanisticFusion, RidgeHead, VariablePrecisionRoughSoft  # noqa: E402
from .reporting import (  # noqa: E402
    log_event, platform_metadata, prepare_output, sha256,
    summarize_feature_impact, write_json,
)


# Pydantic string constraints for expert-correction records.
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Https = Annotated[str, StringConstraints(pattern=r'^https://.+')]


# Vocabulary of allowed expert actions on a boundary review card.
class ExpertAction:
    accept = 'accept'
    challenge = 'challenge'
    defer = 'defer'


# One expert correction record validated by Pydantic.
class ExpertCorrection(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    example_id: NonEmpty
    field: NonEmpty
    value: Literal[0, 1, 'accept', 'challenge', 'defer'] | None
    reason: NonEmpty
    evidence_url: Https


# Type adapter used to parse a list of corrections from JSON.
CORRECTIONS = TypeAdapter(list[ExpertCorrection])


# Load framework settings while accepting configurations from before the rename.
def load_config(path=DEFAULT_CONFIG):
    data = json.loads(Path(path).read_text())
    keys = [key for key in ('framework', 'neurojico') if key in data]
    if len(keys) != 1:
        raise ValueError('config.json must contain exactly one framework or legacy neurojico block')
    return data[keys[0]]


# Options passed from main.py (or the API) into run_framework.
@dataclass(frozen=True)
class FrameworkOptions:
    output: Path
    data_root: Path = PROJECT_ROOT / 'data' / 'regulatory_audit'
    avi_dir: Path = PROJECT_ROOT / 'data' / 'alphagenome_avi'
    tracks_dir: Path = PROJECT_ROOT / 'data' / 'alphagenome_tracks'
    config_path: Path = DEFAULT_CONFIG
    expert_corrections: Path | None = None
    skip_decima: bool = False
    require_tracks: bool = False
    sample_chrom: str | None = None
    sample_per_dataset: tuple[tuple[str, int], ...] = ()


# Standard output directory for a named experiment.
def default_output(name='regulatory-variant-framework'):
    return PROJECT_ROOT / 'output' / 'experiments' / name


# Apply expert corrections restricted to the fitting fold; return (updated frame, applied records).
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


# Coerce a scalar to float or None if non-finite / non-numeric.
def _as_scalar(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


# Build the per-boundary-variant review cards exported to review_cards.json.
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


# Fit the Variable-Precision Rough-Soft model on one fold's fit side.
def _fit_rough_soft(settings, fused_fit, frame_fit):
    return VariablePrecisionRoughSoft(settings).fit(fused_fit, frame_fit)


# Persist one fold's fitted head and rough model to disk.
def _save_fold(output, fold, head, rough):
    joblib.dump({'head': head, 'rough': rough},
                output / f'fold_{fold:02d}.joblib')


# Fit a Ridge head on one fold's fit side.
def fit_fold_head(settings, fused_fit, frame_fit):
    return RidgeHead(settings).fit(fused_fit, frame_fit)



# Maps rough-soft/expert action semantics to a vectorised prediction override.
_SEMANTICS = {
    'keep_ai':            lambda ai, source, const: ai,
    'use_source_effect':  lambda ai, source, const: source,
    'constant':           lambda ai, source, const: np.full_like(ai, const, dtype=float),
}


# Resolve one action/rejection config entry into a full-length array of predictions.
def _resolve_action_value(action_cfg, ai, source):
    return _SEMANTICS[action_cfg['semantics']](ai, source, action_cfg.get('constant_value'))


# Scoring rules the Expert-alone method uses to pick its own B variants to review.
_EXPERT_ALONE_RULES = {
    'random':                   lambda ai, frame, seed: np.random.default_rng(int(seed)).random(len(ai)),
    'largest_abs_ai_prediction': lambda ai, frame, seed: np.abs(np.asarray(ai, dtype=float)),
    'largest_source_se':         lambda ai, frame, seed: frame.source_se.to_numpy(dtype=float),
}


# Pick ``budget`` indices to review under the Expert-alone rule (not VPRS boundary).
def _expert_alone_selection_mask(rule, budget, ai, frame, seed):
    scores = _EXPERT_ALONE_RULES[rule](ai, frame, seed)
    n = len(ai)
    budget = int(min(budget, n))
    mask = np.zeros(n, dtype=bool)
    if budget > 0:
        mask[np.argsort(-scores, kind='stable')[:budget]] = True
    return mask


# Rough-soft AI output: AI where VPRS accepts, else the declared rejected-variant value.
def _rough_soft_ai_prediction(settings, head_prediction, held_rough, held_frame):
    ai = np.asarray(head_prediction, dtype=float)
    source = held_frame.source_effect.to_numpy(dtype=float)
    accepted = held_rough['lower'].sum(axis=1) >= 1
    rejected_value = _resolve_action_value(
        settings['rough_soft_ai_rejected_variant_value'], ai, source)
    return np.where(accepted, ai, rejected_value)


# Return a float R^2 or None when the target variance vanishes.
def _safe_r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = float(np.var(y))
    if denom <= 0:
        return None
    return float(1.0 - np.mean((y - pred) ** 2) / denom)


# Return a copy of ``frame`` whose ``target_effect`` IS the context-transfer delta.
def _delta_frame(frame):
    delta = frame.copy()
    delta['target_effect'] = frame.target_effect.to_numpy(dtype=float) - frame.source_effect.to_numpy(dtype=float)
    return delta


# Equal-locus RMSE + bootstrap CI + R^2 for one prediction vector.
def _method_metrics(frame, prediction, config):
    pred = np.asarray(prediction, dtype=float)
    rmse, per_group = group_rmse(frame, pred)
    lo, hi = group_rmse_bootstrap(per_group, draws=int(config['bootstrap']), seed=int(config['seed']))
    return {'rmse': float(rmse), 'equal_locus_rmse_ci95': [lo, hi],
            'r2': _safe_r2(frame.target_effect.to_numpy(dtype=float), pred)}



# Report every method on both the primary-T and context-transfer-delta endpoints.
def evaluate_four_methods(frame, ai_prediction, rough_prediction, expert_prediction,
                          neurojico_prediction, boundary_mask, expert_alone_mask,
                          review_time_seconds, config, delta_predictions=None):
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


# Overwritten per-run by run_neurojico from config['seed'] so the random selector is reproducible.
_SELECTOR_SEED = 0


# Per-variant confidence scores; higher = more confident = retained first at low coverage.
def _selector_scores(rough_result, head_scores, ensemble_disagreement=None,
                     adaptive_conformal_width=None):
    support = rough_result['support']
    n = support.shape[0]
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


# kNN-weighted calibration residual: ranks uncertainty within a fold.
def _locally_adaptive_conformal_width(fused_cal, residuals_cal, fused_held, k):
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


# Fit n_members bootstrap Ridge heads and return per-held cross-member std on context_transfer.
def _bootstrap_ridge_ensemble_disagreement(settings, fused_fit, frame_fit, fused_held,
                                           source_held, n_members, seed):
    rng = np.random.default_rng(int(seed))
    members = []
    for _ in range(int(n_members)):
        resample = rng.integers(0, len(fused_fit), size=len(fused_fit))
        head = fit_fold_head(settings, fused_fit[resample], frame_fit.iloc[resample])
        members.append(head.predict(fused_held, source_held)['context_transfer'])
    return np.std(np.column_stack(members), axis=1)



# Return the ordered fusion blocks; drop ``tracks`` while NPZs are incomplete.
def _enabled_feature_blocks(settings, tracks_available):
    order = list(settings['fusion'].get('feature_blocks',
                                        ('avi', 'tracks', 'decima', 'mechanistic', 'dhs')))
    if not tracks_available and 'tracks' in order:
        order = [block for block in order if block != 'tracks']
    return tuple(order)


# Return the grouped-locus RMSE of the Ridge head over the configured folds.
def _block_rmse(settings, frame, avi_scores, decima_scores, track_scores, enabled_blocks):
    fusion = MechanisticFusion(settings, enabled_blocks=enabled_blocks)
    fused = fusion.transform(avi_scores, decima_scores, frame, track_scores=track_scores)
    frame = frame.copy()
    frame['split_group'] = locus_groups(frame, gap_bp=int(settings['gap_bp']))
    preds = np.full(len(frame), np.nan)
    for fit, held in grouped_folds(frame, folds=int(settings['folds']), seed=int(settings['seed'])):
        head = fit_fold_head(settings, fused[fit], frame.iloc[fit])
        source_held = frame.iloc[held].source_effect.to_numpy(dtype=float)
        preds[held] = head.predict(fused[held], source_held)['primary_T_reconstructed']
    rmse, _ = group_rmse(frame, preds)
    return float(rmse), fusion.column_names


# Return per-column |coef|*std attributions from the context_transfer Ridge head.
def _coefficient_attribution(settings, frame, avi_scores, decima_scores, track_scores,
                             enabled_blocks):
    fusion = MechanisticFusion(settings, enabled_blocks=enabled_blocks)
    fused = fusion.transform(avi_scores, decima_scores, frame, track_scores=track_scores)
    head = fit_fold_head(settings, fused, frame)
    coefs = head.models['context_transfer'].coef_
    std = np.std(fused, axis=0)
    importance = np.abs(coefs) * std
    denom = importance.sum() if importance.sum() > 0 else 1.0
    normalized = importance / denom
    return [
        {'feature': name, 'coefficient': float(coefs[i]),
         'feature_std': float(std[i]), 'importance': float(importance[i]),
         'importance_fraction': float(normalized[i])}
        for i, name in enumerate(fusion.column_names)
    ]


# Measure block LOO + configured-ablations + per-feature attributions on one dataset.
def _feature_impact_study(settings, frame, avi_scores, decima_scores, track_scores):
    n_loci, feasible_folds = _dataset_fold_feasibility(frame, settings)
    if feasible_folds == 0:
        return {
            'status': 'skipped',
            'skipped_reason': _fold_skip_reason(n_loci, settings),
            'rows': int(len(frame)),
            'independent_loci': int(n_loci),
            'folds_requested': int(settings['folds']),
            'enabled_blocks': list(_enabled_feature_blocks(settings, track_scores is not None)),
        }
    if feasible_folds != int(settings['folds']):
        settings = {**settings, 'folds': feasible_folds}
    tracks_available = track_scores is not None
    full_blocks = _enabled_feature_blocks(settings, tracks_available)
    full_rmse, columns = _block_rmse(settings, frame, avi_scores, decima_scores,
                                     track_scores, full_blocks)
    loo = []
    for block in full_blocks:
        reduced = tuple(b for b in full_blocks if b != block)
        if not reduced:
            continue
        reduced_rmse, _ = _block_rmse(settings, frame, avi_scores, decima_scores,
                                      track_scores, reduced)
        loo.append({
            'removed_block': block,
            'kept_blocks': list(reduced),
            'rmse': reduced_rmse,
            'delta_rmse_vs_full': reduced_rmse - full_rmse,
        })
    ablations = []
    for entry in settings['evaluation'].get('feature_ablations', ()):
        requested = tuple(entry['blocks'])
        active = tuple(b for b in requested if b != 'tracks' or tracks_available)
        if not active:
            continue
        ablation_rmse, _ = _block_rmse(settings, frame, avi_scores, decima_scores,
                                       track_scores, active)
        ablations.append({'name': entry['name'], 'blocks': list(active),
                          'rmse': ablation_rmse})
    attributions = _coefficient_attribution(settings, frame, avi_scores, decima_scores,
                                            track_scores, full_blocks)
    return {
        'enabled_blocks': list(full_blocks),
        'full_rmse': full_rmse,
        'leave_one_block_out': loo,
        'configured_ablations': ablations,
        'feature_attributions': attributions,
        'attribution_method': 'coefficient_magnitude_times_feature_std (gradient*input for Ridge)',
    }


# Explain in one sentence why fold assembly was impossible for this cohort.
def _fold_skip_reason(n_loci, settings):
    return (
        f'Dataset has {n_loci} independent locus/loci after grouping '
        f'(gap_bp={int(settings["gap_bp"])}); GroupKFold with up to '
        f'{int(settings["folds"])} folds followed by an inner GroupShuffleSplit of '
        f'{float(settings["inner_validation_fraction"]):.2f} needs at least 2 loci on the '
        f'fit side of every fold, which requires >=4 independent loci at the configured '
        f'inner_validation_fraction.'
    )



# Tune the rough-soft beta over the dev set and return (best_beta, grid rows).
def _tune_beta_on_development(settings, frame, avi_scores, decima_scores, log_path,
                              track_scores=None):
    n_loci, feasible_folds = _dataset_fold_feasibility(frame, settings)
    if feasible_folds == 0:
        default_beta = float(settings['rough']['beta'])
        log_event(log_path, 'beta_tuning_skipped',
                  rows=int(len(frame)), independent_loci=int(n_loci),
                  folds_requested=int(settings['folds']),
                  default_beta=default_beta,
                  reason=_fold_skip_reason(n_loci, settings))
        return default_beta, []
    if feasible_folds != int(settings['folds']):
        settings = {**settings, 'folds': feasible_folds}
    enabled_blocks = _enabled_feature_blocks(settings, track_scores is not None)
    fusion = MechanisticFusion(settings, enabled_blocks=enabled_blocks)
    fused = fusion.transform(avi_scores, decima_scores, frame, track_scores=track_scores)
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
    rmse_values = [row['equal_locus_rmse_mean_across_folds'] for row in tuning_rows]
    rmse_spread = (max(rmse_values) - min(rmse_values)) if rmse_values else 0.0
    if rmse_values and rmse_spread < 1e-9:
        log_event(log_path, 'beta_tuning_flat_grid',
                  beta_grid=list(settings['tuning']['beta_grid']),
                  rmse=rmse_values[0], rmse_spread=rmse_spread,
                  message=('Every beta on the grid produced the same RMSE to within 1e-9. '
                           'This usually means the VPRS boundary partition is identical '
                           'across the whole grid (e.g. one class dominates support, or the '
                           'dev cohort is too small to differentiate thresholds). '
                           'Consider widening the beta grid, enlarging the dev cohort, or '
                           'checking that cell-state features are non-constant.'))
    return best_beta, tuning_rows


# Produce the expert-only prediction by applying each action's declared semantics vectorised.
def _expert_prediction_from_actions(frame, ai_prediction, config):
    ai = np.asarray(ai_prediction, dtype=float)
    source = frame.source_effect.to_numpy(dtype=float)
    actions = frame.get('expert_action', pd.Series([''] * len(frame))).to_numpy()
    predicted = _resolve_action_value(config['expert_actions']['accept'], ai, source).copy()
    for name, cfg in config['expert_actions'].items():
        rows = (actions == name)
        if rows.any():
            predicted[rows] = _resolve_action_value(cfg, ai, source)[rows]
    return predicted



# Fit GroupKFold+inner-split on one held-out dataset and emit per-variant VPRS, head and cards.
def _evaluate_held_out(settings, frame, avi_scores, decima_scores,
                       corrections_path, output_folds, log_path, track_scores=None):
    n_loci, feasible_folds = _dataset_fold_feasibility(frame, settings)
    if feasible_folds == 0:
        return {
            'status': 'skipped',
            'skipped_reason': _fold_skip_reason(n_loci, settings),
            'rows': int(len(frame)),
            'independent_loci': int(n_loci),
            'folds_requested': int(settings['folds']),
        }
    if feasible_folds != int(settings['folds']):
        settings = {**settings, 'folds': feasible_folds}
    enabled_blocks = _enabled_feature_blocks(settings, track_scores is not None)
    fusion = MechanisticFusion(settings, enabled_blocks=enabled_blocks)
    fused = fusion.transform(avi_scores, decima_scores, frame, track_scores=track_scores)
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
        source_cal = corrected.iloc[calibration].source_effect.to_numpy(dtype=float)
        cal_head = head.predict(fused[calibration], source_cal)
        cal_residuals = np.abs(
            cal_head['context_transfer']
            - (corrected.iloc[calibration].target_effect.to_numpy(dtype=float) - source_cal))
        adaptive_conformal_width[held] = _locally_adaptive_conformal_width(
            fused[calibration], cal_residuals, fused[held], int(ub['conformal_knn']))
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



# Write the per-run audit.json describing config, provenance, timing, and declared limitations.
def _write_audit(output, config, avi_prov, decima_prov, source_paths,
                 wall_time_per_variant, track_prov=None,
                 decima_runtime=None, sampling_audit=None, config_path=DEFAULT_CONFIG):
    decima_block = {'per_dataset': decima_prov}
    if decima_runtime:
        decima_block.update({
            'device': decima_runtime.get('device'),
            'model_identifier': decima_runtime.get('model_identifier'),
            'elapsed_seconds': decima_runtime.get('elapsed_seconds'),
            'peak_rss_bytes': decima_runtime.get('peak_rss_bytes'),
            'peak_rss_mib': decima_runtime.get('peak_rss_mib'),
        })
    if sampling_audit:
        decima_block['dataset_sampling'] = sampling_audit
    audit = {
        'pipeline': 'regulatory-variant-framework v1',
        'platform': platform_metadata(),
        'config_sha256': sha256(config_path),
        'alphagenome_avi': avi_prov,
        'alphagenome_tracks': track_prov or {'status': 'not_loaded'},
        'decima': decima_block,
        'cell_states': config['cell_states'],
        'data_sources_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p)
                                for p in source_paths if p.is_file()},
        'wall_time_per_variant_seconds': wall_time_per_variant,
        'honest_limitations': [
            'Decima context tasks are not donor-matched to Ho 2025; AlphaGenome AVI is a global impact annotation rather than a cell-state-specific measurement.',
            'The rough-soft precision parameter beta is tuned on Siraj (K562 vs HepG2), a cross-cell-type proxy for Ho/Bourges primary T. This keeps Ho/Bourges held out but means beta is optimal for Siraj biology, not Ho biology.',
        ],
        'design_choices_not_limitations': [
            'Foundation models are used inference-only (no fine-tuning) to keep the framework portable across backbones and isolate the contribution of the rough-soft layer.',
            'AlphaGenome contributes its frozen genome-wide AVI raw and PHRED annotations; variants without an archive match are excluded and listed in the audit.',
            'Only the Ridge head and VPRS parameters are fit locally.',
        ],
        'future_work': [
            'Prospective expert study to measure real per-card review time (currently declared as 90 s in config.json).',
        ],
    }
    write_json(output / 'audit.json', audit)


# [1/6] Prepare dev + held-out frames with mechanistic/DHS features attached.
def _step1_prepare_datasets(options, config):
    print('[1/6] Preparing datasets', flush=True)
    dev_frame = _prepare_development(options.data_root, options.output)
    held_frames = _resolve_held_out(options.data_root, config, options.output)
    for name, frame in list(held_frames.items()):
        held_frames[name] = _augment_with_mechanistic(frame, options.data_root)
    dev_frame = _augment_with_mechanistic(dev_frame, options.data_root)
    return dev_frame, held_frames


# [2/6] Load local AlphaGenome AVI scores and filter frames to the matched variants.
def _step2_load_avi(options, log_path, dev_frame, held_frames):
    print('[2/6] Loading local AlphaGenome AVI scores (no API calls)', flush=True)
    all_frames = {'development': dev_frame, **held_frames}
    avi_scores, avi_prov = {}, {}
    t0 = time.monotonic()
    for name, frame in all_frames.items():
        dataset_name = 'siraj' if name == 'development' else name
        prepared_path = options.output / 'prepared' / f'{dataset_name}.csv.gz'
        filtered, provenance = _load_avi_dataset(frame, dataset_name, options.avi_dir, prepared_path)
        avi_scores[name] = _avi_features_from_frame(filtered)
        all_frames[name] = filtered
        avi_prov[name] = provenance
    avi_seconds = time.monotonic() - t0
    total_rows = sum(len(frame) for frame in all_frames.values())
    log_event(log_path, 'alphagenome_avi_loaded', seconds=avi_seconds, variants=total_rows)
    return all_frames, avi_scores, avi_prov, avi_seconds, total_rows


# [2.5/6] Apply --sample-chrom / --sample-per-dataset after AVI matching; recompute AVI tensors in place.
def _step2b_apply_sampling(all_frames, options, config, log_path):
    sampled_frames, sampling_audit = _apply_dataset_sampling(
        all_frames, options, seed=config.get('seed', 42))
    if not sampling_audit.get('applied'):
        return all_frames, {name: _avi_features_from_frame(frame)
                            for name, frame in all_frames.items()}, sampling_audit, sum(len(f) for f in all_frames.values())
    avi_scores = {name: _avi_features_from_frame(frame) for name, frame in sampled_frames.items()}
    total_rows = sum(len(frame) for frame in sampled_frames.values())
    log_event(log_path, 'dataset_sampling_applied',
              chromosome=sampling_audit.get('chromosome'),
              per_dataset_cap=sampling_audit.get('per_dataset_cap'),
              per_dataset_counts=sampling_audit.get('per_dataset_counts'),
              seed=sampling_audit.get('seed'),
              total_after=total_rows)
    return sampled_frames, avi_scores, sampling_audit, total_rows


# [2c/6] Load AlphaGenome cell-state track NPZs (zero-fill missing rows, record status).
def _step2c_load_tracks(options, config, all_frames, log_path):
    print('[2b/6] Loading AlphaGenome cell-state track NPZs', flush=True)
    cell_states = tuple(config['cell_states'].keys())
    modalities = tuple(config['fusion'].get('alphagenome_track_modalities', ()))
    track_scores, track_prov = {}, {}
    t0 = time.monotonic()
    for name, frame in all_frames.items():
        dataset_name = 'siraj' if name == 'development' else name
        scores, provenance = _load_tracks_dataset(
            frame, dataset_name, options.tracks_dir, cell_states, modalities,
            require_complete=options.require_tracks)
        track_scores[name] = scores
        track_prov[name] = provenance
        print(f'  tracks[{name}]: {provenance["status"]}', flush=True)
    tracks_seconds = time.monotonic() - t0
    log_event(log_path, 'alphagenome_tracks_loaded', seconds=tracks_seconds,
              statuses={k: v['status'] for k, v in track_prov.items()})
    return track_scores, track_prov, tracks_seconds


# [3/6] Score Decima per dataset (CPU/MPS), honouring --skip-decima and the per-dataset cache.
def _step3_score_decima(options, config, all_frames, output, log_path):
    print('[3/6] Scoring with Decima (CPU/MPS)', flush=True)
    decima = DecimaGeneScorer(config)
    decima_cache_dir = output / 'decima_scores'
    decima_scores, decima_prov = {}, {}
    t0 = time.monotonic()
    for name, frame in all_frames.items():
        dataset_name = 'siraj' if name == 'development' else name
        if options.skip_decima:
            decima_scores[name] = {state: np.zeros((len(frame), 1), dtype=np.float32)
                                   for state in config['cell_states']}
            decima_prov[name] = {'status': 'skipped'}
        else:
            scores, prov = decima.score(frame, dataset_name=dataset_name,
                                        cache_dir=decima_cache_dir)
            decima_scores[name] = scores
            decima_prov[name] = prov
    decima_seconds = time.monotonic() - t0
    import resource as _resource
    _rusage = _resource.getrusage(_resource.RUSAGE_SELF)
    _peak_rss_raw = int(_rusage.ru_maxrss)
    _peak_rss_bytes = _peak_rss_raw if sys.platform == 'darwin' else _peak_rss_raw * 1024
    decima_runtime = {
        'device': config['secondary'].get('device'),
        'model_identifier': config['secondary'].get('model_identifier', 'ensemble'),
        'elapsed_seconds': decima_seconds,
        'peak_rss_bytes': _peak_rss_bytes,
        'peak_rss_mib': round(_peak_rss_bytes / (1024 * 1024), 1),
    }
    log_event(log_path, 'decima_scoring_complete', seconds=decima_seconds,
              device=decima_runtime['device'],
              model_identifier=decima_runtime['model_identifier'],
              peak_rss_mib=decima_runtime['peak_rss_mib'])
    return decima_scores, decima_prov, decima_seconds, decima_runtime

    write_json(output / 'audit.json', audit)



# [4/6] Tune the rough-soft precision beta on Siraj and persist tuning.json.
def _step4_tune_beta(config, dev_frame, avi_scores, decima_scores, track_scores,
                     log_path, output):
    print('[4/6] Tuning beta on Siraj (development only, no Ho/Bourges label leakage)', flush=True)
    best_beta, tuning_rows = _tune_beta_on_development(
        config, dev_frame, avi_scores['development'], decima_scores['development'], log_path,
        track_scores=track_scores['development'])
    write_json(output / 'tuning.json', {'selected_beta': best_beta, 'rows': tuning_rows})
    log_event(log_path, 'beta_selected', beta=best_beta)
    return best_beta, tuning_rows


# [4b/6] Measure block LOO + per-feature attributions on every dataset.
def _step4b_feature_impact(config, all_frames, avi_scores, decima_scores, track_scores,
                           held_frames, log_path, output):
    print('[4b/6] Feature-impact study (block LOO + coefficient attribution)', flush=True)
    feature_impact = {}
    for name in ('development', *held_frames.keys()):
        feature_impact[name] = _feature_impact_study(
            config, all_frames[name], avi_scores[name], decima_scores[name],
            track_scores[name])
        record = feature_impact[name]
        if record.get('status') == 'skipped':
            print(f'  impact[{name}]: skipped ({record["skipped_reason"]})', flush=True)
        else:
            kept = record['enabled_blocks']
            print(f'  impact[{name}]: blocks={kept} full_rmse={record["full_rmse"]:.4f}',
                  flush=True)
    write_json(output / 'feature_impact.json', feature_impact)
    dev_impact = feature_impact.get('development', {})
    log_event(log_path, 'feature_impact_complete',
              datasets=list(feature_impact),
              scored_datasets=[n for n, r in feature_impact.items() if r.get('status') != 'skipped'],
              blocks=dev_impact.get('enabled_blocks', []))
    return feature_impact


# [5/6] Fit held-out GroupKFold on each held dataset and build four-method evaluations.
def _step5_held_out_evaluation(config, held_frames, avi_scores, decima_scores, track_scores,
                               options, output, log_path):
    print('[5/6] Held-out evaluation on Ho + Bourges', flush=True)
    held_evaluations = {}
    all_cards = []
    for name, frame in held_frames.items():
        print(f'  [5/6] held-out: {name}', flush=True)
        result = _evaluate_held_out(
            config, frame, avi_scores[name], decima_scores[name],
            options.expert_corrections, output / 'folds' / name, log_path,
            track_scores=track_scores[name])
        if result.get('status') == 'skipped':
            print(f'  [5/6] held-out: {name} skipped ({result["skipped_reason"]})',
                  flush=True)
            held_evaluations[name] = {
                'status': 'skipped',
                'skipped_reason': result['skipped_reason'],
                'rows': result['rows'],
                'independent_loci': result['independent_loci'],
                'folds_requested': result['folds_requested'],
            }
            continue
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
    return held_evaluations



# [6/6] Compute per-variant wall-time, write audit.json and evaluation.json, finalise the run.
def _step6_write_outputs(options, config, output, log_path, avi_prov, decima_prov, track_prov,
                         sampling_audit, decima_runtime, best_beta, tuning_rows,
                         feature_impact, held_evaluations, avi_seconds, tracks_seconds,
                         decima_seconds, total_rows):
    print('[6/6] Writing audit.json and evaluation.json', flush=True)
    wall_time_per_variant = {
        'avi_load_seconds_per_variant': avi_seconds / max(total_rows, 1),
        'tracks_load_seconds_per_variant': tracks_seconds / max(total_rows, 1),
        'decima_seconds_per_variant': decima_seconds / max(total_rows, 1),
    }
    source_paths = [options.data_root / p for p in [
        'ho_supp.xlsx', 'bourges_barcode_counts.txt.gz', 'siraj_MPRA.zip',
        'siraj_tables.zip', 'ho_normalized_counts.txt.gz', 'ho_bourges_shared_rsids.txt']]
    source_paths.extend(Path(prov['source']) for prov in avi_prov.values())
    source_paths.extend(Path(prov['summary']) for prov in avi_prov.values())
    _write_audit(output, config, avi_prov, decima_prov, source_paths,
                 wall_time_per_variant, track_prov=track_prov,
                 decima_runtime=decima_runtime,
                 sampling_audit=sampling_audit, config_path=options.config_path)
    scored_impact = {name: record for name, record in feature_impact.items()
                     if record.get('status') != 'skipped'}
    skipped_impact = {name: record for name, record in feature_impact.items()
                      if record.get('status') == 'skipped'}
    feature_impact_summary = summarize_feature_impact(scored_impact)
    for name, record in skipped_impact.items():
        feature_impact_summary[name] = {
            'status': 'skipped',
            'skipped_reason': record['skipped_reason'],
            'rows': record['rows'],
            'independent_loci': record['independent_loci'],
            'folds_requested': record['folds_requested'],
            'enabled_blocks': record['enabled_blocks'],
        }
    evaluation = {
        'held_out': held_evaluations,
        'selected_beta': best_beta,
        'tuning_rows': tuning_rows,
        'wall_time_per_variant': wall_time_per_variant,
        'figure_methods_order': config['evaluation']['four_methods'],
        'feature_impact': feature_impact_summary,
    }
    write_json(output / 'evaluation.json', evaluation)
    log_event(log_path, 'framework_complete')
    return evaluation


# Audit cohort coordinate overlap and locus counts after sampling and AVI filtering.
def _audit_overlap(config, dev_frame, held_frames, output):
    overlap_audit = _cross_dataset_overlap_audit({'siraj': dev_frame, **held_frames},
                                                 int(config['gap_bp']))
    for name, frame in {'siraj': dev_frame, **held_frames}.items():
        groups = locus_groups(frame, gap_bp=int(config['gap_bp']))
        overlap_audit['datasets'][name]['independent_loci_after_grouping'] = int(len(set(groups)))
    write_json(output / 'cross_dataset_overlap_audit.json', overlap_audit)


# Run all framework stages, including all four evaluation methods.
def run_framework(options):
    global _SELECTOR_SEED
    output = prepare_output(
        options.output,
        protected=[options.data_root, options.avi_dir, options.tracks_dir, options.config_path],
        preserve=['prepared', 'decima_scores'],
    )
    options = _options_with_output(options, output)
    config = load_config(options.config_path)
    _SELECTOR_SEED = int(config['seed'])
    log_path = output / 'run_log.jsonl'
    log_event(log_path, 'framework_started', config_sha256=sha256(options.config_path))

    dev_frame, held_frames = _step1_prepare_datasets(options, config)
    all_frames, avi_scores, avi_prov, avi_seconds, _ = _step2_load_avi(
        options, log_path, dev_frame, held_frames)
    all_frames, avi_scores, sampling_audit, total_rows = _step2b_apply_sampling(
        all_frames, options, config, log_path)
    dev_frame = all_frames['development']
    held_frames = {k: v for k, v in all_frames.items() if k != 'development'}
    track_scores, track_prov, tracks_seconds = _step2c_load_tracks(
        options, config, all_frames, log_path)
    _audit_overlap(config, dev_frame, held_frames, output)
    decima_scores, decima_prov, decima_seconds, decima_runtime = _step3_score_decima(
        options, config, all_frames, output, log_path)
    best_beta, tuning_rows = _step4_tune_beta(
        config, dev_frame, avi_scores, decima_scores, track_scores, log_path, output)
    config = {**config, 'rough': {**config['rough'], 'beta': best_beta}}
    feature_impact = _step4b_feature_impact(
        config, all_frames, avi_scores, decima_scores, track_scores,
        held_frames, log_path, output)
    held_evaluations = _step5_held_out_evaluation(
        config, held_frames, avi_scores, decima_scores, track_scores,
        options, output, log_path)
    evaluation = _step6_write_outputs(
        options, config, output, log_path, avi_prov, decima_prov, track_prov,
        sampling_audit, decima_runtime, best_beta, tuning_rows,
        feature_impact, held_evaluations, avi_seconds, tracks_seconds,
        decima_seconds, total_rows)
    return {'output': output, 'evaluation': evaluation}


# Return a copy of options whose ``output`` field is the resolved experiment directory.
def _options_with_output(options, output):
    if options.output == output:
        return options
    from dataclasses import replace
    return replace(options, output=output)



# Public compatibility aliases for workflows created before the project rename.
NeurojicoOptions = FrameworkOptions
run_neurojico = run_framework
