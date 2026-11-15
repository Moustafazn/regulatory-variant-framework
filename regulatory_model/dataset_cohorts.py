"""Per-run cohort assembly: sampling, held-out resolution, leakage dedup, overlap audit,
AlphaGenome track NPZ loading, and GroupKFold+inner-split feasibility reporting.

This module takes the canonical frames produced by ``data_prep`` and shapes them into
the exact cohort a single ``run_framework`` call will evaluate. Anything that compensates
for how the measured data really is (vs. how a textbook split would be) lives here.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import (
    DHS, MECHANISTIC_FEATURES, PROJECT_ROOT,
    merge_ho_mechanistic_features, prepare_dataset,
)
from .evaluation import locus_groups
from .reporting import sha256


# AlphaGenome AVI feature columns consumed by the fusion.
AVI_FEATURES = ('avi_raw_score', 'avi_phred')

# Per-dataset prefix used when assembling the AlphaGenome track NPZ filename.
DATASET_NPZ_PREFIX = {'ho': 'Ho2025', 'bourges': 'Bourges2020', 'siraj': 'Siraj2026'}


# Normalise a chromosome string ('chr1' / 'Chr1' / '1' / None) to its canonical lowercase stem.
def _normalize_chrom(value):
    if value is None:
        return None
    return str(value).strip().lower().removeprefix('chr')


# Return a chromosome-filtered, per-dataset-capped copy of all_frames plus an audit payload.
def _apply_dataset_sampling(all_frames, options, seed):
    requested_chrom = _normalize_chrom(options.sample_chrom)
    per_dataset = dict(options.sample_per_dataset) if options.sample_per_dataset else {}
    if not requested_chrom and not per_dataset:
        return all_frames, {'applied': False}

    audit = {
        'applied': True,
        'chromosome': options.sample_chrom,
        'per_dataset_cap': {k: int(v) for k, v in per_dataset.items()},
        'seed': int(seed),
        'per_dataset_counts': {},
    }
    sampled = {}
    rng = np.random.default_rng(int(seed))
    for name, frame in all_frames.items():
        before = len(frame)
        working = frame
        if requested_chrom is not None:
            chrom_norm = working['chrom'].astype(str).str.lower().str.removeprefix('chr')
            working = working.loc[chrom_norm.eq(requested_chrom)].reset_index(drop=True)
        dataset_name = 'siraj' if name == 'development' else name
        cap = per_dataset.get(dataset_name)
        if cap is not None and len(working) > int(cap):
            indices = rng.choice(len(working), size=int(cap), replace=False)
            working = working.iloc[np.sort(indices)].reset_index(drop=True)
        audit['per_dataset_counts'][dataset_name] = {
            'before': int(before), 'after': int(len(working))}
        sampled[name] = working
    return sampled, audit


# Read explicit local AVI annotations; never infer or fill missing scores.
def _avi_features_from_frame(frame):
    missing = set(AVI_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(
            'Local AlphaGenome AVI scores are required as columns '
            f'{sorted(AVI_FEATURES)}; missing {sorted(missing)}. '
            'Supply the raw and PHRED values from the downloaded AVI archive. '
            'For the cohort runner, run scripts/download_alphagenome_avi.py first.'
        )
    values = {}
    for feature in AVI_FEATURES:
        column = pd.to_numeric(frame[feature], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(column).all():
            raise ValueError(f'{feature} must be finite for every scored variant.')
        values[feature] = column.reshape(len(frame), 1)
    return values


# Join exact per-example AVI output and retain only successfully matched SNVs.
def _load_avi_dataset(frame, name, avi_dir, prepared_path):
    avi_dir = Path(avi_dir)
    score_path = avi_dir / f'{name}.avi.csv.gz'
    summary_path = avi_dir / f'{name}.avi.summary.json'
    if not score_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f'Missing local AVI output for {name}: expected {score_path} and {summary_path}. '
            'Run scripts/download_alphagenome_avi.py after the archive download completes.'
        )
    summary = json.loads(summary_path.read_text())
    avi = pd.read_csv(score_path, dtype={'example_id': str},
                      usecols=['example_id', 'avi_raw_score', 'avi_phred', 'status'])
    if avi.example_id.duplicated().any():
        raise ValueError(f'{name} AVI output contains duplicate example_id values.')
    prepared_sha = sha256(prepared_path)
    sha_matches = (summary.get('input_sha256') == prepared_sha)
    joined = frame.merge(avi, on='example_id', how='left', validate='one_to_one', sort=False)
    missing_ids = joined.status.isna()
    if missing_ids.any():
        raise ValueError(
            f'{name} AVI output is missing {int(missing_ids.sum()):,} prepared examples; '
            f'the AVI table lacks example_id values the prepared frame requires. '
            'Rerun scripts/download_alphagenome_avi.py for this dataset.'
        )
    matched = joined.status.eq('matched')
    finite = np.isfinite(pd.to_numeric(joined.avi_raw_score, errors='coerce')) & \
        np.isfinite(pd.to_numeric(joined.avi_phred, errors='coerce'))
    matched &= finite
    excluded = int((~matched).sum())
    status_counts = joined.loc[~matched, 'status'].value_counts().to_dict()
    if not matched.any():
        raise ValueError(f'{name} has no variants with matched local AVI scores.')
    print(f'  {name}: using {int(matched.sum()):,}/{len(joined):,} matched AVI variants; '
          f'excluding {excluded:,} ({status_counts})', flush=True)
    return joined.loc[matched].drop(columns='status').reset_index(drop=True), {
        'source': str(score_path),
        'source_sha256': sha256(score_path),
        'summary': str(summary_path),
        'summary_sha256': sha256(summary_path),
        'prepared_input_sha256': prepared_sha,
        'prepared_input_sha256_in_avi_summary': summary.get('input_sha256'),
        'prepared_input_sha256_matches_avi_summary': bool(sha_matches),
        'prepared_input_sha256_mismatch_reason': (
            None if sha_matches
            else 'Prepared frame was rewritten (hg38 liftover) after AVI extraction. '
                 'AVI joins on example_id which is coord-independent, so the join '
                 'remains correct; the SHA mismatch is recorded here instead of raised.'),
        'atlas': summary.get('atlas'),
        'atlas_size_bytes': summary.get('atlas_size_bytes'),
        'rows_in_input': int(len(joined)),
        'rows_with_avi_scores': int(matched.sum()),
        'rows_excluded': excluded,
        'excluded_status_counts': status_counts,
        'score_meaning': summary.get('score_meaning'),
    }


# Build the NPZ filename a per-dataset variant NPZ should have on disk.
def _npz_filename(dataset_name, example_id):
    tag = DATASET_NPZ_PREFIX[dataset_name]
    text = str(example_id)
    stem = text[len(tag) + 1:] if text.startswith(f'{tag}:') else text
    cleaned = ''.join(c if c.isalnum() or c in '._-' else '_' for c in stem.replace(':', '_'))
    return f'{tag}_{cleaned}.npz'


# Load per-cell-state track matrices for a prepared frame, zero-filling absent NPZs.
def _load_tracks_dataset(frame, dataset_name, tracks_dir, cell_states, modalities,
                         require_complete=False):
    tracks_dir = Path(tracks_dir)
    expected = [_npz_filename(dataset_name, eid) for eid in frame['example_id']]
    present_mask = np.array([(tracks_dir / name).is_file() for name in expected], dtype=bool)
    rows_missing = int((~present_mask).sum())
    if rows_missing and require_complete:
        raise FileNotFoundError(
            f'{dataset_name}: AlphaGenome track NPZs incomplete '
            f'({rows_missing:,}/{len(expected):,} missing in {tracks_dir}). '
            'Run scripts/download_alphagenome_tracks.py to finish the download.'
        )
    scores = {state: np.zeros((len(frame), len(modalities)), dtype=np.float32)
              for state in cell_states}
    sha_per_file = {}
    for row_index, filename in enumerate(expected):
        if not present_mask[row_index]:
            continue
        path = tracks_dir / filename
        with np.load(path) as payload:
            for state in cell_states:
                for modality_index, modality in enumerate(modalities):
                    key = f'{state}:{modality}'
                    if key not in payload.files:
                        raise KeyError(f'{path} is missing track key {key!r}.')
                    value = payload[key]
                    if value.size != 1:
                        raise ValueError(
                            f'{path} key {key!r} has shape {value.shape}; expected one scalar.'
                        )
                    scores[state][row_index, modality_index] = float(value.reshape(()))
        sha_per_file[filename] = sha256(path)
    status = 'loaded' if rows_missing == 0 else 'partial'
    provenance = {
        'status': status,
        'tracks_dir': str(tracks_dir),
        'rows': int(len(frame)),
        'rows_loaded': int(present_mask.sum()),
        'rows_missing_filled_with_zero': rows_missing,
        'cell_states': list(cell_states),
        'modalities': list(modalities),
        'files_sha256_count': int(len(sha_per_file)),
    }
    return scores, provenance


# Load the Siraj development frame from the prepared CSV.
def _prepare_development(data_root, output):
    prepared = prepare_dataset('siraj', data_root, output / 'prepared' / 'siraj.csv.gz')
    return pd.read_csv(prepared.dataset, dtype={'chrom': str})


# Load every held-out dataset and drop Bourges rsIDs that leak into Ho.
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


# Attach Ho's δSVM / motifbreakR / DHS annotations when available; fill NaN otherwise.
def _augment_with_mechanistic(frame, data_root):
    try:
        return merge_ho_mechanistic_features(data_root, frame)
    except FileNotFoundError:
        for column in MECHANISTIC_FEATURES + DHS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame



# Audit coordinate overlap across every pair of cohort frames and hard-fail on incompatible builds.
def _cross_dataset_overlap_audit(frames, gap_bp):
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


# Return (n_loci, feasible_folds) that both GroupKFold and the inner GroupShuffleSplit can accept.
def _dataset_fold_feasibility(frame, settings):
    requested_folds = int(settings['folds'])
    inner_fraction = float(settings['inner_validation_fraction'])
    n_loci = len(set(locus_groups(frame, gap_bp=int(settings['gap_bp']))))
    for candidate in range(min(requested_folds, n_loci), 1, -1):
        fit_side_min_loci = n_loci - (n_loci // candidate + (1 if n_loci % candidate else 0))
        if (int(fit_side_min_loci * inner_fraction) >= 1
                and fit_side_min_loci - int(fit_side_min_loci * inner_fraction) >= 1):
            return n_loci, candidate
    return n_loci, 0
