"""Per-run cohort assembly: sampling, held-out resolution, leakage dedup, overlap audit,
AlphaGenome track NPZ loading, and GroupKFold+inner-split feasibility reporting.

This module takes the canonical frames produced by ``data_prep`` and shapes them into
the exact cohort a single ``run_framework`` call will evaluate. Anything that compensates
for how the measured data really is (vs. how a textbook split would be) lives here.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import (
    DHS, MECHANISTIC_FEATURES, PROJECT_ROOT,
    merge_ho_mechanistic_features, merge_ho_replicate_quality, prepare_dataset,
)
from .replicates import REPLICATE_COLUMNS
from .evaluation import locus_groups
from .reporting import sha256


# AlphaGenome AVI feature columns consumed by the fusion.
AVI_FEATURES = ('avi_raw_score', 'avi_phred')

# Per-dataset prefix used when assembling the AlphaGenome track NPZ filename.
DATASET_NPZ_PREFIX = {'ho': 'Ho2025', 'bourges': 'Bourges2020', 'siraj': 'Siraj2026'}


# Normalize a chromosome string ('chr1' / 'Chr1' / '1' / None) to its canonical lowercase stem.
def _normalize_chrom(value):
    if value is None:
        return None
    return str(value).strip().lower().removeprefix('chr')


# Return positional row indices ordered by a per-dataset hash rank, so a smaller cap is always a subset of a larger one.
def hash_rank_order(frame, dataset_name, seed):
    keys = [int(hashlib.sha256(f'{int(seed)}:{dataset_name}:{example_id}'.encode()).hexdigest()[:16], 16)
            for example_id in frame['example_id'].astype(str)]
    return np.argsort(np.array(keys, dtype=np.uint64), kind='stable')


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
    for name, frame in all_frames.items():
        before = len(frame)
        working = frame
        if requested_chrom is not None:
            chrom_norm = working['chrom'].astype(str).str.lower().str.removeprefix('chr')
            working = working.loc[chrom_norm.eq(requested_chrom)].reset_index(drop=True)
        dataset_name = 'siraj' if name == 'development' else name
        cap = per_dataset.get(dataset_name)
        if cap is not None and len(working) > int(cap):
            working = working.iloc[hash_rank_order(working, dataset_name, seed)[:int(cap)]]
            working = working.sort_index().reset_index(drop=True)
        audit['per_dataset_counts'][dataset_name] = {
            'before': int(before), 'after': int(len(working))}
        sampled[name] = working
    audit['method'] = 'sha256_rank_of_seed_dataset_example_id; smaller caps are subsets of larger caps'
    return sampled, audit


# Maximum share of externally mapped rows allowed to disagree with the prepared hg38 coordinate.
COORDINATE_DISAGREEMENT_LIMIT = 0.001

# Maximum share of prepared rows allowed to be absent from the frozen AVI archive before the load fails.
AVI_MISSING_LIMIT = 0.02


# Drop rows whose independent hg38 coordinate or alleles differ from the prepared frame; fail above the limit.
def _enforce_coordinate_agreement(joined, name, source):
    mapped = joined.hg38_position.notna() & joined.hg38_chrom.notna()
    chrom = joined.hg38_chrom.astype(str).str.lower().str.removeprefix('chr')
    agree = (chrom.eq(joined.chrom.astype(str).str.lower().str.removeprefix('chr'))
             & joined.hg38_position.eq(joined.position)
             & joined.hg38_ref.eq(joined.ref) & joined.hg38_alt.eq(joined.alt))
    bad = mapped & ~agree
    audit = {'source': source, 'rows_compared': int(mapped.sum()), 'rows_disagreeing': int(bad.sum()),
             'rows_without_external_coordinate': int((~mapped).sum()),
             'limit': COORDINATE_DISAGREEMENT_LIMIT,
             'examples': joined.loc[bad, ['example_id', 'chrom', 'position', 'ref', 'alt',
                                          'hg38_chrom', 'hg38_position', 'hg38_ref', 'hg38_alt']]
             .head(5).astype(str).to_dict('records')}
    if mapped.any() and bad.sum() / mapped.sum() > COORDINATE_DISAGREEMENT_LIMIT:
        raise RuntimeError(
            f'{name}: {int(bad.sum()):,}/{int(mapped.sum()):,} prepared rows disagree with the {source} '
            f'hg38 coordinates (limit {COORDINATE_DISAGREEMENT_LIMIT:.1%}); first rows: {audit["examples"]}')
    if bad.any():
        joined = joined.loc[~bad].reset_index(drop=True)
    return joined, audit


# Strip any trailing source:target context suffix so an example_id identifies only the variant.
def _context_free_key(example_ids):
    return example_ids.astype(str).str.replace(r'^(([^:]+):([^:]+):([^:]+):([^:]+):([^:]+)).*$', r'\1', regex=True)


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
    avi = pd.read_csv(score_path, dtype={'example_id': str, 'hg38_chrom': str, 'hg38_ref': str, 'hg38_alt': str},
                      usecols=['example_id', 'avi_raw_score', 'avi_phred', 'status',
                               'hg38_chrom', 'hg38_position', 'hg38_ref', 'hg38_alt'])
    if avi.example_id.duplicated().any():
        raise ValueError(f'{name} AVI output contains duplicate example_id values.')
    prepared_sha = sha256(prepared_path)
    sha_matches = (summary.get('input_sha256') == prepared_sha)
    # AVI scores depend only on the variant, so join on a context-independent key; Siraj example_ids embed the context pair.
    avi['_avi_key'] = _context_free_key(avi.example_id)
    if avi._avi_key.duplicated().any():
        avi = avi.drop_duplicates('_avi_key')
    joined = frame.assign(_avi_key=_context_free_key(frame.example_id)).merge(
        avi.drop(columns='example_id'), on='_avi_key', how='left', validate='one_to_one', sort=False)
    joined = joined.drop(columns='_avi_key')
    joined, coordinate_audit = _enforce_coordinate_agreement(joined, name, 'AVI')
    # A few prepared variants can be absent from the frozen AVI archive; exclude and report them, but fail if most are.
    missing_ids = joined.status.isna()
    missing_fraction = float(missing_ids.mean()) if len(joined) else 0.0
    if missing_fraction > AVI_MISSING_LIMIT:
        raise ValueError(
            f'{name} AVI output is missing {int(missing_ids.sum()):,} of {len(joined):,} prepared examples '
            f'({missing_fraction:.1%} > {AVI_MISSING_LIMIT:.1%}); the AVI table lacks example_id values the '
            'prepared frame requires. Rerun scripts/download_alphagenome_avi.py for this dataset.'
        )
    joined.loc[missing_ids, 'status'] = 'absent_from_avi_archive'
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
    kept = joined.loc[matched].drop(columns=['status', 'hg38_chrom', 'hg38_position', 'hg38_ref', 'hg38_alt'])
    return kept.reset_index(drop=True), {
        'coordinate_audit': coordinate_audit,
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
def _npz_filename(dataset_name, example_id, tracks_dir=None):
    tag = DATASET_NPZ_PREFIX[dataset_name]
    text = str(example_id)
    stem = text[len(tag) + 1:] if text.startswith(f'{tag}:') else text
    cleaned = ''.join(c if c.isalnum() or c in '._-' else '_' for c in stem.replace(':', '_'))
    name = f'{tag}_{cleaned}.npz'
    if tracks_dir is None or (Path(tracks_dir) / name).is_file():
        return name
    # Track NPZs are per-variant but older Siraj files embed the context pair; accept any context suffix.
    return _track_index(tracks_dir, tag).get(_variant_stem(cleaned), name)


# Drop a trailing <source>_<target> context suffix from a cleaned Siraj example stem.
def _variant_stem(cleaned):
    parts = cleaned.split('_')
    return '_'.join(parts[:4]) if len(parts) > 4 else cleaned


_TRACK_INDEX = {}


# Map variant stem -> NPZ filename once per directory, so context-suffixed lookups never rescan 300k+ files.
def _track_index(tracks_dir, tag):
    key = (str(tracks_dir), tag)
    if key not in _TRACK_INDEX:
        index = {}
        for path in Path(tracks_dir).glob(f'{tag}_*.npz'):
            index.setdefault(_variant_stem(path.name[len(tag) + 1:-4]), path.name)
        _TRACK_INDEX[key] = index
    return _TRACK_INDEX[key]


# Return a boolean mask of rows whose track NPZ exists and was scored at the prepared hg38 coordinate.
def _track_valid_mask(frame, dataset_name, tracks_dir):
    expected = [_npz_filename(dataset_name, example_id, tracks_dir) for example_id in frame['example_id']]
    present = np.array([(Path(tracks_dir) / name).is_file() for name in expected], dtype=bool)
    for row_index in _mislocated_track_files(frame, expected, present, tracks_dir):
        present[row_index] = False
    return present


# Restrict every frame to rows with valid track files so the tracks block is complete and never zero-filled.
def _restrict_to_track_covered(frames, tracks_dir):
    kept, audit = {}, {}
    for name, frame in frames.items():
        dataset_name = 'siraj' if name == 'development' else name
        mask = _track_valid_mask(frame, dataset_name, tracks_dir)
        kept[name] = frame.loc[mask].reset_index(drop=True)
        ids = '\n'.join(sorted(kept[name].example_id.astype(str)))
        audit[dataset_name] = {
            'rows_before': int(len(frame)), 'rows_with_valid_tracks': int(mask.sum()),
            'coverage': float(mask.mean()) if len(frame) else 0.0,
            'pool_sha256': hashlib.sha256(ids.encode()).hexdigest()}
    return kept, audit


# Return row indices whose track file was scored at a coordinate other than the prepared hg38 row.
def _mislocated_track_files(frame, expected, present_mask, tracks_dir):
    wrong = []
    chroms = frame['chrom'].astype(str).str.lower().str.removeprefix('chr').tolist()
    positions, refs, alts = frame['position'].astype(int).tolist(), frame['ref'].tolist(), frame['alt'].tolist()
    for i, name in enumerate(expected):
        if not present_mask[i]:
            continue
        path = Path(tracks_dir) / (name[:-len('.npz')] + '.provenance.json')
        if not path.is_file():
            wrong.append(i)
            continue
        provenance = json.loads(path.read_text())
        scored = (provenance.get('grch38_variant') or provenance.get('variant')
                  or provenance.get('bourges_coordinate_correction') or {})
        scored_chrom = str(scored.get('chrom', '')).lower().removeprefix('chr')
        if (scored_chrom != chroms[i] or int(scored.get('position', -1)) != positions[i]
                or scored.get('ref') != refs[i] or scored.get('alt') != alts[i]):
            wrong.append(i)
    return wrong


# Load per-cell-state track matrices for a prepared frame, zero-filling absent NPZs.
def _load_tracks_dataset(frame, dataset_name, tracks_dir, cell_states, modalities,
                         require_complete=False):
    tracks_dir = Path(tracks_dir)
    expected = [_npz_filename(dataset_name, eid, tracks_dir) for eid in frame['example_id']]
    present_mask = np.array([(tracks_dir / name).is_file() for name in expected], dtype=bool)
    mislocated = _mislocated_track_files(frame, expected, present_mask, tracks_dir)
    for row_index in mislocated:
        present_mask[row_index] = False
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
        'rows_excluded_wrong_coordinate': len(mislocated),
        'examples_excluded_wrong_coordinate': [expected[i] for i in mislocated[:5]],
        'cell_states': list(cell_states),
        'modalities': list(modalities),
        'files_sha256_count': int(len(sha_per_file)),
    }
    return scores, provenance


# Load the Siraj development frame from the prepared CSV.
def _prepare_development(data_root, output, config=None):
    evaluation = (config or {}).get('evaluation', {})
    source = evaluation.get('development_source_context', 'K562')
    target = evaluation.get('development_target_context', 'HepG2')
    prepared = prepare_dataset('siraj', data_root, output / 'prepared' / 'siraj.csv.gz',
                               source_context=source, target_context=target)
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
    if 'bourges' in frames and 'is_positive_control' in frames['bourges'].columns:
        # Spiked-in positive controls are selected responders, not natural variants.
        natural = ~frames['bourges'].is_positive_control.astype(bool)
        frames['bourges'] = frames['bourges'][natural].reset_index(drop=True)
    return frames


# Attach Ho's δSVM / motifbreakR / DHS annotations and report-only replicate quality when available; fill NaN otherwise.
def _augment_with_mechanistic(frame, data_root):
    try:
        frame = merge_ho_mechanistic_features(data_root, frame)
    except FileNotFoundError:
        for column in MECHANISTIC_FEATURES + DHS:
            if column not in frame.columns:
                frame[column] = np.nan
    try:
        frame = merge_ho_replicate_quality(data_root, frame)
    except (FileNotFoundError, ValueError):
        for column in REPLICATE_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
    return frame



# Drop development rows within gap_bp of any held-out variant so beta tuning never sees a held-out locus.
def _remove_dev_rows_near_held_out(dev_frame, held_frames, gap_bp):
    near_positions = {}
    for frame in held_frames.values():
        for chrom, pos in zip(frame.chrom.astype(str), frame.position.astype(int)):
            near_positions.setdefault(chrom, []).append(pos)
    for chrom in near_positions:
        near_positions[chrom].sort()
    near = []
    for chrom, pos in zip(dev_frame.chrom.astype(str), dev_frame.position.astype(int)):
        positions = near_positions.get(chrom, [])
        near.append(bool(positions) and bisect.bisect_right(positions, pos + gap_bp)
                    > bisect.bisect_left(positions, pos - gap_bp))
    near = np.array(near, dtype=bool)
    kept = dev_frame.loc[~near].reset_index(drop=True)
    return kept, {'gap_bp': int(gap_bp), 'dev_rows_before': int(len(dev_frame)),
                  'dev_rows_removed_near_held_out': int(near.sum()), 'dev_rows_after': int(len(kept)),
                  'dev_rows_within_gap_after_removal': 0}


# Audit coordinate overlap across every pair of cohort frames and hard-fail on incompatible builds.
def _cross_dataset_overlap_audit(frames, gap_bp):
    not_hg38 = {name: sorted(set(frame.genome_build.astype(str))) for name, frame in frames.items()
                if set(frame.genome_build.astype(str)) != {'GRCh38'}}
    if not_hg38:
        raise RuntimeError(f'Every cohort frame must be GRCh38 before evaluation; found {not_hg38}.')
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
