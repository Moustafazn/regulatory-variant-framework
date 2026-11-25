"""Coordinate- and context-checked quantitative annotations for external tracks."""
from pathlib import Path

import numpy as np

from .reporting import sha256, write_json

COORDINATES = ['variant_id', 'chrom', 'position', 'ref', 'alt', 'genome_build']
CONTEXTS = ['source_context', 'target_context']


def bigwig_signal(path, frame, window):
    """Exact mean of covered bases at GRCh38 positions; uncovered windows remain NaN."""
    import pyBigWig
    if not frame.genome_build.eq('GRCh38').all():
        raise ValueError('bigWig queries require verified GRCh38 coordinates.')
    if window <= 0:
        raise ValueError('Window must be positive.')
    with pyBigWig.open(str(path)) as handle:
        lengths = handle.chroms()
        for chrom, length in [('chr1', 248956422), ('chr2', 242193529)]:
            if chrom in lengths and lengths[chrom] != length:
                raise ValueError(f'{path}: {chrom} length is inconsistent with GRCh38.')
        values = np.full(len(frame), np.nan)
        for i, row in enumerate(frame.itertuples(index=False)):
            chrom = str(row.chrom)
            chrom = chrom if chrom in lengths else f'chr{chrom}'
            if chrom not in lengths:
                continue
            start = max(0, int(row.position) - 1 - window // 2)
            end = min(lengths[chrom], start + window)
            if end <= start:
                continue
            value = handle.stats(chrom, start, end, type='mean', exact=True)[0]
            if value is not None:
                values[i] = value
    return values


def build_track_schema(frame, tracks, window, prefix=None):
    """tracks maps each mark to (source path, target path); fitted bins are not exported."""
    out = frame[COORDINATES + CONTEXTS].copy()
    provenance = {}
    for mark, paths in tracks.items():
        for role, path in zip(('src', 'tgt'), paths):
            column = f'{mark}_{role}' if prefix is None else f'{prefix}_{"source" if role == "src" else "target"}'
            print(f'Extracting {column}: {Path(path).name} ({len(frame):,} variants)', flush=True)
            out[column] = bigwig_signal(path, frame, window)
            provenance[column] = {'path': str(path), 'sha256': sha256(path),
                                  'coverage': float(out[column].notna().mean())}
        source = out[f'{mark}_src' if prefix is None else f'{prefix}_source']
        target = out[f'{mark}_tgt' if prefix is None else f'{prefix}_target']
        name = mark if prefix is None else prefix
        out[f'{name}_delta'] = target - source
    return out, provenance


def save_schema(path, frame, coordinate_audit, tracks, window):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    write_json(str(path) + '.json', {
        'schema_version': 2, 'coordinate_audit': coordinate_audit, 'tracks': tracks,
        'contexts': frame[CONTEXTS].drop_duplicates().to_dict('records'),
        'rows': len(frame), 'window_bp': window, 'exact_bigwig_means': True,
        'missing_signal': 'NaN; not zero', 'binning': 'Fit on training folds only, downstream.',
        'output_sha256': sha256(path)})


def join_checked_annotations(frame, annotations):
    """Join after verifying coordinates, alleles, and the assay contexts."""
    if annotations.variant_id.duplicated().any():
        raise ValueError('Duplicate annotation variants.')
    if not frame.variant_id.isin(annotations.variant_id).all():
        raise ValueError('Annotation rows missing for cohort variants; uncovered signals must have explicit NaN values.')
    shared = frame[COORDINATES + CONTEXTS].merge(
        annotations[COORDINATES + CONTEXTS], on='variant_id', suffixes=('', '_annotation'))
    for column in COORDINATES[1:] + CONTEXTS:
        left, right = shared[column], shared[column + '_annotation']
        if column == 'chrom':
            left, right = left.astype(str).str.removeprefix('chr'), right.astype(str).str.removeprefix('chr')
        if not left.eq(right).all():
            raise ValueError(f'Annotation {column} mismatch; regenerate for this cohort.')
    features = [c for c in annotations if c not in COORDINATES + CONTEXTS]
    return frame.merge(annotations[['variant_id'] + features], on='variant_id', how='left', validate='one_to_one')
