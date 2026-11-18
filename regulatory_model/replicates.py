"""Replicate-level measurement quality for the Ho T-cell target, computed from the GEO normalized counts.

Everything here is report-only. The returned columns are prefixed ``rep_`` and are deliberately absent from
``ALLOWED_FEATURES``, ``feature_columns`` and ``MECHANISTIC_FEATURES`` so that no replicate quantity can ever
reach the model as an input. Orientation was inferred from correlations against the published workbook, not
from documentation: tag ``R`` is the reference allele, tag ``A`` is the alternate allele, and ``Log2Skew`` is
alt minus ref. The estimator is log2(x+1), per-donor alt minus ref, with the mean plasmid ratio as reference;
the published pipeline uses MPRAmodel/DESeq2, which this does not reproduce, so these are approximations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PLASMID_PREFIX = 'OLJR.C_Plasmid_'
TCELL_PREFIX = 'OLJR.C_Tcell_'

# Per-variant replicate-quality columns merged into the Ho frame; report-only, never model inputs.
REPLICATE_COLUMNS = ['rep_skew', 'rep_skew_se', 'rep_donor_sign_consistency',
                     'rep_n_donors', 'rep_min_donor_count']


# Split the row identifier chr:pos:ref:alt:A|R:wC into fields and flag the two excluded groups.
def parse_identifiers(index):
    raw = pd.Series([str(value) for value in index]).reset_index(drop=True)
    multi = raw.str.startswith('(')
    parts = raw.str.split(':', expand=True)
    for missing in range(parts.shape[1], 6):
        parts[missing] = None
    frame = pd.DataFrame({'chrom': parts[0], 'position': parts[1], 'ref': parts[2],
                          'alt': parts[3], 'tag': parts[4], 'window': parts[5]})
    frame['multi_variant_window'] = multi.to_numpy()
    allele_na = (frame.ref.astype(str) == 'NA') | (frame.alt.astype(str) == 'NA')
    frame['na_allele'] = allele_na.to_numpy() & ~frame.multi_variant_window
    frame['variant_id'] = (frame.chrom.astype(str) + ':' + frame.position.astype(str) + ':'
                           + frame.ref.astype(str) + ':' + frame.alt.astype(str))
    return frame


# Read the gzipped normalized-count matrix and return it with its plasmid and T-cell donor column blocks.
def load_counts(path):
    counts = pd.read_csv(Path(path), sep='\t', index_col=0)
    plasmid = [c for c in counts.columns if c.startswith(PLASMID_PREFIX)]
    donors = [c for c in counts.columns if c.startswith(TCELL_PREFIX)]
    if not plasmid or not donors:
        raise ValueError(f'Expected {PLASMID_PREFIX}* and {TCELL_PREFIX}* columns; found {list(counts.columns)}')
    return counts, plasmid, donors


# Per-donor log2 activity: log2(donor+1) minus log2(mean plasmid+1).
def _donor_activity(block, plasmid, donors):
    reference = np.log2(block[plasmid].to_numpy(float).mean(axis=1) + 1.0)
    return np.log2(block[donors].to_numpy(float) + 1.0) - reference[:, None]


# Spearman-Brown corrected split-half reliability of the replicate skew, with its spread over random splits.
def split_half_reliability(alt, ref, draws=200, seed=42):
    empty = {'split_half_r_mean': None, 'split_half_r_ci95': [None, None],
             'split_half_r_range': [None, None], 'spearman_brown': None, 'draws': 0}
    n_donors = alt.shape[1]
    if n_donors < 4 or len(alt) < 10:
        return empty
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        order = rng.permutation(n_donors)
        left, right = order[: n_donors // 2], order[n_donors // 2:]
        a = np.nanmean(alt[:, left] - ref[:, left], axis=1)
        b = np.nanmean(alt[:, right] - ref[:, right], axis=1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0:
            values.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    if not values:
        return empty
    mean_r = float(np.mean(values))
    return {'split_half_r_mean': mean_r,
            'split_half_r_ci95': [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            'split_half_r_range': [float(np.min(values)), float(np.max(values))],
            'spearman_brown': float(2 * mean_r / (1 + mean_r)) if mean_r > -1 else None,
            'draws': int(len(values))}


# Build the per-variant replicate-quality table keyed by variant_id plus the exclusion log by reason.
def replicate_quality(path, draws=200, seed=42):
    counts, plasmid, donors = load_counts(path)
    identifiers = parse_identifiers(counts.index)
    excluded = {'input_rows': int(len(counts)),
                'plasmid_replicates': len(plasmid),
                'tcell_donors': len(donors),
                'exact_zero_cells': int((counts.to_numpy(float) == 0).sum()),
                'excluded_multi_variant_window_rows': int(identifiers.multi_variant_window.sum()),
                'excluded_na_allele_rows': int(identifiers.na_allele.sum())}
    keep = (~identifiers.multi_variant_window & ~identifiers.na_allele).to_numpy()
    identifiers = identifiers.loc[keep].reset_index(drop=True)
    block = counts.loc[keep].reset_index(drop=True)
    activity = _donor_activity(block, plasmid, donors)
    donor_counts = block[donors].to_numpy(float)
    identifiers['_row'] = np.arange(len(identifiers))
    alt_rows = identifiers[identifiers.tag == 'A'].drop_duplicates('variant_id').set_index('variant_id')
    ref_rows = identifiers[identifiers.tag == 'R'].drop_duplicates('variant_id').set_index('variant_id')
    shared = alt_rows.index.intersection(ref_rows.index)
    excluded['rows_after_exclusions'] = int(len(identifiers))
    excluded['variants_missing_one_allele'] = int(len(set(alt_rows.index) ^ set(ref_rows.index)))
    alt_index = alt_rows.loc[shared, '_row'].to_numpy()
    ref_index = ref_rows.loc[shared, '_row'].to_numpy()
    alt, ref = activity[alt_index], activity[ref_index]
    per_donor = alt - ref
    n_donors = np.isfinite(per_donor).sum(axis=1)
    skew = np.nanmean(per_donor, axis=1)
    spread = np.nanstd(per_donor, axis=1, ddof=1)
    positive = (per_donor > 0).sum(axis=1)
    table = pd.DataFrame({
        'variant_id': list(shared),
        'rep_skew': skew,
        'rep_skew_se': spread / np.sqrt(np.maximum(n_donors, 1)),
        'rep_donor_sign_consistency': np.maximum(positive, n_donors - positive) / np.maximum(n_donors, 1),
        'rep_n_donors': n_donors.astype(float),
        'rep_min_donor_count': np.minimum(donor_counts[alt_index].min(axis=1),
                                          donor_counts[ref_index].min(axis=1)),
    }).reset_index(drop=True)
    excluded['variants_with_both_alleles'] = int(len(table))
    provenance = {'exclusions': excluded,
                  'split_half': split_half_reliability(alt, ref, draws=draws, seed=seed),
                  'method': 'log2(x+1); per-donor alt minus ref; mean plasmid ratio as the reference',
                  'orientation': 'tag R = reference allele; tag A = alternate allele; skew = alt minus ref',
                  'caveat': ('Approximation only: the published pipeline uses MPRAmodel and DESeq2, '
                             'which this estimator does not reproduce.'),
                  'usage': 'Report-only. Never a model input; columns are excluded from feature_columns.'}
    return table, provenance
