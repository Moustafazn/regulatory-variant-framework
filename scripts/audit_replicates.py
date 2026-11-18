"""Read-only audit of Ho replicate measurement quality against the published effects.

    nice -n 19 venv/bin/python scripts/audit_replicates.py

Recomputes per-variant skew, SE, donor sign consistency and split-half reliability from the GEO normalized
counts, compares them with the published Log2Skew / Skew_SE, and reports reliability ceilings with
locus-bootstrap CIs. Nothing here is a model input; it writes output/diagnostics/replicate_reliability.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402

from regulatory_model.evaluation import locus_groups  # noqa: E402
from regulatory_model.replicates import replicate_quality  # noqa: E402


# Yield locus-bootstrap index arrays so every interval respects the grouped design.
def loci_resamples(groups, draws, seed):
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    for _ in range(draws):
        yield np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])


# Pearson correlation of two vectors with a locus-bootstrap 95% interval.
def correlation_ci(x, y, groups, draws, seed):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50:
        return None
    x, y, g = x[ok], y[ok], groups[ok]
    point = float(np.corrcoef(x, y)[0, 1])
    values = [float(np.corrcoef(x[i], y[i])[0, 1]) for i in loci_resamples(g, draws, seed)
              if x[i].std() > 0 and y[i].std() > 0]
    return {'r': round(point, 4), 'n': int(ok.sum()),
            'ci95': [round(float(np.quantile(values, 0.025)), 4),
                     round(float(np.quantile(values, 0.975)), 4)] if values else [None, None]}


# Ordinary-least-squares slope of y on x, reported alongside the agreement correlation.
def slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50 or x[ok].std() == 0:
        return None
    return round(float(np.polyfit(x[ok], y[ok], 1)[0]), 4)


# Reliability implied by a set of standard errors: 1 - mean(SE^2) / var(effect), with a locus-bootstrap CI.
def reliability_ci(effect, se, groups, draws, seed):
    ok = np.isfinite(effect) & np.isfinite(se) & (se > 0)
    if ok.sum() < 50:
        return None
    effect, se, g = effect[ok], se[ok], groups[ok]

    def point(e, s):
        variance = float(np.var(e))
        return None if variance <= 0 else float(1.0 - np.mean(s ** 2) / variance)

    values = [point(effect[i], se[i]) for i in loci_resamples(g, draws, seed)]
    values = [v for v in values if v is not None]
    return {'reliability': round(point(effect, se), 4), 'n': int(ok.sum()),
            'ci95': [round(float(np.quantile(values, 0.025)), 4),
                     round(float(np.quantile(values, 0.975)), 4)] if values else [None, None]}


# Load the published Ho sheet-1 effects keyed by SNP19.
def published_effects(workbook_path):
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    rows = workbook['1. Tcell MPRA results'].values
    sheet = pd.DataFrame(rows, columns=[str(c) for c in next(rows)])
    workbook.close()
    sheet = sheet.drop_duplicates('SNP19').set_index('SNP19')
    for column in ('Log2Skew', 'Skew_SE', 'A_log2FC', 'B_log2FC', 'Skew_logFDR'):
        if column in sheet.columns:
            sheet[column] = pd.to_numeric(sheet[column], errors='coerce')
    return sheet


# Run every replicate check and write the JSON.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path,
                        default=ROOT / 'data' / 'regulatory_audit' / 'ho_normalized_counts.txt.gz')
    parser.add_argument('--workbook', type=Path,
                        default=ROOT / 'data' / 'regulatory_audit' / 'ho_supp.xlsx')
    parser.add_argument('--prepared', type=Path,
                        default=ROOT / 'output' / 'diagnostics' / 'vprs_inputs' / 'prepared' / 'ho.csv.gz')
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--result', type=Path,
                        default=ROOT / 'output' / 'diagnostics' / 'replicate_reliability.json')
    args = parser.parse_args(argv)

    print('Computing replicate quality from the GEO normalized counts', flush=True)
    table, provenance = replicate_quality(args.counts, draws=args.draws)
    print(f'  {len(table):,} variants with both alleles; exclusions {provenance["exclusions"]}', flush=True)

    published = published_effects(args.workbook)
    merged = table.set_index('variant_id').join(published, how='inner')
    print(f'  {len(merged):,} of {len(published):,} workbook variants matched', flush=True)

    frame = merged.reset_index().rename(columns={'index': 'variant_id'})
    parts = frame.variant_id.str.split(':', expand=True)
    frame['chrom'], frame['position'] = parts[0], pd.to_numeric(parts[1], errors='coerce')
    frame['genome_build'] = 'GRCh37'
    frame = frame.dropna(subset=['position']).reset_index(drop=True)
    groups = locus_groups(frame, gap_bp=250_000)

    rep_skew = frame.rep_skew.to_numpy(float)
    rep_se = frame.rep_skew_se.to_numpy(float)
    pub_skew = frame.Log2Skew.to_numpy(float)
    pub_se = frame.Skew_SE.to_numpy(float)

    agreement = {
        'replicate_skew_vs_published_Log2Skew': correlation_ci(rep_skew, pub_skew, groups, args.draws, 42),
        'slope_published_on_replicate': slope(rep_skew, pub_skew),
        'log_se_agreement': correlation_ci(np.log(np.clip(rep_se, 1e-9, None)),
                                           np.log(np.clip(pub_se, 1e-9, None)), groups, args.draws, 42),
        'median_replicate_se': round(float(np.nanmedian(rep_se)), 4),
        'median_published_se': round(float(np.nanmedian(pub_se)), 4),
    }
    ceilings = {
        'from_published_se': reliability_ci(pub_skew, pub_se, groups, args.draws, 42),
        'from_replicate_se': reliability_ci(rep_skew, rep_se, groups, args.draws, 42),
        'split_half_spearman_brown': provenance['split_half'],
    }
    consistency = frame.rep_donor_sign_consistency.to_numpy(float)
    distribution = {
        'mean': round(float(np.nanmean(consistency)), 4),
        'quantiles': {str(q): round(float(np.nanquantile(consistency, q)), 4)
                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        'share_all_donors_agree': round(float(np.nanmean(consistency >= 0.999)), 4),
        'share_at_least_six_of_seven': round(float(np.nanmean(consistency >= 6 / 7)), 4),
    }

    changed = np.isfinite(frame.Skew_logFDR.to_numpy(float)) & (frame.Skew_logFDR.to_numpy(float) >= 1)
    supported = changed & (np.abs(rep_skew) > 2 * rep_se)
    label_support = {
        'published_significant_variants': int(changed.sum()),
        'replicate_supported_share': round(float(supported.sum() / max(changed.sum(), 1)), 4),
        'definition': 'replicate-supported = |replicate skew| exceeds two replicate SEs',
    }

    unmatched = {'workbook_variants': int(len(published)),
                 'matched_to_replicates': int(len(merged)),
                 'workbook_without_replicates': int(len(published) - len(merged))}
    if args.prepared.is_file():
        prepared = pd.read_csv(args.prepared, dtype={'chrom': str}, low_memory=False)
        covered = prepared.variant_id.astype(str).isin(set(table.variant_id))
        unmatched['prepared_rows'] = int(len(prepared))
        unmatched['prepared_rows_with_replicates'] = int(covered.sum())

    result = {'provenance': provenance, 'agreement_with_published': agreement,
              'reliability_ceilings': ceilings, 'donor_sign_consistency': distribution,
              'label_support': label_support, 'coverage': unmatched,
              'usage': 'Report-only diagnostics; no rep_ column is ever a model input.'}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=1, default=float))

    print('\n=== Agreement with the published effect ===')
    print(json.dumps(agreement, indent=1, default=float))
    print('\n=== Reliability ceilings ===')
    print(json.dumps(ceilings, indent=1, default=float))
    print('\n=== Donor sign consistency ===')
    print(json.dumps(distribution, indent=1, default=float))
    print(f'\nSaved {args.result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
