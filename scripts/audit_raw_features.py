"""Read-only audit of every raw column in the Ho workbook as a candidate model feature.

    nice -n 19 venv/bin/python scripts/audit_raw_features.py

Step 1 lists every sheet with its columns, key and variant coverage. Step 2 tests each usable column against
outcomes that were never inputs (T-cell activity, emVar status, skew direction, effect size) with locus-bootstrap
intervals. Every column is tagged SOURCE (Jurkat, allowed), INDEPENDENT (allowed), TARGET (outcome only) or
REVIEW (name looks target-derived). Nothing is added to the model here; it writes output/diagnostics/raw_feature_audit.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from regulatory_model.evaluation import locus_groups  # noqa: E402

REVIEW_PATTERN = re.compile(r'(?i)(mpra|skew|log2fc|emvar|pcre|expression|activity|tcell_|t_cell|bonf|fdr|_bf)')
SELECTED_ON_OUTCOME = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}


# Read every sheet of the workbook that has a header row; unnamed columns get positional names.
def read_sheets(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = {}
    for name in workbook.sheetnames:
        rows = workbook[name].values
        header = next(rows, None)
        if header is None or sum(h is not None for h in header) < 2:
            continue
        columns = [str(h) if h is not None else f'col{i}' for i, h in enumerate(header)]
        sheets[name] = pd.DataFrame(rows, columns=columns)
    workbook.close()
    return sheets


# Return the numeric prefix of a sheet name such as "4. MPRA Functional Annotations".
def sheet_number(name):
    match = re.match(r'\s*(\d+)', name)
    return int(match.group(1)) if match else None


# Yield locus-bootstrap index arrays so intervals respect the grouped design.
def loci_resamples(groups, draws, seed):
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    for _ in range(draws):
        yield np.concatenate([members[i] for i in rng.integers(0, len(labels), len(labels))])


# AUROC of x for a binary outcome on finite rows, with a locus-bootstrap interval when the effect is non-trivial.
def auroc_ci(x, y, groups, draws, seed):
    ok = np.isfinite(x)
    x, y, g = x[ok], y[ok], groups[ok]
    if y.sum() < 20 or (~y).sum() < 20 or np.ptp(x) == 0:
        return None
    point = float(roc_auc_score(y, x))
    low = high = None
    if abs(point - 0.5) >= 0.02:
        values = [roc_auc_score(y[i], x[i]) for i in loci_resamples(g, draws, seed) if 0 < y[i].sum() < len(i)]
        low, high = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    return point, low, high, int(ok.sum())


# Rank correlation of x with a continuous outcome (ranks fixed on the full sample), with a locus-bootstrap interval.
def rho_ci(x, y, groups, draws, seed):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50 or np.ptp(x[ok]) == 0:
        return None
    rx, ry, g = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy(), groups[ok]
    point = float(np.corrcoef(rx, ry)[0, 1])
    low = high = None
    if abs(point) >= 0.03:
        values = [np.corrcoef(rx[i], ry[i])[0, 1] for i in loci_resamples(g, draws, seed)]
        low, high = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    return point, low, high, int(ok.sum())


# Build per-TF motif disruption columns for T cells and Jurkat plus their difference, for the most frequent TFs.
def motif_features(sheets, index, top=40):
    wide = {}
    for key, number in (('t', 10), ('j', 12)):
        name = next((n for n in sheets if sheet_number(n) == number), None)
        if name is None or not {'SNP19', 'geneSymbol', 'alleleDiff'} <= set(sheets[name].columns):
            continue
        s = sheets[name][['SNP19', 'geneSymbol', 'alleleDiff']].copy()
        s['alleleDiff'] = pd.to_numeric(s.alleleDiff, errors='coerce')
        s = s.dropna(subset=['alleleDiff']).assign(a=lambda d: d.alleleDiff.abs())
        s = s.sort_values('a', ascending=False).drop_duplicates(['SNP19', 'geneSymbol'])
        wide[key] = s.pivot(index='SNP19', columns='geneSymbol', values='alleleDiff').reindex(index)
    columns = {}
    if wide:
        counts = pd.concat([w.notna().sum() for w in wide.values()], axis=1).sum(axis=1)
        for tf in counts.sort_values(ascending=False).index[:top]:
            t = wide['t'][tf].fillna(0) if 't' in wide and tf in wide['t'] else pd.Series(0.0, index=index)
            j = wide['j'][tf].fillna(0) if 'j' in wide and tf in wide['j'] else pd.Series(0.0, index=index)
            columns[f'motif:{tf}:tcell'], columns[f'motif:{tf}:jurkat'], columns[f'motif:{tf}:t_minus_j'] = t, j, t - j
    return pd.DataFrame(columns, index=index)


# Print and return the per-sheet inventory with key coverage over the Ho variants.
def inventory(sheets, snp19, snp38_to_19, rsid_to_19):
    rows = []
    for name, sheet in sheets.items():
        number = sheet_number(name)
        key, ids = None, None
        for candidate, mapper in (('SNP19', None), ('SNP38', snp38_to_19), ('rsid', rsid_to_19), ('rsID', rsid_to_19)):
            if candidate in sheet.columns:
                key = candidate
                ids = sheet[candidate].astype(str) if mapper is None else sheet[candidate].astype(str).map(mapper)
                break
        covered = int(ids.isin(snp19).sum()) if ids is not None else 0
        tag = ('TARGET' if number == 1 else 'SOURCE' if number == 9 else
               'SELECTED_ON_OUTCOME' if number in SELECTED_ON_OUTCOME else 'candidate')
        rows.append({'sheet': name[:45], 'rows': len(sheet), 'cols': sheet.shape[1], 'key': key,
                     'variants_covered': int(ids[ids.isin(snp19)].nunique()) if ids is not None else 0,
                     'rows_per_variant': round(covered / max(1, ids[ids.isin(snp19)].nunique()), 2) if covered else 0,
                     'role': tag})
        print(f'  {name[:60]:60s} cols: {", ".join(sheet.columns[:18])}' + (' ...' if sheet.shape[1] > 18 else ''))
    return pd.DataFrame(rows)


# Collect the candidate feature table indexed by SNP19 and tag every column.
def candidate_table(sheets, index, snp38_to_19, rsid_to_19):
    parts, tags = [], {}
    for name, sheet in sheets.items():
        number = sheet_number(name)
        if number in (1, 10, 12) or number in SELECTED_ON_OUTCOME or number is None:
            continue
        for key, mapper in (('SNP19', None), ('SNP38', snp38_to_19), ('rsid', rsid_to_19), ('rsID', rsid_to_19)):
            if key in sheet.columns:
                ids = sheet[key].astype(str) if mapper is None else sheet[key].astype(str).map(mapper)
                break
        else:
            continue
        numeric = sheet.drop(columns=[c for c in ('SNP19', 'SNP38', 'rsid', 'rsID', 'chr', 'pos') if c in sheet.columns])
        numeric = numeric.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        numeric['_id'] = ids.to_numpy()
        numeric = numeric[numeric._id.isin(index)]
        if numeric.empty:
            continue
        grouped = numeric.groupby('_id')
        aggregated = grouped.max() if len(numeric) > numeric._id.nunique() else grouped.first()
        aggregated = aggregated.reindex(index)
        aggregated.columns = [f'{number}:{c}' for c in aggregated.columns]
        for column in aggregated.columns:
            role = 'SOURCE' if number == 9 else ('REVIEW' if REVIEW_PATTERN.search(column.split(':', 1)[1]) else 'INDEPENDENT')
            tags[column] = role
            if aggregated[column].notna().mean() < 0.5:
                aggregated[column + '__present'] = aggregated[column].notna().astype(float)
                tags[column + '__present'] = role
        parts.append(aggregated)
    table = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=index)
    if {'9:A_log2FC', '9:B_log2FC'} <= set(table.columns):
        table['9:jurkat_activity_max'] = table[['9:A_log2FC', '9:B_log2FC']].max(axis=1)
        tags['9:jurkat_activity_max'] = 'SOURCE'
    if {'9:A_logPadj_BF', '9:B_logPadj_BF'} <= set(table.columns):
        table['9:jurkat_activity_p_max'] = table[['9:A_logPadj_BF', '9:B_logPadj_BF']].max(axis=1)
        tags['9:jurkat_activity_p_max'] = 'SOURCE'
    if {'9:A_Ctrl_Mean', '9:B_Ctrl_Mean'} <= set(table.columns):
        table['9:log_dna_depth'] = np.log1p(table['9:A_Ctrl_Mean'] + table['9:B_Ctrl_Mean'])
        tags['9:log_dna_depth'] = 'SOURCE'
    motifs = motif_features(sheets, index)
    for column in motifs.columns:
        tags[column] = 'INDEPENDENT'
    return pd.concat([table, motifs], axis=1), tags


# Run the whole audit and write the JSON.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=ROOT / 'data' / 'regulatory_audit' / 'ho_supp.xlsx')
    parser.add_argument('--prepared', type=Path,
                        default=ROOT / 'output' / 'diagnostics' / 'vprs_inputs' / 'prepared' / 'ho.csv.gz')
    parser.add_argument('--draws', type=int, default=100)
    parser.add_argument('--result', type=Path, default=ROOT / 'output' / 'diagnostics' / 'raw_feature_audit.json')
    args = parser.parse_args(argv)
    sheets = read_sheets(args.workbook)
    frame = pd.read_csv(args.prepared, dtype={'chrom': str}, low_memory=False)
    frame = frame.drop_duplicates('variant_id').reset_index(drop=True)
    index = pd.Index(frame.variant_id.astype(str))
    target = next(s for n, s in sheets.items() if sheet_number(n) == 1).drop_duplicates('SNP19').set_index('SNP19')
    snp38_to_19 = pd.Series(target.index, index=target['SNP38'].astype(str)).to_dict() if 'SNP38' in target else {}
    rsid_col = next((c for c in ('rsid', 'rsID') if c in target.columns), None)
    rsid_to_19 = pd.Series(target.index, index=target[rsid_col].astype(str)).to_dict() if rsid_col else {}
    print('\n=== Sheets and columns ===')
    inv = inventory(sheets, set(index), snp38_to_19, rsid_to_19)
    print('\n' + inv.to_string(index=False))

    t = target.reindex(index)
    for column in ('A_log2FC', 'B_log2FC', 'A_logPadj_BF', 'B_logPadj_BF', 'Skew_logFDR'):
        t[column] = pd.to_numeric(t[column], errors='coerce')
    active = ((t[['A_log2FC', 'B_log2FC']].max(axis=1) >= 1) & (t[['A_logPadj_BF', 'B_logPadj_BF']].max(axis=1) > 2)).to_numpy()
    significant = (t.Skew_logFDR >= 1).to_numpy()
    effect = frame.target_effect.to_numpy(float)
    outcomes_binary = {'T-cell active (pCRE rule)': active, 'emVar (active and skew FDR)': active & significant,
                       'skew direction (alt higher) among emVars': effect > 0}
    masks = {'skew direction (alt higher) among emVars': active & significant}
    groups = locus_groups(frame, gap_bp=250_000)
    table, tags = candidate_table(sheets, index, snp38_to_19, rsid_to_19)
    print(f'\n=== Testing {table.shape[1]} candidate columns on {len(frame)} Ho variants '
          f'({int(active.sum())} active, {int((active & significant).sum())} emVars, {len(set(groups))} loci) ===')
    rows = []
    for column in table.columns:
        x = table[column].to_numpy(float)
        base = {'column': column, 'role': tags.get(column, 'INDEPENDENT'), 'coverage': round(float(np.isfinite(x).mean()), 3)}
        if base['coverage'] < 0.01:
            continue
        for label, y in outcomes_binary.items():
            mask = masks.get(label, np.ones(len(frame), dtype=bool))
            result = auroc_ci(np.where(mask, x, np.nan), y, groups, args.draws, 42)
            if result:
                rows.append({**base, 'outcome': label, 'metric': 'auroc', 'value': result[0], 'ci_low': result[1],
                             'ci_high': result[2], 'n': result[3]})
        for label, y in (('effect size |target|', np.abs(effect)), ('signed target effect', effect)):
            result = rho_ci(x, y, groups, args.draws, 42)
            if result:
                rows.append({**base, 'outcome': label, 'metric': 'spearman', 'value': result[0], 'ci_low': result[1],
                             'ci_high': result[2], 'n': result[3]})
    results = pd.DataFrame(rows)
    results['excludes_null'] = [
        bool(r.ci_low is not None and r.ci_high is not None and
             (r.ci_low > (0.5 if r.metric == 'auroc' else 0) or r.ci_high < (0.5 if r.metric == 'auroc' else 0)))
        for r in results.itertuples()]
    results['strength'] = (results.value - results.metric.map({'auroc': 0.5, 'spearman': 0.0})).abs()
    for outcome, part in results.groupby('outcome'):
        best = part[part.excludes_null].sort_values('strength', ascending=False).head(15)
        print(f'\n--- {outcome}: {int(part.excludes_null.sum())} of {len(part)} tests exclude the null (no multiplicity correction) ---')
        print(best[['column', 'role', 'coverage', 'metric', 'value', 'ci_low', 'ci_high', 'n']].round(3).to_string(index=False)
              if len(best) else '  none')
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps({'inventory': inv.to_dict('records'), 'results': results.to_dict('records')},
                                      indent=1, default=float))
    print(f'\nSaved {args.result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
