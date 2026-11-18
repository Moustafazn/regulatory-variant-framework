"""Deterministically regenerate expert-review evidence from the Ho supplementary workbook.

    nice -n 19 venv/bin/python scripts/build_expert_evidence.py

Reads sheets 4, 5, 6 and the motifbreakR sheets, parses the '#N/A' text columns into explicit flags, and tags
every column SOURCE / INDEPENDENT / TARGET / REVIEW with the same rules as scripts/audit_raw_features.py.
Only INDEPENDENT columns can become evidence; every MPRA-derived column and sheets 14-27 are excluded because
they are selected on MPRA outcomes. Discrimination is pre-declared: a feature must clear BH-FDR in a discovery
half of loci and replicate in the held-out confirmation half. Features that fail confirmation are reported in
their own section, not as evidence. Evidence stays status 'pending_doi' until every source has a DOI.
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
sys.path.insert(0, str(ROOT / 'scripts'))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from regulatory_model.evaluation import locus_groups  # noqa: E402
from regulatory_model.pipeline import EVIDENCE  # noqa: E402

# Same review/target vocabulary and outcome-selected sheet set as the raw-feature audit.
REVIEW_PATTERN = re.compile(r'(?i)(mpra|skew|log2fc|emvar|pcre|expression|activity|tcell_|t_cell|bonf|fdr|_bf)')
SELECTED_ON_OUTCOME = frozenset(range(14, 28))
MISSING_TOKENS = frozenset({'#N/A', 'nan', 'None', '', 'NA', '#VALUE!'})

# Evidence columns parsed as presence flags, with the source key used for DOI lookup.
FLAG_COLUMNS = {
    'atac_qtl_beta': ('caQTL_present', 'atac_qtl', 'chromatin accessibility QTL reported at this variant'),
    'eqtl_beta': ('eQTL_present', 'tcell_eqtl', 'primary T cell eQTL reported at this variant'),
    'ananastra_tf': ('ananastra_tf_present', 'ananastra', 'allele-specific TF binding reported by ANANASTRA'),
    'motifbreakr_tf_2024': ('motifbreakr2024_present', 'motifbreakr_2024',
                            'TF motif disruption predicted by motifbreakR (HOCOMOCO v11)'),
    'tf_motifbreakr': ('tf_motifbreakr_present', 'tf_motifbreakr',
                       'TF motif disruption predicted by motifbreakR (HOCOMOCO v10, ENCODE restricted)'),
}


# Return the numeric prefix of a sheet name such as "4. MPRA Functional Annotations".
def sheet_number(name):
    match = re.match(r'\s*(\d+)', name)
    return int(match.group(1)) if match else None


# Read one workbook sheet into a frame with string column names.
def read_sheet(workbook, name):
    rows = workbook[name].values
    header = next(rows, None)
    if header is None:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[str(h) if h is not None else f'col{i}'
                                       for i, h in enumerate(header)])


# Tag a column SOURCE / TARGET / REVIEW / INDEPENDENT exactly as scripts/audit_raw_features.py does.
def tag_column(number, column):
    if number == 1:
        return 'TARGET'
    if number == 9:
        return 'SOURCE'
    if number in SELECTED_ON_OUTCOME:
        return 'TARGET'
    return 'REVIEW' if REVIEW_PATTERN.search(column) else 'INDEPENDENT'


# Turn a text-valued workbook column into an explicit present/absent flag rather than dropping it.
def presence_flag(series):
    text = series.astype(str).str.strip()
    return (~text.isin(MISSING_TOKENS)).to_numpy()


# Build the candidate evidence flags from sheets 4 and 5 for the given variant index.
def build_flags(sheets, index):
    flags, tags, origins = {}, {}, {}
    annotations = next((s for n, s in sheets.items() if sheet_number(n) == 4), None)
    if annotations is not None:
        table = annotations.drop_duplicates('SNP19').set_index('SNP19').reindex(index)
        for column, (name, source_key, reason) in FLAG_COLUMNS.items():
            if column not in table.columns:
                continue
            role = tag_column(4, column)
            if role != 'INDEPENDENT':
                continue
            flags[name] = presence_flag(table[column])
            tags[name] = role
            origins[name] = {'source_key': source_key, 'sheet': '4. MPRA Functional Annotations',
                             'column': column, 'reason': reason}
    pics = next((s for n, s in sheets.items() if sheet_number(n) == 5), None)
    if pics is not None and 'SNP19' in pics.columns:
        table = pics.drop_duplicates('SNP19').set_index('SNP19').reindex(index)
        if 'CS_95' in table.columns:
            flags['pics_credible_set_95'] = pd.to_numeric(table['CS_95'], errors='coerce').fillna(0).to_numpy() > 0
            tags['pics_credible_set_95'] = 'INDEPENDENT'
            origins['pics_credible_set_95'] = {'source_key': 'pics_finemapping', 'sheet': '5. PICS by MPRA',
                                               'column': 'CS_95',
                                               'reason': 'variant lies in the 95% PICS credible set'}
    ukbb = next((s for n, s in sheets.items() if sheet_number(n) == 6), None)
    if ukbb is not None:
        key = next((c for c in ('SNP', 'SNP19') if c in ukbb.columns), None)
        if key and 'pip' in ukbb.columns:
            pip = ukbb[[key, 'pip']].copy()
            pip['pip'] = pd.to_numeric(pip['pip'], errors='coerce')
            best = pip.dropna(subset=['pip']).groupby(key).pip.max().reindex(index)
            flags['ukbb_pip_above_0.1'] = (best.fillna(0).to_numpy() > 0.1)
            tags['ukbb_pip_above_0.1'] = 'INDEPENDENT'
            origins['ukbb_pip_above_0.1'] = {'source_key': 'ukbb_finemapping', 'sheet': '6. UKBB by MPRA',
                                             'column': 'pip',
                                             'reason': 'UK Biobank fine-mapping posterior inclusion probability above 0.1'}
    return flags, tags, origins


# Benjamini-Hochberg q-values for a vector of p-values.
def bh_qvalues(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / (np.arange(len(pvalues)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(pvalues))
    out[order] = np.clip(ranked, 0, 1)
    return out


# Test every flag on both outcomes in a discovery half of loci, then require replication in the confirmation half.
def discriminating_features(flags, frame, groups, seed, alpha):
    target = frame.target_effect.to_numpy(float)
    target_se = frame.target_se.to_numpy(float)
    detected = np.abs(target) / np.clip(target_se, 1e-9, None) > 2
    outcomes = {'detection': (detected, np.ones(len(frame), bool)),
                'direction_given_detected': (target > 0, detected)}
    labels = np.unique(groups).copy()
    np.random.default_rng(seed).shuffle(labels)
    discovery = np.isin(groups, labels[:len(labels) // 2])
    rows = []
    for name, flag in flags.items():
        for outcome, (y, mask) in outcomes.items():
            record = {'feature': name, 'outcome': outcome}
            for half, half_mask in (('discovery', discovery), ('confirmation', ~discovery)):
                keep = mask & half_mask
                y_half, x_half = y[keep], flag[keep]
                if y_half.sum() < 20 or (~y_half).sum() < 20 or x_half.std() == 0:
                    record[half] = None
                    continue
                table = pd.crosstab(x_half, y_half)
                record[half] = {
                    'auroc': round(float(roc_auc_score(y_half, x_half.astype(float))), 4),
                    'p_value': float(stats.fisher_exact(table.values)[1]) if table.shape == (2, 2) else None,
                    'n': int(keep.sum()), 'positives': int(x_half.sum())}
            rows.append(record)
    testable = [r for r in rows if r.get('discovery') and r['discovery']['p_value'] is not None]
    if testable:
        qvalues = bh_qvalues([r['discovery']['p_value'] for r in testable])
        for record, q in zip(testable, qvalues):
            record['discovery']['bh_q'] = round(float(q), 5)
    passed, failed = [], []
    for record in rows:
        discovery_hit = (record.get('discovery') and record['discovery'].get('bh_q') is not None
                         and record['discovery']['bh_q'] <= alpha)
        confirmation = record.get('confirmation')
        confirmed = bool(discovery_hit and confirmation and confirmation['p_value'] is not None
                         and confirmation['p_value'] <= 0.05
                         and (confirmation['auroc'] - 0.5) * (record['discovery']['auroc'] - 0.5) > 0)
        record['passes_discovery_fdr'] = bool(discovery_hit)
        record['replicates_in_confirmation'] = confirmed
        (passed if confirmed else failed).append(record)
    return passed, failed, {'discovery_loci': int(len(labels) // 2),
                            'confirmation_loci': int(len(labels) - len(labels) // 2),
                            'tests_run': len(rows), 'tests_with_p_value': len(testable),
                            'bh_alpha': alpha, 'seed': int(seed),
                            'rule': ('a feature is evidence only if it clears BH-FDR in the discovery half and '
                                     'replicates with the same direction at p<=0.05 in the confirmation half')}


# Resolve a source key to its DOI URL, or None while the mapping is still empty.
def resolve_doi(sources, key):
    entry = (sources.get('sources') or {}).get(key) or {}
    doi = entry.get('doi')
    if not doi:
        return None
    doi = str(doi).strip()
    return doi if doi.startswith('https://') else f'https://doi.org/{doi.removeprefix("doi:").strip()}'


# Build evidence records for the named flags; field is always one _apply_corrections accepts.
def build_records(frame, flags, origins, names, sources, sampled, call):
    records = []
    for name in sorted(names):
        origin = origins[name]
        url = resolve_doi(sources, origin['source_key'])
        for position in np.flatnonzero(flags[name] & sampled):
            records.append({
                'example_id': str(frame.iloc[position].example_id),
                'field': 'expert_action',
                'value': 'defer',
                'reason': (f"{origin['reason']} [{name}] "
                           f"(workbook sheet {origin['sheet']}, column {origin['column']})"),
                'evidence_url': url, 'independent_of_labels': True,
                'adjudicated_value': None, 'call': call})
    return records


# Regenerate the evidence file and its coverage summary.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=ROOT / 'data' / 'regulatory_audit' / 'ho_supp.xlsx')
    parser.add_argument('--prepared', type=Path,
                        default=ROOT / 'output' / 'diagnostics' / 'vprs_inputs' / 'prepared' / 'ho.csv.gz')
    parser.add_argument('--run-plan', type=Path, default=ROOT / 'output' / 'run_plan.json')
    parser.add_argument('--sources', type=Path,
                        default=ROOT / 'data' / 'regulatory_audit' / 'evidence_sources.json')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--result', type=Path,
                        default=ROOT / 'output' / 'diagnostics' / 'expert_evidence.json')
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.prepared, dtype={'chrom': str}, low_memory=False)
    frame = frame.drop_duplicates('variant_id').reset_index(drop=True)
    index = pd.Index(frame.variant_id.astype(str))

    plan_note, sampled_ids = 'run_plan.json not found; every prepared row treated as sampled', None
    if args.run_plan.is_file():
        plan = json.loads(args.run_plan.read_text())
        sampled_ids = set(plan.get('datasets', {}).get('ho', []))
        plan_note = (f"run_plan.json holds {plan.get('runs')}; "
                     f"{len(sampled_ids)} Ho example_ids are in the sampled cohort")
    sampled = (frame.example_id.astype(str).isin(sampled_ids).to_numpy() if sampled_ids
               else np.ones(len(frame), bool))
    print(f'  {plan_note}', flush=True)

    workbook = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    wanted = {4, 5, 6, 10, 11, 12, 13}
    sheets = {name: read_sheet(workbook, name) for name in workbook.sheetnames
              if sheet_number(name) in wanted}
    workbook.close()

    flags, tags, origins = build_flags(sheets, index)
    print(f'  parsed {len(flags)} INDEPENDENT candidate flags: {sorted(flags)}', flush=True)
    groups = locus_groups(frame, gap_bp=250_000)
    confirmed, failed, design = discriminating_features(flags, frame, groups, args.seed, args.alpha)
    print(f'  {len(confirmed)} of {design["tests_run"]} tests replicate in the confirmation half', flush=True)

    sources = json.loads(args.sources.read_text()) if args.sources.is_file() else {'sources': {}}
    confirmed_names = {record['feature'] for record in confirmed}
    records = build_records(frame, flags, origins, confirmed_names, sources, sampled, 'supports')
    # Direct experimental assays stay reviewable even when they do not discriminate the MPRA outcome.
    candidate_names = {n for n in ('caQTL_present', 'eQTL_present', 'ananastra_tf_present')
                       if n in flags} - confirmed_names
    candidates = build_records(frame, flags, origins, candidate_names, sources, sampled, 'insufficient')
    EVIDENCE.validate_python(records)
    EVIDENCE.validate_python(candidates)

    needed = sorted({origins[name]['source_key'] for name in confirmed_names | candidate_names})
    missing_dois = [key for key in needed if resolve_doi(sources, key) is None]
    all_missing = [key for key in (sources.get('sources') or {}) if resolve_doi(sources, key) is None]
    status = 'final' if (records or candidates) and not missing_dois else 'pending_doi'

    coverage = {
        'prepared_variants': int(len(frame)),
        'sampled_variants': int(sampled.sum()),
        'per_flag_rate_all_prepared': {name: round(float(flag.mean()), 4) for name, flag in flags.items()},
        'per_flag_count_sampled': {name: int((flag & sampled).sum()) for name, flag in flags.items()},
        'sampled_with_any_independent_flag': int(
            np.any(np.column_stack(list(flags.values())), axis=1).astype(bool)[sampled].sum()) if flags else 0,
        'sampled_with_any_strict_flag': int(np.any(np.column_stack(
            [flags[n] for n in ('caQTL_present', 'eQTL_present', 'ananastra_tf_present') if n in flags]
        ), axis=1)[sampled].sum()) if flags else 0,
        'evidence_records_written': len(records),
        'candidate_records_written': len(candidates),
        'variants_with_evidence_records': len({r['example_id'] for r in records}),
        'variants_with_any_record': len({r['example_id'] for r in records + candidates}),
    }
    coverage['share_sampled_with_any_strict_flag'] = round(
        coverage['sampled_with_any_strict_flag'] / max(coverage['sampled_variants'], 1), 4)

    payload = {
        'status': status,
        'status_reason': ('every confirmed source has a DOI' if status == 'final' else
                          f'DOIs still missing for: {missing_dois or all_missing}'),
        'run_plan': plan_note,
        'pre_registration': design,
        'column_roles': tags,
        'evidence_origins': origins,
        'confirmed_features': confirmed,
        'failed_confirmation': failed,
        'coverage': coverage,
        'records': records,
        'candidate_records': candidates,
        'exclusions': {
            'sheets_excluded_selected_on_outcome': sorted(SELECTED_ON_OUTCOME),
            'note': ('Sheets 14-27 (V2G, scCRISPRi, proliferation) are selected on MPRA outcomes. '
                     'Sheet 1 is TARGET and sheet 9 is SOURCE; neither can become independent evidence.')},
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=1, default=float))

    if status != 'final':
        print(f'  STATUS pending_doi -- {payload["status_reason"]}', flush=True)
    print('\n=== Coverage ===')
    print(json.dumps(coverage, indent=1, default=float))
    print('\n=== Features that failed confirmation (not evidence) ===')
    for record in failed:
        discovery = record.get('discovery') or {}
        print(f"  {record['feature']:28s} {record['outcome']:26s} "
              f"discovery_auroc={discovery.get('auroc')} bh_q={discovery.get('bh_q')}")
    print(f'\nSaved {args.result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
