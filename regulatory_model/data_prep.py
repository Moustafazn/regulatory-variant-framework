"""Ho immune-cell assay preparation; explicit endpoint and predictor whitelists."""
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


BASE_FEATURES = ['source_effect', 'source_se', 'ref_A', 'ref_C', 'ref_G', 'ref_T',
                 'alt_A', 'alt_C', 'alt_G', 'alt_T', 'length_delta']


SEQUENCE_FEATURES = ['ref_gc', 'alt_gc', 'gc_delta', 'ref_cpg', 'alt_cpg']


# Whitelisted molecular predictors from Ho supplementary sheets 4, 10 and 12.
MECHANISTIC_FEATURES = ['delta_svm_nCD4', 'motifbreakr_tcell_alleleDiff_max', 'motifbreakr_tcell_alleleDiff_min',
                        'motifbreakr_jurkat_alleleDiff_max', 'motifbreakr_jurkat_alleleDiff_min',
                        'motifbreakr_tcell_tf_count', 'motifbreakr_jurkat_tf_count']


ALLOWED_FEATURES = BASE_FEATURES + SEQUENCE_FEATURES + MECHANISTIC_FEATURES


def read_sheet(workbook, name):
    # Read an Excel sheet without loading workbook formatting into memory.
    rows = workbook[name].values
    return pd.DataFrame(rows, columns=next(rows))


def add_alleles(frame):
    # Encode allele identity without introducing genomic identifiers as predictors.
    frame = frame.copy()
    for side in ['ref', 'alt']:
        for base in 'ACGT':
            frame[f'{side}_{base}'] = (frame[side].str.upper() == base).astype(float)
    frame['length_delta'] = frame.alt.str.len() - frame.ref.str.len()
    return frame


def validate_frame(frame):
    # Normalize the canonical schema and reject malformed metadata, effects, or predictor values.
    required = ['example_id', 'variant_id', 'chrom', 'position', 'genome_build', 'ref', 'alt',
                'source_context', 'target_context', 'source_effect', 'source_se', 'target_effect', 'target_se']
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f'Missing columns: {sorted(absent)}')
    if frame.empty:
        raise ValueError('No valid examples remain after preparation')
    frame = frame.copy()
    metadata = ['example_id', 'variant_id', 'chrom', 'genome_build', 'ref', 'alt', 'source_context', 'target_context']
    if frame[metadata].isna().any().any():
        raise ValueError('Required identifiers, alleles and contexts cannot be missing')
    for column in metadata:
        frame[column] = frame[column].astype(str).str.strip()
    frame['chrom'] = frame.chrom.str.removeprefix('chr')
    numeric = ['position', 'source_effect', 'source_se', 'target_effect', 'target_se']
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise')
    frame['effect_covariance'] = pd.to_numeric(frame.get('effect_covariance', 0.0), errors='raise')
    checks = {
        'duplicate example IDs': frame.example_id.duplicated().any(),
        'empty metadata': frame[metadata].eq('').any().any(),
        'invalid chromosome': (~frame.chrom.str.fullmatch(r'(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)')).any(),
        'invalid genome build': (~frame.genome_build.isin(['GRCh37', 'GRCh38', 'unverified'])).any(),
        'invalid position': ((frame.position <= 0) | (frame.position % 1 != 0)).any(),
        'nonfinite measurement': not np.isfinite(frame[numeric + ['effect_covariance']].to_numpy(float)).all(),
        'nonpositive standard error': (frame[['source_se', 'target_se']] <= 0).any().any(),
        'invalid allele': any((~frame[c].str.fullmatch('[ACGT]+')).any() for c in ['ref', 'alt']),
        'identical alleles': frame.ref.eq(frame.alt).any(),
        'identical source and target contexts': frame.source_context.eq(frame.target_context).any(),
        'invalid covariance': (frame.effect_covariance.abs() > frame.source_se * frame.target_se + 1e-10).any(),
    }
    violations = [name for name, invalid in checks.items() if invalid]
    if violations:
        raise ValueError('Invalid prepared data: ' + '; '.join(violations))
    for column in set(ALLOWED_FEATURES) & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(frame[column].to_numpy(float)).any():
            raise ValueError(f'Infinite predictor values in {column}')
    frame['position'] = frame.position.astype(int)
    return frame.reset_index(drop=True)


def prepare_ho(root):
    # Join paired published allelic effects; preserve the full quantitative cohort.
    workbook = openpyxl.load_workbook(Path(root) / 'ho_supp.xlsx', read_only=True, data_only=True)
    target = read_sheet(workbook, '1. Tcell MPRA results')
    source = read_sheet(workbook, '9. jurkat MPRA results')
    workbook.close()
    paired = target.merge(source, on='SNP19', suffixes=('_T', '_J'), validate='one_to_one')
    if ((paired.ref_allele_T != paired.ref_allele_J) | (paired.alt_allele_T != paired.alt_allele_J)).any():
        raise ValueError('Unharmonized Ho alleles')
    parts = paired.SNP19.str.split(':', expand=True)
    frame = pd.DataFrame({'variant_id': paired.SNP19, 'chrom': parts[0], 'position': parts[1].astype(int),
                          'ref': parts[2], 'alt': parts[3], 'genome_build': 'GRCh37',
                          'rsid': paired.rsid_T, 'lead_group': paired.lead_snp_T,
                          'source_effect': paired.Log2Skew_J, 'source_se': paired.Skew_SE_J,
                          'target_effect': paired.Log2Skew_T, 'target_se': paired.Skew_SE_T,
                          'source_context': 'Jurkat', 'target_context': 'primary_T', 'dataset': 'Ho2025'})
    for column in ['source_effect', 'source_se', 'target_effect', 'target_se']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    keep = np.isfinite(frame[['source_effect', 'source_se', 'target_effect', 'target_se']]).all(axis=1)
    keep &= (frame.source_se > 0) & (frame.target_se > 0)
    # Reproduce the paper's post-model QC for both primary-cell plasmid allele counts.
    keep &= (pd.to_numeric(paired.A_Ctrl_Mean_T, errors='coerce') > 20) & (pd.to_numeric(paired.B_Ctrl_Mean_T, errors='coerce') > 20)
    frame = frame.loc[keep].copy()
    frame['example_id'] = 'Ho2025:' + frame.variant_id
    return validate_frame(add_alleles(frame))



def merge_ho_mechanistic_features(root, frame):
    # Attach Ho's own mechanistic predictors from supplementary sheets 4 (annotations), 10 and 12 (motifbreakR); predictors only, never target-derived fields.
    workbook = openpyxl.load_workbook(Path(root) / 'ho_supp.xlsx', read_only=True, data_only=True)
    try:
        annotations = read_sheet(workbook, '4. MPRA Functional Annotations')
        tcell = read_sheet(workbook, '10. Tcell motifbreakR results')
        jurkat = read_sheet(workbook, '12. Jurkat motifbreakR results')
    finally:
        workbook.close()
    annotations = annotations[['SNP19', 'delta_svm_nCD4']].drop_duplicates('SNP19')
    annotations['delta_svm_nCD4'] = pd.to_numeric(annotations['delta_svm_nCD4'], errors='coerce')
    frame = frame.merge(annotations, left_on='variant_id', right_on='SNP19', how='left', validate='many_to_one').drop(columns='SNP19')
    for assay, sheet in [('tcell', tcell), ('jurkat', jurkat)]:
        sheet = sheet[['SNP19', 'geneSymbol', 'alleleDiff']].copy()
        sheet['alleleDiff'] = pd.to_numeric(sheet['alleleDiff'], errors='coerce')
        sheet = sheet.dropna(subset=['alleleDiff'])
        grouped = sheet.groupby('SNP19')
        summary = pd.DataFrame({f'motifbreakr_{assay}_alleleDiff_max': grouped.alleleDiff.max(),
                                f'motifbreakr_{assay}_alleleDiff_min': grouped.alleleDiff.min(),
                                f'motifbreakr_{assay}_tf_count': grouped.geneSymbol.nunique()}).reset_index()
        frame = frame.merge(summary, left_on='variant_id', right_on='SNP19', how='left', validate='many_to_one').drop(columns='SNP19')
    return frame
