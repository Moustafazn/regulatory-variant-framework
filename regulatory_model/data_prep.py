"""Load, audit and prepare real MPRA data and biological evidence."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'regulatory_model'
from pathlib import Path
from dataclasses import dataclass
import argparse
import gzip
import json
import re
import zipfile
import numpy as np
import pandas as pd
import openpyxl
from .reporting import write_json
from .reporting import sha256

PROJECT_ROOT = Path(__file__).resolve().parent.parent


BASE_FEATURES = ['source_effect', 'source_se', 'ref_A', 'ref_C', 'ref_G', 'ref_T',
                 'alt_A', 'alt_C', 'alt_G', 'alt_T', 'length_delta']


SEQUENCE_FEATURES = ['ref_gc', 'alt_gc', 'gc_delta', 'ref_cpg', 'alt_cpg']


DHS = ['dhs_hTH1', 'dhs_hTH17', 'dhs_hTH2', 'dhs_CD4', 'dhs_CD4pos_N',
       'dhs_hTR', 'dhs_Jurkat', 'dhs_CD8']


# Mechanistic predictors derived from Ho's own supplementary workbook (sheet 4 for delta_svm_nCD4 and DHS; sheets 10 and 12 for motifbreakR per-variant allele difference).
MECHANISTIC_FEATURES = ['delta_svm_nCD4', 'motifbreakr_tcell_alleleDiff_max', 'motifbreakr_tcell_alleleDiff_min',
                        'motifbreakr_jurkat_alleleDiff_max', 'motifbreakr_jurkat_alleleDiff_min',
                        'motifbreakr_tcell_tf_count', 'motifbreakr_jurkat_tf_count']


ALLOWED_FEATURES = BASE_FEATURES + SEQUENCE_FEATURES + DHS


OUTCOMES = ['target_effect', 'target_se', 'effect_covariance']


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


def add_sequences(frame, reference, alternate):
    # Derive simple sequence composition features from experimentally supplied oligos.
    for prefix, sequences in [('ref', reference), ('alt', alternate)]:
        seq = sequences.str.upper()
        frame[prefix + '_gc'] = (seq.str.count('G') + seq.str.count('C')) / seq.str.len()
        frame[prefix + '_cpg'] = seq.str.count('CG') / seq.str.len()
    frame['gc_delta'] = frame.alt_gc - frame.ref_gc
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
    # Join paired published Ho effects and whitelist independent DHS annotations.
    workbook = openpyxl.load_workbook(Path(root) / 'ho_supp.xlsx', read_only=True, data_only=True)
    target = read_sheet(workbook, '1. Tcell MPRA results')
    source = read_sheet(workbook, '9. jurkat MPRA results')
    annotation = read_sheet(workbook, '4. MPRA Functional Annotations')
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
    frame = frame.merge(annotation[['SNP19'] + DHS].rename(columns={'SNP19': 'variant_id'}),
                        on='variant_id', validate='one_to_one')
    for column in ['source_effect', 'source_se', 'target_effect', 'target_se'] + DHS:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    keep = np.isfinite(frame[['source_effect', 'source_se', 'target_effect', 'target_se']]).all(axis=1)
    keep &= (frame.source_se > 0) & (frame.target_se > 0)
    # Reproduce the paper's post-model QC for both primary-cell plasmid allele counts.
    keep &= (pd.to_numeric(paired.A_Ctrl_Mean_T, errors='coerce') > 20) & (pd.to_numeric(paired.B_Ctrl_Mean_T, errors='coerce') > 20)
    frame = frame.loc[keep].copy()
    frame['example_id'] = 'Ho2025:' + frame.variant_id
    return validate_frame(add_alleles(frame))


def prepare_siraj(root, source_context, target_context, genome_build):
    # Read two Siraj contexts and retain one prespecified file-order window per variant.
    tables = []
    with zipfile.ZipFile(Path(root) / 'siraj_MPRA.zip') as archive:
        for context in [source_context, target_context]:
            files = sorted(n for n in archive.namelist() if '/DE/' in n and f'_{context}_' in n and n.endswith('.gz'))
            if not files:
                raise ValueError(f'No Siraj files for {context}')
            pieces = []
            for name in files:
                with gzip.open(archive.open(name), 'rt') as stream:
                    part = pd.read_csv(stream, sep='\t', low_memory=False)
                    part = part[(part.window == 'center') & (part.strand == 'fwd') & (part.haplotype == 'ref')].copy()
                    part['archive_member'] = name
                    pieces.append(part)
            table = pd.concat(pieces, ignore_index=True)
            key = 'SNP' if 'SNP' in table else 'SNP19'
            table['variant_id'] = table[key].astype(str).str.removeprefix('chr')
            table = table[table.variant_id.str.fullmatch(r'(?:[0-9]+|X|Y):[0-9]+:[ACGT]+:[ACGT]+')]
            for col in ['Log2Skew', 'Skew_SE']:
                table[col] = pd.to_numeric(table[col], errors='coerce')
            table = table[np.isfinite(table.Log2Skew) & np.isfinite(table.Skew_SE) & (table.Skew_SE > 0)]
            # File order fixes duplicate resolution without selecting by measured outcome.
            table = table.sort_values(['variant_id', 'archive_member']).drop_duplicates('variant_id')
            tables.append(table[['variant_id', 'Log2Skew', 'Skew_SE']])
    paired = tables[0].merge(tables[1], on='variant_id', suffixes=('_source', '_target'), validate='one_to_one')
    parts = paired.variant_id.str.split(':', expand=True)
    frame = pd.DataFrame({'variant_id': paired.variant_id, 'chrom': parts[0], 'position': parts[1].astype(int),
                          'ref': parts[2], 'alt': parts[3], 'genome_build': genome_build,
                          'source_effect': paired.Log2Skew_source, 'source_se': paired.Skew_SE_source,
                          'target_effect': paired.Log2Skew_target, 'target_se': paired.Skew_SE_target,
                          'source_context': source_context, 'target_context': target_context, 'dataset': 'Siraj2026'})
    with zipfile.ZipFile(Path(root) / 'siraj_tables.zip') as archive:
        with gzip.open(archive.open('tables/stable2.txt.gz'), 'rt') as stream:
            sequences = pd.read_csv(stream, sep='\t')
    sequences['variant_id'] = sequences.variant.str.removeprefix('chr')
    sequences = sequences.drop_duplicates('variant_id')
    frame = frame.merge(sequences[['variant_id', 'allele1_oligo', 'allele2_oligo']], on='variant_id', validate='one_to_one')
    frame = add_sequences(frame, frame.allele1_oligo, frame.allele2_oligo).drop(columns=['allele1_oligo', 'allele2_oligo'])
    frame['example_id'] = 'Siraj2026:' + frame.variant_id + ':' + source_context + ':' + target_context
    return validate_frame(add_alleles(frame))


def prepare_bourges(root, genome_build):
    # Estimate paired donor activation effects from barcode counts without treating windows as donors.
    raw = pd.read_csv(Path(root) / 'bourges_barcode_counts.txt.gz', sep='\t')
    sample_cols = [f'{i}{s}' for i in range(1, 13) for s in ['U', 'S']] + [f'v{i}' for i in range(1, 5)]
    library_sizes = raw[sample_cols].sum()
    variants = raw[raw.rsID.fillna('').str.fullmatch(r'rs\d+')].copy()
    variants['allele'] = variants.allele.astype(str).str.lower()
    if not {'ref', 'alt'} <= set(variants.allele):
        raise ValueError(f'Unsupported allele labels: {variants.allele.unique()}')
    counts = variants.groupby(['rsID', 'elementTag', 'allele'])[sample_cols].sum()
    scaled = counts.div(library_sizes, axis=1) * 1e6
    dna = scaled[[f'v{i}' for i in range(1, 5)]].mean(axis=1)
    # A declared 0.5 CPM pseudocount stabilizes a transparent exploratory estimator.
    activity = np.log2((scaled[[f'{i}{s}' for i in range(1, 13) for s in ['U', 'S']]] + 0.5).div(dna + 0.5, axis=0))
    effects = activity.xs('alt', level='allele') - activity.xs('ref', level='allele')
    effects = effects.groupby(level='rsID').mean()
    records = []
    for rsid, row in effects.iterrows():
        meta = variants[variants.rsID == rsid]
        # Derive the SNP coordinate from the paired oligos rather than using a window start.
        coordinates = []
        for _, window in meta.groupby('elementTag'):
            ref = window[window.allele == 'ref'].iloc[0]
            alt = window[window.allele == 'alt'].iloc[0]
            a, b = str(ref.genomicSequence).upper(), str(alt.genomicSequence).upper()
            differences = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            match = re.fullmatch(r'(chr[^:]+):(\d+)-(\d+)', str(ref.localcoords))
            if len(a) != len(b) or len(differences) != 1 or not match:
                continue
            offset = differences[0]
            coordinates.append((match[1].removeprefix('chr'), int(match[2]) + offset, a[offset], b[offset]))
        if not coordinates or len(set(coordinates)) != 1:
            continue
        chrom, position, ref, alt = coordinates[0]
        source = row[[f'{i}U' for i in range(1, 13)]].to_numpy(float)
        target = row[[f'{i}S' for i in range(1, 13)]].to_numpy(float)
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        records.append(dict(example_id=f'Bourges2020:{rsid}', variant_id=f'{chrom}:{position}:{ref}:{alt}',
                            rsid=rsid, chrom=chrom, position=position, ref=ref, alt=alt, genome_build=genome_build,
                            source_context='primary_T_resting', target_context='primary_T_stimulated', dataset='Bourges2020',
                            source_effect=source.mean(), target_effect=target.mean(),
                            source_se=source.std(ddof=1) / np.sqrt(12), target_se=target.std(ddof=1) / np.sqrt(12),
                            effect_covariance=np.cov(source, target, ddof=1)[0, 1] / 12))
    return validate_frame(add_alleles(pd.DataFrame(records)))


def feature_columns(frame, mode='full'):
    # Return only explicit predictors, excluding identifiers and all target-derived fields.
    allowed = BASE_FEATURES if mode == 'source_only' else ALLOWED_FEATURES
    columns = [c for c in allowed if c in frame and frame[c].notna().any()]
    if mode == 'no_source':
        columns = [c for c in columns if c not in ['source_effect', 'source_se']]
    if mode == 'no_context':
        columns = [c for c in columns if c not in DHS]
    return columns


@dataclass(frozen=True)
class DatasetAdapter:
    # Describe an extraction function together with required files and supported options.
    reader: object
    inputs: tuple[str, ...]
    options: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


ADAPTERS = {
    'ho': DatasetAdapter(prepare_ho, ('ho_supp.xlsx',), warnings=(
        'Both primary-cell plasmid allele means must exceed 20, as specified in the publication.',
        'Quantitative endpoint cohort; not a claim to reproduce the paper significant-variant count.',)),
    'siraj': DatasetAdapter(prepare_siraj, ('siraj_MPRA.zip', 'siraj_tables.zip'),
                           ('source_context', 'target_context', 'genome_build'), (
        'Confirm declared genome build and allele orientation before cross-study evaluation.',
        'Deterministic center/forward/reference-haplotype cohort; not the final publication cohort.')),
    'bourges': DatasetAdapter(prepare_bourges, ('bourges_barcode_counts.txt.gz',), ('genome_build',), (
        'Exploratory CPM/pseudocount estimator; not the original published count model.',
        'Verify interval coordinate convention, genome build, orientation and rs-labelled controls.')),
}


@dataclass(frozen=True)
class PreparedData:
    # Return the canonical dataset, isolated predictor/target files and preparation audit.
    dataset: Path
    inputs: Path
    targets: Path
    report: Path
    rows: int


def prepare_dataset(dataset, root, output, source_context='K562', target_context='HepG2', genome_build='unverified'):
    # Extract actual source files, validate the model schema and persist reproducible preparation artifacts.
    adapter = ADAPTERS[dataset]
    root, output = Path(root), Path(output)
    missing = [name for name in adapter.inputs if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing source data: ' + ', '.join(missing))
    inputs = Path(str(output) + '.inputs.csv.gz')
    targets = Path(str(output) + '.targets.csv.gz')
    report_path = Path(str(output) + '.json')
    source_shas = {name: sha256(root / name) for name in adapter.inputs}
    if all(p.is_file() for p in (output, inputs, targets, report_path)):
        prior = json.loads(report_path.read_text())
        prior_liftover_applied = bool(
            isinstance(prior.get('liftover_to_grch38'), dict)
            and prior['liftover_to_grch38'].get('applied'))
        if prior.get('status') == 'validated' \
                and prior.get('source_files_sha256') == source_shas \
                and prior.get('context_pairs') and prior.get('rows') \
                and prior_liftover_applied:
            print(f'Reusing prepared and validated {prior["rows"]:,} examples at {output}', flush=True)
            return PreparedData(output, inputs, targets, report_path, int(prior['rows']))
    options = dict(source_context=source_context, target_context=target_context, genome_build=genome_build)
    raw = adapter.reader(root, **{name: options[name] for name in adapter.options})
    # Normalize every variant to GRCh38 and verify against the hg38 FASTA Decima reads.
    # Without this step 73-77% of ref alleles disagree with the FASTA (see
    # regulatory_model/liftover.py docstring for the empirical mismatch counts), which
    # silently drives Decima to score the wrong reference base. Rows that cannot be
    # reconciled are dropped and recorded in the dataset report rather than skewing
    # downstream features.
    from .liftover import normalize_to_grch38, default_chain_path, default_hg38_fasta
    hg38_fasta = default_hg38_fasta(PROJECT_ROOT)
    chain = default_chain_path(PROJECT_ROOT)
    liftover_provenance = {'applied': False, 'reason': 'hg38_fasta_absent'}
    if hg38_fasta.is_file():
        raw, liftover_provenance = normalize_to_grch38(raw, hg38_fasta, chain)
        liftover_provenance['applied'] = True
        print(f'  liftover[{dataset}]: '
              f'{liftover_provenance["normalized_rows"]}/'
              f'{liftover_provenance["input_rows"]} variants verified against hg38 '
              f'({liftover_provenance["dropped_rows"]} dropped as unresolvable); '
              f'mapping_methods={liftover_provenance["mapping_method_counts"]}',
              flush=True)
    else:
        print(f'  liftover[{dataset}]: SKIPPED (no hg38 FASTA at '
              f'{hg38_fasta}); variants will be scored on their declared coordinates.',
              flush=True)
    frame = validate_frame(raw)
    features = feature_columns(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    frame[['example_id'] + features].to_csv(inputs, index=False)
    frame[['example_id'] + OUTCOMES].to_csv(targets, index=False)
    write_json(report_path, {
        'status': 'validated', 'dataset': dataset, 'rows': len(frame), 'unique_variants': frame.variant_id.nunique(),
        'features': features, 'target_fields': OUTCOMES,
        'liftover_to_grch38': liftover_provenance,
        'feature_missing_fraction': frame[features].isna().mean().to_dict(),
        'source_files_sha256': source_shas,
        'prepared_files_sha256': {path.name: sha256(path) for path in [output, inputs, targets]},
        'genome_build': frame.genome_build.unique().tolist(),
        'context_pairs': frame[['source_context', 'target_context']].drop_duplicates().to_dict('records'),
        'checks': {'unique_example_ids': True, 'finite_effects': True, 'positive_standard_errors': True,
                   'valid_coordinates': True, 'valid_covariance': True, 'target_free_input_whitelist': True},
        'warnings': list(adapter.warnings),
        'preprocessing': 'Missing-value imputation and scaling are fitted only after training splits.',
    })
    print(f'Prepared and validated {len(frame):,} examples at {output}', flush=True)
    return PreparedData(output, inputs, targets, report_path, len(frame))


def main():
    # Run the same extraction API independently when only prepared data is needed.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=list(ADAPTERS), default='ho')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'regulatory_audit')
    parser.add_argument('--output', type=Path, help='Prepared file; defaults to data/processed/<dataset>.csv.gz')
    parser.add_argument('--source-context', default='K562')
    parser.add_argument('--target-context', default='HepG2')
    parser.add_argument('--genome-build', choices=['GRCh37', 'GRCh38', 'unverified'], default='unverified')
    args = parser.parse_args()
    output = args.output or PROJECT_ROOT / 'data' / 'processed' / f'{args.dataset}.csv.gz'
    prepare_dataset(args.dataset, args.root, output, args.source_context, args.target_context, args.genome_build)


SOURCE_COLUMNS = ['A_log2FC', 'B_log2FC', 'A_log2FC_SE', 'B_log2FC_SE']


def numeric(frame, columns):
    # Convert documented Excel missing cells while rejecting unexpected text.
    return frame[columns].replace('#N/A', np.nan).apply(pd.to_numeric, errors='raise')

def merge_ho_mechanistic_features(root, frame):
    # Attach Ho's own mechanistic predictors from supplementary sheets 4 (annotations), 10 and 12 (motifbreakR); predictors only, never target-derived fields.
    workbook = openpyxl.load_workbook(Path(root) / 'ho_supp.xlsx', read_only=True, data_only=True)
    try:
        annotations = read_sheet(workbook, '4. MPRA Functional Annotations')
        tcell = read_sheet(workbook, '10. Tcell motifbreakR results')
        jurkat = read_sheet(workbook, '12. Jurkat motifbreakR results')
    finally:
        workbook.close()
    annotations = annotations[['SNP19', 'delta_svm_nCD4'] + DHS].drop_duplicates('SNP19')
    annotations['delta_svm_nCD4'] = pd.to_numeric(annotations['delta_svm_nCD4'], errors='coerce')
    for column in DHS:
        annotations[column] = pd.to_numeric(annotations[column], errors='coerce').fillna(0)
    frame = frame.drop(columns=[c for c in DHS if c in frame], errors='ignore')
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
    for column in ['motifbreakr_tcell_tf_count', 'motifbreakr_jurkat_tf_count']:
        frame[column] = frame[column].fillna(0)
    return frame
