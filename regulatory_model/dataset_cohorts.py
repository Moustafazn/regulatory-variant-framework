"""Verified Jurkat-to-primary-CD4 cohort with source-independent annotations."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from .annotations import join_checked_annotations, build_track_schema, save_schema
from .data_prep import PROJECT_ROOT, MECHANISTIC_FEATURES, prepare_ho, merge_ho_mechanistic_features
from .evaluation import locus_groups
from .liftover import normalize_to_grch38
from .reporting import sha256


def prepare_cohort(options, config):
    raw = prepare_ho(options.data_root)
    frame, coordinates = normalize_to_grch38(raw, options.reference_fasta, options.chain, 'GRCh37')
    # Form indivisible groups on the full eligible cohort, before any row sampling.
    frame['split_group'] = locus_groups(frame, gap_bp=int(config['gap_bp']))
    if options.max_rows and len(frame) > options.max_rows:
        ranks = frame.example_id.map(lambda x: hashlib.sha256(f'{config["seed"]}:{x}'.encode()).hexdigest())
        frame = frame.loc[ranks.sort_values().index[:options.max_rows]].sort_values('example_id').reset_index(drop=True)
    frame = merge_ho_mechanistic_features(options.data_root, frame)
    sequence = [f'{side}_{base}' for side in ('ref','alt') for base in 'ACGT'] + ['length_delta'] + MECHANISTIC_FEATURES
    annotation_path = options.annotations
    if annotation_path is None:
        annotation_path = Path(options.output)/'accessibility.csv.gz'
        table, tracks = build_track_schema(frame, {'DNase':(options.source_track, options.target_track)},
                                          150, prefix='access')
        save_schema(annotation_path, table, coordinates, tracks, 150)
    annotations = pd.read_csv(annotation_path, dtype={'chrom': str})
    context = ['access_source', 'access_target', 'access_delta']
    if set(c for c in annotations if c.startswith('access_')) != set(context):
        raise ValueError('Accessibility annotations must supply only access_source, access_target and access_delta predictors.')
    frame = join_checked_annotations(frame, annotations)
    provenance_path = Path(str(annotation_path)+'.json')
    annotation_provenance = json.loads(provenance_path.read_text())
    if annotation_provenance.get('output_sha256') != sha256(annotation_path):
        raise ValueError('Annotation table hash does not match its provenance sidecar.')
    note = ('Jurkat transformed cell line to activated primary CD4 T cells. The generic CD4 DNase track '
            'is a lineage proxy; exact activation-state matching is not established. Reporter assays '
            'do not directly measure native-locus allelic effects.')
    return frame, sequence, context, {
        'coordinates': coordinates,
        'inputs': {str(options.data_root/'ho_supp.xlsx'): sha256(options.data_root/'ho_supp.xlsx'),
                   str(options.reference_fasta): sha256(options.reference_fasta),
                   str(options.chain): sha256(options.chain),
                   str(annotation_path): sha256(annotation_path),
                   str(provenance_path): sha256(provenance_path)},
        'annotation_provenance': annotation_provenance, 'context_limitation': note,
        'feature_coverage': frame[sequence+context].notna().mean().to_dict(),
        'feature_policy': 'Only allele encodings, deltaSVM, motif allele differences/counts and DNase signals. No MPRA significance, skew or activity columns from annotation sheets.',
        'missingness': 'NaN retained; fold-local imputation for ridge, native handling for GBM.',
        'effect_covariance': 'Zero is an unverified working assumption; shared assay normalization can violate it.',
        'scope': 'Ho 2025 reused public data; internal grouped evaluation, not independent experimental validation.'}
