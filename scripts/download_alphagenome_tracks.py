"""Download AlphaGenome's cell-state track scores for prepared variants.

This is the cell-state track downloader. It is separate from the AVI score
utility and never reads AVI archives or AVI result tables. For every prepared
variant it calls ``client.score_variant()`` with the five AlphaGenome variant
scorers (RNA-seq, CAGE, DNase, ATAC, CHIP_TF), averages the per-track raw
scores across each configured cell-state's ontology CURIEs, and writes one
compressed NPZ per variant to data/alphagenome_tracks/ with 10 float32 scalars:

    primary_T:RNA_SEQ, primary_T:CAGE, primary_T:DNASE, primary_T:ATAC,
    primary_T:CHIP_TF, jurkat:RNA_SEQ, jurkat:CAGE, jurkat:DNASE,
    jurkat:ATAC, jurkat:CHIP_TF

The filename is ``<Source>_<example_id with ':' replaced by '_'>.npz``, matching
the historical NPZ files already in data/alphagenome_tracks/. If the target
NPZ already exists the variant is skipped before any API call, so previously
completed downloads (1,353 variants at the time of writing) are never redone.
Datasets run smallest-first by default (bourges then ho then siraj) so the
pipeline can be validated end-to-end within hours.

Run from the repository root after preparing Ho, Bourges, and/or Siraj:
    venv/bin/python scripts/download_alphagenome_tracks.py
    venv/bin/python scripts/download_alphagenome_tracks.py --datasets bourges

The script builds and validates its own GRCh38 coordinate tables in the track
folder. It needs the optional AlphaGenome client and ALPHAGENOME_API_KEY.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREPARED_DIR = ROOT / 'output/experiments/regulatory-variant-framework/prepared'
DEFAULT_TRACK_DIR = ROOT / 'data/alphagenome_tracks'
DEFAULT_BOURGES_OLIGOS = ROOT / 'data/regulatory_audit/bourges_barcode_counts.txt.gz'
DEFAULT_REFERENCE = DEFAULT_TRACK_DIR / 'hg38.2bit'
DEFAULT_CHAIN = DEFAULT_TRACK_DIR / 'hg19ToHg38.over.chain.gz'
DATASETS = ('bourges', 'ho', 'siraj')
DATASET_SOURCE_TAG = {'bourges': 'Bourges2020', 'ho': 'Ho2025', 'siraj': 'Siraj2026'}
MODALITIES = ('RNA_SEQ', 'CAGE', 'DNASE', 'ATAC', 'CHIP_TF')
CELL_STATE_ONTOLOGY_TERMS = {
    # These are the same exact biosample ontology IDs requested in the existing
    # track provenance, grouped by the model's two configured cell states.
    'primary_T': ('CL:0000624', 'CL:0000895', 'CL:0000897'),
    'jurkat': ('CLO:0007045', 'EFO:0002796'),
}
COORDINATE_COLUMNS = (
    'example_id', 'hg38_chrom', 'hg38_position', 'hg38_ref', 'hg38_alt',
    'mapping_method', 'status',
)
COMPLEMENT = str.maketrans('ACGT', 'TGCA')
CHAIN_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz'
REFERENCE_URL = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit'
MAX_FLANK_MISMATCHES = 3
SEARCH_RADIUS = 50_000
SEQUENCE_LENGTH = 1_048_576


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def tls_context():
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def download_resource(url: str, path: Path, label: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    print(f'Downloading {label} into {path.parent}...', flush=True)
    try:
        with urlopen(url, timeout=120, context=tls_context()) as response, temporary.open('wb') as out:
            total = response.headers.get('Content-Length')
            total = int(total) if total and total.isdigit() else None
            written, last_report = 0, time.monotonic()
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
                written += len(chunk)
                if time.monotonic() - last_report >= 15:
                    status = f'{written / 1e9:.1f} GB'
                    if total:
                        status += f' / {total / 1e9:.1f} GB ({written / total:.1%})'
                    print(f'{label} download: {status}', flush=True)
                    last_report = time.monotonic()
        if temporary.stat().st_size == 0:
            raise ValueError(f'{label} download was empty.')
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def chromosome(value) -> str:
    name = str(value).removeprefix('chr')
    return 'chr' + ('M' if name == 'MT' else name)


def load_reference(reference_path: Path):
    from Bio import SeqIO
    if not reference_path.is_file():
        download_resource(REFERENCE_URL, reference_path, 'GRCh38 reference')
    # Biopython's 2bit reader supports keyed, lazy sequence access. Keep the
    # indexed reader instead of loading the whole human genome into RAM.
    reference = SeqIO.parse(reference_path, 'twobit')
    if 'chr22' not in reference.keys() or len(reference['chr22']) != 50_818_468:
        raise ValueError(f'{reference_path} is not the expected GRCh38 reference.')
    return reference


def _paired_oligos(raw: pd.DataFrame, row) -> list[tuple[str, str, int]]:
    windows = []
    for _, group in raw[raw.rsID.eq(row.rsid)].groupby('elementTag'):
        alleles = {
            allele: group.loc[group.allele.eq(allele), 'genomicSequence'].unique()
            for allele in ('ref', 'alt')
        }
        if any(len(values) != 1 for values in alleles.values()):
            raise ValueError('Missing or conflicting source allele sequences')
        ref, alt = (str(alleles[allele][0]).upper() for allele in ('ref', 'alt'))
        differences = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
        if len(ref) != len(alt) or len(differences) != 1:
            raise ValueError('Source window is not a single-nucleotide allele pair')
        offset = differences[0]
        if (ref[offset], alt[offset]) != (str(row.ref).upper(), str(row.alt).upper()):
            raise ValueError('Prepared alleles disagree with the source sequences')
        if set(ref + alt) - set('ACGT'):
            raise ValueError('Source sequence contains ambiguous bases')
        windows.append((ref, alt, offset))
    if not windows:
        raise ValueError(f'No source oligos for {row.rsid}')
    return windows


def _oriented(window, strand):
    ref, alt, offset = window
    if strand == '-':
        return ref.translate(COMPLEMENT)[::-1], alt.translate(COMPLEMENT)[::-1], len(ref) - 1 - offset
    return window


def _match_oligos(sequence: str, windows: list[tuple[str, str, int]]) -> dict:
    central = max(windows, key=lambda w: min(w[2], len(w[0]) - w[2] - 1))
    candidates = []
    for strand in ('+', '-'):
        ref, _, offset = _oriented(central, strand)
        if min(offset, len(ref) - offset - 1) < 30:
            raise ValueError('Source windows lack 30-base flanks on both sides')
        positions = set()
        for seed_offset in (offset - 30, offset + 1):
            seed = ref[seed_offset:seed_offset + 30]
            start = sequence.find(seed)
            while start != -1:
                positions.add(start - seed_offset + offset)
                start = sequence.find(seed, start + 1)
        for position in sorted(positions):
            mismatches = []
            for window in windows:
                window_ref, window_alt, index = _oriented(window, strand)
                start, end = position - index, position - index + len(window_ref)
                if start < 0 or end > len(sequence):
                    break
                actual = sequence[start:end]
                count = sum(a != b for i, (a, b) in enumerate(zip(window_ref, actual)) if i != index)
                if count > MAX_FLANK_MISMATCHES or actual[index] not in (window_ref[index], window_alt[index]):
                    break
                mismatches.append(count)
            else:
                ref, alt, index = _oriented(central, strand)
                assay_ref, assay_alt = ref[index], alt[index]
                genomic_ref = sequence[position]
                swapped = genomic_ref == assay_alt
                candidates.append({
                    'position': position + 1, 'strand': strand, 'ref': genomic_ref,
                    'alt': assay_ref if swapped else assay_alt,
                    'flank_mismatches': mismatches,
                })
    if candidates:
        best = min(sum(item['flank_mismatches']) for item in candidates)
        candidates = [item for item in candidates if sum(item['flank_mismatches']) == best]
    if len(candidates) != 1:
        raise ValueError(f'Expected one best sequence-verified GRCh38 locus; found {len(candidates)}')
    return candidates[0]


def _chain_hits(frame: pd.DataFrame, chain_path: Path):
    from Bio import Align
    required_positions: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    rows = list(frame.itertuples(index=False))
    for i, row in enumerate(rows):
        ref = str(row.ref).upper()
        if not ref or set(ref) - set('ACGT'):
            continue
        required_positions[chromosome(row.chrom)].append((int(row.position) - 1, int(row.position) - 1 + len(ref), i))
    for chrom in required_positions:
        required_positions[chrom].sort()
    starts = {chrom: [item[0] for item in items] for chrom, items in required_positions.items()}
    hits: dict[int, set[tuple[str, int, str]]] = defaultdict(set)
    with gzip.open(chain_path, 'rt') as handle:
        for alignment in Align.parse(handle, 'chain'):
            chrom = alignment.target.id
            if chrom not in required_positions:
                continue
            source, target = alignment.coordinates
            for a, b, c, d in zip(source[:-1], source[1:], target[:-1], target[1:]):
                if b <= a or abs(d - c) != b - a:
                    continue
                items = required_positions[chrom]
                lo, hi = bisect_left(starts[chrom], a), bisect_left(starts[chrom], b)
                for start, end, index in items[lo:hi]:
                    if end > b:
                        continue
                    strand = '+' if d > c else '-'
                    first = c + start - a if strand == '+' else c - (start - a) - 1
                    mapped = tuple(first + j if strand == '+' else first - j for j in range(end - start))
                    if all(abs(x - y) == 1 for x, y in zip(mapped, mapped[1:])):
                        hits[index].add((alignment.query.id, min(mapped) + 1, strand))
    return rows, hits


def build_coordinates(dataset: str, args, reference) -> tuple[pd.DataFrame, Path, Path]:
    reference_names = set(reference.keys())
    prepared_path = args.prepared_dir / f'{dataset}.csv.gz'
    if not prepared_path.is_file():
        raise FileNotFoundError(f'Missing prepared dataset: {prepared_path}')
    coordinates_path = args.output_dir / f'{dataset}.track_coordinates.csv.gz'
    summary_path = args.output_dir / f'{dataset}.track_coordinates.json'
    prepared = pd.read_csv(prepared_path, usecols=[
        'example_id', 'variant_id', 'chrom', 'position', 'ref', 'alt', 'genome_build',
        *(['rsid'] if dataset == 'bourges' else []),
    ], dtype={'chrom': str}, low_memory=False)
    if prepared.example_id.duplicated().any():
        raise ValueError(f'{dataset}: example_id must be unique.')
    if coordinates_path.is_file() and summary_path.is_file():
        cached_summary = json.loads(summary_path.read_text())
        if (cached_summary.get('prepared_sha256') == sha256(prepared_path)
                and cached_summary.get('reference_sha256') == sha256(args.reference)
                and cached_summary.get('chain_sha256') == sha256(args.chain)
                and cached_summary.get('bourges_oligos_sha256') == (
                    sha256(args.bourges_oligos) if dataset == 'bourges' else None
                )
                and cached_summary.get('coordinates_sha256') == sha256(coordinates_path)):
            cached_coordinates = pd.read_csv(
                coordinates_path,
                dtype={'hg38_chrom': str, 'hg38_ref': str, 'hg38_alt': str},
                low_memory=False,
            )
            if (len(cached_coordinates) == len(prepared)
                    and cached_coordinates.status.eq('ready').all()
                    and not cached_coordinates.example_id.duplicated().any()):
                return prepared.merge(
                    cached_coordinates, on='example_id', validate='one_to_one', sort=False,
                ), prepared_path, coordinates_path

    results = []
    if dataset == 'bourges' and not prepared.genome_build.eq('GRCh38').all():
        if not args.bourges_oligos.is_file():
            raise FileNotFoundError(f'Missing Bourges source oligos: {args.bourges_oligos}')
        from Bio import Align
        raw = pd.read_csv(args.bourges_oligos, sep='\t', usecols=[
            'rsID', 'elementTag', 'allele', 'genomicSequence',
        ]).drop_duplicates()
        raw['allele'] = raw.allele.str.lower()
        frame_rows, hits = _chain_hits(prepared, args.chain)
        for i, row in enumerate(frame_rows):
            status, mapping = 'unmapped', {}
            candidates = hits.get(i, set())
            try:
                if len(candidates) != 1:
                    raise ValueError(f'Expected one lifted locus anchor; found {len(candidates)}')
                anchor_chrom, anchor_pos, _ = next(iter(candidates))
                start = max(0, anchor_pos - SEARCH_RADIUS - 1)
                sequence = str(reference[anchor_chrom].seq[start:anchor_pos + SEARCH_RADIUS]).upper()
                match = _match_oligos(sequence, _paired_oligos(raw, row))
                match['position'] += start
                mapping = {
                    'hg38_chrom': anchor_chrom, 'hg38_position': match['position'],
                    'hg38_ref': match['ref'], 'hg38_alt': match['alt'],
                    'mapping_method': 'bourges_sequence_verified', 'status': 'ready',
                }
                status = 'ready'
            except (ValueError, KeyError) as exc:
                mapping = {'hg38_chrom': '', 'hg38_position': '', 'hg38_ref': '', 'hg38_alt': '',
                           'mapping_method': '', 'status': f'unresolved:{exc}'}
            results.append({'example_id': row.example_id, **mapping})
    else:
        # Ho coordinates are GRCh37; the cited Siraj source labels its variant
        # coordinates as hg19/GRCh37. Only these explicit source conventions are
        # used to interpret the prepared tables.
        if dataset == 'siraj':
            builds = prepared.genome_build.replace('unverified', 'GRCh37')
        else:
            builds = prepared.genome_build
        if not builds.isin(['GRCh37', 'GRCh38']).all():
            raise ValueError(f'{dataset}: expected GRCh37/GRCh38 coordinates, got {builds.value_counts().to_dict()}')
        frame_rows, hits = _chain_hits(prepared, args.chain)
        for i, row in enumerate(frame_rows):
            chrom, pos = chromosome(row.chrom), int(row.position)
            ref, alt = str(row.ref).upper(), str(row.alt).upper()
            build = builds.iloc[i]
            status, method = 'ready', 'input_grch38'
            if build == 'GRCh37':
                candidates = hits.get(i, set())
                if len(candidates) != 1:
                    status = 'ambiguous_mapping' if candidates else 'unmapped'
                else:
                    chrom, pos, strand = next(iter(candidates))
                    if strand == '-':
                        ref, alt = ref.translate(COMPLEMENT)[::-1], alt.translate(COMPLEMENT)[::-1]
                    method = 'hg19_to_hg38'
            if status == 'ready':
                if chrom not in reference_names or pos < 1 or pos + len(ref) - 1 > len(reference[chrom]):
                    status = 'out_of_bounds'
                elif str(reference[chrom].seq[pos - 1:pos - 1 + len(ref)]).upper() != ref:
                    status = 'reference_mismatch'
            results.append({
                'example_id': row.example_id, 'hg38_chrom': chrom if status == 'ready' else '',
                'hg38_position': pos if status == 'ready' else '', 'hg38_ref': ref if status == 'ready' else '',
                'hg38_alt': alt if status == 'ready' else '', 'mapping_method': method if status == 'ready' else '',
                'status': status,
            })

    coordinates = pd.DataFrame(results, columns=COORDINATE_COLUMNS)
    if coordinates.example_id.duplicated().any() or len(coordinates) != len(prepared):
        raise ValueError(f'{dataset}: coordinate generation lost or duplicated prepared rows.')
    temporary = coordinates_path.with_name(coordinates_path.name + '.partial')
    coordinates.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(coordinates_path)
    report = {
        'dataset': dataset, 'prepared_sha256': sha256(prepared_path),
        'reference_sha256': sha256(args.reference), 'chain_sha256': sha256(args.chain),
        'bourges_oligos_sha256': sha256(args.bourges_oligos) if dataset == 'bourges' else None,
        'rows': len(coordinates), 'status_counts': coordinates.status.value_counts().to_dict(),
        'coordinates_sha256': sha256(coordinates_path),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(summary_path, report)
    not_ready = coordinates.loc[coordinates.status.ne('ready'), 'status'].value_counts().to_dict()
    ready = coordinates.loc[coordinates.status.eq('ready')]
    if not_ready:
        print(f'{dataset}: skipping {sum(not_ready.values()):,} unmappable variants {not_ready}; '
              f'continuing with {len(ready):,}/{len(coordinates):,} ready variants.',
              flush=True)
    if ready.empty:
        raise ValueError(f'{dataset}: no variants are ready for scoring; status counts: {not_ready}')
    merged = prepared.merge(ready, on='example_id', validate='one_to_one')
    return merged, prepared_path, coordinates_path


def npz_stem(dataset: str, example_id: str) -> str:
    tag = DATASET_SOURCE_TAG[dataset]
    text = str(example_id)
    if text.startswith(f'{tag}:'):
        text = text[len(tag) + 1:]
    cleaned = ''.join(c if c.isalnum() or c in '._-' else '_' for c in text.replace(':', '_'))
    return f'{tag}_{cleaned}'


def aggregate_cell_state_scores(tidy_scores, cell_state_terms, modalities):
    required = {'output_type', 'ontology_curie', 'raw_score'}
    missing = required - set(tidy_scores.columns)
    if missing:
        raise ValueError(f'AlphaGenome score response lacks required fields: {sorted(missing)}')
    work = tidy_scores.copy()
    work['output_type'] = work['output_type'].map(lambda value: str(value).upper().split('.')[-1])
    work['ontology_curie'] = work['ontology_curie'].astype('string')
    work['raw_score'] = pd.to_numeric(work['raw_score'], errors='coerce')
    work = work.loc[np.isfinite(work['raw_score'])]
    arrays, counts = {}, {}
    for context, curies in cell_state_terms.items():
        for modality in modalities:
            values = work.loc[
                work['ontology_curie'].isin(set(curies)) & work['output_type'].eq(modality),
                'raw_score',
            ].to_numpy(dtype=float)
            key = f'{context}:{modality}'
            arrays[key] = np.asarray([float(values.mean()) if values.size else 0.0], dtype=np.float32)
            counts[key] = int(values.size)
    if not any(counts.values()):
        raise ValueError('AlphaGenome returned no scores matching the configured ontology terms.')
    return arrays, counts


def save_prediction(path: Path, provenance_path: Path, arrays: dict, provenance: dict):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    provenance['raw_file_sha256'] = sha256(path)
    atomic_json(provenance_path, provenance)


def request_signature(dataset, prepared_path, coordinates_path, package_version, model_version):
    settings = {
        'script_version': 2,
        'dataset': dataset,
        'prepared_sha256': sha256(prepared_path),
        'coordinates_sha256': sha256(coordinates_path),
        'requested_outputs': list(MODALITIES),
        'cell_state_ontology_terms': {name: list(curie) for name, curie in CELL_STATE_ONTOLOGY_TERMS.items()},
        'sequence_length_bp': SEQUENCE_LENGTH,
        'alphagenome_package_version': package_version,
        'api_model_version': model_version,
        'aggregation': 'mean(raw_score) across ontology_curie per (cell_state, modality)',
    }
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return signature, settings


def restrict_to_plan(frame, dataset, plan_path):
    """Keep only the example IDs a verified run plan lists for this dataset."""
    if plan_path is None:
        return frame
    plan = json.loads(Path(plan_path).read_text())['datasets']
    wanted = set(plan.get(dataset, ()))
    absent = wanted - set(frame.example_id)
    if absent:
        raise ValueError(f'{dataset}: {len(absent):,} planned example IDs are not in the prepared table; '
                         'rerun scripts/verify_run_inputs.py.')
    return frame[frame.example_id.isin(wanted)].reset_index(drop=True)


def retire_mislocated_files(frame, dataset, output_dir):
    """Move track files scored at another coordinate into _stale_mislocated/ so they are re-fetched."""
    stale_dir = output_dir / '_stale_mislocated'
    moved = 0
    for row in frame.itertuples(index=False):
        stem = npz_stem(dataset, row.example_id)
        npz, provenance = output_dir / f'{stem}.npz', output_dir / f'{stem}.provenance.json'
        if not npz.is_file():
            continue
        correct = False
        if provenance.is_file():
            record = json.loads(provenance.read_text())
            scored = (record.get('grch38_variant') or record.get('variant')
                      or record.get('bourges_coordinate_correction') or {})
            correct = (str(scored.get('chrom', '')).removeprefix('chr') == str(row.hg38_chrom).removeprefix('chr')
                       and int(scored.get('position', -1)) == int(row.hg38_position)
                       and scored.get('ref') == str(row.hg38_ref) and scored.get('alt') == str(row.hg38_alt))
        if not correct:
            stale_dir.mkdir(exist_ok=True)
            for path in (npz, provenance):
                if path.is_file():
                    path.replace(stale_dir / path.name)
            moved += 1
    return moved


def download_dataset(dataset, client, args, package_version, variant_scorers):
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    frame, prepared_path, coordinates_path = build_coordinates(dataset, args, args.reference_records)
    frame = restrict_to_plan(frame, dataset, args.example_ids_file)
    retired = retire_mislocated_files(frame, dataset, args.output_dir)
    if retired:
        print(f'{dataset}: moved {retired:,} mislocated track files to _stale_mislocated/ for re-fetching.',
              flush=True)
    signature, settings = request_signature(
        dataset, prepared_path, coordinates_path, package_version,
        str(getattr(client, '_model_version', 'server_default')),
    )
    manifest_path = args.output_dir / f'track_predictions_{dataset}.manifest.json'
    manifest = {
        'dataset': dataset, 'signature': signature, 'settings': settings,
        'total_variants': len(frame), 'completed_variants': 0,
        'status': 'running', 'updated_utc': datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('signature') != signature:
            print(f'{manifest_path.name}: previous signature differs; resuming per-NPZ since on-disk files '
                  f'remain the source of truth.', flush=True)
        else:
            manifest.update(previous)

    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[name] for name in MODALITIES]
    errors_path = args.output_dir / f'track_predictions_{dataset}.errors.csv'
    total = len(frame)
    already = sum(1 for row in frame.itertuples(index=False)
                  if (args.output_dir / f'{npz_stem(dataset, row.example_id)}.npz').is_file())
    print(f'{dataset}: {already:,}/{total:,} NPZs already on disk; downloading {total - already:,} remaining.',
          flush=True)
    done, downloaded, skipped = already, 0, 0
    started = time.monotonic()
    for row in frame.itertuples(index=False):
        stem = npz_stem(dataset, row.example_id)
        output_path = args.output_dir / f'{stem}.npz'
        provenance_path = args.output_dir / f'{stem}.provenance.json'
        if output_path.is_file():
            continue
        variant = genome.Variant(
            chromosome=str(row.hg38_chrom), position=int(row.hg38_position),
            reference_bases=str(row.hg38_ref).upper(), alternate_bases=str(row.hg38_alt).upper(),
            name=str(row.variant_id),
        )
        interval = variant.reference_interval.resize(SEQUENCE_LENGTH)
        try:
            scores = client.score_variant(
                interval=interval, variant=variant,
                variant_scorers=scorers,
                organism=dna_client.Organism.HOMO_SAPIENS,
            )
            tidy = variant_scorers.tidy_scores(scores, include_extended_metadata=True)
            if tidy is None:
                raise ValueError('AlphaGenome returned no score rows.')
            arrays, counts = aggregate_cell_state_scores(tidy, CELL_STATE_ONTOLOGY_TERMS, MODALITIES)
            provenance = {
                'dataset': dataset, 'example_id': str(row.example_id),
                'variant_id': str(row.variant_id), 'signature': signature,
                'input_variant': {
                    'chrom': str(row.chrom), 'position': int(row.position),
                    'ref': str(row.ref), 'alt': str(row.alt),
                    'genome_build': str(row.genome_build),
                },
                'grch38_variant': {
                    'chrom': str(row.hg38_chrom), 'position': int(row.hg38_position),
                    'ref': str(row.hg38_ref), 'alt': str(row.hg38_alt),
                    'mapping_method': str(row.mapping_method),
                },
                'requested_outputs': list(MODALITIES),
                'cell_state_ontology_terms': {k: list(v) for k, v in CELL_STATE_ONTOLOGY_TERMS.items()},
                'score_counts_per_key': counts,
            }
            save_prediction(output_path, provenance_path, arrays, provenance)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            with errors_path.open('a') as handle:
                handle.write(json.dumps({
                    'example_id': str(row.example_id),
                    'variant': f'{row.hg38_chrom}:{row.hg38_position}:{row.hg38_ref}>{row.hg38_alt}',
                    'error': message,
                }) + '\n')
            if 'Unrecognized chromosome' in str(exc) and 'INVALID_ARGUMENT' in message:
                skipped += 1
                print(f'{dataset}: skipping {row.example_id} '
                      f'({row.hg38_chrom}:{row.hg38_position} not on primary assembly); '
                      f'logged to {errors_path.name}.', flush=True)
                continue
            manifest.update(
                completed_variants=done, status='failed',
                failed_example_id=str(row.example_id),
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(manifest_path, manifest)
            raise RuntimeError(f'Track download failed for {row.example_id}; completed NPZs remain in {args.output_dir}.') from exc
        done += 1
        downloaded += 1
        if downloaded % 10 == 0 or done == total:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = downloaded / elapsed
            remaining = total - done
            eta_hours = remaining / rate / 3600 if rate else float('inf')
            percent = 100.0 * done / total if total else 100.0
            print(f'{dataset}: {done:,}/{total:,} ({percent:.1f}%); '
                  f'{rate:.3f}/s this session; about {eta_hours:.1f} h remaining',
                  flush=True)
            manifest.update(
                completed_variants=done,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(manifest_path, manifest)
    manifest.update(status='complete', completed_variants=done,
                    skipped_unsupported_contig=skipped,
                    updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(manifest_path, manifest)
    skipped_suffix = f'; skipped {skipped:,} unsupported-contig variants' if skipped else ''
    print(f'Saved {dataset} track scores to {args.output_dir} '
          f'({done:,}/{total:,} variants{skipped_suffix}).', flush=True)
    return done


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--prepared-dir', type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE,
                        help='GRCh38 2bit reference; downloaded into the track folder if absent.')
    parser.add_argument('--chain', type=Path, default=DEFAULT_CHAIN,
                        help='hg19-to-hg38 chain; downloaded into the track folder if absent.')
    parser.add_argument('--bourges-oligos', type=Path, default=DEFAULT_BOURGES_OLIGOS)
    parser.add_argument('--example-ids-file', type=Path, default=None,
                        help='Run plan written by scripts/verify_run_inputs.py; downloads only its example IDs.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get('ALPHAGENOME_API_KEY')
    if not key:
        print('Set ALPHAGENOME_API_KEY in the shell to use AlphaGenome predictions.', file=sys.stderr)
        return 2
    try:
        from alphagenome.models import dna_client, variant_scorers
        from Bio import Align  # noqa: F401 - verify Biopython is available before starting.
        package_version = importlib.metadata.version('alphagenome')
    except ImportError as exc:
        print('Install the API and coordinate dependencies with: venv/bin/python -m pip install -e ".[alphagenome]" biopython', file=sys.stderr)
        return 2
    reference_records = None
    try:
        download_resource(CHAIN_URL, args.chain, 'hg19-to-hg38 coordinate chain')
        reference_records = load_reference(args.reference)
        args.reference_records = reference_records
        client = dna_client.create(key)
        total, failed = 0, []
        for dataset in dict.fromkeys(args.datasets):
            try:
                total += download_dataset(dataset, client, args, package_version, variant_scorers)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failed.append((dataset, f'{type(exc).__name__}: {exc}'))
                print(f'{dataset}: FAILED ({exc}); continuing with the next dataset.',
                      file=sys.stderr, flush=True)
        print(f'Finished AlphaGenome track scores: {total:,} variants.', flush=True)
        if failed:
            print(f'Datasets with errors: {failed}', file=sys.stderr, flush=True)
            return 1
        return 0
    except KeyboardInterrupt:
        print('\nInterrupted. Completed per-variant track files are saved; rerun the same command to resume.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        stream = getattr(reference_records, 'stream', None)
        if stream is not None:
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
