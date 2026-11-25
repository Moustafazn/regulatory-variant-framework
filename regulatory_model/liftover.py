"""Normalize variants from a declared source build and verify GRCh38 reference alleles.

Every GRCh37 row is lifted before reference verification. Failed mappings are
excluded with recorded reasons; alleles are never swapped to force agreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .reporting import sha256

# Bump when the mapping rules change so stale prepared tables and score caches are rejected.
LIFTOVER_VERSION = 'declared-build-v2'

_COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')
_PRIMARY_CHROMS = frozenset([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM'])


# Normalize a chromosome spelling to the ``chr`` form used by the hg38 FASTA.
def _chr(name) -> str:
    text = str(name).strip()
    return text if text.startswith('chr') else 'chr' + text


# Watson-Crick reverse complement of an allele string.
def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# One resolved variant on hg38 plus the rule that produced it.
@dataclass(frozen=True)
class _Resolved:
    chrom: str
    position: int
    ref: str
    alt: str
    method: str


# Return the single hg38 (chrom, 1-based start, strand) of an hg19 locus, or None if unmapped or ambiguous.
def _lift_locus(lift_over, chrom, position, length):
    hits = lift_over.convert_coordinate(_chr(chrom), int(position) - 1)
    if not hits or len(hits) != 1:
        return None
    new_chrom, new_pos0, strand = hits[0][0], int(hits[0][1]), hits[0][2]
    if new_chrom not in _PRIMARY_CHROMS:
        return None
    start = new_pos0 + 1 if strand == '+' else new_pos0 + 1 - (length - 1)
    return new_chrom, start, strand


# Read ``length`` hg38 bases at a 1-based position, or '' when out of range.
def _fasta_bases(fasta, chrom, position, length):
    if chrom not in fasta or position < 1:
        return ''
    return str(fasta[chrom][position - 1:position - 1 + length]).upper()


# Resolve one row under its declared build; return (_Resolved, None) or (None, reason).
def _resolve_row(chrom, position, ref, alt, source_build, lift_over, fasta):
    chrom, ref, alt = _chr(chrom), str(ref).upper(), str(alt).upper()
    position = int(position)
    if source_build == 'GRCh37':
        lifted = _lift_locus(lift_over, chrom, position, len(ref))
        if lifted is None:
            return None, 'unmapped_or_ambiguous_liftover'
        new_chrom, new_pos, strand = lifted
        if strand == '+':
            resolved = _Resolved(new_chrom, new_pos, ref, alt, 'hg19_to_hg38')
        elif len(ref) != 1 or len(alt) != 1:
            return None, 'strand_flipped_indel_unsupported'
        else:
            resolved = _Resolved(new_chrom, new_pos, _revcomp(ref), _revcomp(alt),
                                 'hg19_to_hg38_strand_flip')
    elif source_build == 'GRCh38':
        resolved = _Resolved(chrom, position, ref, alt, 'input_grch38')
    else:
        return None, f'unsupported_source_build_{source_build}'
    if _fasta_bases(fasta, resolved.chrom, resolved.position, len(resolved.ref)) != resolved.ref:
        return None, 'reference_mismatch_after_liftover'
    return resolved, None


# Normalize a prepared frame to GRCh38 and return (frame_hg38, provenance); failed rows are dropped, never rewritten.
def normalize_to_grch38(frame: pd.DataFrame, fasta_path, chain_path, source_build: str) -> tuple[pd.DataFrame, dict]:
    import pyfaidx
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f'hg38 reference FASTA not found at {fasta_path}.')
    if source_build not in ('GRCh37', 'GRCh38'):
        raise ValueError('Declare source_build as GRCh37 or GRCh38.')
    lift_over = None
    if source_build == 'GRCh37':
        if chain_path is None or not Path(chain_path).is_file():
            raise FileNotFoundError('hg19ToHg38 chain file is required to lift a GRCh37 dataset.')
        from pyliftover import LiftOver
        lift_over = LiftOver(str(chain_path))
    fasta = pyfaidx.Fasta(str(fasta_path))

    frame = frame.reset_index(drop=True)
    resolved, dropped, keep = [], [], []
    columns = zip(frame.example_id, frame.chrom, frame.position, frame.ref, frame.alt)
    for index, (example_id, chrom, position, ref, alt) in enumerate(columns):
        result, reason = _resolve_row(chrom, position, ref, alt, source_build, lift_over, fasta)
        if result is None:
            dropped.append({'index': index, 'example_id': str(example_id), 'reason': reason,
                            'chrom': str(chrom), 'position': int(position)})
        else:
            keep.append(index)
            resolved.append(result)

    fasta.close()

    out = frame.iloc[keep].copy().reset_index(drop=True)
    out['orig_chrom'], out['orig_position'] = out['chrom'].astype(str), out['position'].astype('int64')
    out['orig_ref'], out['orig_alt'], out['orig_build'] = out['ref'], out['alt'], source_build
    out['chrom'] = pd.Series([r.chrom.removeprefix('chr') for r in resolved], dtype=object)
    out['position'] = pd.Series([r.position for r in resolved], dtype='int64')
    out['ref'] = pd.Series([r.ref for r in resolved], dtype=object)
    out['alt'] = pd.Series([r.alt for r in resolved], dtype=object)
    out['genome_build'] = 'GRCh38'
    for side in ('ref', 'alt'):
        for base in 'ACGT':
            column = f'{side}_{base}'
            if column in out.columns:
                out[column] = (out[side].astype(str).str.upper() == base).astype(float)
    if 'length_delta' in out.columns:
        out['length_delta'] = out.alt.str.len() - out.ref.str.len()

    methods, reasons = {}, {}
    for item in resolved:
        methods[item.method] = methods.get(item.method, 0) + 1
    for item in dropped:
        reasons[item['reason']] = reasons.get(item['reason'], 0) + 1
    provenance = {
        'version': LIFTOVER_VERSION,
        'declared_source_build': source_build,
        'reference_fasta': str(fasta_path),
        'chain_file': str(chain_path) if chain_path is not None else None,
        'chain_sha256': sha256(Path(chain_path)) if source_build == 'GRCh37' else None,
        'input_rows': int(len(frame)),
        'normalized_rows': int(len(out)),
        'dropped_rows': int(len(dropped)),
        'mapping_method_counts': methods,
        'dropped_reason_counts': reasons,
        'examples_dropped': dropped[:5],
    }
    return out, provenance
