"""Normalize prepared variant frames to GRCh38 against the hg38 FASTA Decima scores against.

Why this module exists
----------------------
The published source datasets are not all on hg38:
  * Ho is tagged ``GRCh37`` in sheet 1 of ``ho_supp.xlsx``.
  * Siraj is tagged ``unverified`` by the paper's own supplementary metadata.
  * Bourges is tagged ``unverified`` by this project's adapter.

An empirical check against the FASTA at ``data/decima_cache/hg38/hg38.fa`` (the file
Decima actually reads) shows that if the prepared frames are passed through unchanged:

    siraj   : 76.2% of declared ref alleles disagree with the hg38 base
    ho      : 76.7% disagree
    bourges : 73.0% disagree

Decima's ``predict_variant_effect`` happily consumes those rows, but every disagreement
is a wrong-reference prediction. Applying a UCSC hg19->hg38 lift-over and verifying the
lifted ref base against the actual hg38 FASTA recovers:

    siraj   : 1986/1999 (99.3%)
    ho      : 2000/2000 (100%)
    bourges : 199/241  (82.6%); the residual 42 split into 20 rows whose ``alt`` matches
              the hg38 base (strand flip or alt-indexed source) and 22 rows that match
              neither ref nor alt (dropped with an explicit reason).

See the sibling investigation records in run_log.jsonl for the exact spot-check used
to arrive at these numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


# Watson-Crick complement, used when a lifted alignment lands on the minus strand.
_COMPLEMENT = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')


# Canonical hg38 primary-assembly chromosomes Decima and validate_frame both accept.
# Lifts that land on alt / random / unlocalized contigs (chr14_KI270846v1_alt etc.)
# are dropped rather than passed through, because Decima's gene annotation and this
# project's validate_frame regex both restrict to the primary assembly.
_PRIMARY_CHROMS = frozenset(
    [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY', 'chrM'])


# Return True if chrom (``chr`` prefixed) is on the hg38 primary assembly.
def _is_primary(chrom: str) -> bool:
    return chrom in _PRIMARY_CHROMS


# Normalize a chromosome spelling to the ``chr`` form used by the hg38 FASTA.
def _chr(name) -> str:
    s = str(name).strip()
    return s if s.startswith('chr') else 'chr' + s


# Watson-Crick reverse complement for a short allele string.
def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# One normalized variant plus its provenance tag.
@dataclass(frozen=True)
class _NormalizedRow:
    index: int                   # index in the original frame
    chrom: str                   # hg38 chromosome, 'chr'-prefixed
    position: int                # 1-based hg38 position
    ref: str                     # hg38 reference base at `position`
    alt: str                     # declared alternate allele, strand-complemented if needed
    mapping_method: str          # 'input_grch38' | 'hg19_to_hg38' | 'alleles_swapped' | ...
    strand: str                  # '+' or '-' (chain-file strand of the lifted hit)
    alleles_swapped: bool        # True when input ref/alt were swapped to match hg38


# Resolve a 1-based hg19 position to its matching hg38 (chrom, pos, strand); None if unmappable.
def _lift_hg19_to_hg38(lift_over, chrom: str, position_1based: int):
    hits = lift_over.convert_coordinate(_chr(chrom), int(position_1based) - 1)
    if not hits:
        return None
    new_chrom, new_pos0, new_strand, _ = hits[0]
    return _chr(new_chrom), int(new_pos0) + 1, new_strand


# Read a short base from an hg38 FASTA at (chrom, 1-based position), or '' if out of range.
def _fasta_base(fasta, chrom: str, position_1based: int, length: int) -> str:
    chrom = _chr(chrom)
    if chrom not in fasta:
        return ''
    try:
        return str(fasta[chrom][position_1based - 1:position_1based - 1 + length]).upper()
    except Exception:  # noqa: BLE001
        return ''


# Try to accept one row either as hg38-native or via hg19->hg38; return _NormalizedRow or None.
def _try_normalize_row(index: int, row, fasta, lift_over) -> _NormalizedRow | None:
    try:
        chrom = str(row['chrom'])
        pos = int(row['position'])
        ref = str(row['ref']).upper()
        alt = str(row['alt']).upper()
    except (KeyError, TypeError, ValueError):
        return None

    # Path A: the row is already consistent with hg38 at its declared coordinates.
    hg38_base_here = _fasta_base(fasta, chrom, pos, len(ref))
    if hg38_base_here == ref:
        return _NormalizedRow(
            index=index, chrom=_chr(chrom), position=pos, ref=ref, alt=alt,
            mapping_method='input_grch38', strand='+', alleles_swapped=False)
    if hg38_base_here == alt and len(ref) == 1 and len(alt) == 1:
        # Row is on hg38 but with the ref/alt convention reversed. Swap orientations.
        return _NormalizedRow(
            index=index, chrom=_chr(chrom), position=pos, ref=alt, alt=ref,
            mapping_method='alleles_swapped_input_grch38', strand='+', alleles_swapped=True)

    # Path B: treat the declared coordinates as hg19 and lift to hg38.
    lifted = _lift_hg19_to_hg38(lift_over, chrom, pos)
    if lifted is None:
        return None
    new_chrom, new_pos, new_strand = lifted
    if not _is_primary(new_chrom):
        return None
    ref_lifted = ref if new_strand == '+' else _revcomp(ref)
    alt_lifted = alt if new_strand == '+' else _revcomp(alt)
    hg38_base_lifted = _fasta_base(fasta, new_chrom, new_pos, len(ref_lifted))

    if hg38_base_lifted == ref_lifted:
        return _NormalizedRow(
            index=index, chrom=new_chrom, position=new_pos,
            ref=ref_lifted, alt=alt_lifted,
            mapping_method='hg19_to_hg38' if new_strand == '+' else 'hg19_to_hg38_strand_flip',
            strand=new_strand, alleles_swapped=False)
    if (hg38_base_lifted == alt_lifted
            and len(ref_lifted) == 1 and len(alt_lifted) == 1):
        return _NormalizedRow(
            index=index, chrom=new_chrom, position=new_pos,
            ref=alt_lifted, alt=ref_lifted,
            mapping_method='hg19_to_hg38_alleles_swapped',
            strand=new_strand, alleles_swapped=True)
    return None


# Fallback used when no chain file is available: no row ever gets lifted off its hg19 slot.
class _NoLiftOver:
    @staticmethod
    def convert_coordinate(_chrom: str, _pos0: int) -> Iterable:
        return ()


# Normalize an entire prepared frame to GRCh38 and return (frame_hg38, provenance_dict).
#
# ``frame_hg38`` is a copy of ``frame`` restricted to rows that could be verified against
# the hg38 FASTA (via Path A or Path B above). Verified rows have their ``chrom`` /
# ``position`` / ``ref`` / ``alt`` / ``genome_build`` columns rewritten to hg38 values.
# Rows that could not be reconciled are dropped (never silently rewritten).
#
# ``provenance_dict`` records the counts and the top-5 dropped rows for ``audit.json``.
def normalize_to_grch38(frame: pd.DataFrame,
                        fasta_path: str | Path,
                        chain_path: str | Path | None) -> tuple[pd.DataFrame, dict]:
    import pyfaidx  # local import keeps data_prep importable without the FASTA on disk
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(
            f'hg38 reference FASTA not found at {fasta_path}; required to verify every '
            'variant used for Decima scoring. Make sure data/decima_cache/hg38/hg38.fa '
            'has been downloaded.')
    fasta = pyfaidx.Fasta(str(fasta_path))

    lift_over = _NoLiftOver()
    if chain_path is not None:
        chain_path = Path(chain_path)
        if chain_path.is_file():
            from pyliftover import LiftOver
            lift_over = LiftOver(str(chain_path))

    normalized: list[_NormalizedRow] = []
    dropped: list[dict] = []
    total = len(frame)
    for index, row in frame.reset_index(drop=True).iterrows():
        result = _try_normalize_row(index, row, fasta, lift_over)
        if result is None:
            dropped.append({
                'index': int(index),
                'original_chrom': str(row.get('chrom', '')),
                'original_position': int(row.get('position', 0) or 0),
                'original_ref': str(row.get('ref', '')),
                'original_alt': str(row.get('alt', '')),
                'original_build': str(row.get('genome_build', 'unverified')),
                'variant_id': str(row.get('variant_id', '')),
                'reason': 'unresolvable_against_hg38_fasta',
            })
        else:
            normalized.append(result)

    kept_indices = [r.index for r in normalized]
    out = frame.reset_index(drop=True).iloc[kept_indices].copy().reset_index(drop=True)
    # Write back the hg38 coordinates. Use the plain ('1', '2', ...) chromosome
    # convention validate_frame expects downstream; _chr() is only used while
    # talking to the FASTA.
    out['chrom'] = pd.Series(
        [('MT' if r.chrom == 'chrM' else r.chrom.removeprefix('chr')) for r in normalized],
        dtype=object)
    out['position'] = pd.Series([r.position for r in normalized], dtype='int64')
    out['ref'] = pd.Series([r.ref for r in normalized], dtype=object)
    out['alt'] = pd.Series([r.alt for r in normalized], dtype=object)
    out['genome_build'] = 'GRCh38'

    # Rebuild derived allele / sequence columns so they reflect the hg38 orientation
    # that downstream feature blocks see. We only rebuild columns that already existed
    # in the input frame; other columns (DHS, mechanistic, source/target_effect) are
    # left untouched because they key off `variant_id`, not coordinates.
    for side in ('ref', 'alt'):
        for base in 'ACGT':
            col = f'{side}_{base}'
            if col in out.columns:
                out[col] = (out[side].astype(str).str.upper() == base).astype(float)
    if 'length_delta' in out.columns:
        out['length_delta'] = out.alt.str.len() - out.ref.str.len()

    counts: dict[str, int] = {}
    for r in normalized:
        counts[r.mapping_method] = counts.get(r.mapping_method, 0) + 1

    provenance = {
        'reference_fasta': str(fasta_path),
        'chain_file': str(chain_path) if chain_path is not None else None,
        'input_rows': int(total),
        'normalized_rows': int(len(normalized)),
        'dropped_rows': int(len(dropped)),
        'mapping_method_counts': counts,
        'examples_dropped': dropped[:5],
    }
    return out, provenance


# Convenience: find the UCSC hg19ToHg38 chain file in whatever location the AVI
# download script already put it (data/alphagenome_avi/hg19ToHg38.over.chain.gz).
def default_chain_path(project_root: Path) -> Path | None:
    for candidate in (
        project_root / 'data' / 'alphagenome_avi' / 'hg19ToHg38.over.chain.gz',
        project_root / 'data' / 'decima_cache' / 'hg19ToHg38.over.chain.gz',
    ):
        if candidate.is_file():
            return candidate
    return None


# Convenience: hg38 FASTA shipped under the Decima cache.
def default_hg38_fasta(project_root: Path) -> Path:
    return project_root / 'data' / 'decima_cache' / 'hg38' / 'hg38.fa'
