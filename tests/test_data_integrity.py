"""Offline coordinate and annotation integrity checks."""
import json

import numpy as np
import pandas as pd
import pytest

from regulatory_model.liftover import normalize_to_grch38

SEQUENCE = 'ACGTACGTTTGACCAGTCAGGATCCTAGGCTA' * 4


class FakeLift:
    # Map chr1 hg19 position p (0-based) to hg38 position p + 100 on '+', and chr2 to a '-' strand hit.
    @staticmethod
    def convert_coordinate(chrom, pos0):
        if chrom == 'chr1':
            return [('chr1', pos0 + 10, '+', 1.0)]
        if chrom == 'chr2':
            return [('chr1', pos0 + 10, '-', 1.0)]
        return None


@pytest.fixture
def fasta(tmp_path):
    path = tmp_path / 'hg38.fa'
    path.write_text('>chr1\n' + SEQUENCE + '\n')
    return path


def _frame(rows):
    return pd.DataFrame(rows, columns=['example_id', 'chrom', 'position', 'ref', 'alt']).assign(genome_build='GRCh37')


def _patched_lift(monkeypatch):
    import pyliftover
    monkeypatch.setattr(pyliftover, 'LiftOver', lambda path: FakeLift())


def test_liftover_lifts_first_and_never_swaps_alleles(tmp_path, fasta, monkeypatch):
    _patched_lift(monkeypatch)
    chain = tmp_path / 'chain.gz'
    chain.write_text('x')
    hg19_position = 5
    hg38_position = hg19_position + 10
    ref = SEQUENCE[hg38_position - 1]
    other = next(b for b in 'ACGT' if b != ref)
    unlucky_ref = SEQUENCE[hg19_position - 1]
    frame = _frame([
        ('ok', '1', hg19_position, ref, other),
        # Declared ref equals the hg38 base at the UNLIFTED position: must still be lifted, then rejected.
        ('coincidence', '1', hg19_position, unlucky_ref if unlucky_ref != ref else other, ref),
        ('swapped', '1', hg19_position, other, ref),
    ])
    out, provenance = normalize_to_grch38(frame, fasta, chain, 'GRCh37')
    assert out.example_id.tolist() == ['ok']
    assert int(out.position.iloc[0]) == hg38_position
    assert provenance['mapping_method_counts'] == {'hg19_to_hg38': 1}
    assert provenance['dropped_reason_counts'] == {'reference_mismatch_after_liftover': 2}
    assert out.genome_build.eq('GRCh38').all() and out.orig_position.iloc[0] == hg19_position


def test_liftover_drops_strand_flipped_indels_and_requires_chain(tmp_path, fasta, monkeypatch):
    _patched_lift(monkeypatch)
    chain = tmp_path / 'chain.gz'
    chain.write_text('x')
    frame = _frame([('indel', '2', 5, 'AC', 'A')])
    out, provenance = normalize_to_grch38(frame, fasta, chain, 'GRCh37')
    assert out.empty and provenance['dropped_reason_counts'] == {'strand_flipped_indel_unsupported': 1}
    with pytest.raises(FileNotFoundError):
        normalize_to_grch38(frame, fasta, None, 'GRCh37')



def test_annotation_join_rejects_wrong_coordinate_and_context():
    from regulatory_model.annotations import join_checked_annotations
    frame=pd.DataFrame({'variant_id':['v'],'chrom':['1'],'position':[100],
        'ref':['A'],'alt':['G'],'genome_build':['GRCh38'],'source_context':['Jurkat'],'target_context':['primary_T']})
    for changed in (frame.assign(position=101),frame.assign(target_context='K562')):
        with pytest.raises(ValueError,match='mismatch'):
            join_checked_annotations(frame,changed.assign(access_source=1.))
    with pytest.raises(ValueError,match='missing'):
        join_checked_annotations(frame,frame.iloc[:0].assign(access_source=1.))


def test_bigwig_requires_verified_build_and_preserves_missing(tmp_path):
    import pyBigWig
    from regulatory_model.annotations import bigwig_signal
    path=tmp_path/'signal.bw'
    with pyBigWig.open(str(path),'w') as bw:
        bw.addHeader([('chr1',248956422)])
        bw.addEntries(['chr1'],[99],ends=[100],values=[7.])
    frame=pd.DataFrame({'chrom':['1','1'],'position':[100,500],'genome_build':'GRCh37'})
    with pytest.raises(ValueError,match='GRCh38'): bigwig_signal(path,frame,1)
    values=bigwig_signal(path,frame.assign(genome_build='GRCh38'),1)
    assert values[0]==7 and np.isnan(values[1])


def test_reference_fasta_must_match_its_verified_archive(tmp_path):
    import gzip
    from scripts.verify_run_inputs import unpack_reference
    from regulatory_model.reporting import sha256
    archive=tmp_path/'hg38.fa.gz'
    with gzip.open(archive,'wb') as stream:stream.write(b'>chr1\nACGT\n')
    fasta=archive.with_suffix('')
    assert unpack_reference(archive)==sha256(fasta)
    assert unpack_reference(archive)==sha256(fasta)
    fasta.write_bytes(b'>chr1\nAAAA\n')
    with pytest.raises(ValueError,match='differ'):
        unpack_reference(archive)
    assert fasta.read_bytes()==b'>chr1\nAAAA\n'
