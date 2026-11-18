"""Offline proofs for liftover, coordinate guards, nested sampling and the per-variant Decima store."""
import json

import numpy as np
import pandas as pd
import pytest

from regulatory_model import dataset_cohorts as cohorts
from regulatory_model.decima_scorer import DecimaGeneScorer
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


def test_coordinate_agreement_guard_fails_above_limit():
    joined = pd.DataFrame({
        'example_id': [f'e{i}' for i in range(2000)], 'chrom': '1', 'position': np.arange(2000) + 100,
        'ref': 'A', 'alt': 'G', 'hg38_chrom': 'chr1', 'hg38_position': np.arange(2000) + 100.0,
        'hg38_ref': 'A', 'hg38_alt': 'G'})
    clean, audit = cohorts._enforce_coordinate_agreement(joined.copy(), 'x', 'AVI')
    assert audit['rows_disagreeing'] == 0 and len(clean) == 2000
    joined.loc[:0, 'hg38_position'] += 5
    clean, audit = cohorts._enforce_coordinate_agreement(joined.copy(), 'x', 'AVI')
    assert audit['rows_disagreeing'] == 1 and len(clean) == 1999
    joined.loc[:50, 'hg38_position'] += 5
    with pytest.raises(RuntimeError, match='disagree'):
        cohorts._enforce_coordinate_agreement(joined.copy(), 'x', 'AVI')


def test_mislocated_track_files_are_rejected(tmp_path):
    frame = pd.DataFrame({'chrom': ['1', '1'], 'position': [100, 200], 'ref': ['A', 'C'], 'alt': ['G', 'T']})
    names = ['Ho2025_a.npz', 'Ho2025_b.npz']
    for name, position in zip(names, (100, 999)):
        (tmp_path / name).write_bytes(b'')
        (tmp_path / name.replace('.npz', '.provenance.json')).write_text(json.dumps({
            'grch38_variant': {'chrom': 'chr1', 'position': position, 'ref': frame.ref.iloc[names.index(name)],
                               'alt': frame.alt.iloc[names.index(name)]}}))
    wrong = cohorts._mislocated_track_files(frame, names, np.array([True, True]), tmp_path)
    assert wrong == [1]


def _example_frame(prefix, count):
    return pd.DataFrame({'example_id': [f'{prefix}:{i}' for i in range(count)]})


def test_sampling_is_nested_order_free_and_independent_per_dataset():
    frame = _example_frame('Ho2025', 3000)
    small = set(frame.example_id.iloc[cohorts.hash_rank_order(frame, 'ho', 42)[:600]])
    large = set(frame.example_id.iloc[cohorts.hash_rank_order(frame, 'ho', 42)[:2000]])
    assert small < large
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    again = set(shuffled.example_id.iloc[cohorts.hash_rank_order(shuffled, 'ho', 42)[:600]])
    assert again == small
    other = set(frame.example_id.iloc[cohorts.hash_rank_order(frame, 'siraj', 42)[:600]])
    assert other != small


class _Options:
    sample_chrom = None
    sample_per_dataset = (('siraj', 40), ('ho', 60))


class _BiggerOptions:
    sample_chrom = None
    sample_per_dataset = (('siraj', 150), ('ho', 600))


def test_apply_dataset_sampling_small_run_is_subset_of_large_run():
    frames = {'development': _example_frame('Siraj2026', 800).assign(chrom='1'),
              'ho': _example_frame('Ho2025', 900).assign(chrom='1')}
    small, _ = cohorts._apply_dataset_sampling(frames, _Options, 42)
    large, audit = cohorts._apply_dataset_sampling(frames, _BiggerOptions, 42)
    for name in frames:
        assert set(small[name].example_id) < set(large[name].example_id)
    assert (len(small['ho']), len(large['ho']), len(small['development'])) == (60, 600, 40)
    assert 'sha256_rank' in audit['method']


def test_leakage_guard_removes_dev_rows_near_held_out():
    dev = pd.DataFrame({'chrom': ['1', '1', '2'], 'position': [1_000, 900_000, 5]})
    held = {'ho': pd.DataFrame({'chrom': ['1'], 'position': [100_000]})}
    kept, audit = cohorts._remove_dev_rows_near_held_out(dev, held, 250_000)
    assert kept.position.tolist() == [900_000, 5]
    assert audit['dev_rows_removed_near_held_out'] == 1


class FakeDecima(DecimaGeneScorer):
    # Bypass model loading and genomepy configuration; count how many variants reach the fake Decima.
    def __init__(self, cache_identifier='rep0'):
        self.settings = {'model_identifier': cache_identifier, 'batch_size': 1, 'chunksize': 10,
                         'float_precision': '32', 'reference_cache': True}
        self.cell_states = {'primary_T': {}, 'jurkat': {}}
        self._task_queries = {
            'primary_T': {'query': "cell_type == 'T'", 'on_missing': 'fail', 'proxy': None, 'note': ''},
            'jurkat': {'query': "cell_type == 'Jurkat'", 'on_missing': 'proxy_zero', 'proxy': 'none', 'note': ''}}
        self.scored_batches = []

    def _ensure_model(self):
        self._model = object()

    def _resolve_lightning_device_for_trainer(self):
        return 'cpu'

    def _release_device_memory(self):
        return None

    def _num_workers(self):
        return 1

    def _resolve_task_query(self, task_query):
        return ('t1', 't2') if 'Jurkat' not in task_query else ()


@pytest.fixture
def fake_decima(monkeypatch):
    import decima

    def fake_predict(df, output_pq, **_):
        keys = df.variant_id.tolist()
        FakeDecima.last_batch = keys
        rows = []
        for key in keys:
            if key == 'chr1:20:A:G':
                continue
            base = (int.from_bytes(key.encode()[-6:], 'big') % 1000) / 1000.0
            for gene, shift in (('G1', 0.0), ('G2', 0.2)):
                rows.append({'chrom': 'chr1', 'pos': 1, 'ref': 'A', 'alt': 'G', 'gene': gene, 'start': 0,
                             'end': 1, 'strand': '+', 'gene_mask_start': 0, 'gene_mask_end': 1, 'rel_pos': 1,
                             'ref_tx': 'A', 'alt_tx': 'G', 'tss_dist': 1, 'variant_id': key,
                             't1': base + shift, 't2': base - shift})
        pd.DataFrame(rows).to_parquet(output_pq, index=False)

    monkeypatch.setattr(decima, 'predict_variant_effect', fake_predict)
    FakeDecima.last_batch = None


def _variants(positions):
    return pd.DataFrame({'chrom': '1', 'position': positions, 'ref': 'A', 'alt': 'G',
                         'variant_id': [f'1:{p}:A:G' for p in positions]})


def test_decima_store_is_incremental_reusable_and_order_free(tmp_path, fake_decima):
    scorer = FakeDecima()
    first, _ = scorer.score(_variants([10, 20, 30]), 'ho', tmp_path)
    assert sorted(FakeDecima.last_batch) == ['chr1:10:A:G', 'chr1:20:A:G', 'chr1:30:A:G']
    second, provenance = scorer.score(_variants([40, 30, 10, 20, 50]), 'ho_bigger', tmp_path)
    assert sorted(FakeDecima.last_batch) == ['chr1:40:A:G', 'chr1:50:A:G']
    assert provenance['per_state']['primary_T']['variants_reused_from_cache'] == 3
    lookup = dict(zip([40, 30, 10, 20, 50], second['primary_T'][:, 0]))
    assert [lookup[p] for p in (10, 20, 30)] == pytest.approx(first['primary_T'][:, 0].tolist())
    again, provenance = scorer.score(_variants([10, 20]), 'ho', tmp_path)
    assert provenance['per_state']['primary_T']['status'] == 'reused_from_cache'
    assert np.allclose(again['primary_T'][:, 0], first['primary_T'][:2, 0])
    assert (second['jurkat'] == 0).all()
    assert list((tmp_path / 'raw').glob('decima_raw_*.parquet'))


def test_decima_store_is_invalidated_by_changed_model_and_retires_legacy_files(tmp_path, fake_decima):
    (tmp_path / 'decima_ho_primary_T.parquet').write_bytes(b'legacy')
    scorer = FakeDecima('rep0')
    scorer.score(_variants([10, 20]), 'ho', tmp_path)
    assert not (tmp_path / 'decima_ho_primary_T.parquet').exists()
    assert list((tmp_path / '_stale').glob('decima_ho_primary_T.*.parquet'))
    FakeDecima('ensemble').score(_variants([10, 20]), 'ho', tmp_path)
    assert sorted(FakeDecima.last_batch) == ['chr1:10:A:G', 'chr1:20:A:G']
    assert list((tmp_path / '_stale').glob('decima_store_primary_T.*.parquet'))


def test_decima_flags_variants_without_gene_overlap(tmp_path, fake_decima):
    scores, provenance = FakeDecima().score(_variants([10, 20, 30]), 'ho', tmp_path)
    state = provenance['per_state']['primary_T']
    assert (state['variants_with_gene_overlap'], state['variants_without_gene_overlap']) == (2, 1)
    assert scores['primary_T'][1, 0] == 0.0 and scores['primary_T'][0, 0] != 0.0


def _corrections_file(tmp_path, records):
    path = tmp_path / 'corrections.json'
    path.write_text(json.dumps(records))
    return path


def _record(example_id, field, value, independent=True):
    return {'example_id': example_id, 'field': field, 'value': value, 'reason': 'binding assay',
            'evidence_url': 'https://doi.org/10.0000/test', 'independent_of_labels': independent}


def test_corrections_reach_the_evaluated_frame_and_cover_held_out_rows(tmp_path):
    from regulatory_model.pipeline import _apply_corrections, _expert_prediction_from_actions, load_config
    frame = pd.DataFrame({'example_id': ['a', 'b', 'c'], 'source_effect': [0.5, -0.4, 0.1],
                          'dhs_CD4': [1.0, 1.0, 0.0]})
    path = _corrections_file(tmp_path, [
        _record('a', 'expert_action', 'challenge'), _record('b', 'dhs_CD4', 0),
        _record('zzz', 'expert_action', 'accept'),
        _record('c', 'expert_action', 'challenge', independent=False)])
    corrected, outcome = _apply_corrections(frame, path)
    assert [r['example_id'] for r in outcome['applied']] == ['a', 'b']
    assert [r['example_id'] for r in outcome['skipped_not_in_cohort']] == ['zzz']
    assert outcome['rejected'][0]['rejected_because'] == 'evidence_not_independent_of_mpra_labels'
    assert corrected.loc[corrected.example_id == 'b', 'dhs_CD4'].iloc[0] == 0.0
    assert frame.loc[1, 'dhs_CD4'] == 1.0
    config = load_config()
    ai = np.array([9.0, 9.0, 9.0])
    prediction = _expert_prediction_from_actions(corrected, ai, config)
    assert prediction.tolist() == [0.5, 9.0, 9.0]


def test_corrections_without_a_file_leave_every_row_on_accept(tmp_path):
    from regulatory_model.pipeline import _apply_corrections
    frame = pd.DataFrame({'example_id': ['a'], 'source_effect': [0.2]})
    corrected, outcome = _apply_corrections(frame, None)
    assert corrected.expert_action.tolist() == [''] and outcome['applied'] == []


def test_correction_records_must_declare_label_independence(tmp_path):
    from pydantic import ValidationError
    from regulatory_model.pipeline import CORRECTIONS
    record = _record('a', 'expert_action', 'accept')
    del record['independent_of_labels']
    with pytest.raises(ValidationError):
        CORRECTIONS.validate_python([record])
