"""One-shot: download AlphaGenome track NPZs for exactly the 3-day-plan chr1 subset.

Stops the full download of Siraj (which would need ~83 h). Instead, deterministically
re-selects the same chr1 subset that main.py --sample-chrom chr1 --sample-per-dataset
siraj=200 --sample-per-dataset ho=300 will score, then downloads only the still-missing
NPZs for that subset. Reuses the production downloader's internals so the on-disk
format and provenance exactly match previously-downloaded files.

Run from the repository root. Delete this file after it finishes successfully.

Expected cost at the measured 0.68 variants/s: a few minutes for ~150 Siraj NPZs on chr1.
Ho and Bourges chr1 subsets are already fully covered on disk and need no new downloads.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from download_alphagenome_tracks import (  # noqa: E402
    CELL_STATE_ONTOLOGY_TERMS,
    DATASET_SOURCE_TAG,
    DEFAULT_PREPARED_DIR,
    DEFAULT_TRACK_DIR,
    MODALITIES,
    SEQUENCE_LENGTH,
    aggregate_cell_state_scores,
    build_coordinates,
    npz_stem,
    save_prediction,
)

CAPS = {'siraj': 200, 'ho': 300, 'bourges': None}
SEED = 42


def select_sampled(df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    is_chr1 = df['chrom'].astype(str).str.lower().str.removeprefix('chr').eq('1')
    sub = df.loc[is_chr1].reset_index(drop=True)
    if cap is not None and len(sub) > cap:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(sub), size=cap, replace=False)
        sub = sub.iloc[np.sort(idx)].reset_index(drop=True)
    return sub


class _Args:
    def __init__(self, output_dir: Path, prepared_dir: Path, reference, chain):
        self.output_dir = output_dir
        self.prepared_dir = prepared_dir
        self.reference = reference
        self.chain = chain
        self.bourges_oligos = ROOT / 'data/regulatory_audit/bourges_barcode_counts.txt.gz'
        self.reference_records = None


def main() -> int:
    key = os.environ.get('ALPHAGENOME_API_KEY')
    if not key:
        print('Set ALPHAGENOME_API_KEY in the shell to use AlphaGenome predictions.',
              file=sys.stderr)
        return 2

    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    from download_alphagenome_tracks import load_reference, download_resource, CHAIN_URL

    output_dir = DEFAULT_TRACK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    chain_path = output_dir / 'hg19ToHg38.over.chain.gz'
    reference_path = output_dir / 'hg38.2bit'
    download_resource(CHAIN_URL, chain_path, 'hg19-to-hg38 coordinate chain')
    reference_records = load_reference(reference_path)

    args = _Args(output_dir=output_dir, prepared_dir=DEFAULT_PREPARED_DIR,
                 reference=reference_path, chain=chain_path)
    args.reference_records = reference_records

    client = dna_client.create(key)
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[name] for name in MODALITIES]

    total_requested = 0
    total_downloaded = 0

    for dataset, cap in CAPS.items():
        prepared_df = pd.read_csv(DEFAULT_PREPARED_DIR / f'{dataset}.csv.gz', low_memory=False)
        sampled = select_sampled(prepared_df, cap)
        sampled_ids = set(sampled['example_id'].astype(str))
        tag = DATASET_SOURCE_TAG[dataset]
        print(f'{dataset}: three-day chr1 subset = {len(sampled_ids)} variants', flush=True)

        frame, _, _ = build_coordinates(dataset, args, reference_records)
        frame = frame.loc[frame['example_id'].astype(str).isin(sampled_ids)].reset_index(drop=True)

        on_disk = sum(1 for row in frame.itertuples(index=False)
                      if (output_dir / f'{npz_stem(dataset, row.example_id)}.npz').is_file())
        missing = len(frame) - on_disk
        print(f'{dataset}: resolved {len(frame)} ready; {on_disk} present, {missing} to download.',
              flush=True)
        total_requested += len(sampled_ids)

        started = time.monotonic()
        done_this_run = 0
        for row in frame.itertuples(index=False):
            stem = npz_stem(dataset, row.example_id)
            npz_path = output_dir / f'{stem}.npz'
            if npz_path.is_file():
                continue
            variant = genome.Variant(
                chromosome=str(row.hg38_chrom), position=int(row.hg38_position),
                reference_bases=str(row.hg38_ref).upper(),
                alternate_bases=str(row.hg38_alt).upper(),
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
            except Exception as exc:  # noqa: BLE001 - log and continue; the sample is small
                print(f'{dataset}: skip {row.example_id} ({type(exc).__name__}: {exc})',
                      file=sys.stderr, flush=True)
                continue
            provenance = {
                'dataset': dataset, 'example_id': str(row.example_id),
                'variant_id': str(row.variant_id),
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
                'source': 'scripts/download_chr1_sample_only.py (one-shot top-up)',
            }
            provenance_path = output_dir / f'{stem}.provenance.json'
            save_prediction(npz_path, provenance_path,
                            {k: v.astype(np.float32) for k, v in arrays.items()},
                            provenance)
            done_this_run += 1
            if done_this_run % 10 == 0:
                elapsed = time.monotonic() - started
                rate = done_this_run / elapsed if elapsed > 0 else 0
                print(f'{dataset}: downloaded {done_this_run}/{missing} this run '
                      f'({rate:.2f}/s)', flush=True)
        total_downloaded += done_this_run
        print(f'{dataset}: done. {done_this_run} new NPZs written.', flush=True)

    print(f'\nFinished. Covered the 3-day chr1 subset: {total_requested} variants requested, '
          f'{total_downloaded} newly downloaded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
