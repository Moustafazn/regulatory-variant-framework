"""Read-only pre-run verifier: build the exact cohorts for the requested sample caps and prove they are aligned.

    venv/bin/python scripts/verify_run_inputs.py
    venv/bin/python scripts/verify_run_inputs.py --run siraj=400,ho=600 --run siraj=1500,ho=6000

It calls the same ``build_cohort`` the pipeline uses, never calls Decima or the AlphaGenome API, and exits
non-zero if any check fails. The prepared tables it writes are reused by the real run.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyfaidx  # noqa: E402

from regulatory_model.dataset_cohorts import (  # noqa: E402
    _dataset_fold_feasibility, _load_tracks_dataset, _remove_dev_rows_near_held_out,
)
from regulatory_model.evaluation import locus_groups  # noqa: E402
from regulatory_model.liftover import default_hg38_fasta  # noqa: E402
from regulatory_model.pipeline import (  # noqa: E402
    DEFAULT_CONFIG, FrameworkOptions, build_cohort, default_output, load_config,
)

DECIMA_SECONDS_PER_VARIANT = 46.1


# Parse "siraj=400,ho=600" into the tuple form FrameworkOptions expects.
def parse_run(text):
    pairs = []
    for item in text.split(','):
        name, _, count = item.partition('=')
        pairs.append((name.strip().lower(), int(count)))
    return tuple(pairs)


# Collect named pass/fail results with a short detail string.
class Report:
    def __init__(self):
        self.rows = []

    def add(self, label, passed, detail='', warn_only=False):
        passed = bool(passed)
        self.rows.append((label, passed or warn_only, detail))
        tag = 'PASS' if passed else ('WARN' if warn_only else 'FAIL')
        print(f'  [{tag}] {label}' + (f' - {detail}' if detail else ''), flush=True)

    @property
    def failed(self):
        return [row for row in self.rows if not row[1]]


# Count variants a Decima store does not yet hold for one cell state, using the pipeline's key format.
def uncached_variants(frames, cache_dir, state):
    path = Path(cache_dir) / f'decima_store_{state}.parquet'
    held = set(pd.read_parquet(path, columns=['variant_id']).variant_id) if path.is_file() else set()
    keys = set()
    for frame in frames.values():
        keys |= set('chr' + frame.chrom.astype(str) + ':' + frame.position.astype(int).astype(str)
                    + ':' + frame.ref + ':' + frame.alt)
    return len(keys - held), len(keys)


# Verify one sample specification end to end and return its example IDs per dataset.
def verify_run(spec, config, output, report, fasta):
    caps = dict(spec)
    print(f'\n=== run: {",".join(f"{k}={v}" for k, v in spec)} ===', flush=True)
    options = FrameworkOptions(output=output, sample_per_dataset=spec)
    with tempfile.TemporaryDirectory() as scratch:
        frames, _, _, _, sampling, _, leakage = build_cohort(options, config, Path(scratch) / 'verify_log.jsonl')
    # track_coverage_restriction is absent unless the tracks block is enabled under restrict_to_covered.
    restriction = sampling.get('track_coverage_restriction') or {}
    pools = {('siraj' if name == 'development' else name):
             (restriction[name]['rows_with_valid_tracks'] if name in restriction else len(frame))
             for name, frame in frames.items()}
    track_states = tuple(config['fusion'].get('track_cell_states') or config['cell_states'].keys())
    tracks_enabled = 'tracks' in config['fusion'].get('feature_blocks', ())
    held = {k: v for k, v in frames.items() if k != 'development'}
    for name, frame in frames.items():
        key = 'siraj' if name == 'development' else name
        expected = min(caps.get(key, pools[key]), pools[key])
        report.add(f'{key}: sample size', len(frame) == expected,
                   f'got {len(frame)}, requested {caps.get(key)}, track-covered pool {pools[key]}')
        report.add(f'{key}: unique example ids', not frame.example_id.duplicated().any())
        report.add(f'{key}: all rows GRCh38', set(frame.genome_build) == {'GRCh38'})
        mismatches = sum(str(fasta['chr' + str(c)][int(p) - 1:int(p) - 1 + len(r)]).upper() != r
                         for c, p, r in zip(frame.chrom.astype(str), frame.position, frame.ref))
        report.add(f'{key}: ref allele equals hg38 FASTA', mismatches == 0, f'{mismatches} mismatches')
        report.add(f'{key}: finite targets and positive SEs', bool(np.isfinite(frame[['source_effect', 'target_effect']]).all().all()
                                                                    and (frame[['source_se', 'target_se']] > 0).all().all()))
        if tracks_enabled:
            _, tracks = _load_tracks_dataset(frame, key, options.tracks_dir, track_states,
                                             tuple(config['fusion']['alphagenome_track_modalities']))
            report.add(f'{key}: tracks complete and correctly located',
                       tracks['status'] == 'loaded' and tracks['rows_excluded_wrong_coordinate'] == 0,
                       f'{tracks["rows_loaded"]}/{tracks["rows"]} loaded for states {list(track_states)}',
                       warn_only=True)
    if 'is_positive_control' in frames.get('bourges', pd.DataFrame()).columns:
        report.add('bourges: no positive controls', not frames['bourges'].is_positive_control.any())
    _, final = _remove_dev_rows_near_held_out(frames['development'], held, int(config['gap_bp']))
    report.add('no development row within the leakage gap of a held-out row',
               final['dev_rows_removed_near_held_out'] == 0,
               f'{leakage["dev_rows_removed_near_held_out"]} dev rows removed before sampling')
    settings = {**config}
    for name, frame in held.items():
        loci, folds = _dataset_fold_feasibility(frame, settings)
        report.add(f'{name}: {config["folds"]}-fold evaluation feasible', folds == int(config['folds']),
                   f'{loci} independent loci, {folds} feasible folds' + (
                       '; will be reported descriptively' if folds == 0 else ''),
                   warn_only=(name == 'bourges'))
    chroms = frames['development'].chrom.astype(str).value_counts()
    print(f'  development chromosomes: {chroms.head(8).to_dict()}', flush=True)
    todo, total = uncached_variants(frames, output / 'decima_scores', 'primary_T')
    print(f'  Decima: {todo:,} of {total:,} variants need scoring '
          f'(~{todo * DECIMA_SECONDS_PER_VARIANT / 3600:.1f} h at {DECIMA_SECONDS_PER_VARIANT}s each)', flush=True)
    return {('siraj' if k == 'development' else k): sorted(v.example_id) for k, v in frames.items()}


# Run every requested specification, check nesting across them, and write the run plan.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', default=None, help='e.g. siraj=400,ho=600 (repeatable)')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=default_output())
    parser.add_argument('--plan-out', type=Path, default=ROOT / 'output' / 'run_plan.json')
    args = parser.parse_args(argv)
    runs = [parse_run(text) for text in (args.run or ['siraj=400,ho=600', 'siraj=1500,ho=6000'])]
    config = load_config(args.config)
    report = Report()
    fasta = pyfaidx.Fasta(str(default_hg38_fasta(ROOT)))
    plans = [verify_run(spec, config, args.output, report, fasta) for spec in runs]
    ordered = sorted(range(len(runs)), key=lambda i: sum(len(v) for v in plans[i].values()))
    for small, large in zip(ordered[:-1], ordered[1:]):
        for name in plans[small]:
            report.add(f'nesting: {name} in run {small + 1} is a subset of run {large + 1}',
                       set(plans[small][name]) <= set(plans[large][name]))
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(json.dumps({'runs': [dict(r) for r in runs],
                                         'datasets': plans[ordered[-1]]}, indent=1))
    print(f'\n{len(report.rows) - len(report.failed)}/{len(report.rows)} checks passed; plan: {args.plan_out}')
    return 1 if report.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
