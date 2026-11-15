"""Run the Regulatory Variant Interpretation Framework, including NeuroJICO review."""
import argparse
import sys
from pathlib import Path

from regulatory_model.training import (
    DEFAULT_CONFIG, FrameworkOptions, default_output, run_framework,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(prog='regulatory-variants', description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data' / 'regulatory_audit',
                        help='Directory containing Ho / Siraj / Bourges source files')
    parser.add_argument('--avi-dir', type=Path, default=PROJECT_ROOT / 'data' / 'alphagenome_avi',
                        help='Directory containing extracted per-dataset AlphaGenome AVI CSVs')
    parser.add_argument('--tracks-dir', type=Path,
                        default=PROJECT_ROOT / 'data' / 'alphagenome_tracks',
                        help='Directory with per-variant AlphaGenome cell-state track NPZs')
    parser.add_argument('--require-tracks', action='store_true',
                        help='Fail if the track NPZ coverage is incomplete for any dataset.')
    parser.add_argument('--output', type=Path, default=default_output(),
                        help='Experiment directory; previous contents are overwritten')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='JSON config with a "framework" block (legacy "neurojico" accepted)')
    parser.add_argument('--expert-corrections', type=Path, default=None,
                        help='Optional expert_corrections.json produced by a reviewer')
    parser.add_argument('--skip-decima', action='store_true',
                        help='Skip Decima scoring (declared in audit.json)')
    parser.add_argument('--sample-chrom', type=str, default=None,
                        help='Restrict every dataset to a single chromosome before Decima scoring '
                             '(e.g. chr1); recorded in audit.json -> decima.dataset_sampling.')
    parser.add_argument('--sample-per-dataset', action='append', default=[],
                        metavar='NAME=N',
                        help='Cap variants per dataset after chromosome filtering, e.g. '
                             '--sample-per-dataset siraj=200 --sample-per-dataset ho=300. '
                             'Deterministic given config.seed; recorded in audit.json.')
    return parser


def _parse_sample_per_dataset(entries):
    parsed = []
    for raw in entries or ():
        if '=' not in raw:
            raise SystemExit(f'--sample-per-dataset expects NAME=N, got {raw!r}')
        name, _, count = raw.partition('=')
        name = name.strip().lower()
        if name not in ('siraj', 'ho', 'bourges'):
            raise SystemExit(f"--sample-per-dataset name must be one of 'siraj', 'ho', 'bourges'; got {name!r}")
        try:
            cap = int(count)
        except ValueError as exc:
            raise SystemExit(f'--sample-per-dataset cap must be an integer; got {count!r}') from exc
        if cap <= 0:
            raise SystemExit(f'--sample-per-dataset cap must be positive; got {cap}')
        parsed.append((name, cap))
    return tuple(parsed)


def main(argv=None):
    args = build_parser().parse_args(argv)
    options = FrameworkOptions(
        output=args.output, data_root=args.data_root, avi_dir=args.avi_dir,
        tracks_dir=args.tracks_dir, config_path=args.config,
        expert_corrections=args.expert_corrections, skip_decima=args.skip_decima,
        require_tracks=args.require_tracks,
        sample_chrom=args.sample_chrom,
        sample_per_dataset=_parse_sample_per_dataset(args.sample_per_dataset),
    )
    try:
        result = run_framework(options)
    except FileNotFoundError as exc:
        print(f'Missing required resource: {exc}', file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        sys.exit(3)
    print(f'Regulatory variant framework output: {result["output"]}')
    print(f'  evaluation.json:    {result["output"] / "evaluation.json"}')
    print(f'  audit.json:         {result["output"] / "audit.json"}')
    print(f'  review_cards:       {result["output"] / "review_cards.json"}')
    print(f'  feature_impact.json:{result["output"] / "feature_impact.json"}')
    return result


if __name__ == '__main__':
    main()
