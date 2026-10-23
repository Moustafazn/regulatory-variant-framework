"""Neurojico pipeline entrypoint. Runs the single human-AI framework end-to-end:
AlphaGenome API scoring + Decima gene-expression scoring + rough-soft wrapping +
boundary expert review + four-method evaluation on Ho 2025 + Bourges 2020.
"""
import argparse
import sys
from pathlib import Path

from regulatory_model.training import (
    DEFAULT_CONFIG, NeurojicoOptions, default_output, run_neurojico,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data' / 'regulatory_audit',
                        help='Directory containing Ho / Siraj / Bourges source files')
    parser.add_argument('--output', type=Path, default=default_output('neurojico'),
                        help='Experiment directory; previous contents are overwritten')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='JSON config with a top-level "neurojico" block')
    parser.add_argument('--expert-corrections', type=Path, default=None,
                        help='Optional expert_corrections.json produced by a reviewer')
    parser.add_argument('--skip-decima', action='store_true',
                        help='Skip Decima scoring (declared in audit.json)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    options = NeurojicoOptions(
        output=args.output, data_root=args.data_root, config_path=args.config,
        expert_corrections=args.expert_corrections, skip_decima=args.skip_decima,
    )
    try:
        result = run_neurojico(options)
    except FileNotFoundError as exc:
        print(f'Missing required resource: {exc}', file=sys.stderr)
        sys.exit(2)
    except RuntimeError as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        sys.exit(3)
    print(f'Neurojico output: {result["output"]}')
    print(f'  evaluation.json: {result["output"] / "evaluation.json"}')
    print(f'  audit.json:      {result["output"] / "audit.json"}')
    print(f'  review_cards:    {result["output"] / "review_cards.json"}')
    return result


if __name__ == '__main__':
    main()
