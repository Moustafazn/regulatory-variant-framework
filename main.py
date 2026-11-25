"""Run the public-data Jurkat to activated primary-CD4 benchmark."""
import argparse
from pathlib import Path
from regulatory_model.pipeline import DEFAULT_CONFIG, FrameworkOptions, default_output, run_framework
from regulatory_model.data_prep import PROJECT_ROOT


def build_parser():
    parser=argparse.ArgumentParser(prog='regulatory-variants',description=__doc__)
    parser.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'data'/'regulatory_audit')
    parser.add_argument('--output',type=Path,default=default_output(),help='Fresh/empty artifact directory; existing results are never erased.')
    parser.add_argument('--config',dest='config_path',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--reference-fasta',type=Path,default=PROJECT_ROOT/'data'/'reference'/'hg38.fa')
    parser.add_argument('--chain',type=Path,default=PROJECT_ROOT/'data'/'reference'/'hg19ToHg38.over.chain.gz')
    parser.add_argument('--annotations',type=Path,help='Optional precomputed verified accessibility CSV with .json provenance sidecar.')
    parser.add_argument('--source-track',type=Path,default=PROJECT_ROOT/'data'/'accessibility'/'jurkat_ENCFF916JRN.bigWig')
    parser.add_argument('--target-track',type=Path,default=PROJECT_ROOT/'data'/'accessibility'/'cd4_ENCFF124RFI.bigWig')
    parser.add_argument('--max-rows',type=int,default=0,help='Deterministic smoke-test cap; 0 uses the complete eligible cohort.')
    return parser


def main(argv=None):
    parser=build_parser(); args=parser.parse_args(argv)
    try:
        result=run_framework(FrameworkOptions(**vars(args)))
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        parser.exit(2,f'{exc}\n')
    print(f'Results and paper tables: {result["output"]}')
    return 0


if __name__=='__main__': main()
