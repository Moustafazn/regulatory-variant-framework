"""Prepare verified Ho accessibility annotations from public workbook, reference and tracks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from main import build_parser
from regulatory_model.pipeline import FrameworkOptions,load_config
from regulatory_model.dataset_cohorts import prepare_cohort
from regulatory_model.reporting import write_json


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.annotations is not None:raise ValueError('This builder expects raw tracks; omit --annotations.')
    args.output.mkdir(parents=True,exist_ok=True)
    frame,_,_,provenance=prepare_cohort(FrameworkOptions(**vars(args)),load_config(args.config_path))
    frame.to_csv(args.output/'prepared.csv.gz',index=False)
    write_json(args.output/'preparation_audit.json',provenance)
    print(args.output/'accessibility.csv.gz')

if __name__=='__main__':main()
