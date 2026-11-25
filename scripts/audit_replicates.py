"""Reproduce the primary-CD4 replicate sensitivity diagnostic without fitting models."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from regulatory_model.replicates import replicate_diagnostics
from regulatory_model.reporting import write_json,sha256


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--draws',type=int,default=50)
    parser.add_argument('--seed',type=int,default=20261122)
    args=parser.parse_args(argv)
    result=replicate_diagnostics(args.counts,args.draws,args.seed)
    result['input_sha256']=sha256(args.counts)
    write_json(args.output,result)
    print(result)

if __name__=='__main__': main()
