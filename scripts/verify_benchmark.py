"""Run the benchmark, or regenerate tables, figures and architecture from a saved run."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from main import main
from regulatory_model.reporting import regenerate_report

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--report-only':
        regenerate_report(Path(sys.argv[2]))
        print(f'Regenerated tables, figures and architecture in {sys.argv[2]}')
    else:
        main()
