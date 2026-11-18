# Regulatory Variant Interpretation Framework

### Context-dependent prediction, integrated molecular evidence, uncertainty, and expert review

A Python framework for predicting how a non-coding regulatory variant's allelic effect changes
between cellular contexts, with an explicit uncertainty layer that separates *confident*
predictions from those that require human review.

---

## Scientific scope and motivation

Most variants associated with human disease fall outside protein-coding regions. Their effect is
regulatory: they change how much a nearby gene is transcribed, and that change is often
**context-dependent** — a variant may alter expression in one cell type and do nothing in another.

Massively parallel reporter assays (MPRAs) measure these allelic effects directly by testing both
alleles of thousands of variants inside living cells. But an MPRA measures one context at a time,
and the measurement error on a single variant is frequently as large as the effect itself. This
creates a practical problem for interpretation:

> When a variant's measured effect differs between two contexts, is that a **true biological
> context difference**, or simply **insufficient evidence** in one of the measurements?

This framework addresses that question directly. It does three things:

1. **Predicts** the context-transfer effect (target effect minus source effect) for a variant,
   fusing genome foundation-model annotations with mechanistic sequence evidence.
2. **Quantifies uncertainty** using a variable-precision rough-soft set (Ziarko 1993), which
   partitions variants into a confidently classified region and a *boundary region* where the
   evidence cannot separate the classes.
3. **Routes the boundary region to expert review**, producing structured review cards and
   accepting corrections that must be declared independent of the measured outcome labels.

The design goal is honesty about uncertainty rather than maximal apparent accuracy. Measurement
error is propagated into the class labels, predictors are strictly separated from outcomes, and
evaluation is grouped by genomic locus so that linkage disequilibrium cannot leak between folds.

---

## Architecture

![Framework architecture](docs/images/neurojico_model.png)

### Pipeline stages

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                           │
│  Regulatory DNA segment ── REF allele ──┐                                        │
│                          └─ ALT allele ──┴─► both alleles tested in each context  │
│                                             (source context ─► target context)   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1 EVIDENCE    │──►│ 2 PREDICTOR   │──►│ 3 ROUGH-SOFT   │──►│ 4 EXPERT     │──►│ 5 OUTPUT     │
│               │   │               │   │   UNCERTAINTY  │   │   REVIEW     │   │              │
│ AlphaGenome   │   │ Ridge heads:  │   │ VPRS over kNN  │   │ review cards │   │ maintained   │
│   AVI         │   │  • source     │   │ neighbourhoods │   │ for boundary │   │ weakened     │
│ AlphaGenome   │   │  • target     │   │                │   │ variants     │   │ strengthened │
│   tracks      │   │  • context    │   │ lower / upper  │   │              │   │ reversed     │
│ Decima        │   │    transfer   │   │ class support  │   │ accept /     │   │ ───────────  │
│ Mechanistic   │   │               │   │                │   │ challenge /  │   │ boundary ►   │
│   (δSVM,      │   │ precision-    │   │ β threshold    │   │ defer        │   │ unresolved   │
│   motifbreakR)│   │ weighted fit  │   │                │   │              │   │              │
│ DHS chromatin │   │               │   │                │   │ independence │   │ + per-class  │
│               │   │               │   │                │   │ enforced     │   │   support    │
└───────────────┘   └───────────────┘   └────────────────┘   └──────────────┘   └──────────────┘
        ▲                                                                              │
        └──────────── development feedback only; held-out labels never seen ───────────┘
```

### Module map

| Module | Responsibility |
| --- | --- |
| `regulatory_model/data_prep.py` | Dataset adapters, schema validation, feature whitelist |
| `regulatory_model/liftover.py` | GRCh37 → GRCh38 lift-over with per-base FASTA verification |
| `regulatory_model/dataset_cohorts.py` | Cohort assembly, AVI/track joins, leakage removal, sampling |
| `regulatory_model/model.py` | `MechanisticFusion`, `RidgeHead`, `VariablePrecisionRoughSoft` |
| `regulatory_model/evaluation.py` | Locus grouping, class labels, grouped CV, bootstrap metrics |
| `regulatory_model/replicates.py` | Replicate-level measurement quality (report-only) |
| `regulatory_model/decima_scorer.py` | Decima inference wrapper with per-state caching |
| `regulatory_model/pipeline.py` | End-to-end orchestration and audit writing |
| `regulatory_model/api.py` | In-memory `fit` / `predict` interface |

---

## Repository structure

```
regulatory-variant-framework/
├── main.py                      # CLI entry point for the full pipeline
├── regulatory_model/            # Library package
│   ├── api.py                   # RegulatoryVariantModel (fit / predict)
│   ├── config.json              # All settings: seeds, margins, cell states, blocks
│   ├── data_prep.py             # Dataset adapters and schema validation
│   ├── dataset_cohorts.py       # Cohort assembly and leakage control
│   ├── decima_scorer.py         # Decima inference and caching
│   ├── evaluation.py            # Splits, labels, grouped metrics
│   ├── liftover.py              # Genome-build normalisation
│   ├── model.py                 # Fusion, Ridge heads, rough-soft set
│   ├── pipeline.py              # Orchestration, audit, evaluation outputs
│   ├── replicates.py            # Replicate quality (never a model input)
│   └── reporting.py             # JSON writing, hashing, provenance
├── scripts/                     # Standalone utilities (read-only unless noted)
│   ├── verify_run_inputs.py     # Pre-run cohort verification; writes the run plan
│   ├── download_alphagenome_avi.py
│   ├── download_alphagenome_tracks.py
│   ├── audit_replicates.py      # Replicate reliability report
│   ├── audit_raw_features.py    # Candidate-feature audit
│   ├── build_expert_evidence.py # Expert-review evidence generator
│   └── diagnose_vprs*.py        # Uncertainty-layer diagnostics
├── docs/
│   ├── METHODS.md               # Detailed methods reference
│   ├── regulatory_variant_proposal.md
│   ├── images/                  # Architecture diagram
│   └── templates/               # Input CSV templates
├── tests/                       # Offline test suite
├── data/                        # Input data (not tracked)
└── output/                      # Generated artefacts (not tracked)
```

---

## Installation

Requires **Python ≥ 3.11**.

```bash
git clone https://github.com/Moustafazn/regulatory-variant-framework.git
cd regulatory-variant-framework

python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -e .                  # core library
pip install -e ".[test]"          # plus the test suite
pip install -e ".[alphagenome]"   # plus the AlphaGenome API client
```

Verify the install:

```bash
venv/bin/python -m pytest -q
```

### Optional external resources

| Resource | Needed for | How to obtain |
| --- | --- | --- |
| AlphaGenome AVI archive | `avi` feature block | `scripts/download_alphagenome_avi.py` |
| AlphaGenome track NPZs | `tracks` feature block | `scripts/download_alphagenome_tracks.py` (needs `ALPHAGENOME_API_KEY`) |
| Decima weights | `decima` feature block | Downloaded automatically from Hugging Face on first use |
| hg38 FASTA / chain file | Lift-over verification | Fetched by the download scripts |

Every block is optional. Blocks without data are reported as disabled in `audit.json` rather than
silently filled with zeros.

---

## How to run

### 1. Verify inputs (recommended first step)

Builds the exact cohort the pipeline will evaluate, checks coordinates, alleles, leakage and fold
feasibility, then writes `output/run_plan.json`. Read-only; makes no API calls.

```bash
venv/bin/python scripts/verify_run_inputs.py --run siraj=400,ho=600
```

### 2. Run the pipeline

```bash
venv/bin/python main.py --sample-per-dataset siraj=400 --sample-per-dataset ho=600
```

Useful options:

| Option | Effect |
| --- | --- |
| `--sample-per-dataset NAME=N` | Cap variants per dataset (deterministic, nested across caps) |
| `--sample-chrom chr1` | Restrict every dataset to one chromosome |
| `--skip-decima` | Skip Decima scoring; declared in `audit.json` |
| `--expert-corrections FILE` | Apply reviewer corrections |
| `--output DIR` | Experiment directory |
| `--config FILE` | Alternative configuration |

### 3. Outputs

| File | Contents |
| --- | --- |
| `evaluation.json` | Metrics for every method and baseline, both endpoints, with CIs |
| `audit.json` | Config hash, input hashes, provenance, timings, declared limitations |
| `review_cards.json` | One card per boundary variant for expert review |
| `feature_impact.json` | Per-block leave-one-out and coefficient attribution |

Runs are reproducible from `config.json`'s seed: sampling uses a SHA-256 rank of
`seed:dataset:example_id`, so a smaller cap is always a subset of a larger one.

---

## How to use it as a package

### Input

`fit()` takes measured variant pairs — the same variant assayed in a source and a target context.
All coordinates must be GRCh38.

| Column | Meaning |
| --- | --- |
| `example_id`, `variant_id` | Identifiers |
| `chrom`, `position`, `ref`, `alt`, `genome_build` | GRCh38 coordinates and alleles |
| `source_context`, `target_context` | Context labels (one pair per model) |
| `source_effect`, `source_se` | Measured effect and standard error in the source context |
| `target_effect`, `target_se` | Measured effect and standard error in the target context |
| `effect_covariance` | Covariance of the two errors; `0` only if genuinely independent |
| `avi_raw_score`, `avi_phred` | AlphaGenome AVI annotations |

`predict()` takes **unlabeled** candidates — `chrom`, `position`, `genome_build`, `ref`, `alt`,
plus AVI columns. Supplying `target_effect` or `target_se` raises an error, which prevents
accidental label leakage. Adding `source_effect` additionally returns a target-effect estimate.

Templates: `docs/templates/training_variants.csv`, `docs/templates/candidate_variants.csv`.

### Process

```python
import pandas as pd
from regulatory_model import RegulatoryVariantModel

training = pd.read_csv("docs/templates/training_variants.csv")
candidates = pd.read_csv("docs/templates/candidate_variants.csv")

model = RegulatoryVariantModel(use_decima=False).fit(training)
predictions = model.predict(candidates)
```

`fit()` validates the schema, fuses the enabled evidence blocks, fits the Ridge heads, and
estimates class support by propagating the reported measurement errors through Monte-Carlo
sampling. `predict()` fuses the same blocks for the candidates and returns both the point
prediction and its rough-soft support.

Constructor options:

```python
RegulatoryVariantModel(
    use_decima=True,     # set False to skip Decima inference
    beta=None,           # override the rough-soft precision threshold (0 < beta < 1)
    config_path=None,    # alternative config.json
)
```

### Output

`predict()` returns one row per candidate:

| Column | Meaning |
| --- | --- |
| `example_id`, `variant_id` | Identifiers carried through |
| `source_context`, `target_context` | The fitted context pair |
| `predicted_context_transfer_delta` | Predicted target effect minus source effect |
| `predicted_target_effect` | Only when `source_effect` was supplied |
| `in_rough_soft_boundary` | `True` when evidence cannot confidently classify the variant |
| `mean_neighbor_distance` | Distance to the k nearest training variants |
| `support_*`, `lower_*`, `upper_*` | Per-class support and rough-set bounds |

Classes are `maintained`, `weakened`, `strengthened` and `reversed`, assigned relative to a
configurable effect margin.

**Interpreting the boundary flag.** `in_rough_soft_boundary=True` does not mean the prediction is
wrong. It means the available evidence does not separate the classes at the configured precision
threshold β, and the variant is a candidate for expert review rather than automated interpretation.

---

## Configuration

All settings live in `regulatory_model/config.json`:

| Key | Purpose |
| --- | --- |
| `seed` | Global determinism |
| `margin` | Effect difference defining a context change |
| `folds`, `gap_bp` | Grouped cross-validation and locus-linkage distance |
| `rough.beta`, `rough.neighbors` | Rough-soft precision threshold and neighbourhood size |
| `head.*` | Ridge α grid, source shrinkage, precision weighting |
| `cell_states` | Ontology terms and context aliases per cell state |
| `fusion.feature_blocks` | Which evidence blocks are active |
| `secondary.task_queries` | Decima task selection per cell state |

---

## Testing

```bash
venv/bin/python -m pytest -q          # full suite
venv/bin/python -m pytest -q -k api   # a subset
```

Tests run entirely offline with synthetic fixtures — no API keys, no downloads, no network.

---

## Citation

If you use this framework, please cite it:

```bibtex
@software{regulatory_variant_framework,
  title  = {Regulatory Variant Interpretation Framework: context-dependent
            regulatory variant interpretation with integrated molecular
            evidence, uncertainty, and expert review},
  author = {Zein, Moustafa and Hassanien, Aboul Ella},
  year   = {2026},
  url    = {https://github.com/Moustafazn/regulatory-variant-framework},
  version = {1.0.0}
}
```

### Methods and data this framework builds on

| Component | Reference | DOI |
| --- | --- | --- |
| Variable-precision rough sets | Ziarko, *J. Comput. Syst. Sci.* (1993) | 10.1016/0022-0000(93)90048-2 |
| deltaSVM | Lee et al., *Nat Genet* (2015) | 10.1038/ng.3331 |
| motifbreakR | Coetzee et al., *Bioinformatics* (2015) | 10.1093/bioinformatics/btv470 |
| HOCOMOCO v11 | Kulakovskiy et al., *NAR* (2018) | 10.1093/nar/gkx1106 |
| DHS index | Meuleman et al., *Nature* (2020) | 10.1038/s41586-020-2559-3 |
| PICS fine-mapping | Farh et al., *Nature* (2015) | 10.1038/nature13835 |
| UK Biobank fine-mapping | Kanai et al., *Nat Genet* (2021) | 10.1038/s41588-021-00870-7 |
| DICE T cell eQTL | Schmiedel et al., *Cell* (2018) | 10.1016/j.cell.2018.10.022 |
| Immune caQTL | Gate et al., *Nat Genet* (2018) | 10.1038/s41588-018-0156-2 |
| Stimulation-responsive chromatin | Calderon et al., *Nat Genet* (2019) | 10.1038/s41588-019-0505-9 |
| ANANASTRA | Abramov et al., *NAR* (2022) | 10.1093/nar/gkac262 |
| T cell MPRA | Mouri et al., *Nat Genet* (2022) | 10.1038/s41588-022-01056-5 |

Foundation models used inference-only: **AlphaGenome** (Google DeepMind) and **Decima**.
