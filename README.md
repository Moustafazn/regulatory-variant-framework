# Neurojico

**Human-AI augmented regulatory variant interpretation with rough-soft uncertainty.**

Neurojico combines a state-of-the-art cell-type-aware DNA foundation model (**AlphaGenome**)
with a gene-expression foundation model (**Decima**) and wraps the fused prediction in a
**Ziarko variable-precision rough-soft set** (Ziarko, 1993). Variants whose predictions fall in
the rough-soft **boundary** are routed to a human expert for review; variants whose predictions
fall in the **lower approximation** are trusted. The framework is evaluated on primary
CD4+ T-cell MPRA data from Ho 2025 and Bourges 2020.

The full scientific proposal is in [`proposal/regulatory_variant_proposal.md`](proposal/regulatory_variant_proposal.md).

---

## The five stages (one pipeline, no fallback)

```
+------------------+     +---------------------+     +-------------------+
| 1. Evidence      |     | 2. AI predictor     |     | 3. Rough-soft set |
| AlphaGenome +    | --> | Ridge head per cell | --> | Lower / Upper /   |
| Decima + Ho mech.|     | state on fused rep. |     | Boundary per class|
+------------------+     +---------------------+     +---------+---------+
                                                              | boundary only
                                                              v
                   +--------------------------+     +----------------------+
                   | 5. Prediction            | <-- | 4. Expert review     |
                   | Maintained / Changed /   |     | accept / challenge / |
                   | Unresolved               |     | defer                |
                   +--------------------------+     +----------------------+
```

Target effects are never shown on review cards. Expert corrections apply only to the
fitting fold. Four methods are compared on held-out data: **AI alone**, **Rough-soft AI**,
**Expert alone**, **Neurojico**.

---

## Repository layout

```
neurojico-rough-set/
├── main.py                       # CLI entrypoint
├── pyproject.toml                # dependencies
├── requirements-lock.txt
├── proposal/
│   └── regulatory_variant_proposal.md
├── regulatory_model/
│   ├── config.json               # single source of truth for every hyperparameter
│   ├── data_prep.py              # Ho / Siraj / Bourges extractors
│   ├── evaluation.py             # leakage-aware splits, metrics, selective curves
│   ├── model.py                  # VariablePrecisionRoughSoft, MechanisticFusion, RidgeHead
│   ├── reporting.py              # JSON writer, SHA-256, event log, workspace prep
│   └── training.py               # AlphaGenomeVariantScorer, DecimaGeneScorer, run_neurojico
├── data/
│   ├── hg38.2bit                 # GRCh38 reference (optional, for coordinate validation)
│   ├── processed/                # pre-prepared Ho / Siraj / Bourges CSVs
│   └── regulatory_audit/         # raw source files
└── output/experiments/neurojico/
    ├── alphagenome_cache/        # one .npz + .provenance.json per variant
    ├── decima_cache/
    ├── folds/{ho,bourges}/       # per-fold Ridge + VPRS joblib snapshots
    ├── review_cards.json         # boundary cases only, no target effects
    ├── audit.json                # every SHA-256 + honest limitations
    └── evaluation.json           # four-method comparison + selective curves
```

---

## Install

```bash
cd neurojico-rough-set
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Required credentials

AlphaGenome inference uses the free non-commercial DeepMind API. Register at the AlphaGenome
portal to obtain a key, then export it before running:

```bash
export ALPHA_GENOME_API_KEY=<your key>
```

Decima's HuggingFace weights (`Genentech/decima-model`) are downloaded automatically on first
run into `output/experiments/neurojico/decima_cache/checkpoint/` and SHA-256 hashed into
`audit.json`.

### Required source data

Place these files under `data/regulatory_audit/`:

| File | Source |
|------|--------|
| `ho_supp.xlsx` | Ho 2025, *Nat Genet* (supplementary workbook) |
| `ho_normalized_counts.txt.gz` | Ho 2025 |
| `bourges_barcode_counts.txt.gz` | Bourges 2020, *EMBO Mol Med* |
| `siraj_MPRA.zip` | Siraj 2026, *Nature* |
| `siraj_tables.zip` | Siraj 2026 |
| `ho_bourges_shared_rsids.txt` | Precomputed leakage-control file |

If any of these is missing the pipeline exits non-zero with the exact file name required.

---

## Run

```bash
python main.py --output output/experiments/neurojico
```

With an expert-review file:

```bash
python main.py \\
  --output output/experiments/neurojico \\
  --expert-corrections path/to/expert_corrections.json
```

Skip Decima (declared as a caveat in `audit.json`):

```bash
python main.py --skip-decima
```

On completion the pipeline prints the paths to `evaluation.json`, `audit.json` and
`review_cards.json`.

---

## Outputs

- `review_cards.json` — one card per boundary-case variant. Each card contains AlphaGenome track
  deltas, Decima gene-expression deltas, Ho mechanistic predictors (motifbreakR, delta-SVM), DHS
  context, VPRS lower/upper/support vectors, and the Jurkat-vs-primary-T contrast. **Target
  effects are never on the card.**
- `expert_corrections.template.json` — schema template emitted when no corrections file is
  supplied; a reviewer fills it in and reruns.
- `evaluation.json` — four-method comparison per held-out dataset, selective-risk curves per
  selector, `strict_lower_coverage`, boundary rate, and per-method RMSE 95% CI.
- `audit.json` — every pinned SHA (config, source files, Decima checkpoint directory,
  AlphaGenome API model version), per-variant wall-time, and honest limitations.
- `tuning.json` — beta selection trace on Siraj (no Ho/Bourges label leakage).

---

## Scientific guardrails

- Every numeric hyperparameter is read from `regulatory_model/config.json`.
- No backbone is fine-tuned; only the Ridge head and VPRS parameters are fit.
- No synthetic data, no silent fallbacks.
- Ho-Bourges rsID overlap is enforced before held-out evaluation.
- Expert corrections are rejected if they target a row outside the fitting fold.
- Review cards never contain target effects or standard errors.

---

## limitations

Two biological / methodological limitations apply:

- Primary CD4+ T is addressed via UBERON/CL ontology terms, not donor-matched tissue from Ho.
  Jurkat is addressed via a cell-line proxy ontology term.
- The rough-soft precision parameter beta is tuned on Siraj 2026 (K562 vs HepG2), a
  cross-cell-type proxy for Ho/Bourges primary T. This keeps Ho/Bourges fully held out but
  means beta is optimal for Siraj's biology rather than Ho's.

The following are intentional design choices, documented so they are not mistaken for
limitations: foundation models are used inference-only (no fine-tuning, to keep the framework
portable across backbones); AlphaGenome is served via API (frozen reproducibility is via the
per-response cache + API `model_version`); only the Ridge head and VPRS parameters are fit
locally. Future work: a prospective study to measure real per-card expert review time (currently
declared as 90 s in `config.json`).

---

## License & citation

If you use Neurojico in published work, please cite:

- This repository (DOI to be minted on first release).
- Avsec et al., *Nature* 2026 (AlphaGenome).
- Lal et al., *Nat Methods* 2026 (Decima).
- Ziarko, *JCSS* 1993 (variable-precision rough sets).
- Ho et al., *Nat Genet* 2025; Bourges et al., *EMBO Mol Med* 2020 (MPRA datasets).
