# Regulatory Variant Interpretation Framework

**Context-dependent prediction, integrated molecular evidence, uncertainty, and expert review.**

**Short title.** Regulatory Variant Interpretation Framework.

**Project scope.** This framework covers data harmonization, molecular-evidence fusion, context-transfer prediction, uncertainty analysis, feature-impact studies, expert review, and evaluation. **NeuroJICO** names the human–AI collaboration component and its boundary-selected review arm, not the entire project. The implementation uses the `regulatory_model` package; the repository and distribution are `regulatory-variant-framework`.

**Status.** This document is a research proposal. Local development evaluations have been produced, but the prospective numeric-estimate expert study described below has not been completed. The software currently supports categorical expert actions. Proposed superiority and novelty statements below are hypotheses to substantiate, not established results.

---

## Lay abstract

Most autoimmune-disease risk variants fall in non-coding DNA that controls when and where genes
are turned on. Whether a variant that alters gene regulation in a laboratory T-cell line (Jurkat)
behaves the same way in a patient's own primary T cells is often unclear, and getting the answer
experimentally is slow and expensive. Modern deep-learning models can predict the effect of such
variants in different cell types, but their predictions have no built-in notion of "I'm not sure."
We propose a **Regulatory Variant Interpretation Framework** that combines molecular evidence and AI prediction with a
**rough-soft set**, which splits predictions into *supported*, *plausible*, and *uncertain
(boundary)*. The **NeuroJICO** component routes boundary cases to a human expert. We will test whether this
hybrid is more accurate than the AI alone, than the expert alone, and than the AI with simpler
uncertainty measures, on two independent primary-T-cell MPRA datasets. If supported, this framework offers
a scalable way to combine deep-learning scale with expert judgment where it is actually needed.

---

## 1. Background and motivation

### 1.1 The regulatory variant interpretation gap

Genome-wide association studies have identified thousands of autoimmune-disease risk loci. The
causal variants are overwhelmingly non-coding and are presumed to act by altering regulatory
element activity in disease-relevant cell types. Massively parallel reporter assays (MPRAs)
can quantify the effect of thousands of candidate variants on reporter transcription in a single
experiment. Ho et al. (2025, *Nat Genet*) and Bourges et al. (2020, *EMBO Mol Med*) provide two
of the most carefully controlled MPRA studies in primary CD4⁺ T cells, including the Jurkat
cell line as a model. A central finding across both studies is that **a variant's effect in Jurkat
does not reliably transfer to primary T cells**, and the probability of transfer depends on
chromatin context, motif-level evidence, and transcription-factor availability in a way that is
not captured by any single feature.

### 1.2 What foundation models give us — and what they don't

Three recent foundation models change what is feasible for this problem:

- **AlphaGenome** (Avsec et al., *Nature* 2026) predicts regulatory effects from sequence. This
  project uses two outputs: (i) the public, precomputed AlphaGenome Variant Impact (AVI) SNV
  archive as a global raw-impact and PHRED annotation and (ii) `score_variant()` requests via
  the live API for the five recommended scorers (RNA-seq, CAGE, DNase, ATAC, CHIP_TF), averaged
  across each cell-state's ontology CURIEs into a 10-scalar per-variant NPZ in
  `data/alphagenome_tracks/`. AVI is used genome-wide (keyless, no API calls at prediction
  time); the cell-state track scores are consumed as a dynamic fusion block that is activated
  per dataset only when every variant has an NPZ on disk.
- **Decima** (Lal et al., *Nat Methods* 2026) predicts per-gene expression across >22 million
  single-cell transcriptomes and exposes variant-level effects at single-cell-type resolution.
  Weights are public on HuggingFace.
- **Variable-precision rough sets** (Ziarko, 1993) provide a mathematically grounded way to
  split a classifier's output into *lower* (supported with confidence ≥ β), *upper* (plausible,
  support > 1 − β), and *boundary* (upper minus lower) regions, when combined with a learned
  neighbourhood.

None of these three solves the problem alone. Foundation models score every variant quickly but
produce point estimates with no principled uncertainty, and benchmarking shows systematic
over-confidence on cell-type-transfer cases. Expert review is accurate but does not scale.
Rough sets are a formal framework but need a representation and a human loop.

### 1.3 The gap this proposal addresses

> **Deep-learning predictions of regulatory variant effects in primary T cells are produced at
> scale but without principled uncertainty quantification; expert review is accurate but
> unscalable; no published framework combines them with a declared boundary rule and a
> head-to-head comparison against the AI-alone and expert-alone baselines on independent
> primary-T-cell MPRA data.**

The framework studies this gap, with NeuroJICO supplying the boundary-selected expert-review component.

---

## 2. Hypothesis and specific aims

### 2.1 Primary hypothesis

The **primary biological endpoint** is the *change* in measured allele effect between the
source and target cellular contexts, i.e. `target_effect − source_effect`. This is the one
well-defined MPRA quantity relevant to the figure's central question *"true context difference
or insufficient evidence?"*. Primary-T allele effects are reported as a secondary endpoint.
Ho 2025 (Jurkat → primary CD4⁺ T) and Bourges 2020 (resting → stimulated primary T) are
evaluated as **separate context-transfer problems** and never pooled.

> On each of the two held-out context-transfer tasks (Ho and Bourges) independently, the
> framework with the NeuroJICO component — AlphaGenome + Decima + rough-soft uncertainty + expert review of
> boundary cases — achieves a **lower locus-grouped RMSE on the context-transfer delta at a
> matched reviewed-variant budget** than any of (a) AI alone, (b) Rough-soft AI without expert,
> (c) Expert alone at the same budget, or (d) any of the uncertainty-ranking baselines in §3.4b.

### 2.2 Specific aims

**Aim 1 — Fuse foundation-model evidence into a per-variant representation.** For variants with
an exact record in the downloaded AlphaGenome AVI archive, use the global raw and PHRED impact
scores together with the per-cell-state AlphaGenome track scalars (RNA-seq, CAGE, DNase, ATAC,
CHIP_TF for `primary_T` and `jurkat`), Decima per-cell-state gene-expression deltas, Ho
motifbreakR allele-difference scores, δSVM scores and DHS context annotations. The fusion is
built from an explicit `FeatureBlock` registry (`avi`, `tracks`, `decima`, `mechanistic`,
`dhs`), so blocks can be enabled or disabled per run without touching the model code and the
active block list is written into `evaluation.json`. A **feature-impact study** (block
leave-one-out ΔRMSE and `|coefficient| × σ(feature)` attributions, equivalent to gradient ×
input for the Ridge head) quantifies the contribution of each block and each column on every
dataset.

**Aim 2 — Wrap the AI predictor with a rough-soft uncertainty layer.** We will fit a Ridge
regressor (per cell state) on the fused representation and feed it into a Ziarko variable-
precision rough-soft set to obtain, per variant, the three regions *supported / plausible /
boundary*. The precision parameter β is tuned on Siraj 2026 only — no Ho or Bourges labels
touch the tuning.

**Aim 3 — Route only boundary variants to expert review.** Boundary variants (`upper − lower > 0
∧ max(support) < β`) are exported to `review_cards.json`. Cards expose evidence — AlphaGenome
AVI scores, Decima deltas, Ho mechanistic predictors, DHS, and the Jurkat vs primary-T contrast
— but never the target effect. The reviewer records one of `{accept, challenge, defer}` with a
reason and an `https://` evidence URL. Corrections are applied only to the fitting fold.

**Aim 4 — Benchmark four methods on independent held-out data.** On Ho 2025 and Bourges 2020
(Ho↔Bourges rsID overlap removed) we compare **AI alone / Rough-soft AI / Expert alone /
NeuroJICO** on locus-grouped RMSE with 300-draw equal-locus bootstrap and on selective-risk
curves across five selectors (VPRS lower support, ordinary neighbour support, ensemble spread,
representation distance, random) at matched coverage and matched review time.

---

## 3. Methods

### 3.1 Data

| Role | Dataset | Rationale | Leakage control |
|------|---------|-----------|-----------------|
| Development (β tuning only) | **Siraj 2026** (Nature) | Multi-cell-type MPRA, no overlap with Ho/Bourges primary T loci | No Ho/Bourges label used |
| Held-out | **Ho 2025** (Nat Genet) | Primary CD4⁺ T MPRA with paired Jurkat | Locus-grouped K-fold with 250 kb gap |
| Held-out | **Bourges 2020** (*EMBO Mol Med*) | Resting vs stimulated primary T MPRA | Shared rsIDs removed; cross-dataset locus overlap audited (see below) |

All extractors reject non-finite effects, invalid alleles, invalid genome builds and malformed
coordinates (`validate_frame`). Locus grouping (`locus_groups`) unions variants within 250 kb,
sharing an rsID, sharing a lead SNP, or inside the same user-supplied LD block.

**Independent-validation audit** (addresses reviewer concern about "cross-dataset" claims
being overstated). The pipeline writes `cross_dataset_overlap_audit.json` on every run,
reporting per held-out pair: exact coordinate matches, variants within 250 kb of each other,
shared rsID count, and whether declared genome builds are compatible. A claim of
*independent cross-dataset validation* is only made when exact-coordinate matches are zero
and within-250 kb overlap is a small declared fraction of each dataset. Ho and Bourges are
evaluated as **separate** context-transfer problems and never pooled. The number of
independent loci remaining after exclusions is logged per fold.

### 3.2 Foundation model evidence

- **AlphaGenome AVI**: use the precomputed SNV archive in Tabix format, downloaded from
  the [AlphaGenome Atlas](https://deepmind.google.com/science/alphagenome/_/download/atlas/avi_scores_snvs_tabix.zip)
  and queried locally by `../scripts/download_alphagenome_avi.py`. The 88 GB ZIP is read
  in place; only its small Tabix index is extracted. The pipeline uses the archive's
  global `raw_score` and `PHRED` values as variant-impact features. These scores are not
  cell-state-specific or directional expression deltas. Only reference-matched variants
  with an exact AVI record enter the main cohort analysis; unsupported variants and
  lookup misses are excluded and counted in the audit.
- **Decima**: HuggingFace `Genentech/decima-model` pinned by revision + local SHA-256; CPU-only
  inference, per-cell-state scalar gene-expression delta.
- Neither backbone is fine-tuned; only the Ridge head and VPRS parameters are fit.

### 3.2a Model design and implementation map

The current framework overview is shown in the [README](../README.md#scientific-scope). The [historical NeuroJICO diagram](images/neurojico_model.png) is preserved as an earlier design artifact; its project-wide branding predates the rename.

Figure 1 follows the implemented path: paired source/target MPRA measurements and their
uncertainty provide fitting labels and rough-soft class support, while AlphaGenome, Decima, and
available motif/DHS evidence form the predictor inputs. `MechanisticFusion` combines those
predictors; Ridge estimates the target-minus-source effect; and the rough-soft layer assigns
support to maintained, weakened, strengthened, or reversed effects before defining lower, upper,
and boundary regions. Boundary examples are exported as review cards with target effects
omitted. The diagram distinguishes software outputs from measured expert outcomes. The final
evaluation code writes locus-grouped comparisons and selective-risk curves. The AlphaGenome
predictor input is the AVI archive's global raw and PHRED scores, not cell-state track deltas.

### 3.3 Rough-soft set

Formally, given fused representation `x_i` and target class membership `m_i(c)` (derived from
the measured target effect and its SE via a Gaussian error model), the support in class *c* is

```
S_c(x) = Σ_j w(d(x, x_j)) · m_j(c)  /  Σ_j w(d(x, x_j))
```

with `w(d) = exp(-d / h)` and `h` the median neighbour distance. Then
`lower_c(x) = 1[S_c(x) ≥ β]`, `upper_c(x) = 1[S_c(x) > 1 − β]`, boundary = `upper − lower`.
β is tuned on Siraj 2026 by minimising the equal-locus RMSE of the rough-soft-AI
prediction across K folds; no Ho or Bourges label participates in tuning.

**Why variable-precision rough-soft over plain kNN confidence.** Plain kNN confidence is
`max_c S_c(x)` — the probability of the top class. VPRS instead keeps the full class-support
vector S(x) and uses two orthogonal pieces of information:

- The **top-vs-runner-up margin** `S_{(1)}(x) − S_{(2)}(x)`, used as the ranking signal for
  selective risk. This penalises multimodal neighbourhoods where two classes are near-tied,
  which is the regime VPRS boundary is designed for and where plain `max` is insensitive
  (both classes can have high `max`).
- The **lower / upper / boundary partition** at precision β, used as the hard trust rule for
  the Rough-soft-AI and NeuroJICO methods.

The empirical test of this claim is §3.4b, which compares VPRS head-to-head against
`ordinary_neighbor_support` (`max_c S_c(x)`) and against three further established baselines.
Calibration of the VPRS predicted confidence is not asserted but **measured** per fold via a
reliability diagram written to `evaluation.json`.

### 3.4 Four-method comparison and the matched-budget expert protocol (primary outcome)

For every held-out dataset, under a fixed reviewed-variant budget B (B = number of variants
in the VPRS boundary region at the chosen β), four methods are compared:

- **AI alone** = `RidgeHead.predict` on every variant. B = 0.
- **Rough-soft AI** = AI where VPRS accepts (`lower.sum ≥ 1`); on rejected variants the
  method is defined to output the source-context effect (declared in
  `config.rough_soft_ai_rejected_variant_value`, biological reading *"no evidence of a
  cell-type change"*). B = 0.
- **Expert alone at budget B** = expert's numeric estimate replaces AI on B variants selected
  by `config.evaluation.expert_alone_selection_rule` (default `largest_abs_ai_prediction`;
  alternatives: `random`, `largest_source_se`). On the other variants, the AI prediction is
  used. The selection rule is deliberately **non-VPRS** so that the Expert-alone and
  NeuroJICO arms differ in **which** B variants they review, not just by labelling.
- **NeuroJICO** = expert's numeric estimate on B variants selected by **VPRS boundary**
  (`upper − lower > 0 ∧ max(support) < β`); AI on the rest.

> Both methods review the same number B of variants and spend `t × B` seconds. The only
> difference is which B variants they look at: VPRS boundary for NeuroJICO vs an
> uncertainty-blind rule for Expert-alone. If NeuroJICO wins, it is because VPRS selects the
> variants on which the AI is least reliable and expert review has the most information value.
> This is the paper's claim.

Scoring is reported on **two endpoints**:

1. **Context-transfer endpoint** (primary): RMSE on `target_effect − source_effect`.
2. **Primary-T endpoint** (secondary): RMSE on `target_effect`.

Per endpoint, per dataset: locus-grouped RMSE with equal-locus 95 % CI (300-draw bootstrap
over loci); paired-locus bootstrap of RMSE differences (1 000 draws) between method pairs;
reliability diagram per fold.

### 3.4a Frozen expert-study protocol

The headline claim depends on real expert input. The expert study is specified in advance:

- **Reviewers.** 2–3 independent domain experts (molecular immunologists with MPRA experience),
  blinded to the measured target effect, blinded to each other, blinded to dataset origin.
- **Response.** For every card: (i) a **numeric estimate** of the context-transfer delta
  `Δ = target − source` on the log₂ reporter-activity scale, (ii) a short rationale,
  (iii) one or more `https://` evidence URLs, (iv) `seconds` spent.
- **Budget.** Each reviewer gets exactly B cards per dataset, where B is the NeuroJICO boundary
  count; within-expert cards are a random permutation of the boundary set.
- **Aggregation.** Trimmed mean across reviewers (drop min and max for k ≥ 3; mean otherwise).
- **Current state.** The pipeline emits the frozen card schema in `review_cards.json`; a real
  expert panel has not yet been run. The present proposal commits to running it before the
  four-method comparison is reported as evaluated. The categorical `accept/challenge/defer`
  actions currently in `expert_corrections.json` are a development-time schema; they do not
  substitute for the numeric expert responses required by the headline claim.

### 3.4b Uncertainty-ranking baselines (novelty controls)

VPRS is compared against established genomic-uncertainty baselines at matched coverage on the
same held-out variants and the same AI point prediction. VPRS is scored as the **top-vs-runner-up
support margin** `S_{(1)}(x) − S_{(2)}(x)`, which is mathematically distinct from the max
support used by the plain-kNN baseline; the paper's novelty claim rests on this distinction.

- `ordinary_neighbor_support` = `max_c S_c(x)` — plain weighted-kNN top-class probability.
- `degu_style_ensemble_disagreement` = negative cross-member standard deviation of a
  **5-member Ridge ensemble** fitted on bootstrap resamples of the fit fold, predicting the
  **single `context_transfer` endpoint** (so the baseline measures genuine model-prediction
  variability on the same quantity, not variability across different biological endpoints;
  Toneyan & Koo 2024).
- `representation_distance` = negative mean distance to k-nearest fitting-fold representations.
- `split_conformal` = negative **locally-adaptive** split-conformal residual width: inside
  each fold the fit set is split into inner-fit (train) and calibration (compute absolute
  residuals of the context-transfer prediction); each held-out variant's conformal width is
  the kNN-weighted mean calibration residual of its nearest calibration variants (Vovk et al.
  2005; Papadopoulos 2008). This produces a per-variant ranking, not a constant per fold.
- `random` = control.

A defensible novelty claim requires VPRS to strictly dominate `ordinary_neighbor_support` and
to be competitive with or better than `split_conformal` and `degu_style_ensemble_disagreement`
on both held-out tasks.

### 3.5 Pre-registered success criterion

On each held-out dataset (Ho and Bourges) **independently**, and on the primary
context-transfer endpoint, the paired-locus bootstrap 95 % CI on
`RMSE(NeuroJICO) − RMSE(AI alone)` has its upper bound strictly below zero. A positive
result on one dataset without the other is reported as such, not pooled.

### 3.6 Correctness and validity controls

Given that we use model-defined cell-type tasks (not donor-matched tissue) and that β is tuned on
Siraj (not Ho), reviewers will reasonably ask: *how do we know the AI predictions are
biologically correct for Ho's primary T cells?* We answer that with four layered controls, each
of which stands on its own even if the others fail.

1. **The ground truth is Ho's measured target effect.** The entire evaluation is against
   Ho/Bourges MPRA labels on held-out loci. The AI never sees them during tuning. AVI is a
   global impact annotation rather than a donor-matched, cell-specific measurement; that limit
   is reflected in held-out RMSE rather than hidden. The headline result reports how good or bad the
   predictions are in units of log₂ reporter activity.
2. **Locus-grouped K-fold with a 250 kb gap and rsID-based leakage control.** Any variant
   whose locus or rsID appears in training cannot appear in the held-out fold. Ho↔Bourges
   shared rsIDs are removed before cross-study evaluation. This is enforced by
   `locus_groups` and the shared-rsID file, not by a splitting convention.
3. **Cross-dataset generalisation.** The pipeline is evaluated on *two independent* MPRA
   studies (Ho 2025 and Bourges 2020), from different labs, with different library designs,
   different donor panels, and different stimulation conditions. A result that holds across
   both is unlikely to be an artefact of a single study's experimental idiosyncrasies.
4. **The rough-soft layer makes the AI say "I don't know."** Even if the AI's point
prediction is biased by the model-to-assay context mismatch, the rough-soft lower approximation only
   *accepts* a prediction when the weighted class support reaches β. Variants in the boundary
   region are flagged for expert review precisely so that known model weaknesses (unusual
   motif context, missing TF, cell-state mismatch) are triaged out of the AI-trusted set
   before any downstream claim is made.

Together these controls mean that even in the worst case — the model's cell-type task is a poor
proxy for Ho's donor T cells — the pipeline reports an honest held-out RMSE, flags the
uncertain cases, and does not claim correctness it has not earned. The alternative framing
is: *we cannot wait for a donor-matched foundation model to exist to study primary-T MPRA
variants; we must build the uncertainty-aware wrapper now, with the models and context labels
terms that actually exist, and declare both the controls and their limits.*

---

## 4. Preliminary data and feasibility

### 4.1 Data already in-tree

- Ho 2025 supplementary workbook with validated extractor (`prepare_ho`), including the
  publication post-model QC (both primary-cell plasmid allele means > 20) and donor-level
  normalized counts.
- Bourges 2020 barcode counts with a transparent CPM-pseudocount estimator (`prepare_bourges`).
- Siraj 2026 MPRA archives with deterministic cross-context pairing.
- Pre-computed Ho↔Bourges shared-rsID list for leakage removal.

### 4.2 Compute envelope

- AlphaGenome AVI: the local archive supports large-scale SNV lookup without live model calls.
  It supplies global impact annotations and does not provide cell-state-specific track deltas.
- Decima: CPU/MPS inference. Wall time per variant is measured, not estimated.
- Only Ridge heads and VPRS fits run locally — minutes per fold.

### 4.3 Why this pipeline, now

- AlphaGenome and Decima are both 2026 releases; no earlier benchmark has combined them.
- The rough-soft set, originally proposed in 1993, has not previously been applied to
  foundation-model regulatory-variant predictions.
- The Ho/Bourges held-out setup with pre-removed shared rsIDs is an unusually clean
  cross-study generalisation test.

---

## 5. Significance

- **Methodological.** First reported application of variable-precision rough-soft sets to
  foundation-model regulatory-variant predictions, with a declared boundary rule and a
  four-method comparison.
- **Translational.** A route to triage GWAS fine-mapped candidate variants into high-confidence
  "trust the model" and low-confidence "needs expert eyes" buckets, enabling MPRA follow-up to
  focus on the second group.
- **Methodological transferability.** The same pipeline — swap datasets, update model context
  labels — generalises to any cell-type-matched MPRA with a published foundation-model-queryable
  context.

---

## 6. Limitations

Two limitations are intrinsic to the biology and the data available, and we declare them
up front in `audit.json` so no reviewer can be surprised:

- **Model context is not donor-matched tissue.** Decima's primary CD4⁺ T task is not
  donor-matched to Ho's three primary-T donors; its Jurkat task is a cell-type label rather than
  a sample from the exact cell line used in Ho. AlphaGenome AVI itself is global and has no
  cell-state-specific output. This is a key biological limitation of the study.
- **β is tuned on a cross-cell-type proxy.** We tune the rough-soft precision parameter β on
  Siraj 2026 using the K562-vs-HepG2 contrast, as declared in `config.json`. The cell-type
  distance from Ho/Bourges primary T is intentional (keeps Ho/Bourges labels fully held out)
  but it means the chosen β is optimal for Siraj's biology, not Ho's.

### Design choices (not limitations)

Three decisions are sometimes mistaken for limitations but are actually scope choices,
documented here so a reviewer can see they are deliberate:

- The foundation models are used **inference-only**, with no gradient fine-tuning. This keeps
  the framework portable across backbones (swap AlphaGenome for a successor model without
  retraining), isolates the contribution of the rough-soft layer, and lets any reader
  reproduce exactly our results without GPU hardware or weeks of training.
- **AlphaGenome evidence has two inputs declared in `config.json`.** `backbone` points at the
  frozen AVI SNV archive (`avi_raw_score`, `avi_phred`; keyless, genome-wide, no API calls at
  prediction time). `tracks` points at the live-API score block (`requested_outputs`:
  `RNA_SEQ`, `CAGE`, `DNASE`, `ATAC`, `CHIP_TF`) averaged across each cell-state's ontology
  CURIEs declared in `cell_states` (primary_T: `CL:0000624`, `CL:0000895`, `CL:0000897`;
  jurkat: `CLO:0007045`, `EFO:0002796`) and cached as one 10-scalar NPZ per variant in
  `data/alphagenome_tracks/`. The AVI CSV identities and the track NPZ provenance are both
  audited in `audit.json → alphagenome_avi` and `audit.json → alphagenome_tracks`.
- Only the **Ridge head and VPRS parameters** are fit locally. Both are small, deterministic
  given a seed, and reproducible on any hardware.

### Future work

- Measuring real per-card expert review time (currently a declared 90 s constant in
  `config.json`) under the §3.4a protocol; this complements, not replaces, the committed
  expert study of §3.4a.
- Transfer to additional cell contexts (stimulated vs resting B cells, myeloid lineages) by
  updating the model's configured cell-type tasks in `config.json`.
- A multi-backbone fusion (AlphaGenome AVI + Decima + a successor model) to test whether the
  rough-soft boundary rule generalises across predictor families.
- Numeric-estimate ingestion in `expert_corrections.json`: the current categorical
  `accept/challenge/defer` schema is a development-time harness only. The real four-method
  evaluation reads numeric expert estimates per §3.4a; the proposal commits to running that
  expert panel before reporting the headline comparison as evaluated.

---

## 7. References (short list)

1. Ho J. S. Y. et al. Genetic and epigenetic screens in primary human T cells. *Nat Genet* (2025). doi:10.1038/s41588-025-02301-3
2. Bourges C. et al. Resting and activated primary T cell MPRA. *EMBO Mol Med* (2020). doi:10.15252/emmm.202012112
3. Siraj L. et al. Functional dissection of complex trait variants at single-nucleotide resolution. *Nature* (2026). doi:10.1038/s41586-026-10121-6
4. Avsec Ž. et al. AlphaGenome. *Nature* (2026).
5. Lal A. et al. Decima: variant effects on gene expression from scRNA-seq. *Nat Methods* (2026).
6. Ziarko W. Variable precision rough set model. *J Comput Syst Sci* (1993). doi:10.1016/0022-0000(93)90048-2
7. Coetzee S. G., Coetzee G. A. motifbreakR v2. *Bioinform Adv* (2024). doi:10.1093/bioadv/vbae162
8. Lee D. et al. A method to predict the impact of regulatory variants from DNA sequence (δSVM). *Nat Genet* (2015). doi:10.1038/ng.3331
9. Zain, M., & Hassanien, A. E. *Regulatory Variant Interpretation Framework: context-dependent prediction, integrated molecular evidence, uncertainty, and expert review.* Manuscript in preparation.
