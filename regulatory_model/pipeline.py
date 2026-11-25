"""Reproducible immune-cell transfer benchmark and outcome-blind review prioritization."""
from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .benchmarking import (locus_weights, regression_summary, paired_mse_gain, selective_summary,
    paired_selective_gain, review_mask, review_cards, conformal_quantile, transfer_evidence)
from .data_prep import PROJECT_ROOT
from .dataset_cohorts import prepare_cohort
from .model import EffectRegressor, VariablePrecisionRoughSoft
from .reporting import write_json, sha256, platform_metadata
from .replicates import replicate_diagnostics

DEFAULT_CONFIG = Path(__file__).with_name('config.json')
BASELINES = ['train_mean','source_copy','source_calibrated','source_shrunk','source:ridge','source:gbm']
CANDIDATES = BASELINES + [f'{family}:{kind}' for family in ('sequence','source_sequence','source_sequence_context')
                         for kind in ('ridge','gbm')] + ['source_sequence:binned_gbm']


def load_config(path=DEFAULT_CONFIG):
    data = json.loads(Path(path).read_text())
    keys = [k for k in ('framework','neurojico') if k in data]
    if len(keys) != 1:
        raise ValueError('Configuration must contain exactly one framework or neurojico block.')
    cfg = data[keys[0]]
    if cfg['folds'] < 2 or cfg['bootstrap'] < 100 or cfg['gap_bp'] <= 0:
        raise ValueError('Need at least two folds, 100 bootstrap draws and a positive locus gap.')
    for name in ('minimum_prediction_gain','minimum_routing_gain'):
        if not 0 <= cfg[name] < 1:
            raise ValueError(f'{name} must lie in [0,1).')
    if not 0 < cfg['primary_review_budget'] < 1 or not all(0 < b < 1 for b in cfg['review_budgets']):
        raise ValueError('Review budgets must lie in (0,1).')
    return cfg


def default_output(name='regulatory-variant-framework'):
    return PROJECT_ROOT/'output'/'experiments'/name


@dataclass(frozen=True)
class FrameworkOptions:
    output: Path
    data_root: Path = PROJECT_ROOT/'data'/'regulatory_audit'
    config_path: Path = DEFAULT_CONFIG
    annotations: Path | None = None
    reference_fasta: Path = PROJECT_ROOT/'data'/'reference'/'hg38.fa'
    chain: Path = PROJECT_ROOT/'data'/'reference'/'hg19ToHg38.over.chain.gz'
    source_track: Path = PROJECT_ROOT/'data'/'accessibility'/'jurkat_ENCFF916JRN.bigWig'
    target_track: Path = PROJECT_ROOT/'data'/'accessibility'/'cd4_ENCFF124RFI.bigWig'
    max_rows: int = 0


def predict_candidate(name, train, held, sequence, context, seed):
    return EffectRegressor(name, sequence, context, seed).fit(train).predict(held)


def routing_experiment(dev, confirm_inputs, sequence, context, config, output, context_note):
    """Fit, score-training, policy-selection and calibration loci are disjoint.

    This function receives NO confirmation outcomes. The source-shrinkage predictor
    is fixed in advance, independent of the primary prediction model competition.
    """
    seed = int(config['seed'])
    fit_idx, rest = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(dev, groups=dev.split_group))
    scale_rel, rest_rel = next(GroupShuffleSplit(n_splits=1, test_size=.6, random_state=seed+1).split(dev.iloc[rest], groups=dev.iloc[rest].split_group))
    remainder = rest[rest_rel]
    valid_rel, cal_rel = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed+2).split(dev.iloc[remainder], groups=dev.iloc[remainder].split_group))
    partitions = {'fit':fit_idx, 'score_training':rest[scale_rel], 'policy_selection':remainder[valid_rel],
                  'calibration':remainder[cal_rel]}
    manifest = dev[['example_id','split_group']].copy()
    for label, idx in partitions.items(): manifest.loc[idx,'routing_partition'] = label
    manifest.to_csv(output/'routing_development_partitions.csv.gz', index=False)
    fit, scale, valid, cal = [dev.iloc[partitions[k]] for k in partitions]
    predictor = EffectRegressor(config['routing_predictor'], sequence, context, seed).fit(fit)
    # Use the same source + molecular features for learned difficulty and rough-set scores.
    representation = EffectRegressor('source_sequence_context:ridge', sequence, context, seed)
    representation.prior = predictor.prior
    x, xs, xv, xc, xh = [representation.design(f) for f in (fit,scale,valid,cal,confirm_inputs)]
    ps, pv, pc, ph = [predictor.predict(f) for f in (scale,valid,cal,confirm_inputs)]
    weights = locus_weights(scale.split_group); weights /= weights.mean()
    scale_model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=30,
        l2_regularization=10, early_stopping=False, random_state=seed).fit(
            xs, abs(scale.target_effect.to_numpy()-ps), sample_weight=weights)
    floor = max(float(np.median(abs(scale.target_effect.to_numpy()-ps)))*.1, 1e-4)
    sv, sc, sh = [np.maximum(scale_model.predict(z),floor) for z in (xv,xc,xh)]
    scores_valid = {'source_se':valid.source_se.to_numpy(), 'predicted_absolute_error':sv}
    scores_held = {'source_se':confirm_inputs.source_se.to_numpy(), 'predicted_absolute_error':sh,
                   'random':np.random.default_rng(seed).random(len(confirm_inputs))}
    for membership in ('crisp','soft'):
        rough = VariablePrecisionRoughSoft({**config,'rough':{**config['rough'],'membership':membership}}).fit(x,fit)
        for z, scores in ((xv,scores_valid),(xh,scores_held)):
            scores['vprs_'+membership] = 1-rough.predict(z)['support'].max(axis=1)
    budget = config['primary_review_budget']
    validation = {name: selective_summary(valid.target_effect, pv, valid.split_group,
                         review_mask(score,valid.example_id,budget)) for name,score in scores_valid.items()}
    reference = min(('source_se','predicted_absolute_error'), key=lambda n: validation[n]['retained_mse'])
    challenger = min(('vprs_crisp','vprs_soft'), key=lambda n: validation[n]['retained_mse'])
    selection = {'reference':reference,'challenger':challenger,'validation':validation,
                 'primary_budget':budget,'predictor':config['routing_predictor'],
                 'partition_loci':{k:int(dev.iloc[v].split_group.nunique()) for k,v in partitions.items()}}
    write_json(output/'routing_selection.json',selection)
    cal_scores = abs(cal.target_effect.to_numpy()-pc)/sc
    per_locus = pd.Series(cal_scores).groupby(cal.split_group.to_numpy()).max().to_numpy()
    q = conformal_quantile(per_locus, config['conformal_alpha'])
    saved = confirm_inputs[['example_id','split_group','source_effect','source_se']].copy()
    saved['prediction'] = ph
    for name, score in scores_held.items(): saved['score_'+name] = score
    saved['interval_half_width'] = q*sh
    saved.to_csv(output/'routing_predictions.csv.gz',index=False)
    # Use the selected ordinary reference policy operationally; promotion of rough
    # scores is a research comparison and is not made using confirmation outcomes.
    cards = review_cards(confirm_inputs, ph, scores_held[reference], reference, budget, context_note)
    write_json(output/'review_cards.json', cards)
    return selection, saved


def evaluate_routing(confirm, selection, saved, config):
    y, p, groups = confirm.target_effect.to_numpy(), saved.prediction.to_numpy(), confirm.split_group.to_numpy()
    curves=[]
    for column in saved:
        if not column.startswith('score_'): continue
        for budget in sorted(set([0.,*config['review_budgets'],config['primary_review_budget']])):
            mask = review_mask(saved[column].to_numpy(),confirm.example_id,budget)
            curves.append({'selector':column[6:],'budget':budget,**selective_summary(y,p,groups,mask)})
    budget = config['primary_review_budget']
    a = review_mask(saved['score_'+selection['reference']].to_numpy(),confirm.example_id,budget)
    b = review_mask(saved['score_'+selection['challenger']].to_numpy(),confirm.example_id,budget)
    gain = paired_selective_gain(y,p,groups,a,b,draws=config['bootstrap'],seed=config['seed'])
    passed = (gain['relative_reduction'] is not None and gain['relative_reduction'] >= config['minimum_routing_gain']
              and gain['ci95'][0] is not None and gain['ci95'][0] > 0)
    covered = abs(y-p) <= saved.interval_half_width.to_numpy()
    return {'selection':selection,'curves':curves,'challenger_gain_vs_selected_reference':gain,
            'useful_routing_improvement_verified':bool(passed),
            'interpretation':'Observed-error prioritization only. Retained error excludes review candidates; no corrected outcomes or human benefit assumed.',
            'conformal':{'target_whole_locus_coverage':1-config['conformal_alpha'],
                         'empirical_row_coverage':float(covered.mean()),
                         'empirical_whole_locus_coverage':float(pd.Series(covered).groupby(groups).all().mean()),
                         'mean_interval_width':float(2*saved.interval_half_width.mean()),
                         'unbounded':bool(np.isinf(saved.interval_half_width).any()),
                         'assumption':'Exchangeability of independent loci; not guaranteed in this selected genomic cohort.'}}


def run_framework(options):
    config = load_config(options.config_path)
    if options.max_rows < 0: raise ValueError('max_rows must be nonnegative.')
    output = Path(options.output).resolve()
    # Never clear an arbitrary user path; each run must have a fresh artifact directory.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}. Choose a fresh run directory.')
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config['seed'])
    print('Preparing public Ho data and verifying GRCh38 reference alleles...',flush=True)
    frame, sequence, context, provenance = prepare_cohort(options, config)
    frame.to_csv(output/'prepared.csv.gz',index=False)
    di, ci = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(frame,groups=frame.split_group))
    dev, confirm = frame.iloc[di].reset_index(drop=True), frame.iloc[ci].reset_index(drop=True)
    if dev.split_group.nunique() < max(20, config['folds']) or confirm.split_group.nunique() < 2:
        raise ValueError('Too few independent loci for this protocol; use more rows.')
    manifest=frame[['example_id','variant_id','chrom','position','genome_build','split_group']].copy()
    manifest['partition']='development'; manifest.loc[ci,'partition']='confirmation'
    manifest.to_csv(output/'split_manifest.csv.gz',index=False)
    code = [PROJECT_ROOT/'main.py', *sorted(Path(__file__).parent.glob('*.py')), options.config_path,
            *sorted((PROJECT_ROOT/'scripts').glob('*.py')),
            PROJECT_ROOT/'requirements-lock.txt', PROJECT_ROOT/'pyproject.toml']
    protocol={'study':config['study'],'config':config,'dataset':'Ho2025',
        'primary_endpoint':'target allelic Log2Skew given source effect and SE; not reporter activity',
        'primary_metric':'equal-locus mean squared error',
        'prediction_gate':'At least 5% lower MSE and paired locus bootstrap 95% lower bound above zero versus development-selected strongest source baseline.',
        'routing_gate':'Secondary: at least 5% lower retained MSE and paired locus bootstrap 95% lower bound above zero at the same 5% row-review budget.',
        'confirmation_scope':'Previously explored public cohort. Fixed internal evaluation; not a preregistered or independent validation experiment.',
        'candidates':CANDIDATES,'sequence_columns':sequence,'context_columns':context,
        'development_rows':len(dev),'confirmation_rows':len(confirm),
        'development_loci':int(dev.split_group.nunique()),'confirmation_loci':int(confirm.split_group.nunique()),
        'code_sha256':{str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p):sha256(p) for p in code},
        'runtime':platform_metadata(),'provenance':provenance}
    write_json(output/'protocol.json',protocol)
    print(f'{len(dev):,} development / {len(confirm):,} confirmation rows; {dev.split_group.nunique()} / {confirm.split_group.nunique()} loci',flush=True)
    pred={name:np.full(len(dev),np.nan) for name in CANDIDATES}
    for fold,(fit,held) in enumerate(GroupKFold(n_splits=config['folds']).split(dev,groups=dev.split_group)):
        for name in CANDIDATES:
            pred[name][held]=predict_candidate(name,dev.iloc[fit],dev.iloc[held],sequence,context,seed)
        print(f'Development fold {fold+1}/{config["folds"]} complete',flush=True)
    summaries={name:regression_summary(dev.target_effect,p,dev.split_group) for name,p in pred.items()}
    baseline=min(BASELINES,key=lambda n:summaries[n]['equal_locus_mse'])
    augmented=[n for n in CANDIDATES if n.startswith('source_sequence')]
    best=min(augmented,key=lambda n:summaries[n]['equal_locus_mse'])
    overall=min(CANDIDATES,key=lambda n:summaries[n]['equal_locus_mse'])
    context_best=min([n for n in CANDIDATES if n.startswith('source_sequence_context:')],key=lambda n:summaries[n]['equal_locus_mse'])
    parent='source_sequence:'+context_best.split(':')[1]
    selection={'selected_source_baseline':baseline,'selected_augmented':best,'overall_development_winner':overall,
               'context_candidate':context_best,'context_parent':parent,'development':summaries}
    write_json(output/'selection.json',selection)
    dev[['example_id','split_group','source_effect','source_se','target_effect','target_se']].assign(**pred).to_csv(output/'development_predictions.csv.gz',index=False)
    print(f'Selected augmented: {best}; source baseline: {baseline}',flush=True)
    # Confirmation predictor inputs exclude all measured target values.
    inputs=confirm.drop(columns=['target_effect','target_se','effect_covariance'])
    chosen=list(dict.fromkeys([best,baseline,context_best,parent,overall,'source_calibrated','source_copy','train_mean']))
    cp={name:predict_candidate(name,dev,inputs,sequence,context,seed) for name in chosen}
    route_selection, route_predictions=routing_experiment(dev,inputs,sequence,context,config,output,provenance['context_limitation'])
    # Both predictor and policy choices are frozen before confirmation scoring.
    gain=paired_mse_gain(confirm.target_effect,cp[baseline],cp[best],confirm.split_group,draws=config['bootstrap'],seed=seed)
    gate=gain['relative_reduction'] is not None and gain['relative_reduction']>=config['minimum_prediction_gain'] and gain['ci95'][0]>0
    confirm[['example_id','split_group','source_effect','source_se','target_effect','target_se']].assign(**cp).to_csv(output/'confirmation_predictions.csv.gz',index=False)
    routing=evaluate_routing(confirm,route_selection,route_predictions,config)
    evidence=transfer_evidence(confirm,margin=config['margin'])
    evidence.insert(0,'example_id',confirm.example_id.to_numpy())
    evidence.to_csv(output/'retrospective_transfer_evidence.csv.gz',index=False)
    result={'selection':selection,'confirmation':{n:regression_summary(confirm.target_effect,p,confirm.split_group) for n,p in cp.items()},
        'primary_gain_vs_source':gain,'useful_prediction_improvement_verified':bool(gate),
        'context_increment':paired_mse_gain(confirm.target_effect,cp[parent],cp[context_best],confirm.split_group,draws=config['bootstrap'],seed=seed),
        'routing':routing,'transfer_evidence_counts':evidence.evidence_status.value_counts().to_dict(),
        'transfer_evidence_caveat':'Retrospective Gaussian uncertainty sensitivity with assumed zero covariance; not ground-truth classes.',
        'expert_benefit_measured':False,
        'conclusion':'Useful prediction improvement established on this internal partition.' if gate else 'Useful prediction improvement not established on this internal partition.'}
    write_json(output/'results.json',result)
    print('Computing report-only primary-cell replicate diagnostics...',flush=True)
    counts=options.data_root/'ho_normalized_counts.txt.gz'
    diagnostics=replicate_diagnostics(counts,draws=config['replicate_splits'],seed=seed)
    diagnostics['input_sha256']=sha256(counts)
    write_json(output/'replicate_diagnostics.json',diagnostics)
    joblib.dump(EffectRegressor(overall,sequence,context,seed).fit(dev),output/'effect_regressor.joblib')
    from .reporting import write_study_summary
    write_study_summary(output,protocol,result,diagnostics)
    write_json(output/'artifact_manifest.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(result['conclusion'],gain,flush=True)
    return {'output':output,'evaluation':result}


NeurojicoOptions=FrameworkOptions
run_neurojico=run_framework
