"""Public interfaces and a complete offline synthetic-cohort benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from main import build_parser
from regulatory_model import RegulatoryVariantModel, NeurojicoModel
from regulatory_model.pipeline import FrameworkOptions, load_config, run_framework
from regulatory_model.reporting import sha256


def cohort():
    rng=np.random.default_rng(123);n=360
    source=rng.normal(size=n)
    return pd.DataFrame({'example_id':[f'e{i}' for i in range(n)],'variant_id':[f'1:{i+1}:A:G' for i in range(n)],
        'chrom':'1','position':np.arange(n)+1,'genome_build':'GRCh38','ref':'A','alt':'G',
        'source_context':'Jurkat','target_context':'primary_T','source_effect':source,
        'source_se':rng.uniform(.05,.5,n),'target_effect':.7*source+rng.normal(0,.2,n),
        'target_se':.2,'effect_covariance':0.,'split_group':np.repeat([f'l{i}' for i in range(120)],3),
        'ref_gc':rng.uniform(.2,.8,n),'access_source':rng.normal(size=n)})


def test_api_target_leakage_and_context_guards():
    frame=cohort();model=RegulatoryVariantModel().fit(frame)
    candidates=frame.drop(columns=['target_effect','target_se','effect_covariance'])
    pred=model.predict(candidates)
    np.testing.assert_allclose(pred.predicted_target_effect-frame.source_effect,pred.predicted_context_transfer_delta)
    with pytest.raises(ValueError,match='unlabeled'): model.predict(frame)
    with pytest.raises(ValueError,match='contexts'): model.predict(candidates.assign(target_context='resting_CD4'))
    with pytest.raises(ValueError,match='whitelist'): RegulatoryVariantModel(sequence=['target_effect'])
    assert NeurojicoModel is RegulatoryVariantModel


def test_unfitted_api_fails():
    with pytest.raises(RuntimeError,match='fit'): RegulatoryVariantModel().predict(cohort())


def test_cli_all_input_paths_are_overridable():
    args=build_parser().parse_args(['--data-root','/tmp/public','--reference-fasta','/tmp/hg38.fa','--chain','/tmp/chain.gz'])
    assert args.data_root==Path('/tmp/public') and args.reference_fasta==Path('/tmp/hg38.fa')
    assert FrameworkOptions(**vars(args)).config_path == args.config_path


def test_cli_returns_a_success_exit_code(monkeypatch,capsys):
    import main as cli
    monkeypatch.setattr(cli,'run_framework',lambda _: {'output':'fixture-run'})
    assert cli.main([])==0
    assert 'fixture-run' in capsys.readouterr().out


def test_end_to_end_outputs_are_consistent_and_outcome_blind(tmp_path,monkeypatch):
    import regulatory_model.pipeline as pipeline
    frame=cohort()
    monkeypatch.setattr(pipeline,'prepare_cohort',lambda *_:(frame,['ref_gc'],['access_source'],{'context_limitation':'test proxy'}))
    monkeypatch.setattr(pipeline,'replicate_diagnostics',lambda *a,**k:{'usage':'report-only'})
    counts=tmp_path/'ho_normalized_counts.txt.gz';counts.write_bytes(b'fixture')
    config=load_config();config.update(folds=2,bootstrap=100,replicate_splits=2)
    path=tmp_path/'config.json';path.write_text(json.dumps({'framework':config}))
    out=tmp_path/'run'
    run_framework(FrameworkOptions(output=out,data_root=tmp_path,config_path=path))
    cards=json.loads((out/'review_cards.json').read_text())
    assert not any('target_effect' in c or 'target_se' in c for c in cards)
    assert sum(c['status']=='review_candidate' for c in cards)==int(np.ceil(.05*len(cards)))
    results=json.loads((out/'results.json').read_text())
    assert results['expert_benefit_measured'] is False
    predictions=pd.read_csv(out/'confirmation_predictions.csv.gz')
    name=results['selection']['selected_source_baseline']
    actual=((predictions.target_effect-predictions[name])**2).groupby(predictions.split_group).mean().mean()
    assert results['confirmation'][name]['equal_locus_mse']==pytest.approx(actual)
    manifest=json.loads((out/'artifact_manifest.json').read_text())
    assert all(sha256(out/name)==digest for name,digest in manifest.items())
    from regulatory_model.reporting import regenerate_report
    results_hash=sha256(out/'results.json')
    regenerate_report(out)
    assert sha256(out/'results.json')==results_hash
    regenerated=json.loads((out/'artifact_manifest.json').read_text())
    assert all(sha256(out/name)==digest for name,digest in regenerated.items())
    paper=json.loads((out/'paper_manifest.json').read_text())
    assert [f['number'] for f in paper['figures']]==['Figure 1','Figure 2','Figure 3','Figure 4']
    assert [t['number'] for t in paper['tables']]==['Table 1','Table 2','Table S1','Table S2','Table S3']
    assert all((out/name).is_file() for figure in paper['figures'] for name in figure['files'])
    assert all((out/table['file']).is_file() for table in paper['tables'])
    assert not (out/'benchmark.png').exists()
    saved_bytes=(out/'routing_predictions.csv.gz').read_bytes()
    (out/'routing_predictions.csv.gz').write_bytes(saved_bytes+b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        regenerate_report(out)
    (out/'routing_predictions.csv.gz').write_bytes(saved_bytes)
    split=pd.read_csv(out/'split_manifest.csv.gz')
    assert split.groupby('split_group').partition.nunique().max()==1
    route=pd.read_csv(out/'routing_development_partitions.csv.gz')
    assert route.groupby('split_group').routing_partition.nunique().max()==1
    with pytest.raises(ValueError,match='not empty'):
        run_framework(FrameworkOptions(output=out,config_path=path))


def test_selective_risk_preserves_original_locus_weights():
    from regulatory_model.benchmarking import selective_summary,paired_selective_gain
    y=np.array([0.,2.,4.]);groups=['a','a','b'];mask=[False,True,False]
    out=selective_summary(y,np.zeros(3),groups,mask)
    assert out['retained_mse']==pytest.approx(16/1.5)
    assert out['weighted_coverage']==.75
    gain=paired_selective_gain(y,np.zeros(3),groups,mask,mask,draws=100)
    assert gain['ci95']==[0.,0.]


def test_cards_cannot_use_target_measurements():
    from regulatory_model.benchmarking import review_cards
    frame=cohort().iloc[:20]
    args=(np.zeros(20),np.arange(20),'source_se',.05,'context limit')
    assert review_cards(frame,*args)==review_cards(frame.assign(target_effect=1e20,target_se=1e20),*args)


def test_routing_ties_are_order_invariant_and_reject_nonfinite():
    from regulatory_model.benchmarking import review_mask
    ids=np.array(['a','b','c','d']);scores=np.ones(4)
    a=review_mask(scores,ids,.25);b=review_mask(scores[::-1],ids[::-1],.25)[::-1]
    np.testing.assert_array_equal(a,b)
    assert a.sum()==1
    with pytest.raises(ValueError):review_mask([np.nan],['x'],.1)


def test_quantiles_are_fitted_only_on_training_values():
    from regulatory_model.benchmarking import QuantileDiscretizer
    x=np.array([[0.,np.nan],[0.,2.],[0.,2.],[1.,3.],[2.,4.],[3.,5.]])
    model=QuantileDiscretizer(3).fit(x);before=[v.copy() for v in model.edges_]
    assert len(set(model.transform(x)[:3,0]))==1
    assert model.transform(x)[0,1]==-1
    model.transform([[1e15,-1e15]])
    for a,b in zip(before,model.edges_):np.testing.assert_array_equal(a,b)


def test_small_calibration_samples_cannot_claim_finite_coverage():
    from regulatory_model.benchmarking import conformal_quantile
    assert conformal_quantile([1,2,3],.25)==3
    assert np.isinf(conformal_quantile([1,2,3],.1))


@pytest.mark.parametrize('beta',[.35,.5,float('nan'),1.01])
def test_rough_membership_rejects_invalid_beta(beta):
    from regulatory_model.model import validate_beta
    with pytest.raises(ValueError):validate_beta(beta)


def test_readme_api_examples_run():
    root=Path(__file__).resolve().parents[1]
    training=pd.read_csv(root/'docs/templates/training_variants.csv')
    candidates=pd.read_csv(root/'docs/templates/candidate_variants.csv')
    predictions=RegulatoryVariantModel().fit(training).predict(candidates)
    assert len(predictions)==len(candidates) and np.isfinite(predictions.predicted_target_effect).all()
