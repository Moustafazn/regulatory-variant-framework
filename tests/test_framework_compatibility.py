"""Offline regression checks for the framework identity and legacy interfaces."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main import build_parser
from regulatory_model import NeurojicoModel, RegulatoryVariantModel
from regulatory_model.env import configure_torch_env_from_config
from regulatory_model.pipeline import (
    DEFAULT_CONFIG, FrameworkOptions, NeurojicoOptions, _write_audit,
    default_output, load_config, run_framework, run_neurojico,
)
from regulatory_model.reporting import sha256


@pytest.fixture
def training():
    count = 36
    evidence = np.linspace(-2, 2, count)
    return pd.DataFrame({
        'example_id': [f'example-{i}' for i in range(count)],
        'variant_id': [f'1:{1000+i}:A:G' for i in range(count)],
        'chrom': '1', 'position': np.arange(1000, 1000+count),
        'genome_build': 'GRCh38', 'ref': 'A', 'alt': 'G',
        'avi_raw_score': evidence, 'avi_phred': np.arange(count) / 3,
        'source_context': 'jurkat', 'target_context': 'primary_T',
        'source_effect': np.sin(evidence),
        'target_effect': np.sin(evidence) + 0.3 * evidence,
        'source_se': 0.08, 'target_se': 0.1, 'effect_covariance': 0.0,
    })


def test_legacy_config_and_api_preserve_predictions(tmp_path, training):
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps({'neurojico': load_config()}))
    candidates = training.iloc[::5].drop(columns=['target_effect', 'target_se'])
    candidates = candidates.assign(position=lambda x: x.position + 10000)
    current = RegulatoryVariantModel(use_decima=False).fit(training).predict(candidates)
    old = NeurojicoModel(use_decima=False, config_path=legacy).fit(training).predict(candidates)
    pd.testing.assert_frame_equal(current, old, check_exact=True)
    assert np.isfinite(current.predicted_context_transfer_delta).all()
    np.testing.assert_allclose(current.filter(regex='^support_').sum(axis=1), 1)
    assert np.isfinite(current.predicted_target_effect).all()
    assert not np.allclose(current.predicted_target_effect,
                           candidates.source_effect.to_numpy() + current.predicted_context_transfer_delta)


def test_existing_public_imports_still_work():
    from regulatory_model import training as compatibility
    assert NeurojicoModel is RegulatoryVariantModel
    assert NeurojicoOptions is FrameworkOptions
    assert run_neurojico is run_framework
    assert compatibility.run_neurojico is run_framework
    assert compatibility.FrameworkOptions is FrameworkOptions


@pytest.mark.parametrize('config', [{}, {'framework': {}, 'neurojico': {}}])
def test_ambiguous_or_missing_config_is_rejected(tmp_path, config):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='exactly one'):
        load_config(path)


@pytest.mark.parametrize('key', ['framework', 'neurojico'])
def test_cpu_configuration_is_applied_for_both_formats(tmp_path, monkeypatch, key):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({key: {'secondary': {'device': 'cpu'}}}))
    monkeypatch.delenv('DISABLE_MPS', raising=False)
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    configure_torch_env_from_config(path)
    assert __import__('os').environ['DISABLE_MPS'] == '1'
    assert __import__('os').environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'


def test_cli_defaults_and_explicit_old_output_path():
    root = Path(__file__).resolve().parents[1]
    parser = build_parser()
    args = parser.parse_args([])
    assert args.output == root / 'output/experiments/regulatory-variant-framework'
    assert args.config == DEFAULT_CONFIG
    assert args.data_root == root / 'data/regulatory_audit'
    assert default_output('neurojico') == root / 'output/experiments/neurojico'
    custom = parser.parse_args(['--output', 'output/experiments/neurojico'])
    assert custom.output == Path('output/experiments/neurojico')


def test_audit_is_written_with_active_config_hash(tmp_path):
    config_path = tmp_path / 'custom.json'
    settings = load_config()
    settings['seed'] += 1
    config_path.write_text(json.dumps({'framework': settings}))
    _write_audit(tmp_path, settings, {'source': 'test'}, {}, [], {}, config_path=config_path)
    audit = json.loads((tmp_path / 'audit.json').read_text())
    assert audit['pipeline'] == 'regulatory-variant-framework v1'
    assert audit['config_sha256'] == sha256(config_path)
    assert audit['config_sha256'] != sha256(DEFAULT_CONFIG)
    assert audit['alphagenome_avi'] == {'source': 'test'}


def test_candidate_target_leakage_is_rejected(training):
    model = RegulatoryVariantModel(use_decima=False).fit(training)
    with pytest.raises(ValueError, match='unlabeled'):
        model.predict(training)


def test_predict_requires_fitting(training):
    with pytest.raises(RuntimeError, match='fit'):
        RegulatoryVariantModel(use_decima=False).predict(training)
