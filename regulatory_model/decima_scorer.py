"""Decima variant-effect scorer: real public API, chunk-wise prediction, per-state cache."""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import PROJECT_ROOT


# Wrap Decima's load_decima_model + predict_variant_effect with device, cache, and tasks resolution.
class DecimaGeneScorer:
    """Score variants with the Decima model via its real public API."""

    _VALID_ON_MISSING = frozenset({'fail', 'proxy_zero'})
    _REQUIRED_STANZA_KEYS = frozenset({'query', 'on_missing'})

    def __init__(self, settings):
        self.settings = settings['secondary']
        self.cell_states = settings['cell_states']
        self._model = None
        self._task_queries = self._load_task_queries()
        self._configure_genomes_dir()

    def _load_task_queries(self):
        raw = self.settings.get('task_queries') or {}
        missing_states = [s for s in self.cell_states if s not in raw]
        if missing_states:
            raise ValueError(
                f"config.secondary.task_queries is missing cell_states {missing_states}; "
                "every entry in config.cell_states must have a matching task_queries stanza "
                "with keys {'query', 'on_missing'[, 'proxy', 'note']}.")
        malformed = {
            state: sorted(self._REQUIRED_STANZA_KEYS - set(raw[state]))
            for state in self.cell_states
            if not isinstance(raw[state], dict)
            or self._REQUIRED_STANZA_KEYS - set(raw[state])
        }
        if malformed:
            raise ValueError(
                f"config.secondary.task_queries stanzas have missing required keys: {malformed}.")
        bad_on_missing = {
            state: raw[state]['on_missing']
            for state in self.cell_states
            if raw[state]['on_missing'] not in self._VALID_ON_MISSING
        }
        if bad_on_missing:
            raise ValueError(
                f"config.secondary.task_queries[*].on_missing must be one of "
                f"{sorted(self._VALID_ON_MISSING)}; got {bad_on_missing}.")
        return {
            state: {
                'query': raw[state]['query'],
                'on_missing': raw[state]['on_missing'],
                'proxy': raw[state].get('proxy'),
                'note': raw[state].get('note'),
            }
            for state in self.cell_states
        }

    # Persist genomepy's genomes_dir so every subprocess reads it on import.
    def _configure_genomes_dir(self):
        self._apply_mps_memory_cap()
        self._patch_lightning_cpu_devices()
        cache_dir = self.settings.get('decima_cache_dir')
        if not cache_dir:
            return
        absolute = (PROJECT_ROOT / cache_dir).resolve()
        absolute.mkdir(parents=True, exist_ok=True)
        try:
            from appdirs import user_config_dir
            from genomepy.config import config as genomepy_config
        except ImportError:
            return
        genomepy_config['genomes_dir'] = str(absolute)
        config_dir = Path(user_config_dir('genomepy'))
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'genomepy.yaml'
        desired_line = f'genomes_dir: {absolute}\n'
        if not config_file.is_file() or desired_line not in config_file.read_text():
            config_file.write_text(desired_line)

    # Monkey-patch Lightning Trainer so Decima's ``devices=[int]`` is accepted on CPU.
    def _patch_lightning_cpu_devices(self):
        import pytorch_lightning as pl
        if getattr(pl.Trainer, '_neurojico_cpu_patch', False):
            return
        original_init = pl.Trainer.__init__

        def patched_init(trainer_self, *args, **kwargs):
            devices = kwargs.get('devices')
            if (isinstance(devices, (list, tuple)) and len(devices) == 1
                    and isinstance(devices[0], int)
                    and os.environ.get('DISABLE_MPS') == '1'):
                kwargs['devices'] = devices[0]
            return original_init(trainer_self, *args, **kwargs)

        pl.Trainer.__init__ = patched_init
        pl.Trainer._neurojico_cpu_patch = True

    # Cap Apple MPS unified-memory usage from config.secondary.mps_memory_fraction.
    def _apply_mps_memory_cap(self):
        if 'mps_memory_fraction' not in self.settings:
            raise ValueError(
                "config.secondary.mps_memory_fraction is required; set a float "
                "between 0.1 and 1.0 (e.g. 0.8 keeps 20% of unified memory for the OS)."
            )
        value = float(self.settings['mps_memory_fraction'])
        if not 0.1 <= value <= 1.0:
            raise ValueError(f'mps_memory_fraction must be between 0.1 and 1.0, got {value}')
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = f'{value:.3f}'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = f'{value * 0.9:.3f}'

    # Clamp Decima's DataLoader workers to what the current machine has.
    def _num_workers(self):
        configured = int(self.settings.get('num_workers', 0) or 0)
        available = os.cpu_count() or 1
        if configured > 0:
            return min(configured, available)
        return max(1, available // 2)

    # Suppress Lightning marketing tips and upstream deprecation warnings during scoring.
    @staticmethod
    @contextlib.contextmanager
    def _quiet_decima_output():
        nuisance_substrings = (
            'try installing [litlogger]',
            'try installing [litmodels]',
            'persistent_workers=True',
            'LeafSpec',
        )

        class _Filter:
            def __init__(self, inner):
                self._inner = inner
                self._buffer = ''

            def write(self, text):
                self._buffer += text
                while '\n' in self._buffer:
                    line, self._buffer = self._buffer.split('\n', 1)
                    if not any(token in line for token in nuisance_substrings):
                        self._inner.write(line + '\n')

            def flush(self):
                if self._buffer and not any(token in self._buffer for token in nuisance_substrings):
                    self._inner.write(self._buffer)
                self._buffer = ''
                self._inner.flush()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*persistent_workers=True.*')
            warnings.filterwarnings('ignore', message='.*LeafSpec.*')
            warnings.filterwarnings('ignore', message='.*DataLoader will create .* worker processes.*')
            original_stderr, original_stdout = sys.stderr, sys.stdout
            sys.stderr = _Filter(original_stderr)
            sys.stdout = _Filter(original_stdout)
            try:
                yield
            finally:
                sys.stderr.flush()
                sys.stdout.flush()
                sys.stderr = original_stderr
                sys.stdout = original_stdout

    # Return the torch.device Decima's load_decima_model must land weights on.
    def _resolve_torch_device_for_weights(self):
        import torch
        raw = self.settings.get('device', 'cpu')
        if raw in (None, 'cpu'):
            os.environ['DISABLE_MPS'] = '1'
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
            return torch.device('cpu')
        os.environ.pop('DISABLE_MPS', None)
        if raw == 'mps':
            return torch.device('mps')
        if raw in ('gpu', 'cuda'):
            return torch.device('cuda:0')
        if isinstance(raw, int):
            return torch.device(f'cuda:{int(raw)}')
        if isinstance(raw, str) and raw.isdigit():
            return torch.device(f'cuda:{int(raw)}')
        raise ValueError(
            f"Unsupported Decima device {raw!r}; use 'cpu', 'mps', 'cuda', or an integer index."
        )

    # Return the integer Decima/Lightning's Trainer expects (not a torch.device).
    def _resolve_lightning_device_for_trainer(self):
        raw = self.settings.get('device', 'cpu')
        if raw in (None, 'cpu'):
            os.environ['DISABLE_MPS'] = '1'
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
            return 1
        os.environ.pop('DISABLE_MPS', None)
        if raw in ('mps', 'gpu', 'cuda', 0, '0'):
            return 0
        if isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
            return int(raw)
        raise ValueError(
            f"Unsupported Decima device {raw!r}; use 'cpu', 'mps', 'cuda', or an integer index."
        )

    # Set train_params[key] on the ensemble and every underlying LightningModel.
    def _propagate_train_param(self, key, value):
        model = self._model
        if model is None:
            return
        if hasattr(model, 'train_params'):
            model.train_params[key] = value
        for submodel in getattr(model, 'models', ()) or ():
            if hasattr(submodel, 'train_params'):
                submodel.train_params[key] = value

    # Return accelerator memory to the OS between cell-state passes.
    @staticmethod
    def _release_device_memory():
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    # Load the Decima model weights lazily on the configured device.
    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from decima.hub import load_decima_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'decima package not installed. Install with `pip install decima`.'
            ) from exc
        weights_device = self._resolve_torch_device_for_weights()
        self._model = load_decima_model(
            model=self.settings.get('model_identifier', 'ensemble'),
            device=weights_device)
        expected = weights_device.type
        for name, parameter in self._model.named_parameters():
            if parameter.device.type != expected:
                raise RuntimeError(
                    f'Decima weight {name!r} loaded on {parameter.device} '
                    f'but config.secondary.device requested {expected!r}.'
                )
        if 'pin_memory' in self.settings:
            self._propagate_train_param('pin_memory', bool(self.settings['pin_memory']))


    # Score every variant in ``frame`` and return (per-state scores, provenance).
    def score(self, frame, dataset_name=None, cache_dir=None):
        try:
            from decima import predict_variant_effect  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'decima package not installed. Install with `pip install decima`.'
            ) from exc
        df_variant = pd.DataFrame({
            'chrom': ['chr' + str(c).lstrip('chr') for c in frame.chrom],
            'pos':   frame.position.astype(int).to_numpy(),
            'ref':   frame.ref.astype(str).to_numpy(),
            'alt':   frame.alt.astype(str).to_numpy(),
            'variant_id': frame.variant_id.astype(str).to_numpy(),
        })
        per_state, per_state_prov = {}, {}
        workers = self._num_workers()
        batch_size = int(self.settings.get('batch_size', 1))
        chunksize = int(self.settings.get('chunksize', 1000))
        float_precision = str(self.settings.get('float_precision', '32'))
        reference_cache = bool(self.settings.get('reference_cache', True))
        from decima.data.dataset import VariantDataset
        metadata_columns = set(VariantDataset.DEFAULT_COLUMNS) | {'variant_id'}
        with tempfile.TemporaryDirectory(prefix='neurojico-decima-') as temporary_dir:
            model_ready = False
            for state_name in self.cell_states:
                cached = self._load_cached_state(cache_dir, dataset_name, state_name,
                                                 df_variant.variant_id.tolist())
                if cached is not None:
                    per_state[state_name] = cached.reshape(-1, 1)
                    per_state_prov[state_name] = {'status': 'reused_from_cache',
                                                  'variants_scored': int(len(cached))}
                    print(f'  decima[{state_name}]: reused {len(cached)} cached scores for '
                          f'{dataset_name!r}', flush=True)
                    continue
                stanza = self._task_queries[state_name]
                task_query = stanza['query']
                task_names = self._resolve_task_query(task_query)
                if not task_names:
                    if stanza['on_missing'] == 'fail':
                        raise RuntimeError(
                            f"Decima has no matching tasks for cell_state {state_name!r} "
                            f"(query {task_query!r}). Fix the query in "
                            "config.secondary.task_queries, or set on_missing to 'proxy_zero' "
                            "with an explicit proxy label if a declared zero-fill is intended.")
                    proxy_label = stanza['proxy'] or 'proxy_zero'
                    print(f'  decima[{state_name}]: proxy_zero ({proxy_label!r}) for '
                          f'query {task_query!r}; {len(df_variant)} variants set to 0.0 '
                          f'as declared in config.secondary.task_queries.', flush=True)
                    values = np.zeros(len(df_variant), dtype=np.float32)
                    per_state[state_name] = values.reshape(-1, 1)
                    per_state_prov[state_name] = {
                        'status': 'proxy_zero',
                        'proxy': proxy_label,
                        'proxy_note': stanza['note'],
                        'task_query': task_query,
                        'variants_scored': int(len(df_variant)),
                    }
                    self._save_cached_state(cache_dir, dataset_name, state_name,
                                            df_variant.variant_id.tolist(), values)
                    continue
                if not model_ready:
                    with self._quiet_decima_output():
                        self._ensure_model()
                    model_ready = True
                output_path = Path(temporary_dir) / f'{state_name}.parquet'
                with self._quiet_decima_output():
                    predict_variant_effect(
                        df_variant, output_pq=str(output_path),
                        tasks=task_query, model=self._model,
                        device=self._resolve_lightning_device_for_trainer(),
                        num_workers=workers, batch_size=batch_size, chunksize=chunksize,
                        float_precision=float_precision, reference_cache=reference_cache,
                        include_cols=['variant_id'])
                self._release_device_memory()
                predictions = pd.read_parquet(output_path)
                task_columns = [c for c in predictions.columns if c not in metadata_columns]
                if not task_columns:
                    raise RuntimeError(
                        f'Decima parquet for {state_name} has no task-prediction columns; '
                        f'got {list(predictions.columns)}.')
                predictions['_mean_effect'] = predictions[task_columns].mean(axis=1)
                per_variant_mean = predictions.groupby('variant_id')['_mean_effect'].mean()
                missing = sorted(set(df_variant.variant_id) - set(per_variant_mean.index))
                if missing:
                    print(f'  decima[{state_name}]: no gene overlap for {len(missing)}/'
                          f'{len(df_variant)} variants ({len(missing)/len(df_variant):.1%}); '
                          f'these are set to 0.0 so downstream features stay dense '
                          f'(examples: {missing[:3]})', flush=True)
                per_variant_mean = per_variant_mean.reindex(df_variant.variant_id).fillna(0.0)
                values = per_variant_mean.to_numpy(dtype=np.float32)
                if not np.isfinite(values).all():
                    raise RuntimeError(f'Decima returned nonfinite scores for {state_name}.')
                per_state[state_name] = values.reshape(-1, 1)
                per_state_prov[state_name] = {
                    'status': 'scored', 'task_query': task_query,
                    'task_count': len(task_names),
                    'variants_scored': int(len(df_variant)),
                    'variants_with_gene_overlap': int(len(df_variant) - len(missing)),
                    'variants_without_gene_overlap': int(len(missing)),
                    'gene_overlap_fraction': float(1.0 - len(missing) / max(1, len(df_variant))),
                    'examples_without_gene_overlap': missing[:5],
                }
                self._save_cached_state(cache_dir, dataset_name, state_name,
                                        df_variant.variant_id.tolist(), values)
        return per_state, {
            'model_identifier': self.settings.get('model_identifier', 'ensemble'),
            'variants_scored': int(len(frame)),
            'task_queries': {s: self._task_queries[s]['query'] for s in self.cell_states},
            'task_queries_full': self._task_queries,
            'per_state': per_state_prov}


    # Resolve a Decima metadata query to the matching task names, or () if none / unavailable.
    def _resolve_task_query(self, task_query):
        if task_query is None:
            return ()
        try:
            from decima.hub import load_decima_metadata  # type: ignore
        except ImportError:
            return ()
        if not hasattr(self, '_cell_metadata'):
            self._cell_metadata = load_decima_metadata().obs
        try:
            return tuple(self._cell_metadata.query(task_query).index.tolist())
        except Exception:  # noqa: BLE001
            return ()

    # Return the parquet path used to cache a (dataset, state) scoring result.
    @staticmethod
    def _cache_path(cache_dir, dataset_name, state_name):
        if cache_dir is None or dataset_name is None:
            return None
        return Path(cache_dir) / f'decima_{dataset_name}_{state_name}.parquet'

    # Load a previously cached (dataset, state) score vector if the variant_id list matches exactly.
    def _load_cached_state(self, cache_dir, dataset_name, state_name, expected_ids):
        path = self._cache_path(cache_dir, dataset_name, state_name)
        if path is None or not path.is_file():
            return None
        cached = pd.read_parquet(path)
        if list(cached['variant_id']) != list(expected_ids):
            return None
        return cached['score'].to_numpy(dtype=np.float32)

    # Persist a (dataset, state) score vector to the parquet cache.
    def _save_cached_state(self, cache_dir, dataset_name, state_name, variant_ids, values):
        path = self._cache_path(cache_dir, dataset_name, state_name)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'variant_id': variant_ids,
                      'score': values}).to_parquet(path, index=False)
