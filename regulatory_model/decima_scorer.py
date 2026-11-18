"""Decima variant-effect scorer: real public API, chunk-wise prediction, per-state cache."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import PROJECT_ROOT
from .liftover import LIFTOVER_VERSION


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


    # Score every variant in ``frame`` and return (per-state scores, provenance); only uncached hg38 variants reach Decima.
    def score(self, frame, dataset_name=None, cache_dir=None):
        try:
            from decima import predict_variant_effect  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'decima package not installed. Install with `pip install decima`.'
            ) from exc
        df_all = pd.DataFrame({
            'chrom': ['chr' + str(c).removeprefix('chr') for c in frame.chrom],
            'pos': frame.position.astype(int).to_numpy(),
            'ref': frame.ref.astype(str).to_numpy(),
            'alt': frame.alt.astype(str).to_numpy(),
        })
        df_all['variant_id'] = (df_all.chrom + ':' + df_all.pos.astype(str) + ':'
                                + df_all.ref + ':' + df_all.alt)
        keys = df_all.variant_id.to_numpy()
        per_state, per_state_prov = {}, {}
        workers = self._num_workers()
        batch_size = int(self.settings.get('batch_size', 1))
        chunksize = int(self.settings.get('chunksize', 1000))
        float_precision = str(self.settings.get('float_precision', '32'))
        reference_cache = bool(self.settings.get('reference_cache', True))
        from decima.data.dataset import VariantDataset
        metadata_columns = set(VariantDataset.DEFAULT_COLUMNS) | {'variant_id'}
        retired = self._retire_legacy_files(cache_dir)
        with tempfile.TemporaryDirectory(prefix='neurojico-decima-') as temporary_dir:
            model_ready = False
            for state_name in self.cell_states:
                stanza = self._task_queries[state_name]
                task_query = stanza['query']
                store = self._load_store(cache_dir, state_name)
                todo = (df_all.loc[~pd.Series(keys).isin(store.index).to_numpy()]
                        .drop_duplicates('variant_id').reset_index(drop=True))
                reused = int(len(set(keys)) - len(todo))
                task_names = ()
                if len(todo):
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
                              f'query {task_query!r}; {len(df_all)} variants set to 0.0 '
                              f'as declared in config.secondary.task_queries.', flush=True)
                        per_state[state_name] = np.zeros((len(df_all), 1), dtype=np.float32)
                        per_state_prov[state_name] = {
                            'status': 'proxy_zero', 'proxy': proxy_label,
                            'proxy_note': stanza['note'], 'task_query': task_query,
                            'variants_scored': int(len(df_all))}
                        continue
                    if not model_ready:
                        with self._quiet_decima_output():
                            self._ensure_model()
                        model_ready = True
                    output_path = Path(temporary_dir) / f'{state_name}.parquet'
                    with self._quiet_decima_output():
                        predict_variant_effect(
                            todo[['chrom', 'pos', 'ref', 'alt', 'variant_id']], output_pq=str(output_path),
                            tasks=task_query, model=self._model,
                            device=self._resolve_lightning_device_for_trainer(),
                            num_workers=workers, batch_size=batch_size, chunksize=chunksize,
                            float_precision=float_precision, reference_cache=reference_cache,
                            include_cols=['variant_id'])
                    self._release_device_memory()
                    self._save_raw_output(cache_dir, dataset_name, state_name, output_path)
                    store = self._append_to_store(cache_dir, state_name, store, todo, output_path,
                                                  metadata_columns)
                values = store.loc[keys, 'score'].to_numpy(dtype=np.float32)
                overlap = store.loc[keys, 'gene_overlap'].to_numpy(dtype=bool)
                if not np.isfinite(values).all():
                    raise RuntimeError(f'Decima returned nonfinite scores for {state_name}.')
                per_state[state_name] = values.reshape(-1, 1)
                per_state_prov[state_name] = {
                    'status': 'scored' if reused == 0 else (
                        'reused_from_cache' if not len(todo) else 'scored_incremental'),
                    'task_query': task_query, 'task_count': len(task_names) or None,
                    'variants_scored': int(len(todo)), 'variants_reused_from_cache': reused,
                    'variants_with_gene_overlap': int(overlap.sum()),
                    'variants_without_gene_overlap': int((~overlap).sum()),
                    'gene_overlap_fraction': float(overlap.mean()),
                    'legacy_files_retired': retired,
                }
                print(f'  decima[{state_name}]: {len(todo)} scored, {reused} reused for '
                      f'{dataset_name!r}', flush=True)
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

    # Return the settings that define what a stored score means; any change invalidates the store.
    def _store_signature(self, state_name):
        payload = {'model_identifier': self.settings.get('model_identifier', 'ensemble'),
                   'task_query': self._task_queries[state_name]['query'],
                   'liftover_version': LIFTOVER_VERSION,
                   'aggregation': 'mean_over_tasks_then_mean_over_gene_rows'}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload

    # Return the per-state store paths (scores parquet and its signature sidecar), or (None, None) without a cache dir.
    @staticmethod
    def _store_paths(cache_dir, state_name):
        if cache_dir is None:
            return None, None
        base = Path(cache_dir) / f'decima_store_{state_name}'
        return base.with_suffix('.parquet'), base.with_suffix('.signature.json')

    # Return an empty store frame indexed by hg38 variant key.
    @staticmethod
    def _empty_store():
        index = pd.Index([], name='variant_id', dtype=object)
        return pd.DataFrame({'score': pd.Series([], dtype='float32'),
                             'gene_overlap': pd.Series([], dtype=bool),
                             'source': pd.Series([], dtype=object)}, index=index)

    # Load a state's score store; a store written under different settings is moved to _stale/ and ignored.
    def _load_store(self, cache_dir, state_name):
        path, sidecar = self._store_paths(cache_dir, state_name)
        if path is None or not path.is_file():
            return self._empty_store()
        signature, _ = self._store_signature(state_name)
        recorded = json.loads(sidecar.read_text()).get('signature') if sidecar.is_file() else None
        if recorded != signature:
            self._move_to_stale(path, sidecar)
            return self._empty_store()
        store = pd.read_parquet(path).set_index('variant_id')
        if store.index.duplicated().any():
            raise RuntimeError(f'Decima store {path} contains duplicate variant keys.')
        return store

    # Move a cache file and its sidecar into _stale/ with a timestamp; nothing is ever deleted.
    @staticmethod
    def _move_to_stale(*paths):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        for path in paths:
            if path is not None and Path(path).is_file():
                stale = Path(path).parent / '_stale'
                stale.mkdir(exist_ok=True)
                shutil.move(str(path), str(stale / f'{Path(path).stem}.{stamp}{Path(path).suffix}'))

    # Write the store and its signature sidecar atomically.
    def _write_store(self, cache_dir, state_name, store):
        path, sidecar = self._store_paths(cache_dir, state_name)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        signature, payload = self._store_signature(state_name)
        temporary = path.with_name(path.name + '.partial')
        store.reset_index().to_parquet(temporary, index=False)
        temporary.replace(path)
        sidecar.write_text(json.dumps({'signature': signature, 'settings': payload, 'rows': int(len(store))}))

    # Aggregate Decima's per-gene, per-task output into one score per variant and append it to the store.
    def _append_to_store(self, cache_dir, state_name, store, todo, output_path, metadata_columns):
        predictions = pd.read_parquet(output_path)
        task_columns = [c for c in predictions.columns if c not in metadata_columns]
        if not task_columns:
            raise RuntimeError(f'Decima parquet for {state_name} has no task-prediction columns; '
                               f'got {list(predictions.columns)}.')
        predictions['_mean_effect'] = predictions[task_columns].mean(axis=1)
        per_variant = predictions.groupby('variant_id')['_mean_effect'].mean()
        scored = pd.DataFrame({'score': per_variant.reindex(todo.variant_id).fillna(0.0).to_numpy(dtype=np.float32),
                               'gene_overlap': todo.variant_id.isin(per_variant.index).to_numpy(),
                               'source': 'decima_scored'}, index=pd.Index(todo.variant_id.to_numpy(), name='variant_id'))
        if not np.isfinite(scored.score.to_numpy()).all():
            raise RuntimeError(f'Decima returned nonfinite scores for {state_name}.')
        missing = int((~scored.gene_overlap).sum())
        if missing:
            print(f'  decima[{state_name}]: no gene overlap for {missing}/{len(scored)} new variants; '
                  'set to 0.0 and flagged gene_overlap=False.', flush=True)
        store = pd.concat([store, scored])
        self._write_store(cache_dir, state_name, store)
        return store

    # Move pre-store cache files (keyed by variant_id, scored before the liftover fix) into _stale/.
    def _retire_legacy_files(self, cache_dir):
        if cache_dir is None or not Path(cache_dir).is_dir():
            return []
        retired = []
        for path in sorted(Path(cache_dir).glob('decima_*.parquet')) + sorted(Path(cache_dir).glob('decima_*.signature.json')):
            if path.name.startswith('decima_store_') or path.name.startswith('decima_raw_'):
                continue
            self._move_to_stale(path)
            retired.append(path.name)
        return retired

    # Keep Decima's raw per-gene, per-task output so other aggregations need no rescoring.
    @staticmethod
    def _save_raw_output(cache_dir, dataset_name, state_name, output_path):
        if cache_dir is None or dataset_name is None:
            return
        raw_dir = Path(cache_dir) / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copyfile(output_path, raw_dir / f'decima_raw_{dataset_name}_{state_name}.{stamp}.parquet')
