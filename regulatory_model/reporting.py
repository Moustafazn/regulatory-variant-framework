"""Infrastructure: strict JSON writer, SHA-256 hashing, event logging and workspace preparation."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np


def clean_json(value):
    # Convert NumPy objects and nonfinite numbers into interoperable JSON values.
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    # Atomically write strict JSON without NaN or Infinity extensions.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    # Hash input and output files for reproducibility and review binding.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def platform_metadata():
    # Record Python and library versions on every run.
    import importlib.metadata
    versions = {}
    for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'joblib']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions}


def prepare_output(output, protected=(), preserve=()):
    # Clear the output directory while refusing to touch protected input paths.
    # Entries whose basename is listed in ``preserve`` are kept across runs.
    output = Path(output).resolve()
    protected = {Path(p).resolve() for p in protected if p is not None}
    for guard in protected:
        if guard == output or output in guard.parents:
            raise ValueError(f'Refusing to overwrite protected path: {guard}')
    preserve = {str(name) for name in preserve}
    if output.exists():
        import shutil
        for child in output.iterdir():
            if child.name in preserve:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


def summarize_feature_impact(feature_impact, top_k=10):
    """Compact per-dataset summary of block LOO and coefficient attributions for evaluation.json."""
    summary = {}
    for dataset, record in feature_impact.items():
        full_rmse = record['full_rmse']
        loo_ranked = sorted(record['leave_one_block_out'],
                            key=lambda entry: -entry['delta_rmse_vs_full'])
        attributions = sorted(record['feature_attributions'],
                              key=lambda entry: -entry['importance_fraction'])
        block_rollup = {}
        for entry in record['feature_attributions']:
            block = entry['feature'].split(':', 1)[0]
            block_rollup[block] = block_rollup.get(block, 0.0) + entry['importance_fraction']
        summary[dataset] = {
            'enabled_blocks': record['enabled_blocks'],
            'full_rmse': full_rmse,
            'block_leave_one_out': [
                {'removed_block': row['removed_block'],
                 'rmse': row['rmse'],
                 'delta_rmse_vs_full': row['delta_rmse_vs_full']}
                for row in loo_ranked
            ],
            'configured_ablations': record.get('configured_ablations', []),
            'top_features': attributions[:int(top_k)],
            'importance_fraction_by_block': dict(sorted(block_rollup.items(),
                                                       key=lambda kv: -kv[1])),
            'attribution_method': record['attribution_method'],
        }
    return summary


def log_event(path, event, **details):
    # Append a structured diagnostic record and echo a compact progress line.
    from datetime import datetime, timezone
    record = clean_json({'event': event, 'utc': datetime.now(timezone.utc).isoformat(), **details})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(f'[{event}] ' + json.dumps(clean_json(details), allow_nan=False), flush=True)
