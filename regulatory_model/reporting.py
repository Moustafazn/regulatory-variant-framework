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


def sha256_bytes(payload):
    # Hash in-memory bytes used for API-response provenance.
    return hashlib.sha256(payload).hexdigest()


def sha256_directory(path):
    # Deterministic SHA over sorted file hashes inside a model checkpoint directory.
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b'\x00')
        digest.update(sha256(file).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def platform_metadata():
    # Record Python and library versions on every run.
    import importlib.metadata
    versions = {}
    for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'alphagenome', 'torch', 'joblib']:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions}


def prepare_output(output, protected=(), preserve=()):
    # Clear the output directory while refusing to touch protected input paths.
    # Entries whose basename is listed in ``preserve`` are kept across runs; this
    # is how the long-lived caches (alphagenome_cache/, decima_cache/, prepared/)
    # survive a re-run instead of being deleted and re-downloaded.
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


def log_event(path, event, **details):
    # Append a structured diagnostic record and echo a compact progress line.
    from datetime import datetime, timezone
    record = clean_json({'event': event, 'utc': datetime.now(timezone.utc).isoformat(), **details})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    print(f'[{event}] ' + json.dumps(clean_json(details), allow_nan=False), flush=True)
