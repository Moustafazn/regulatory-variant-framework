"""Env-var hygiene that must run before any ``import torch`` happens anywhere."""
from __future__ import annotations

import json
import os
from pathlib import Path


# Read config.secondary.device and set DISABLE_MPS/PYTORCH_ENABLE_MPS_FALLBACK before torch imports.
def configure_torch_env_from_config(config_path):
    try:
        settings = json.loads(Path(config_path).read_text())
    except (OSError, ValueError):
        return
    framework = settings.get('framework', settings.get('neurojico', {}))
    device = framework.get('secondary', {}).get('device')
    if device in (None, '', 'cpu'):
        os.environ['DISABLE_MPS'] = '1'
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '0')
