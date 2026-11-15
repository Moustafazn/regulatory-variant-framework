"""Regulatory variant interpretation: evidence, prediction, uncertainty, and review."""
from pathlib import Path as _Path

from .env import configure_torch_env_from_config as _configure_torch_env_from_config

# Honour config.secondary.device = "cpu" before any transitive torch import happens.
_configure_torch_env_from_config(_Path(__file__).parent / 'config.json')

__version__ = "1.0.0"

from .api import NeurojicoModel, RegulatoryVariantModel

__all__ = ["RegulatoryVariantModel", "NeurojicoModel", "__version__"]
