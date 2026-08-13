"""Public API for vLLM Ascend HBM planning."""

from .config import DEFAULT_CONFIG, load_config, validate_config
from .engine import calculate
from .profiles import get_profile, list_auto_families, list_profiles
from .recommender import recommend

__all__ = [
    "DEFAULT_CONFIG", "load_config", "validate_config", "calculate", "recommend",
    "get_profile", "list_profiles", "list_auto_families",
]

__version__ = "0.3.1"
