"""KV-cache estimator selection."""

from __future__ import annotations

from typing import Any

from .deepseek_v4_flash import estimate_deepseek_v4
from .homogeneous import estimate_homogeneous


def estimate_kv(c: dict[str, Any]):
    strategy = c["model"]["kv_cache_strategy"]
    if strategy == "deepseek_v4_flash":
        return estimate_deepseek_v4(c)
    return estimate_homogeneous(c)


__all__ = ["estimate_kv"]
