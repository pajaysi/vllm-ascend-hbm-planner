"""Small dependency-free helpers."""

from __future__ import annotations

import copy
from typing import Any, Optional, Union

from .constants import GIB


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def gib(value: Union[int, float]) -> float:
    return value / GIB


def bytes_from_gib(
    value: Optional[Union[float, int]],
) -> Optional[int]:
    return None if value is None else round(float(value) * GIB)


def distribute(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        return []
    q, r = divmod(total, buckets)
    return [q + (index < r) for index in range(buckets)]


def deep_merge(base: Any, update: Any, path: str = "") -> Any:
    """Strict recursive merge; empty mappings accept user-defined labels."""
    if not isinstance(base, dict):
        return copy.deepcopy(update)
    if not isinstance(update, dict):
        raise ValueError(f"{path or 'JSON root'} must be an object")
    result = copy.deepcopy(base)
    if base:
        unknown = sorted(set(update) - set(base))
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path or 'root'}: {', '.join(unknown)}"
            )
    for key, value in update.items():
        child = f"{path}.{key}" if path else key
        result[key] = deep_merge(base[key], value, child) if key in base else copy.deepcopy(value)
    return result


def set_if_missing(target: dict[str, Any], key: str, value: Any) -> None:
    if target.get(key) is None and value is not None:
        target[key] = value
