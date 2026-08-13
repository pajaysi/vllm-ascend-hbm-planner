"""Platform HBM capacity resolution without mixing nominal and visible sizes."""

from __future__ import annotations

from typing import Any, Optional

from .utils import bytes_from_gib


def resolve_capacity_bytes(
    config: dict[str, Any],
    *,
    visible_gib: Optional[float] = None,
    startup_free_gib: Optional[float] = None,
    requested_gib: Optional[float] = None,
) -> tuple[int, int, int]:
    platform = config["platform"]
    fallback = float(platform["hbm_gib_per_die"])
    visible = (
        visible_gib
        if visible_gib is not None
        else platform.get("visible_hbm_gib_per_die")
    )
    if visible is None:
        visible = fallback
    startup_free = (
        startup_free_gib
        if startup_free_gib is not None
        else platform.get("startup_free_hbm_gib_per_die")
    )
    if startup_free is None:
        startup_free = visible
    visible_bytes = bytes_from_gib(visible) or 0
    startup_free_bytes = bytes_from_gib(startup_free) or 0
    requested_bytes = (
        bytes_from_gib(requested_gib)
        if requested_gib is not None
        else round(
            visible_bytes * float(platform["gpu_memory_utilization"])
        )
    )
    return visible_bytes, startup_free_bytes, requested_bytes or 0
