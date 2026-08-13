"""DeepSeek-V4-Flash heterogeneous cache/BlockPool adapter.

Both v0.20 and v0.23 use a common global block ID across heterogeneous cache
groups on A3/910C, so planner capacity can exceed useful tensor payload
substantially.  v0.23 adds block sizes 32 and 64 while retaining the block-128
page geometry used by v0.20.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..types import KVEstimate
from ..utils import ceildiv, distribute
from .deepseek_v4_v023 import minimum_kv_admission


@dataclass(frozen=True)
class ModelShape:
    c4_layers: int = 21
    c128_layers: int = 20
    swa_layers: int = 44
    c4_history_bytes: int = 512 * 2 + 128 + 2
    c128_history_bytes: int = 512 * 2
    swa_state_bytes: int = 512 * 2
    c4_main_state_bytes: int = (2 * 2 * 512) * 4
    c4_index_state_bytes: int = (2 * 2 * 128) * 4
    c128_state_bytes: int = (2 * 1 * 512) * 4
    sliding_window: int = 128


@dataclass(frozen=True)
class Layout:
    version: str
    device: str
    logical_block_size: int
    c4_entries: int
    c128_entries: int
    c4_state_entries: int
    c128_state_entries: int
    small_page_bytes: int
    large_page_bytes: int


@dataclass(frozen=True)
class Geometry:
    group_size: int
    swa_groups: tuple[int, ...]
    non_mtp_tuple_count: int
    planner_tuple_count: int
    planner_bytes_per_global_block: int
    actual_bytes_per_global_block: int


def _normalize_version(version: str) -> str:
    version = version.lower().lstrip("v")
    if version.startswith("0.20"):
        return "0.20"
    if version.startswith("0.23") or version in {"latest", "new"}:
        return "0.23"
    raise ValueError(
        "DeepSeek-V4-Flash custom layout currently recognizes 0.20 and 0.23/latest; "
        "add a versioned adapter before using another allocator layout"
    )


def _layout(version: str, device: str, block_size: int) -> Layout:
    version = _normalize_version(version)
    if device.lower() not in {"a3", "910c"}:
        raise ValueError("DeepSeek-V4-Flash custom cache adapter supports A3/910C")
    if version == "0.20" and block_size != 128:
        raise ValueError("v0.20 DeepSeek-V4-Flash layout requires block_size=128")
    table = {
        128: (128, 128, 8, 32, 16_640, 131_072),
        64: (64, 64, 4, 16, 8_320, 65_536),
        32: (32, 32, 2, 8, 4_160, 32_768),
    }
    if block_size not in table:
        raise ValueError("DeepSeek-V4-Flash block_size must be 32, 64, or 128")
    return Layout(version, "A3/910C", block_size, *table[block_size])


def _geometry(layout: Layout, mtp_layers: int) -> Geometry:
    group_size = 22
    canonical = layout.small_page_bytes + layout.large_page_bytes
    planner_tuples = group_size + mtp_layers
    return Geometry(
        group_size=group_size,
        swa_groups=(22, 22),
        non_mtp_tuple_count=group_size,
        planner_tuple_count=planner_tuples,
        planner_bytes_per_global_block=canonical * planner_tuples,
        actual_bytes_per_global_block=group_size * canonical + mtp_layers * layout.large_page_bytes,
    )


def _history_counts(length: int, layout: Layout) -> tuple[int, int, int, int]:
    c4_entries, c128_entries = length // 4, length // 128
    return (
        c4_entries,
        c128_entries,
        ceildiv(c4_entries, layout.c4_entries) if c4_entries else 0,
        ceildiv(c128_entries, layout.c128_entries) if c128_entries else 0,
    )


def _swa_blocks(previous: int, query: int, max_len: int, shape: ModelShape, layout: Layout) -> int:
    final = min(previous + query, max_len)
    if final <= 0:
        return 0
    if previous <= 0:
        return ceildiv(final, layout.logical_block_size)
    live = min(final, shape.sliding_window - 1 + query)
    return ceildiv(live, layout.logical_block_size) + 1


def _workload(c: dict[str, Any], local: int) -> list[tuple[int, int, int]]:
    scheduler, workload, par = c["scheduler"], c["workload"], c["parallelism"]
    if workload["mode"] == "admission":
        q = min(
            ceildiv(
                scheduler["max_num_batched_tokens"],
                par["pcp_size"],
            ),
            ceildiv(scheduler["max_model_len"], par["pcp_size"]),
        )
        history = ceildiv(workload["context_len"], par["pcp_size"] * par["dcp_size"])
        previous = max(ceildiv(workload["context_len"], par["pcp_size"]) - q, 0)
        return [(q, history, previous)] * local
    result = []
    for query in distribute(scheduler["max_num_batched_tokens"], local):
        final = min(query, scheduler["max_model_len"]) if workload["mode"] == "fresh" else min(workload["context_len"], scheduler["max_model_len"])
        final_rank = ceildiv(final, par["pcp_size"])
        q_rank = min(
            ceildiv(query, par["pcp_size"]),
            final_rank,
        )
        history = ceildiv(final, par["pcp_size"] * par["dcp_size"])
        previous = max(final_rank - q_rank, 0)
        result.append((q_rank, history, previous))
    return result


def _estimate_row(c: dict[str, Any], concurrency: int, shape: ModelShape, layout: Layout, geometry: Geometry) -> KVEstimate:
    par, workload, scheduler = c["parallelism"], c["workload"], c["scheduler"]
    local = concurrency if workload["concurrency_scope"] == "per-dp" else ceildiv(concurrency, min(concurrency, par["dp_size"]))
    rows = _workload(c, local)
    c4_entries = c128_entries = c4_history = c128_history = 0
    swa_per_group = c4_state = c128_state = q_total = 0
    max_len_pcp = ceildiv(scheduler["max_model_len"], par["pcp_size"])
    admission_cap = ceildiv(
        min(shape.sliding_window - 1 + ceildiv(scheduler["max_num_batched_tokens"], par["pcp_size"]), max_len_pcp),
        layout.logical_block_size,
    ) + 1
    for q_rank, history, previous in rows:
        e4, e128, b4, b128 = _history_counts(history, layout)
        c4_entries += e4
        c128_entries += e128
        c4_history += b4
        c128_history += b128
        q_total += q_rank
        swa = _swa_blocks(previous, q_rank, max_len_pcp, shape, layout)
        if workload["mode"] == "admission":
            swa = min(ceildiv(ceildiv(workload["context_len"], par["pcp_size"]), layout.logical_block_size), admission_cap)
        swa_per_group += swa
        c4_state += ceildiv(q_rank, layout.c4_state_entries) + (1 if q_rank else 0)
        c128_state += ceildiv(q_rank, layout.c128_state_entries) + (1 if q_rank else 0)
    swa_blocks = swa_per_group * len(geometry.swa_groups)
    history_blocks = c4_history + c128_history
    global_blocks = history_blocks + swa_blocks + c4_state + c128_state
    physical = global_blocks * geometry.actual_bytes_per_global_block
    planner_history = history_blocks * geometry.planner_bytes_per_global_block
    planner_transient = (swa_blocks + c4_state + c128_state) * geometry.planner_bytes_per_global_block
    planner = planner_history + planner_transient
    raw_history = c4_entries * shape.c4_layers * shape.c4_history_bytes + c128_entries * shape.c128_layers * shape.c128_history_bytes
    raw_transient = q_total * (
        shape.swa_layers * shape.swa_state_bytes
        + shape.c4_layers * (shape.c4_main_state_bytes + shape.c4_index_state_bytes)
        + shape.c128_layers * shape.c128_state_bytes
    )
    return KVEstimate(
        concurrency_input=concurrency,
        max_local_concurrency=local,
        q_tokens_per_rank=q_total,
        physical_pool_tensor_bytes=physical,
        planner_total_bytes=planner,
        details={
            "global_blocks": global_blocks,
            "history_blocks": history_blocks,
            "swa_blocks": swa_blocks,
            "c4_state_blocks": c4_state,
            "c128_state_blocks": c128_state,
            "planner_history_bytes": planner_history,
            "planner_transient_bytes": planner_transient,
            "raw_history_bytes": raw_history,
            "raw_transient_bytes": raw_transient,
            "raw_total_bytes": raw_history + raw_transient,
        },
        source="deepseek-v4-flash-heterogeneous-blockpool",
    )


def estimate_deepseek_v4(c: dict[str, Any]) -> tuple[dict[str, Any], list[KVEstimate]]:
    layout = _layout(
        c["platform"]["vllm_ascend_version"], c["platform"]["device"], c["scheduler"]["block_size"]
    )
    shape = ModelShape()
    geometry = _geometry(layout, int(c["model"]["mtp_layers"]))
    rows = [_estimate_row(c, concurrency, shape, layout, geometry) for concurrency in c["workload"]["concurrency"]]
    profile: dict[str, Any] = {
        "layout": asdict(layout),
        "geometry": asdict(geometry),
        "model_shape": asdict(shape),
    }
    if layout.version == "0.23":
        admission = minimum_kv_admission(
            max_model_len=int(c["scheduler"]["max_model_len"]),
            max_num_batched_tokens=int(
                c["scheduler"]["max_num_batched_tokens"]
            ),
            block_size=layout.logical_block_size,
        )
        profile["minimum_admission"] = asdict(admission)
    return profile, rows
