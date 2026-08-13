"""Activation, workspace, graph and runtime estimators."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import GIB
from .types import ActivationEstimate, ComponentEstimate
from .utils import bytes_from_gib, ceildiv


PROFILE_RE = re.compile(
    r"Actual usage is\s+([0-9.]+)\s+GiB for weight,\s*"
    r"([0-9.]+)\s+GiB for peak activation,\s*"
    r"([0-9.]+)\s+GiB for non[- ]torch memory"
    r"(?:,\s*and\s*([0-9.]+)\s+GiB for "
    r"(?:CUDA\s*Graph|ACL\s*Graph|CUDAGraph|NPUGraph|graph) memory)?",
    re.IGNORECASE,
)


def parse_profile_text(
    text: str,
) -> Optional[dict[str, Optional[float]]]:
    matches = list(PROFILE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "weight_gib_per_rank": float(match.group(1)),
        "peak_activation_gib_per_rank": float(match.group(2)),
        "non_torch_gib_per_rank": float(match.group(3)),
        "graph_gib_per_rank": None if match.group(4) is None else float(match.group(4)),
    }


def resolve_profile_calibration(
    c: dict[str, Any],
) -> dict[str, Optional[float]]:
    cfg = c["profile_calibration"]
    result: dict[str, Optional[float]] = {
        "weight_gib_per_rank": None,
        "peak_activation_gib_per_rank": None,
        "non_torch_gib_per_rank": None,
        "graph_gib_per_rank": None,
        "profiled_max_num_batched_tokens": None,
    }
    if cfg["vllm_log_path"]:
        path = Path(str(cfg["vllm_log_path"]))
        try:
            parsed = parse_profile_text(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            raise ValueError(f"cannot read vLLM profile log {path}: {exc}") from exc
        if parsed is None:
            raise ValueError("vLLM profile log does not contain the expected memory line")
        result.update(parsed)
    for key in result:
        if cfg[key] is not None:
            result[key] = float(cfg[key])
    return result


def _deepseek_v4_activation(c: dict[str, Any], q_tokens: int) -> tuple[int, int, int, dict[str, int]]:
    m, a, par = c["model"], c["activation"], c["parallelism"]
    b, hidden = int(a["dtype_bytes"]), int(m["hidden_size"])
    hidden_one = q_tokens * hidden * b
    mhc_residual = hidden_one * int(m["mhc_expansion_factor"])
    hidden_buffers = round(hidden_one * float(a["hidden_buffer_count"]))
    persistent = mhc_residual + hidden_buffers
    local_q_heads = ceildiv(int(m["num_attention_heads"]), par["tp_size"])
    local_index_heads = ceildiv(int(m["indexer_heads"]), par["tp_size"])
    local_output_groups = ceildiv(int(m["output_projection_groups"]), par["tp_size"])
    attention_q = q_tokens * local_q_heads * int(m["head_dim"]) * b
    compressed_q = q_tokens * int(m["query_compression_dim"]) * b
    indexer_q = q_tokens * local_index_heads * int(m["indexer_head_dim"]) * b
    grouped_output = q_tokens * local_output_groups * int(m["attention_output_intermediate_dim"]) * b
    sparse_selection = q_tokens * int(m["attention_topk"]) * (4 + b)
    attention = attention_q + compressed_q + indexer_q + grouped_output + sparse_selection
    routed_tokens = q_tokens * int(m["num_experts_per_token"])
    router_logits = q_tokens * int(m["num_routed_experts"]) * 4
    dispatch = round(routed_tokens * hidden * b * float(a["moe_dispatch_buffer_copies"]))
    expert_intermediate = round(
        routed_tokens
        * int(m["moe_intermediate_size"])
        * b
        * float(a["moe_intermediate_buffer_count"])
    )
    moe = round((router_logits + dispatch + expert_intermediate) * float(a["moe_capacity_factor"]))
    components = {
        "mhc_residual": mhc_residual,
        "hidden_io_buffers": hidden_buffers,
        "attention_q": attention_q,
        "attention_compressed_q": compressed_q,
        "attention_indexer_q": indexer_q,
        "attention_grouped_output": grouped_output,
        "attention_sparse_topk": sparse_selection,
        "moe_router_logits": router_logits,
        "moe_dispatch_buffers": dispatch,
        "moe_expert_intermediate": expert_intermediate,
    }
    return persistent, attention, moe, components


def _generic_activation(c: dict[str, Any], q_tokens: int) -> tuple[int, int, int, dict[str, int]]:
    m, a, par = c["model"], c["activation"], c["parallelism"]
    b, hidden = int(a["dtype_bytes"]), int(m["hidden_size"])
    hidden_one = q_tokens * hidden * b
    hidden_buffers = round(hidden_one * float(a["hidden_buffer_count"]))
    persistent = hidden_one + hidden_buffers
    local_heads = ceildiv(int(m.get("num_attention_heads") or 1), par["tp_size"])
    head_dim = int(m.get("head_dim") or ceildiv(hidden, int(m.get("num_attention_heads") or 1)))
    q_heads = q_tokens * local_heads * head_dim * b
    attention_io = round((q_heads * 2 + hidden_one) * float(a["attention_buffer_factor"]))
    attention = q_heads + attention_io
    experts = int(m.get("num_routed_experts") or 0)
    active = int(m.get("num_experts_per_token") or 0)
    moe_intermediate = int(m.get("moe_intermediate_size") or m.get("intermediate_size") or 0)
    router_logits = q_tokens * experts * 4 if experts else 0
    routed_tokens = q_tokens * active
    dispatch = round(routed_tokens * hidden * b * float(a["moe_dispatch_buffer_copies"]))
    intermediate = round(
        routed_tokens * moe_intermediate * b * float(a["moe_intermediate_buffer_count"])
    )
    moe = round((router_logits + dispatch + intermediate) * float(a["moe_capacity_factor"]))
    components = {
        "hidden_input": hidden_one,
        "hidden_io_buffers": hidden_buffers,
        "attention_q_heads": q_heads,
        "attention_io": attention_io,
        "moe_router_logits": router_logits,
        "moe_dispatch_buffers": dispatch,
        "moe_expert_intermediate": intermediate,
    }
    return persistent, attention, moe, components


def estimate_activation(
    c: dict[str, Any],
    q_tokens: int,
    profile: dict[str, Optional[float]],
) -> ActivationEstimate:
    if c["model"]["activation_model"] == "deepseek_v4_flash":
        effective_q_tokens = q_tokens
        if c.get("vllm_ascend", {}).get("enable_flashcomm1", False):
            tp_size = int(c["parallelism"]["tp_size"])
            if tp_size > 2:
                effective_q_tokens = ceildiv(q_tokens * 2, tp_size)
        persistent, attention, moe, components = _deepseek_v4_activation(
            c,
            effective_q_tokens,
        )
        components["scheduled_tokens"] = q_tokens
        components["sequence_parallel_effective_tokens"] = (
            effective_q_tokens
        )
    else:
        persistent, attention, moe, components = _generic_activation(c, q_tokens)
    analytical = round(
        persistent + max(attention, moe) * float(c["activation"]["branch_live_fraction"])
    )
    manual = profile.get("peak_activation_gib_per_rank")
    source = "vllm-profile"
    uncertainty = float(c["uncertainty"]["profile_activation_fraction"])
    if manual is None:
        manual = c["activation"]["manual_peak_gib_per_rank"]
        source = "manual-peak" if manual is not None else "structural-analytical"
    if manual is None:
        used = analytical
        uncertainty = float(c["uncertainty"]["analytical_activation_fraction"])
    else:
        used = bytes_from_gib(manual) or 0
        profiled_q = profile.get("profiled_max_num_batched_tokens")
        if source == "vllm-profile" and profiled_q:
            profiled_q_rank = ceildiv(int(profiled_q), c["parallelism"]["pcp_size"])
            used = round(used * q_tokens / profiled_q_rank)
            if q_tokens != profiled_q_rank:
                source += "-scaled-linearly-by-Q"
    return ActivationEstimate(
        analytical_peak_bytes=analytical,
        used_peak_bytes=used,
        source=source,
        persistent_hidden_bytes=persistent,
        attention_branch_bytes=attention,
        moe_branch_bytes=moe,
        components=components,
        uncertainty_fraction=uncertainty,
    )


def estimate_workspace(c: dict[str, Any], activation: ActivationEstimate) -> ComponentEstimate:
    wc = c["operator_workspace"]
    if activation.source.startswith("vllm-profile") and wc["assume_in_profile_activation"]:
        return ComponentEstimate(0, 0, "included-in-profile-activation", 0.0)
    if wc["manual_peak_gib_per_rank"] is not None:
        value = bytes_from_gib(wc["manual_peak_gib_per_rank"]) or 0
        return ComponentEstimate(value, value, "manual-workspace-peak", 0.15)
    if wc["components_gib_per_rank"]:
        peak = max(float(value) for value in wc["components_gib_per_rank"].values())
        value = bytes_from_gib(peak * float(wc["concurrent_factor"])) or 0
        return ComponentEstimate(
            value, value, "max-known-operator-workspace", float(c["uncertainty"]["workspace_fraction"])
        )
    return ComponentEstimate(
        0, 0, "unresolved-no-kernel-data", float(c["uncertainty"]["workspace_fraction"]), True
    )


def estimate_graph(
    c: dict[str, Any],
    profile: dict[str, Optional[float]],
) -> ComponentEstimate:
    graph = c["graph_cache"]
    if profile.get("graph_gib_per_rank") is not None:
        value = bytes_from_gib(profile["graph_gib_per_rank"]) or 0
        return ComponentEstimate(value, value, "vllm-profile", 0.05)
    if graph["mode"] == "eager":
        return ComponentEstimate(0, 0, "eager-no-graph-cache", 0.0)
    if graph["manual_gib_per_rank"] is not None:
        value = bytes_from_gib(graph["manual_gib_per_rank"]) or 0
        return ComponentEstimate(value, value, "manual-acl-graph", 0.10)
    estimate = (
        len(graph["capture_sizes"]) * float(graph["fixed_gib_per_graph"]) * GIB
        + sum(int(value) for value in graph["capture_sizes"])
        * float(graph["bytes_per_captured_token"])
    )
    value = round(estimate)
    return ComponentEstimate(value, value, "capture-size-coefficients", 0.25)


def estimate_runtime(
    c: dict[str, Any],
    q_tokens: int,
    profile: dict[str, Optional[float]],
) -> ComponentEstimate:
    runtime = c["runtime"]
    if profile.get("non_torch_gib_per_rank") is not None:
        value = bytes_from_gib(profile["non_torch_gib_per_rank"]) or 0
        return ComponentEstimate(
            value, value, "vllm-profile-non-torch", float(c["uncertainty"]["profile_runtime_fraction"])
        )
    if runtime["manual_non_torch_gib_per_rank"] is not None:
        value = bytes_from_gib(runtime["manual_non_torch_gib_per_rank"]) or 0
        return ComponentEstimate(value, value, "manual-non-torch", 0.15)
    scheduler, model = c["scheduler"], c["model"]
    input_buffers = q_tokens * int(runtime["bytes_per_scheduled_token"])
    block_table = (
        scheduler["max_num_seqs"]
        * ceildiv(scheduler["max_model_len"], scheduler["block_size"])
        * int(runtime["block_table_entry_bytes"])
    )
    sampler = scheduler["max_num_seqs"] * int(model["vocab_size"]) * int(runtime["sampler_logit_bytes"])
    fixed = bytes_from_gib(
        float(runtime["base_persistent_gib_per_rank"])
        + float(runtime["hccl_and_cann_persistent_gib_per_rank"])
    ) or 0
    ascend = c.get("vllm_ascend", {})
    hccl_buffsize_mib = ascend.get("hccl_buffsize_mib")
    hccl_domains = int(
        ascend.get("hccl_communication_domains_per_rank", 1)
    )
    hccl = (
        0
        if hccl_buffsize_mib is None
        else round(
            2
            * float(hccl_buffsize_mib)
            * hccl_domains
            * 2**20
        )
    )
    value = fixed + hccl + input_buffers + block_table + sampler
    return ComponentEstimate(
        value,
        value,
        "analytical-persistent-buffers-with-hccl-domains",
        float(c["uncertainty"]["analytical_runtime_fraction"]),
    )


def uncertainty_bounds(
    values: Iterable[tuple[int, float]], fragmentation: int, safety: int
) -> tuple[int, int, float]:
    values = list(values)
    low = sum(round(value * max(0.0, 1 - uncertainty)) for value, uncertainty in values)
    high = sum(round(value * (1 + uncertainty)) for value, uncertainty in values)
    subtotal = sum(value for value, _ in values)
    confident = sum(value for value, uncertainty in values if uncertainty <= 0.10)
    return low + fragmentation + safety, high + fragmentation + safety, confident / subtotal if subtotal else 1.0
