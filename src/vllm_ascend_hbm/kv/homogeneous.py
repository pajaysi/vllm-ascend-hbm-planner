"""Generic homogeneous KV cache models for GQA/MHA, MLA and manual profiles."""

from __future__ import annotations

from typing import Any

from ..types import KVEstimate
from ..utils import bytes_from_gib, ceildiv, distribute


def _align(value: int, alignment: int) -> int:
    return ceildiv(value, alignment) * alignment


def _max_local_concurrency(concurrency: int, scope: str, dp: int) -> int:
    if scope == "per-dp":
        return concurrency
    active_dp = min(concurrency, dp)
    return ceildiv(concurrency, active_dp) if active_dp else 0


def _geometry(c: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    model, par, scheduler, kv = c["model"], c["parallelism"], c["scheduler"], c["kv_cache"]
    strategy = model["kv_cache_strategy"]
    layers_per_rank = ceildiv(int(model["num_cache_layers"]), par["pp_size"])
    dtype_bytes = int(kv["dtype_bytes"])
    if strategy == "none":
        token_layer_bytes = 0
        description = "no generation KV cache"
    elif strategy == "standard_gqa":
        kv_heads = int(model["num_key_value_heads"])
        local_kv_heads = max(1, ceildiv(kv_heads, par["tp_size"]))
        token_layer_bytes = 2 * local_kv_heads * int(model["head_dim"]) * dtype_bytes
        description = "2(K,V) * local_kv_heads * head_dim * dtype"
    elif strategy == "mla":
        # The compressed latent and RoPE key are stored once per token/layer;
        # they are normally replicated across TP ranks.
        local_kv_heads = None
        token_layer_bytes = (
            int(model["kv_lora_rank"]) + int(model.get("qk_rope_head_dim") or 0)
        ) * dtype_bytes
        description = "(kv_lora_rank + qk_rope_head_dim) * dtype"
    elif strategy == "manual":
        local_kv_heads = None
        value = kv["manual_bytes_per_token_per_rank"]
        token_layer_bytes = 0 if value is None else ceildiv(int(value), layers_per_rank)
        description = "manual bytes per token per rank"
    else:
        raise ValueError(f"homogeneous adapter cannot handle strategy {strategy!r}")
    page_per_layer = _align(
        token_layer_bytes * scheduler["block_size"], int(kv["page_alignment_bytes"])
    ) if token_layer_bytes else 0
    physical_block = page_per_layer * layers_per_rank
    planner_block = round(physical_block * (1 + float(kv["planner_overhead_fraction"])))
    details = {
        "strategy": strategy,
        "layers_per_rank": layers_per_rank,
        "local_kv_heads": local_kv_heads,
        "bytes_per_token_per_layer": token_layer_bytes,
        "page_bytes_per_layer": page_per_layer,
        "physical_bytes_per_block": physical_block,
        "planner_bytes_per_block": planner_block,
        "formula": description,
    }
    return physical_block, planner_block, details


def estimate_homogeneous(c: dict[str, Any]) -> tuple[dict[str, Any], list[KVEstimate]]:
    model, scheduler, par, workload, kv = (
        c["model"], c["scheduler"], c["parallelism"], c["workload"], c["kv_cache"]
    )
    physical_block, planner_block, geometry = _geometry(c)
    rows: list[KVEstimate] = []
    for concurrency in workload["concurrency"]:
        local = _max_local_concurrency(concurrency, workload["concurrency_scope"], par["dp_size"])
        q_per_request = distribute(scheduler["max_num_batched_tokens"], local)
        q_rank = sum(ceildiv(value, par["pcp_size"]) for value in q_per_request)
        if workload["mode"] == "fresh":
            request_lengths = [
                ceildiv(min(value, scheduler["max_model_len"]), par["pcp_size"] * par["dcp_size"])
                for value in q_per_request
            ]
        else:
            length = ceildiv(
                min(workload["context_len"], scheduler["max_model_len"]),
                par["pcp_size"] * par["dcp_size"],
            )
            request_lengths = [length] * local
        blocks = sum(ceildiv(length, scheduler["block_size"]) for length in request_lengths if length)

        if kv["manual_gib_per_rank"] is not None:
            physical = bytes_from_gib(kv["manual_gib_per_rank"]) or 0
            planner = physical
            source = "manual-kv-gib-per-rank"
        else:
            physical = blocks * physical_block
            planner = blocks * planner_block
            source = f"generic-{model['kv_cache_strategy']}-block-model"
        rows.append(
            KVEstimate(
                concurrency_input=concurrency,
                max_local_concurrency=local,
                q_tokens_per_rank=q_rank,
                physical_pool_tensor_bytes=physical,
                planner_total_bytes=planner,
                details={
                    **geometry,
                    "request_lengths_on_rank": request_lengths,
                    "global_blocks": blocks,
                    "history_blocks": blocks,
                    "planner_history_bytes": planner,
                    "planner_transient_bytes": 0,
                },
                source=source,
            )
        )
    return geometry, rows
