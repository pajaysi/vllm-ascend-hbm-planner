"""End-to-end per-rank HBM calculation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional

from .components import (
    estimate_activation,
    estimate_graph,
    estimate_runtime,
    estimate_workspace,
    resolve_profile_calibration,
    uncertainty_bounds,
)
from .capacity import resolve_capacity_bytes
from .kv import estimate_kv
from .types import ComponentEstimate, TotalEstimate, WeightEstimate
from .utils import bytes_from_gib
from .weights import estimate_weights
from .startup import evaluate_startup, startup_model_supported


def calculate(
    c: dict[str, Any],
    *,
    resolved_profile: Optional[dict[str, Optional[float]]] = None,
    weight_estimate: Optional[WeightEstimate] = None,
    graph_estimate: Optional[ComponentEstimate] = None,
) -> dict[str, Any]:
    profile = resolved_profile or resolve_profile_calibration(c)
    weights = weight_estimate or estimate_weights(c, profile)
    graph = graph_estimate or estimate_graph(c, profile)
    kv_profile, kv_rows = estimate_kv(c)
    physical_hbm, _, requested_budget = resolve_capacity_bytes(c)
    safety = bytes_from_gib(c["runtime"]["safety_reserve_gib_per_rank"]) or 0
    rows: list[TotalEstimate] = []
    for kv_row in kv_rows:
        activation = estimate_activation(c, kv_row.q_tokens_per_rank, profile)
        workspace = estimate_workspace(c, activation)
        runtime = estimate_runtime(c, kv_row.q_tokens_per_rank, profile)
        central = [
            (weights.per_rank_bytes, weights.uncertainty_fraction),
            (kv_row.physical_pool_tensor_bytes, float(c["uncertainty"]["kv_tensor_fraction"])),
            (activation.used_peak_bytes, activation.uncertainty_fraction),
            (workspace.used_bytes, workspace.uncertainty_fraction),
            (graph.used_bytes, graph.uncertainty_fraction),
            (runtime.used_bytes, runtime.uncertainty_fraction),
        ]
        subtotal_actual = sum(value for value, _ in central)
        fragmentation = round(
            subtotal_actual * float(c["runtime"]["allocator_fragmentation_fraction"])
        )
        actual = subtotal_actual + fragmentation + safety
        planning = actual - kv_row.physical_pool_tensor_bytes + kv_row.planner_total_bytes
        low, high, confident = uncertainty_bounds(central, fragmentation, safety)
        unresolved = ["operator_workspace"] if workspace.unresolved else []
        rows.append(
            TotalEstimate(
                concurrency_input=kv_row.concurrency_input,
                max_local_concurrency=kv_row.max_local_concurrency,
                q_tokens_per_rank=kv_row.q_tokens_per_rank,
                weights_bytes=weights.per_rank_bytes,
                kv_pool_tensor_bytes=kv_row.physical_pool_tensor_bytes,
                kv_planner_bytes=kv_row.planner_total_bytes,
                activation_bytes=activation.used_peak_bytes,
                operator_workspace_bytes=workspace.used_bytes,
                graph_cache_bytes=graph.used_bytes,
                runtime_bytes=runtime.used_bytes,
                fragmentation_bytes=fragmentation,
                safety_reserve_bytes=safety,
                actual_total_bytes=actual,
                planning_total_bytes=planning,
                lower_bound_bytes=low,
                upper_bound_bytes=high,
                requested_hbm_budget_bytes=requested_budget,
                physical_hbm_bytes=physical_hbm,
                planning_headroom_bytes=requested_budget - planning,
                fits_requested_budget=planning <= requested_budget,
                fits_physical_hbm_upper_bound=high <= physical_hbm,
                high_confidence_fraction=confident,
                coverage_complete=not unresolved,
                unresolved_components=unresolved,
                kv_details=kv_row.details,
                activation_details={
                    "analytical_peak_bytes": activation.analytical_peak_bytes,
                    "persistent_hidden_bytes": activation.persistent_hidden_bytes,
                    "attention_branch_bytes": activation.attention_branch_bytes,
                    "moe_branch_bytes": activation.moe_branch_bytes,
                    "components": activation.components,
                },
                component_sources={
                    "weights": weights.source,
                    "kv": kv_row.source,
                    "activation": activation.source,
                    "operator_workspace": workspace.source,
                    "graph_cache": graph.source,
                    "runtime": runtime.source,
                },
            )
        )
    result = {
        "operation": "estimate",
        "config": c,
        "model": {
            key: c["model"].get(key)
            for key in (
                "profile", "name", "family", "architecture", "kv_cache_strategy",
                "activation_model", "resolved_config_path", "multimodal_text_backbone_only"
            )
        },
        "profile_calibration_resolved": profile,
        "weight_estimate": asdict(weights),
        "graph_estimate": asdict(graph),
        "kv_profile": kv_profile,
        "estimates": [asdict(row) for row in rows],
    }
    if startup_model_supported(c):
        result["startup_estimate"] = asdict(
            evaluate_startup(
                c,
                int(c["scheduler"]["max_num_batched_tokens"]),
                int(c["scheduler"]["max_num_seqs"]),
            )
        )
    else:
        result["startup_estimate"] = None
    return result
