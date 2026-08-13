"""Search the Q/max_num_seqs feasible region against a per-rank HBM budget."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from typing import Any, Optional

from .components import estimate_graph, resolve_profile_calibration
from .capacity import resolve_capacity_bytes
from .engine import calculate
from .utils import bytes_from_gib


def _normalized_log(value: int, minimum: int, maximum: int) -> float:
    if maximum <= minimum:
        return 1.0
    return math.log(value / minimum) / math.log(maximum / minimum)


def candidate_score(q: int, seqs: int, rec: dict[str, Any]) -> float:
    q_values, seq_values = rec["candidate_max_num_batched_tokens"], rec["candidate_max_num_seqs"]
    q_score = _normalized_log(q, min(q_values), max(q_values))
    seq_score = _normalized_log(seqs, min(seq_values), max(seq_values))
    if rec["objective"] == "prefill_throughput":
        q_weight, seq_weight = 0.90, 0.10
    elif rec["objective"] == "concurrency":
        q_weight, seq_weight = 0.10, 0.90
    else:
        q_weight, seq_weight = float(rec["balanced_q_weight"]), float(rec["balanced_seq_weight"])
        total = q_weight + seq_weight
        q_weight, seq_weight = q_weight / total, seq_weight / total
    return 100.0 * (q_weight * q_score + seq_weight * seq_score)


def _candidate(
    c: dict[str, Any], scenario: dict[str, Any], q: int, seqs: int,
    profile: dict[str, Optional[float]], graph,
) -> dict[str, Any]:
    rec = c["recommendation"]
    if q < seqs:
        return {
            "max_num_batched_tokens": q,
            "max_num_seqs": seqs,
            "score": candidate_score(q, seqs, rec),
            "feasible": False,
            "failure_reasons": ["max_num_batched_tokens < max_num_seqs"],
        }
    pair = copy.deepcopy(c)
    pair["operation"] = "estimate"
    pair["scheduler"]["max_num_batched_tokens"] = q
    pair["scheduler"]["max_num_seqs"] = seqs
    pair["workload"]["context_len"] = scenario["context_len"]
    pair["workload"]["concurrency"] = [seqs]
    pair["workload"]["concurrency_scope"] = "per-dp"
    estimate = calculate(
        pair,
        resolved_profile=profile,
        graph_estimate=graph,
    )
    runtime_estimate = estimate["estimates"][0]
    startup = estimate["startup_estimate"]
    _, _, requested = resolve_capacity_bytes(c)
    unresolved_reserve = (
        bytes_from_gib(rec["unresolved_workspace_reserve_gib_per_rank"]) or 0
        if "operator_workspace" in runtime_estimate["unresolved_components"]
        else 0
    )
    kv_u = float(c["uncertainty"]["kv_tensor_fraction"])
    physical_kv_high = round(
        runtime_estimate["kv_pool_tensor_bytes"] * (1 + kv_u)
    )
    planner_kv_high = round(
        runtime_estimate["kv_planner_bytes"] * (1 + kv_u)
    )
    planning_center = (
        runtime_estimate["planning_total_bytes"] + unresolved_reserve
    )
    planning_upper = (
        runtime_estimate["upper_bound_bytes"]
        - physical_kv_high
        + planner_kv_high
        + unresolved_reserve
    )
    fit_value = planning_upper if rec["fit_basis"] == "planning_upper" else planning_center
    minimum_headroom = bytes_from_gib(rec["minimum_headroom_gib_per_rank"]) or 0
    runtime_budget_safe = fit_value + minimum_headroom <= requested
    startup_feasible = (
        None if startup is None else bool(startup["startup_feasible"])
    )
    startup_gate_passed = (
        True if startup_feasible is None else startup_feasible
    )
    runtime_safe = runtime_budget_safe and startup_gate_passed
    failure_reasons = []
    if startup_feasible is False:
        failure_reasons.append(
            "startup minimum-KV admission exceeds available KV"
        )
    if not runtime_budget_safe:
        failure_reasons.append(
            f"{rec['fit_basis']} plus minimum headroom exceeds requested HBM budget"
        )
    return {
        "max_num_batched_tokens": q,
        "max_num_seqs": seqs,
        "score": candidate_score(q, seqs, rec),
        "feasible": runtime_safe,
        "startup_feasible": startup_feasible,
        "runtime_budget_safe": runtime_budget_safe,
        "runtime_safe": runtime_safe,
        "fit_basis": rec["fit_basis"],
        "context_len": scenario["context_len"],
        "actual_total_bytes": runtime_estimate["actual_total_bytes"] + unresolved_reserve,
        "actual_upper_bytes": runtime_estimate["upper_bound_bytes"] + unresolved_reserve,
        "planning_total_bytes": planning_center,
        "planning_upper_bytes": planning_upper,
        "requested_hbm_budget_bytes": requested,
        "minimum_headroom_bytes": minimum_headroom,
        "planning_headroom_bytes": requested - planning_center,
        "planning_upper_headroom_bytes": requested - planning_upper,
        "fit_headroom_bytes": requested - fit_value,
        "unresolved_workspace_reserve_bytes": unresolved_reserve,
        "coverage_complete": runtime_estimate["coverage_complete"],
        "failure_reasons": failure_reasons,
        "startup_available_kv_bytes": (
            None if startup is None else startup["available_kv_bytes"]
        ),
        "startup_required_min_kv_bytes": (
            None if startup is None else startup["minimum_kv_bytes"]
        ),
        "startup_headroom_bytes": (
            None
            if startup is None
            else startup["available_kv_bytes"] - startup["minimum_kv_bytes"]
        ),
        "startup_limiting_stage": (
            None if startup is None else startup["limiting_stage"]
        ),
        "breakdown_bytes": {
            "weights": runtime_estimate["weights_bytes"],
            "kv_pool_tensor": runtime_estimate["kv_pool_tensor_bytes"],
            "kv_planner": runtime_estimate["kv_planner_bytes"],
            "activation": runtime_estimate["activation_bytes"],
            "operator_workspace": runtime_estimate["operator_workspace_bytes"],
            "graph_cache": runtime_estimate["graph_cache_bytes"],
            "runtime": runtime_estimate["runtime_bytes"],
            "fragmentation": runtime_estimate["fragmentation_bytes"],
            "safety_reserve": runtime_estimate["safety_reserve_bytes"],
        },
        "component_sources": runtime_estimate["component_sources"],
    }


def recommend(c: dict[str, Any]) -> dict[str, Any]:
    rec = c["recommendation"]
    profile = resolve_profile_calibration(c)
    graph = estimate_graph(c, profile)
    baseline_weight = calculate(
        c,
        resolved_profile=profile,
        graph_estimate=graph,
    )["weight_estimate"]
    scenario_results: list[dict[str, Any]] = []
    for scenario in rec["scenarios"]:
        candidates = [
            _candidate(c, scenario, q, seqs, profile, graph)
            for seqs in rec["candidate_max_num_seqs"]
            for q in rec["candidate_max_num_batched_tokens"]
        ]
        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        feasible.sort(
            key=lambda item: (
                item["score"], item["fit_headroom_bytes"],
                item["max_num_batched_tokens"], item["max_num_seqs"]
            ),
            reverse=True,
        )
        recommended = feasible[0] if feasible else None
        frontier_seqs = []
        for seqs in rec["candidate_max_num_seqs"]:
            options = [item for item in feasible if item["max_num_seqs"] == seqs]
            best = max(options, key=lambda item: item["max_num_batched_tokens"], default=None)
            frontier_seqs.append({
                "max_num_seqs": seqs,
                "max_feasible_num_batched_tokens": None if best is None else best["max_num_batched_tokens"],
                "planning_total_bytes": None if best is None else best["planning_total_bytes"],
                "planning_upper_bytes": None if best is None else best["planning_upper_bytes"],
                "fit_headroom_bytes": None if best is None else best["fit_headroom_bytes"],
            })
        frontier_q = []
        for q in rec["candidate_max_num_batched_tokens"]:
            options = [item for item in feasible if item["max_num_batched_tokens"] == q]
            best = max(options, key=lambda item: item["max_num_seqs"], default=None)
            frontier_q.append({
                "max_num_batched_tokens": q,
                "max_feasible_num_seqs": None if best is None else best["max_num_seqs"],
                "planning_total_bytes": None if best is None else best["planning_total_bytes"],
                "planning_upper_bytes": None if best is None else best["planning_upper_bytes"],
                "fit_headroom_bytes": None if best is None else best["fit_headroom_bytes"],
            })
        infeasible = [item for item in candidates if not item["feasible"] and "planning_total_bytes" in item]
        closest = min(
            infeasible,
            key=lambda item: max(
                0,
                (item["planning_upper_bytes"] if rec["fit_basis"] == "planning_upper" else item["planning_total_bytes"])
                + item["minimum_headroom_bytes"] - item["requested_hbm_budget_bytes"],
            ),
            default=None,
        )
        scenario_results.append({
            "name": scenario["name"],
            "context_len": scenario["context_len"],
            "assumption": "all max_num_seqs slots on one DP engine reach this context length",
            "recommended": recommended,
            "top_alternatives": feasible[: int(rec["top_k"])],
            "frontier_by_max_num_seqs": frontier_seqs,
            "frontier_by_max_num_batched_tokens": frontier_q,
            "closest_infeasible": closest,
            "num_candidates": len(candidates),
            "num_feasible_candidates": len(feasible),
            "candidates": candidates,
        })

    maps = [
        {(item["max_num_batched_tokens"], item["max_num_seqs"]): item for item in scenario["candidates"]}
        for scenario in scenario_results
    ]
    aggregate = []
    for q in rec["candidate_max_num_batched_tokens"]:
        for seqs in rec["candidate_max_num_seqs"]:
            items = [mapping[(q, seqs)] for mapping in maps]
            if not all(item["feasible"] for item in items):
                continue
            limiting = min(range(len(items)), key=lambda index: items[index]["fit_headroom_bytes"])
            aggregate.append({
                "max_num_batched_tokens": q,
                "max_num_seqs": seqs,
                "score": candidate_score(q, seqs, rec),
                "limiting_scenario": scenario_results[limiting]["name"],
                "minimum_fit_headroom_bytes": items[limiting]["fit_headroom_bytes"],
                "startup_feasible": True,
                "runtime_budget_safe": True,
                "runtime_safe": True,
                "per_scenario": [
                    {
                        "name": scenario["name"],
                        "context_len": scenario["context_len"],
                        "planning_total_bytes": item["planning_total_bytes"],
                        "planning_upper_bytes": item["planning_upper_bytes"],
                        "fit_headroom_bytes": item["fit_headroom_bytes"],
                    }
                    for scenario, item in zip(scenario_results, items)
                ],
            })
    aggregate.sort(
        key=lambda item: (
            item["score"], item["minimum_fit_headroom_bytes"],
            item["max_num_batched_tokens"], item["max_num_seqs"]
        ),
        reverse=True,
    )
    startup_candidates = [
        item
        for item in scenario_results[0]["candidates"]
        if item.get("startup_feasible") is True
    ]
    startup_candidates.sort(
        key=lambda item: (
            item["score"],
            item.get("startup_headroom_bytes") or 0,
            item["max_num_batched_tokens"],
            item["max_num_seqs"],
        ),
        reverse=True,
    )
    startup_limit = startup_candidates[0] if startup_candidates else None
    _, _, requested = resolve_capacity_bytes(c)
    return {
        "operation": "recommend",
        "config": c,
        "model": {
            key: c["model"].get(key)
            for key in ("profile", "name", "family", "architecture", "kv_cache_strategy")
        },
        "method": {
            "objective": rec["objective"],
            "fit_basis": rec["fit_basis"],
            "requested_hbm_budget_bytes": requested,
            "minimum_headroom_bytes": bytes_from_gib(rec["minimum_headroom_gib_per_rank"]) or 0,
            "unresolved_workspace_reserve_bytes": bytes_from_gib(rec["unresolved_workspace_reserve_gib_per_rank"]) or 0,
            "score_description": "weighted logarithmic normalization of Q and max_num_seqs",
        },
        "profile_calibration_resolved": profile,
        "weight_estimate": baseline_weight,
        "graph_estimate": asdict(graph),
        "startup_limit_recommended": startup_limit,
        "runtime_safe_recommended": aggregate[0] if aggregate else None,
        "single_service_recommended": aggregate[0] if aggregate else None,
        "single_service_top_alternatives": aggregate[: int(rec["top_k"])],
        "scenarios": scenario_results,
    }
