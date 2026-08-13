"""Typed result objects shared by estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    file: str


@dataclass
class WeightEstimate:
    per_rank_bytes: int
    source: str
    total_checkpoint_bytes: Optional[int] = None
    routed_expert_checkpoint_bytes: Optional[int] = None
    dense_sharded_checkpoint_bytes: Optional[int] = None
    replicated_checkpoint_bytes: Optional[int] = None
    max_pp_stage: Optional[int] = None
    uncertainty_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)
    details: dict[str, int] = field(default_factory=dict)


@dataclass
class ActivationEstimate:
    analytical_peak_bytes: int
    used_peak_bytes: int
    source: str
    persistent_hidden_bytes: int
    attention_branch_bytes: int
    moe_branch_bytes: int
    components: dict[str, int]
    uncertainty_fraction: float


@dataclass
class ComponentEstimate:
    used_bytes: int
    analytical_bytes: int
    source: str
    uncertainty_fraction: float
    unresolved: bool = False


@dataclass
class KVEstimate:
    concurrency_input: int
    max_local_concurrency: int
    q_tokens_per_rank: int
    physical_pool_tensor_bytes: int
    planner_total_bytes: int
    details: dict[str, Any]
    source: str


@dataclass
class TotalEstimate:
    concurrency_input: int
    max_local_concurrency: int
    q_tokens_per_rank: int
    weights_bytes: int
    kv_pool_tensor_bytes: int
    kv_planner_bytes: int
    activation_bytes: int
    operator_workspace_bytes: int
    graph_cache_bytes: int
    runtime_bytes: int
    fragmentation_bytes: int
    safety_reserve_bytes: int
    actual_total_bytes: int
    planning_total_bytes: int
    lower_bound_bytes: int
    upper_bound_bytes: int
    requested_hbm_budget_bytes: int
    physical_hbm_bytes: int
    planning_headroom_bytes: int
    fits_requested_budget: bool
    fits_physical_hbm_upper_bound: bool
    high_confidence_fraction: float
    coverage_complete: bool
    unresolved_components: list[str]
    kv_details: dict[str, Any]
    activation_details: dict[str, Any]
    component_sources: dict[str, str]


@dataclass(frozen=True)
class StartupEstimate:
    max_num_batched_tokens: int
    max_num_seqs: int
    requested_memory_bytes: int
    visible_hbm_bytes: int
    startup_free_hbm_bytes: int
    theoretical_model_load_bytes: int
    model_load_bytes: int
    measured_weight_residual_bytes: int
    profile_activation_bytes: int
    non_torch_bytes: int
    graph_bytes: int
    available_kv_bytes: int
    minimum_kv_bytes: int
    reported_required_kv_bytes: Optional[int]
    current_kv_cache_bytes: Optional[int]
    minimum_kv_check_passed: bool
    graph_capture_passed: Optional[bool]
    startup_feasible: bool
    limiting_stage: Optional[str]
    component_sources: dict[str, str]
    details: dict[str, Any]
