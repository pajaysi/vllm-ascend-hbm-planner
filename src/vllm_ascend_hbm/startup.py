"""Startup lifecycle model for source-equivalent capacity checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

from .capacity import resolve_capacity_bytes
from .components import estimate_activation
from .kv.deepseek_v4_v023 import minimum_kv_admission
from .logs import parse_startup_log
from .types import StartupEstimate
from .utils import bytes_from_gib
from .weight_models.deepseek_v4_w8a8 import estimate_deepseek_v4_w8a8


def startup_model_supported(c: dict[str, Any]) -> bool:
    return (
        c["model"].get("kv_cache_strategy")
        == "deepseek_v4_flash"
        and str(c["platform"].get("vllm_ascend_version", "")).startswith(
            "0.23"
        )
        and int(c["scheduler"].get("block_size", 0)) == 128
    )


def _configured_non_torch(
    c: dict[str, Any],
) -> tuple[int, dict[str, int]]:
    runtime = c["runtime"]
    manual = runtime.get("manual_non_torch_gib_per_rank")
    if manual is not None:
        value = bytes_from_gib(manual) or 0
        return value, {"manual_non_torch_bytes": value}
    fixed = bytes_from_gib(
        float(runtime.get("base_persistent_gib_per_rank", 0.0))
        + float(runtime.get("hccl_and_cann_persistent_gib_per_rank", 0.0))
    ) or 0
    ascend = c.get("vllm_ascend", {})
    buffsize_mib = ascend.get("hccl_buffsize_mib")
    domains = int(
        ascend.get("hccl_communication_domains_per_rank", 1)
    )
    # Huawei documents independent send and receive buffers for every HCCL
    # communication domain: 2 * HCCL_BUFFSIZE per domain.
    hccl = (
        0
        if buffsize_mib is None
        else round(2 * float(buffsize_mib) * domains * 2**20)
    )
    return fixed + hccl, {
        "configured_fixed_bytes": fixed,
        "hccl_buffer_bytes": hccl,
        "hccl_communication_domains_per_rank": domains,
    }


def evaluate_startup(
    c: dict[str, Any],
    q: int,
    seqs: int,
    log_text: Optional[str] = None,
) -> StartupEstimate:
    pair = copy.deepcopy(c)
    pair["scheduler"]["max_num_batched_tokens"] = int(q)
    pair["scheduler"]["max_num_seqs"] = int(seqs)
    calibration = pair.get("profile_calibration", {})
    if log_text is None and calibration.get("vllm_log_path"):
        path = Path(str(calibration["vllm_log_path"]))
        try:
            log_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(
                f"cannot read vLLM startup log {path}: {exc}"
            ) from exc
    parsed = parse_startup_log(log_text)

    visible, startup_free, requested = resolve_capacity_bytes(
        pair,
        visible_gib=parsed.visible_hbm_gib,
        startup_free_gib=parsed.startup_free_gib,
        requested_gib=parsed.requested_memory_gib,
    )
    theoretical_weight = estimate_deepseek_v4_w8a8(pair)
    calibrated_weight = (
        parsed.weight_gib
        if parsed.weight_gib is not None
        else calibration.get("weight_gib_per_rank")
    )
    if calibrated_weight is not None:
        model_load = bytes_from_gib(calibrated_weight) or 0
        profiled_q = calibration.get(
            "profiled_max_num_batched_tokens"
        )
        if profiled_q:
            reference = copy.deepcopy(pair)
            reference["scheduler"]["max_num_batched_tokens"] = int(
                profiled_q
            )
            reference_weight = estimate_deepseek_v4_w8a8(reference)
            model_load += (
                theoretical_weight.per_rank_bytes
                - reference_weight.per_rank_bytes
            )
        weight_source = "measured"
    else:
        model_load = theoretical_weight.per_rank_bytes
        weight_source = theoretical_weight.source

    calibrated_activation = (
        parsed.activation_gib
        if parsed.activation_gib is not None
        else calibration.get("peak_activation_gib_per_rank")
    )
    if calibrated_activation is not None:
        activation = bytes_from_gib(calibrated_activation) or 0
        profiled_q = calibration.get(
            "profiled_max_num_batched_tokens"
        )
        if profiled_q:
            activation = round(
                activation * int(q) / int(profiled_q)
            )
        activation_source = "measured"
    else:
        activation_estimate = estimate_activation(pair, q, {})
        activation = activation_estimate.used_peak_bytes
        activation_source = activation_estimate.source

    calibrated_non_torch = (
        parsed.non_torch_gib
        if parsed.non_torch_gib is not None
        else calibration.get("non_torch_gib_per_rank")
    )
    if calibrated_non_torch is not None:
        non_torch = bytes_from_gib(calibrated_non_torch) or 0
        non_torch_source = "measured"
        non_torch_details = {
            "measured_non_torch_bytes": non_torch,
        }
    else:
        non_torch, non_torch_details = _configured_non_torch(pair)
        non_torch_source = "source-analytical-hccl-plus-configured-fixed"

    calibrated_graph = (
        parsed.graph_gib
        if parsed.graph_gib is not None
        else calibration.get("graph_gib_per_rank")
    )
    graph = bytes_from_gib(calibrated_graph) or 0
    computed_available = requested - model_load - activation - non_torch
    available = (
        bytes_from_gib(parsed.available_kv_gib)
        if parsed.available_kv_gib is not None
        else computed_available
    )
    available = available or 0

    minimum = minimum_kv_admission(
        int(pair["scheduler"]["max_model_len"]),
        int(q),
        int(pair["scheduler"]["block_size"]),
    )
    minimum_passed = minimum.total_bytes <= available
    if not minimum_passed:
        limiting_stage = "minimum_kv_check"
        graph_passed: Optional[bool] = None
        feasible = False
    else:
        limiting_stage = None
        graph_passed = (
            True
            if parsed.graph_capture_finished
            else None
        )
        feasible = True

    measured_residual = (
        model_load - theoretical_weight.per_rank_bytes
        if calibrated_weight is not None
        else 0
    )
    return StartupEstimate(
        max_num_batched_tokens=int(q),
        max_num_seqs=int(seqs),
        requested_memory_bytes=requested,
        visible_hbm_bytes=visible,
        startup_free_hbm_bytes=startup_free,
        theoretical_model_load_bytes=theoretical_weight.per_rank_bytes,
        model_load_bytes=model_load,
        measured_weight_residual_bytes=measured_residual,
        profile_activation_bytes=activation,
        non_torch_bytes=non_torch,
        graph_bytes=graph,
        available_kv_bytes=available,
        minimum_kv_bytes=minimum.total_bytes,
        reported_required_kv_bytes=bytes_from_gib(
            parsed.required_kv_gib
        ),
        current_kv_cache_bytes=bytes_from_gib(parsed.current_kv_gib),
        minimum_kv_check_passed=minimum_passed,
        graph_capture_passed=graph_passed,
        startup_feasible=feasible,
        limiting_stage=limiting_stage,
        component_sources={
            "model_load": weight_source,
            "profile_activation": activation_source,
            "non_torch": non_torch_source,
            "minimum_kv": "source_exact:vllm-ascend-0.23",
            "graph": "measured" if calibrated_graph is not None else "unknown",
        },
        details={
            "minimum_kv": minimum.__dict__,
            "theoretical_weight": theoretical_weight.details,
            "non_torch": non_torch_details,
            "computed_available_kv_bytes": computed_available,
            "estimated_max_model_len": parsed.estimated_max_model_len,
        },
    )
