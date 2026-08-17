"""JSON configuration loading, profile resolution and validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

from .hf_config import enrich_model_from_hf
from .profiles import get_profile
from .utils import ceildiv, deep_merge


MODEL_SCHEMA: dict[str, Any] = {
    "profile": "deepseek-v4-flash",
    "name": None,
    "config_path": None,
    "resolved_config_path": None,
    "model_path": None,
    "family": None,
    "architecture": None,
    "task": "generate",
    "total_parameters": None,
    "num_hidden_layers": None,
    "num_cache_layers": None,
    "hidden_size": None,
    "vocab_size": None,
    "num_attention_heads": None,
    "num_key_value_heads": None,
    "head_dim": None,
    "sliding_window": None,
    "max_position_embeddings": None,
    "intermediate_size": 0,
    "num_routed_experts": 0,
    "num_shared_experts": 0,
    "num_experts_per_token": 0,
    "num_moe_layers": None,
    "moe_intermediate_size": 0,
    "kv_lora_rank": None,
    "qk_rope_head_dim": None,
    "kv_cache_strategy": "auto",
    "activation_model": "auto",
    "query_compression_dim": 0,
    "indexer_heads": 0,
    "indexer_head_dim": 0,
    "attention_topk": 0,
    "output_projection_groups": 0,
    "attention_output_intermediate_dim": 0,
    "mhc_expansion_factor": 1,
    "mtp_layers": 0,
    "multimodal_text_backbone_only": False,
}


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 2,
    "operation": "recommend",
    "platform": {
        "device": "910c",
        "vllm_ascend_version": "0.20",
        "hbm_gib_per_die": 64.0,
        "nominal_hbm_gib_per_die": None,
        "visible_hbm_gib_per_die": None,
        "startup_free_hbm_gib_per_die": None,
        "gpu_memory_utilization": 0.90,
        "server_count": None,
        "physical_cards_per_server": None,
        "dies_per_card": None,
        "logical_devices_per_server": None,
        "logical_device_count": None,
    },
    "vllm_ascend": {
        "enable_shared_expert_dp": False,
        "multistream_overlap_shared_expert": False,
        "enable_flashcomm1": False,
        "hccl_buffsize_mib": None,
        "hccl_communication_domains_per_rank": 1,
    },
    "model": MODEL_SCHEMA,
    "weights": {
        "manual_gib_per_rank": None,
        "dense_weight_bits": 8.0,
        "routed_expert_weight_bits": 8.0,
        "checkpoint_to_hbm_overhead_fraction": 0.03,
        "pp_imbalance_factor": 1.0,
    },
    "kv_cache": {
        "dtype_bytes": 2,
        "page_alignment_bytes": 256,
        "planner_overhead_fraction": 0.0,
        "manual_bytes_per_token_per_rank": None,
        "manual_gib_per_rank": None,
    },
    "scheduler": {
        "block_size": 128,
        "max_model_len": 1_048_576,
        "max_num_batched_tokens": 81_920,
        "max_num_seqs": 64,
    },
    "parallelism": {
        "dp_size": 4,
        "tp_size": 4,
        "pp_size": 1,
        "ep_size": 16,
        "pcp_size": 1,
        "dcp_size": 1,
    },
    "workload": {
        "mode": "late",
        "context_len": 1_048_576,
        "concurrency": [1, 2, 4, 8, 16, 32, 64],
        "concurrency_scope": "global",
    },
    "activation": {
        "dtype_bytes": 2,
        "hidden_buffer_count": 3.0,
        "attention_buffer_factor": 2.0,
        "moe_dispatch_buffer_copies": 2.0,
        "moe_intermediate_buffer_count": 2.0,
        "moe_capacity_factor": 1.10,
        "branch_live_fraction": 0.70,
        "manual_peak_gib_per_rank": None,
    },
    "operator_workspace": {
        "manual_peak_gib_per_rank": None,
        "components_gib_per_rank": {},
        "concurrent_factor": 1.0,
        "assume_in_profile_activation": True,
    },
    "graph_cache": {
        "mode": "eager",
        "manual_gib_per_rank": None,
        "capture_sizes": [],
        "fixed_gib_per_graph": None,
        "bytes_per_captured_token": None,
    },
    "runtime": {
        # Keep the theory baseline at zero.  HCCL is derived separately from
        # HCCL_BUFFSIZE; a CANN/ACL residual must be supplied as calibration
        # instead of being hidden in an arbitrary default.
        "base_persistent_gib_per_rank": 0.0,
        "hccl_and_cann_persistent_gib_per_rank": 0.0,
        "bytes_per_scheduled_token": 32,
        "block_table_entry_bytes": 4,
        "sampler_logit_bytes": 4,
        "manual_non_torch_gib_per_rank": None,
        "allocator_fragmentation_fraction": 0.03,
        "safety_reserve_gib_per_rank": 0.50,
    },
    "profile_calibration": {
        "vllm_log_path": None,
        "profiled_max_num_batched_tokens": None,
        "weight_gib_per_rank": None,
        "peak_activation_gib_per_rank": None,
        "non_torch_gib_per_rank": None,
        "graph_gib_per_rank": None,
    },
    "uncertainty": {
        "analytical_weight_fraction": 0.08,
        "parsed_weight_sharding_fraction": 0.05,
        "analytical_activation_fraction": 0.35,
        "profile_activation_fraction": 0.05,
        "workspace_fraction": 0.35,
        "analytical_runtime_fraction": 0.50,
        "profile_runtime_fraction": 0.10,
        "kv_tensor_fraction": 0.01,
    },
    "recommendation": {
        "objective": "balanced",
        "candidate_max_num_batched_tokens": [
            1024, 2048, 4096, 8192, 10240, 16384, 32768, 49152, 65536, 81920
        ],
        "candidate_max_num_seqs": [1, 2, 4, 8, 16, 32, 64],
        "scenarios": [
            {"name": "max_context", "context_len": 1_048_576},
            {"name": "32k_typical", "context_len": 32_768},
        ],
        "fit_basis": "planning_upper",
        "minimum_headroom_gib_per_rank": 2.0,
        "unresolved_workspace_reserve_gib_per_rank": 2.0,
        "balanced_q_weight": 0.50,
        "balanced_seq_weight": 0.50,
        "top_k": 5,
    },
    "validation": {
        "profile_calibration_by_tp": {},
    },
    "output": {"format": "text"},
}


def _read_user(path: Optional[str]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read config {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path!r}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    if value.get("schema_version", 2) not in {1, 2}:
        raise ValueError("schema_version must be 1 or 2")
    value = copy.deepcopy(value)
    # v1 used the DeepSeek-specific name ``attention_head_dim``.  v2 uses the
    # architecture-neutral Hugging Face spelling ``head_dim``.
    user_model = value.get("model")
    if isinstance(user_model, dict) and "attention_head_dim" in user_model:
        user_model.setdefault("head_dim", user_model["attention_head_dim"])
        del user_model["attention_head_dim"]
    value["schema_version"] = 2
    return value


def _read_hardware(path: Optional[str]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read hardware config {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path!r}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("hardware JSON root must be an object")
    if value.get("schema_version", 1) != 1:
        raise ValueError("hardware schema_version must be 1")
    hardware = value.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("hardware JSON needs a hardware object")
    hardware = copy.deepcopy(hardware)
    if (
        hardware.get("hbm_gib_per_die") is None
        and hardware.get("nominal_hbm_gib_per_die") is not None
    ):
        hardware["hbm_gib_per_die"] = hardware["nominal_hbm_gib_per_die"]
    if (
        hardware.get("server_count") is not None
        and hardware.get("logical_devices_per_server") is not None
    ):
        hardware["logical_device_count"] = (
            int(hardware["server_count"])
            * int(hardware["logical_devices_per_server"])
        )
    return hardware


def load_config(
    path: Optional[str] = None,
    hardware_path: Optional[str] = None,
) -> dict[str, Any]:
    user = _read_user(path)
    hardware = _read_hardware(hardware_path)
    if hardware:
        user.setdefault("platform", {}).update(hardware)
    if path is not None:
        root = Path(path).resolve().parent
        for section, key in (
            ("model", "config_path"),
            ("model", "model_path"),
            ("profile_calibration", "vllm_log_path"),
        ):
            value = user.get(section, {}).get(key)
            if value:
                candidate = Path(str(value))
                if not candidate.is_absolute():
                    user[section][key] = str((root / candidate).resolve())
    profile_name = str(user.get("model", {}).get("profile", "deepseek-v4-flash"))
    base = copy.deepcopy(DEFAULT_CONFIG)
    if profile_name.lower() != "auto":
        profile = get_profile(profile_name)
        base["model"] = deep_merge(base["model"], profile.defaults, "model-profile")
        base["model"]["profile"] = profile.profile_id
        base["model"]["name"] = profile.display_name

    c = deep_merge(base, user)
    model = c["model"]
    config_path = model.get("config_path")
    if config_path is None and model.get("model_path"):
        candidate = Path(str(model["model_path"])) / "config.json"
        if candidate.is_file():
            config_path = str(candidate)
            model["config_path"] = config_path
    if config_path:
        enrich_model_from_hf(model, str(config_path))

    raw_scheduler = user.get("scheduler", {})
    raw_workload = user.get("workload", {})
    raw_rec = user.get("recommendation", {})
    model_max = model.get("max_position_embeddings")
    if model_max and "max_model_len" not in raw_scheduler:
        c["scheduler"]["max_model_len"] = int(model_max)
    max_len = int(c["scheduler"]["max_model_len"])
    if "context_len" not in raw_workload:
        c["workload"]["context_len"] = max_len
    if "scenarios" not in raw_rec:
        typical = min(32_768, max_len)
        scenarios = [{"name": "max_context", "context_len": max_len}]
        if typical != max_len:
            scenarios.append({"name": "32k_typical", "context_len": typical})
        c["recommendation"]["scenarios"] = scenarios
    validate_config(c)
    return c


def normalize_concurrency(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        values = value
    elif isinstance(value, str):
        values = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = (int(item) for item in part.split("-", 1))
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))
    else:
        raise ValueError("workload.concurrency must be an integer, list, or range string")
    if not values or any(value <= 0 for value in values):
        raise ValueError("all workload.concurrency values must be positive")
    return sorted(set(values))


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")


def validate_config(c: dict[str, Any]) -> None:
    if c["operation"] not in {"estimate", "recommend"}:
        raise ValueError("operation must be estimate or recommend")
    p, m, s, par, workload = (
        c["platform"], c["model"], c["scheduler"], c["parallelism"], c["workload"]
    )
    for name in ("num_hidden_layers", "num_cache_layers", "hidden_size", "vocab_size"):
        _positive_int(f"model.{name}", m.get(name))
    if m.get("total_parameters") is None and m.get("model_path") is None and c["weights"]["manual_gib_per_rank"] is None and c["profile_calibration"]["weight_gib_per_rank"] is None:
        raise ValueError(
            "weight modeling needs model.total_parameters, model.model_path, "
            "weights.manual_gib_per_rank, or a profiled weight value"
        )
    strategy = m["kv_cache_strategy"]
    if strategy not in {"standard_gqa", "mla", "deepseek_v4_flash", "manual", "none"}:
        raise ValueError(f"unsupported model.kv_cache_strategy={strategy!r}")
    if strategy == "standard_gqa":
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            _positive_int(f"model.{name}", m.get(name))
    elif strategy == "mla":
        _positive_int("model.kv_lora_rank", m.get("kv_lora_rank"))
        _positive_int("model.qk_rope_head_dim", m.get("qk_rope_head_dim"), allow_zero=True)
    elif strategy == "manual":
        kv = c["kv_cache"]
        if kv["manual_bytes_per_token_per_rank"] is None and kv["manual_gib_per_rank"] is None:
            raise ValueError(
                "manual KV strategy needs kv_cache.manual_bytes_per_token_per_rank "
                "or kv_cache.manual_gib_per_rank"
            )

    for name in ("block_size", "max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        _positive_int(f"scheduler.{name}", s[name])
    for name, value in par.items():
        _positive_int(f"parallelism.{name}", value)
    for name in (
        "server_count",
        "physical_cards_per_server",
        "dies_per_card",
        "logical_devices_per_server",
    ):
        if p.get(name) is not None:
            _positive_int(f"platform.{name}", p[name])
    if all(
        p.get(name) is not None
        for name in (
            "physical_cards_per_server",
            "dies_per_card",
            "logical_devices_per_server",
        )
    ):
        derived_per_server = (
            p["physical_cards_per_server"] * p["dies_per_card"]
        )
        if derived_per_server != p["logical_devices_per_server"]:
            raise ValueError(
                "physical_cards_per_server*dies_per_card="
                f"{derived_per_server} does not match "
                "logical_devices_per_server="
                f"{p['logical_devices_per_server']}"
            )
    logical_device_count = p.get("logical_device_count")
    if logical_device_count is not None:
        _positive_int("platform.logical_device_count", logical_device_count)
        world_size = par["dp_size"] * par["tp_size"] * par["pp_size"]
        if world_size > logical_device_count:
            raise ValueError(
                f"DP*TP*PP={world_size} exceeds hardware "
                f"logical_device_count={logical_device_count}"
            )
    if par["pcp_size"] > par["tp_size"] or par["dcp_size"] > par["tp_size"]:
        raise ValueError("PCP/DCP cannot exceed TP")
    if par["tp_size"] % par["pcp_size"] or par["tp_size"] % par["dcp_size"]:
        raise ValueError("PCP and DCP must divide TP")
    if par["ep_size"] > par["dp_size"] * par["tp_size"]:
        raise ValueError("parallelism.ep_size cannot exceed DP*TP within a PP stage")
    if float(p["hbm_gib_per_die"]) <= 0 or not 0 < float(p["gpu_memory_utilization"]) <= 1:
        raise ValueError("HBM must be positive and gpu_memory_utilization must be in (0,1]")

    if workload["mode"] not in {"fresh", "late", "admission"}:
        raise ValueError("workload.mode must be fresh, late, or admission")
    if workload["concurrency_scope"] not in {"global", "per-dp"}:
        raise ValueError("workload.concurrency_scope must be global or per-dp")
    workload["concurrency"] = normalize_concurrency(workload["concurrency"])
    if not 0 < int(workload["context_len"]) <= s["max_model_len"]:
        raise ValueError("workload.context_len must be in (0, max_model_len]")
    for concurrency in workload["concurrency"]:
        local = concurrency if workload["concurrency_scope"] == "per-dp" else ceildiv(concurrency, min(concurrency, par["dp_size"]))
        if local > s["max_num_seqs"]:
            raise ValueError(
                f"workload concurrency {concurrency} implies {local} sequences per DP, "
                f"above max_num_seqs={s['max_num_seqs']}"
            )

    graph = c["graph_cache"]
    if graph["mode"] not in {"eager", "acl_graph"}:
        raise ValueError("graph_cache.mode must be eager or acl_graph")
    if graph["mode"] == "acl_graph" and graph["manual_gib_per_rank"] is None:
        if not graph["capture_sizes"] or graph["fixed_gib_per_graph"] is None or graph["bytes_per_captured_token"] is None:
            raise ValueError(
                "ACL Graph needs manual_gib_per_rank, or capture_sizes plus two coefficients"
            )

    rec = c["recommendation"]
    if rec["objective"] not in {"balanced", "prefill_throughput", "concurrency"}:
        raise ValueError("invalid recommendation.objective")
    if rec["fit_basis"] not in {"planning_center", "planning_upper"}:
        raise ValueError("invalid recommendation.fit_basis")
    for key in ("candidate_max_num_batched_tokens", "candidate_max_num_seqs"):
        values = rec[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"recommendation.{key} must be a non-empty list")
        for value in values:
            _positive_int(f"recommendation.{key}[]", value)
        rec[key] = sorted(set(values))
    if not isinstance(rec["scenarios"], list) or not rec["scenarios"]:
        raise ValueError("recommendation.scenarios must be non-empty")
    names: set[str] = set()
    for scenario in rec["scenarios"]:
        if set(scenario) != {"name", "context_len"}:
            raise ValueError("each scenario needs exactly name and context_len")
        if scenario["name"] in names:
            raise ValueError(f"duplicate scenario name: {scenario['name']}")
        names.add(scenario["name"])
        if not 0 < int(scenario["context_len"]) <= s["max_model_len"]:
            raise ValueError(f"scenario {scenario['name']} exceeds max_model_len")
