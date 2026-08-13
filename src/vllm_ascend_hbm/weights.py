"""Weight HBM estimation from profile, safetensors metadata or model shape."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any, Optional

from .types import TensorRecord, WeightEstimate
from .utils import bytes_from_gib
from .weight_models.deepseek_v4_w8a8 import estimate_deepseek_v4_w8a8


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
EXPERT_RE = re.compile(r"(?:^|\.)experts(?:\.|$)")
SHARED_EXPERT_RE = re.compile(r"(?:^|\.)shared_experts?(?:\.|$)")
REPLICATED_RE = re.compile(
    r"(?:norm|layernorm|rms_norm)(?:\.|$)|e_score_correction_bias", re.IGNORECASE
)


def resolve_safetensor_files(model_path: str) -> list[Path]:
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"model path does not exist: {model_path}")
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    if path.is_file() and path.suffix == ".json":
        index_path, root = path, path.parent
    elif path.is_dir():
        candidates = [path / "model.safetensors.index.json", *sorted(path.glob("*.safetensors.index.json"))]
        index_path = next((candidate for candidate in candidates if candidate.exists()), None)
        root = path
        if index_path is None:
            files = sorted(path.glob("*.safetensors"))
            if not files:
                raise ValueError(f"no safetensors files found under {model_path}")
            return files
    else:
        raise ValueError("model_path must be a safetensors file, index, or directory")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid safetensors index {index_path}: {exc}") from exc
    return [root / name for name in sorted(set(weight_map.values()))]


def read_safetensor_header(path: Path) -> list[TensorRecord]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("file is shorter than the safetensors header")
            header_len = struct.unpack("<Q", raw)[0]
            if not 0 < header_len <= 512 * 1024 * 1024:
                raise ValueError(f"unreasonable header length: {header_len}")
            header = json.loads(stream.read(header_len))
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise ValueError(f"cannot parse safetensors header {path}: {exc}") from exc
    rows: list[TensorRecord] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        rows.append(
            TensorRecord(
                name=name,
                dtype=str(info["dtype"]),
                shape=tuple(int(value) for value in info["shape"]),
                nbytes=int(end) - int(start),
                file=str(path),
            )
        )
    return rows


def _tensor_stage(name: str, pp: int, layers: int) -> int:
    match = LAYER_RE.search(name)
    if match:
        layer = min(int(match.group(1)), layers - 1)
        return min(pp - 1, layer * pp // layers)
    low = name.lower()
    if "lm_head" in low or "output_layer" in low or re.search(r"(?:^|\.)model\.norm", low):
        return pp - 1
    return 0


def parsed_weight_estimate(c: dict[str, Any], model_path: str) -> WeightEstimate:
    files = resolve_safetensor_files(model_path)
    records = [record for file in files for record in read_safetensor_header(file)]
    par, model, wc = c["parallelism"], c["model"], c["weights"]
    pp = par["pp_size"]
    stage_expert = [0.0] * pp
    stage_sharded = [0.0] * pp
    stage_replicated = [0.0] * pp
    total_expert = total_sharded = total_replicated = 0
    for record in records:
        stage = _tensor_stage(record.name, pp, model["num_hidden_layers"])
        is_expert = bool(EXPERT_RE.search(record.name)) and not SHARED_EXPERT_RE.search(record.name)
        if is_expert:
            total_expert += record.nbytes
            stage_expert[stage] += record.nbytes / par["ep_size"]
        elif REPLICATED_RE.search(record.name):
            total_replicated += record.nbytes
            stage_replicated[stage] += record.nbytes
        else:
            total_sharded += record.nbytes
            stage_sharded[stage] += record.nbytes / par["tp_size"]
    per_stage = [
        (stage_expert[index] + stage_sharded[index] + stage_replicated[index])
        * (1 + float(wc["checkpoint_to_hbm_overhead_fraction"]))
        * float(wc["pp_imbalance_factor"])
        for index in range(pp)
    ]
    max_stage = max(range(pp), key=per_stage.__getitem__)
    return WeightEstimate(
        per_rank_bytes=round(per_stage[max_stage]),
        source="safetensors-header+name-based-sharding",
        total_checkpoint_bytes=sum(record.nbytes for record in records),
        routed_expert_checkpoint_bytes=total_expert,
        dense_sharded_checkpoint_bytes=total_sharded,
        replicated_checkpoint_bytes=total_replicated,
        max_pp_stage=max_stage,
        uncertainty_fraction=float(c["uncertainty"]["parsed_weight_sharding_fraction"]),
        notes=[
            f"parsed {len(records):,} tensors from {len(files)} shard(s)",
            "checkpoint bytes are exact; rank placement is inferred from names",
        ],
    )


def analytical_weight_estimate(c: dict[str, Any]) -> WeightEstimate:
    model, wc, par = c["model"], c["weights"], c["parallelism"]
    total = int(model["total_parameters"])
    experts = int(model.get("num_routed_experts") or 0)
    moe_layers = int(model.get("num_moe_layers") or model["num_hidden_layers"])
    moe_hidden = int(model.get("moe_intermediate_size") or 0)
    routed_params = 0
    if experts and moe_hidden:
        routed_params = moe_layers * experts * 3 * int(model["hidden_size"]) * moe_hidden
        routed_params = min(total, routed_params)
    dense_params = max(0, total - routed_params)
    routed_bytes = routed_params * float(wc["routed_expert_weight_bits"]) / 8
    dense_bytes = dense_params * float(wc["dense_weight_bits"]) / 8
    local = (routed_bytes / par["ep_size"] + dense_bytes / par["tp_size"]) / par["pp_size"]
    local *= 1 + float(wc["checkpoint_to_hbm_overhead_fraction"])
    local *= float(wc["pp_imbalance_factor"])
    return WeightEstimate(
        per_rank_bytes=round(local),
        source="architecture-parameter-count",
        total_checkpoint_bytes=round(routed_bytes + dense_bytes),
        routed_expert_checkpoint_bytes=round(routed_bytes),
        dense_sharded_checkpoint_bytes=round(dense_bytes),
        uncertainty_fraction=float(c["uncertainty"]["analytical_weight_fraction"]),
        notes=[
            "routed expert weights divide by EP; remaining weights divide by TP",
            "use model_path or measured weight HBM when tensor placement is irregular",
        ],
    )


def estimate_weights(
    c: dict[str, Any],
    profile: dict[str, Optional[float]],
) -> WeightEstimate:
    theoretical: Optional[WeightEstimate] = None
    if (
        c["model"].get("profile") == "deepseek-v4-flash"
        and str(c["platform"].get("vllm_ascend_version", "")).lower()
        .lstrip("v")
        .startswith("0.23")
    ):
        theoretical = estimate_deepseek_v4_w8a8(c)

    measured = profile.get("weight_gib_per_rank")
    if measured is not None:
        measured_bytes = bytes_from_gib(measured) or 0
        if theoretical is not None:
            details = dict(theoretical.details)
            details["theoretical_model_load_bytes"] = (
                theoretical.per_rank_bytes
            )
            details["measured_model_load_bytes"] = measured_bytes
            details["measured_residual_bytes"] = (
                measured_bytes - theoretical.per_rank_bytes
            )
            return WeightEstimate(
                per_rank_bytes=measured_bytes,
                source="vllm-profile+theory-residual",
                total_checkpoint_bytes=theoretical.total_checkpoint_bytes,
                uncertainty_fraction=0.02,
                notes=[
                    *theoretical.notes,
                    "measured model-load memory is used for capacity checks",
                    "the theoretical estimate and residual remain visible",
                ],
                details=details,
            )
        return WeightEstimate(
            per_rank_bytes=measured_bytes,
            source="vllm-profile",
            uncertainty_fraction=0.02,
            notes=["profile value overrides checkpoint and analytical estimates"],
        )
    manual = c["weights"]["manual_gib_per_rank"]
    if manual is not None:
        return WeightEstimate(
            per_rank_bytes=bytes_from_gib(manual) or 0,
            source="manual-per-rank",
            uncertainty_fraction=0.02,
            notes=["manual loaded-weight value"],
        )
    if theoretical is not None:
        return theoretical
    if c["model"].get("model_path"):
        return parsed_weight_estimate(c, str(c["model"]["model_path"]))
    return analytical_weight_estimate(c)
