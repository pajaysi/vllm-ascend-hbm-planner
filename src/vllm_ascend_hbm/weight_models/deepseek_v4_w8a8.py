"""Tensor-aware DeepSeek-V4-Flash W8A8_DYNAMIC model-load estimate.

The formulas mirror the v0.23 model module classes: routed experts are local
to the EP rank, selected attention/MLP tensors are TP-sharded, and
ReplicatedLinear tensors remain present on every rank.  The result also
includes post-load FP32 scale copies and model-owned Q-dependent buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import WeightEstimate
from .persistent_buffers import deepseek_v4_persistent_buffers


@dataclass(frozen=True)
class DeepSeekV4W8A8Geometry:
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    routed_experts: int = 256
    shared_experts: int = 1
    layers: int = 43
    mtp_layers: int = 1
    vocab_size: int = 129_280
    q_lora_rank: int = 1024
    head_dim: int = 512
    attention_heads: int = 64
    output_groups: int = 8
    output_lora_rank: int = 1024
    index_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    c4_layers: int = 21
    c128_layers: int = 20


def _dynamic_linear_bytes(input_size: int, output_size: int) -> int:
    # INT8 weight + BF16 scale + BF16 offset + post-load FP32 scale copy.
    return input_size * output_size + output_size * (2 + 2 + 4)


def _geometry(c: dict[str, Any]) -> DeepSeekV4W8A8Geometry:
    model = c["model"]
    return DeepSeekV4W8A8Geometry(
        hidden_size=int(model.get("hidden_size") or 4096),
        moe_intermediate_size=int(
            model.get("moe_intermediate_size") or 2048
        ),
        routed_experts=int(model.get("num_routed_experts") or 256),
        shared_experts=int(model.get("num_shared_experts") or 1),
        layers=int(model.get("num_hidden_layers") or 43),
        mtp_layers=int(model.get("mtp_layers") or 0),
        vocab_size=int(model.get("vocab_size") or 129_280),
        q_lora_rank=int(model.get("query_compression_dim") or 1024),
        head_dim=int(model.get("head_dim") or 512),
        attention_heads=int(model.get("num_attention_heads") or 64),
        output_groups=int(model.get("output_projection_groups") or 8),
        output_lora_rank=int(
            model.get("attention_output_intermediate_dim") or 1024
        ),
        index_heads=int(model.get("indexer_heads") or 64),
        index_head_dim=int(model.get("indexer_head_dim") or 128),
        index_topk=int(model.get("attention_topk") or 512),
        hc_mult=int(model.get("mhc_expansion_factor") or 4),
    )


def estimate_deepseek_v4_w8a8(c: dict[str, Any]) -> WeightEstimate:
    geometry = _geometry(c)
    parallel = c["parallelism"]
    tp = int(parallel["tp_size"])
    ep = int(parallel["ep_size"])
    pp = int(parallel["pp_size"])
    if pp != 1:
        raise ValueError(
            "the exact DeepSeek-V4 W8A8 estimator currently requires pp_size=1"
        )
    if geometry.routed_experts % ep:
        raise ValueError("num_routed_experts must be divisible by ep_size")

    hidden = geometry.hidden_size
    intermediate = geometry.moe_intermediate_size
    layer_count = geometry.layers + geometry.mtp_layers
    local_experts = geometry.routed_experts // ep
    details: dict[str, int] = {}

    # FusedMoE: W13 contains gate+up; W2 is the down projection.
    expert_weight = 3 * hidden * intermediate
    expert_scale_offset = (2 * intermediate + hidden) * 4
    expert_fp32_scale_copy = 2 * intermediate * 4
    details["routed_expert_bytes"] = (
        layer_count
        * local_experts
        * (expert_weight + expert_scale_offset + expert_fp32_scale_copy)
    )

    shared_one = (
        3 * hidden * intermediate
        + (2 * intermediate + hidden) * 8
    )
    shared_dp = bool(
        c.get("vllm_ascend", {}).get("enable_shared_expert_dp", False)
    )
    shared_divisor = 1 if shared_dp else tp
    details["shared_expert_bytes"] = (
        layer_count
        * geometry.shared_experts
        * shared_one
        // shared_divisor
    )

    replicated_per_layer = (
        _dynamic_linear_bytes(hidden, geometry.q_lora_rank)
        + _dynamic_linear_bytes(hidden, geometry.head_dim)
    )
    details["attention_replicated_bytes"] = (
        layer_count * replicated_per_layer
    )

    q_output = geometry.attention_heads * geometry.head_dim
    q_output_local = q_output // tp
    wo_a_input = q_output // geometry.output_groups
    wo_a_output = geometry.output_groups * geometry.output_lora_rank
    details["attention_tp_sharded_bytes"] = layer_count * (
        _dynamic_linear_bytes(geometry.q_lora_rank, q_output_local)
        + (
            wo_a_input * (wo_a_output // tp)
            + (wo_a_output // tp) * hidden
        )
        * 2
    )
    details["attention_norm_and_sink_bytes"] = layer_count * (
        geometry.q_lora_rank * 2
        + geometry.head_dim * 2
        + geometry.attention_heads * 4
    )

    compressor_bytes = 0
    for count, ratio, overlap in (
        (geometry.c4_layers, 4, 2),
        (geometry.c128_layers, 128, 1),
    ):
        output = overlap * geometry.head_dim
        compressor_bytes += count * (
            # quant_model_description marks compressor.wkv/wgate FLOAT.
            2 * hidden * output * 4
            + ratio * output * 4
            + geometry.head_dim * 2
        )
    details["compressor_bytes"] = compressor_bytes
    details["compressor_projection_dtype_bytes"] = 4

    index_output = geometry.index_heads * geometry.index_head_dim
    index_compressor_output = 2 * geometry.index_head_dim
    details["indexer_bytes"] = geometry.c4_layers * (
        _dynamic_linear_bytes(geometry.q_lora_rank, index_output)
        + hidden * geometry.index_heads * 2
        + 2 * hidden * index_compressor_output * 4
        + 4 * index_compressor_output * 4
        + geometry.index_head_dim * 2
    )

    router = 0
    for layer_index in range(geometry.layers):
        # AscendUnquantizedLinearMethod keeps the original BF16 parameter and
        # adds weight_fp32 because DeepseekV4MoE sets precast_fp32_weight.
        router += hidden * geometry.routed_experts * (2 + 4)
        router += (
            geometry.vocab_size * 6 * 4
            if layer_index < 3
            else geometry.routed_experts * 4
        )
    if geometry.mtp_layers:
        router += geometry.mtp_layers * (
            hidden * geometry.routed_experts * (2 + 4)
            + geometry.routed_experts * 4
        )
    details["router_bytes"] = router
    details["router_retains_bf16_and_fp32"] = 1

    mix_hc = (2 + geometry.hc_mult) * geometry.hc_mult
    hc_dim = geometry.hc_mult * hidden
    hc_per_layer = (
        2 * mix_hc * hc_dim * 4
        + 2 * mix_hc * 4
        + 2 * 3 * 4
    )
    details["mhc_bytes"] = (
        layer_count * hc_per_layer
        + geometry.hc_mult * hc_dim * 4
        + geometry.hc_mult * 4
        + 4
    )
    details["layer_norm_bytes"] = layer_count * 2 * hidden * 2

    target_embedding_head = 2 * geometry.vocab_size * hidden * 2 // tp
    # The compressed DSV4 MTP keeps its own embedding. Its head is compared
    # after load and aliased to the target LM head when equal.
    mtp_embedding_head = (
        geometry.mtp_layers * geometry.vocab_size * hidden * 2 // tp
    )
    details["embedding_and_head_bytes"] = (
        target_embedding_head + mtp_embedding_head
    )
    details["mtp_lm_head_shared"] = 1
    details["mtp_projection_and_norm_bytes"] = geometry.mtp_layers * (
        2 * hidden * hidden * 2 + 4 * hidden * 2
    )

    persistent = deepseek_v4_persistent_buffers(
        max_num_batched_tokens=int(
            c["scheduler"]["max_num_batched_tokens"]
        ),
        hidden_size=hidden,
        hc_mult=geometry.hc_mult,
        index_topk=geometry.index_topk,
        dtype_bytes=2,
        mtp_layers=geometry.mtp_layers,
        max_position_embeddings=int(
            c["model"]["max_position_embeddings"]
        ),
        rope_dim=int(c["model"].get("qk_rope_head_dim") or 64),
    )
    details["mtp_hidden_buffer_bytes"] = (
        persistent.mtp_hidden_buffer_bytes
    )
    details["topk_buffers_bytes"] = persistent.topk_buffers_bytes
    details["mtp_topk_buffer_shared"] = int(
        persistent.mtp_topk_buffer_shared
    )
    details["rope_full_cache_bytes"] = (
        persistent.rope_full_cache_bytes
    )
    details["rope_runtime_buffer_bytes"] = (
        persistent.rope_runtime_buffer_bytes
    )
    details["model_owned_buffer_bytes"] = persistent.total_bytes

    total = sum(
        value
        for key, value in details.items()
        if key not in {
            "mtp_hidden_buffer_bytes",
            "topk_buffers_bytes",
            "rope_full_cache_bytes",
            "rope_runtime_buffer_bytes",
        }
    )
    return WeightEstimate(
        per_rank_bytes=total,
        source="deepseek-v4-w8a8-module-placement",
        total_checkpoint_bytes=300_002_377_534,
        uncertainty_fraction=0.05,
        notes=[
            "W8A8_DYNAMIC tensor placement follows vLLM Ascend v0.23 modules",
            "model-owned Q-dependent buffers are included separately",
            "allocator and undocumented operator buffers are not fitted",
        ],
        details=details,
    )
