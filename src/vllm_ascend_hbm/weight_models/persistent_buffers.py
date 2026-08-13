"""Persistent buffers created while constructing inference models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepSeekV4PersistentBuffers:
    mtp_hidden_buffer_bytes: int
    topk_buffers_bytes: int
    mtp_topk_buffer_shared: bool
    rope_full_cache_bytes: int
    rope_runtime_buffer_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.mtp_hidden_buffer_bytes
            + self.topk_buffers_bytes
            + self.rope_full_cache_bytes
            + self.rope_runtime_buffer_bytes
        )


def deepseek_v4_persistent_buffers(
    *,
    max_num_batched_tokens: int,
    hidden_size: int,
    hc_mult: int,
    index_topk: int,
    dtype_bytes: int,
    mtp_layers: int,
    max_position_embeddings: int,
    rope_dim: int,
) -> DeepSeekV4PersistentBuffers:
    mtp_hidden = 0
    topk = 0
    topk_shared = False
    if mtp_layers > 0:
        mtp_hidden = (
            max_num_batched_tokens
            * hc_mult
            * hidden_size
            * dtype_bytes
        )
        # Both models construct a buffer, but AscendSpecDecodeBaseProposer
        # replaces the draft reference with the target buffer and releases the
        # duplicate before DeviceMemoryProfiler records model_memory_usage.
        topk = max_num_batched_tokens * index_topk * 4
        topk_shared = True

    # ComplexExpRotaryEmbedding maintains two source-keyed FP32 cos/sin
    # tables: ordinary RoPE (theta=10000) and compressed RoPE
    # (compress_rope_theta=160000). C4/C128 layers share the latter.
    rope_full = 2 * max_position_embeddings * rope_dim * 4 * 2
    # Runtime buffers are keyed by (config, group): one ordinary "default"
    # group plus compressed "default", "c4", and "c128".
    rope_runtime = (
        4 * max_num_batched_tokens * rope_dim * 4 * 2
    )
    return DeepSeekV4PersistentBuffers(
        mtp_hidden,
        topk,
        topk_shared,
        rope_full,
        rope_runtime,
    )
