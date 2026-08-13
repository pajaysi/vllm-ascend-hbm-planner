"""Source-equivalent DeepSeek-V4 KV admission model for vLLM 0.23.

This module models the minimum cache capacity check performed before the
physical KV tensors are created.  It intentionally does not reuse the physical
pool allocation tuple count: those are separate source paths in vLLM Ascend.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils import ceildiv


@dataclass(frozen=True)
class DeepSeekV4V023KVLayout:
    logical_block_size: int
    c4_state_block_size: int
    c128_state_block_size: int
    small_page_bytes: int
    large_page_bytes: int
    minimum_admission_tuple_count: int = 22
    sliding_window: int = 128
    c4_ratio: int = 4
    c128_ratio: int = 128

    @property
    def bytes_per_tuple(self) -> int:
        return self.small_page_bytes + self.large_page_bytes


@dataclass(frozen=True)
class KVAdmissionEstimate:
    max_model_len: int
    max_num_batched_tokens: int
    block_size: int
    tuple_count: int
    bytes_per_tuple: int
    c4_history_pages: int
    c128_history_pages: int
    swa_pages_per_group: int
    swa_total_pages: int
    c4_state_pages: int
    c128_state_pages: int
    total_pages: int
    total_bytes: int


def layout_for_block_size(block_size: int) -> DeepSeekV4V023KVLayout:
    layouts = {
        128: DeepSeekV4V023KVLayout(128, 8, 32, 16_640, 131_072),
        64: DeepSeekV4V023KVLayout(64, 4, 16, 8_320, 65_536),
        32: DeepSeekV4V023KVLayout(32, 2, 8, 4_160, 32_768),
    }
    try:
        return layouts[block_size]
    except KeyError as exc:
        raise ValueError(
            "DeepSeek-V4 v0.23 block_size must be 32, 64, or 128"
        ) from exc


def minimum_kv_admission(
    max_model_len: int,
    max_num_batched_tokens: int,
    block_size: int = 128,
) -> KVAdmissionEstimate:
    """Return the v0.23 minimum KV capacity for one maximum-length request."""

    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")

    layout = layout_for_block_size(block_size)
    c4_history = ceildiv(
        max_model_len,
        layout.logical_block_size * layout.c4_ratio,
    )
    c128_history = ceildiv(
        max_model_len,
        layout.logical_block_size * layout.c128_ratio,
    )
    swa_tokens = min(
        layout.sliding_window - 1 + max_num_batched_tokens,
        max_model_len,
    )
    swa_per_group = ceildiv(swa_tokens, layout.logical_block_size) + 1
    c4_state_tokens = min(
        layout.c4_state_block_size - 1 + max_num_batched_tokens,
        max_model_len,
    )
    c4_state = ceildiv(c4_state_tokens, layout.c4_state_block_size) + 1
    c128_state_tokens = min(
        layout.sliding_window - 1 + max_num_batched_tokens,
        max_model_len,
    )
    c128_state = (
        ceildiv(c128_state_tokens, layout.c128_state_block_size) + 1
    )
    swa_total = 2 * swa_per_group
    total_pages = (
        c4_history
        + c128_history
        + swa_total
        + c4_state
        + c128_state
    )
    total_bytes = (
        total_pages
        * layout.minimum_admission_tuple_count
        * layout.bytes_per_tuple
    )
    return KVAdmissionEstimate(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        block_size=block_size,
        tuple_count=layout.minimum_admission_tuple_count,
        bytes_per_tuple=layout.bytes_per_tuple,
        c4_history_pages=c4_history,
        c128_history_pages=c128_history,
        swa_pages_per_group=swa_per_group,
        swa_total_pages=swa_total,
        c4_state_pages=c4_state,
        c128_state_pages=c128_state,
        total_pages=total_pages,
        total_bytes=total_bytes,
    )
