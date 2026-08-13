"""Parse stable vLLM Ascend startup-capacity log fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ParsedStartupLog:
    startup_free_gib: Optional[float] = None
    visible_hbm_gib: Optional[float] = None
    gpu_memory_utilization: Optional[float] = None
    requested_memory_gib: Optional[float] = None
    weight_gib: Optional[float] = None
    activation_gib: Optional[float] = None
    non_torch_gib: Optional[float] = None
    graph_gib: Optional[float] = None
    current_kv_gib: Optional[float] = None
    required_kv_gib: Optional[float] = None
    available_kv_gib: Optional[float] = None
    estimated_max_model_len: Optional[int] = None
    graph_capture_finished: bool = False


def _float_match(
    pattern: str, text: str, group: int = 1
) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return None if match is None else float(match.group(group))


def parse_startup_log(text: Optional[str]) -> ParsedStartupLog:
    if not text:
        return ParsedStartupLog()
    free_match = re.search(
        r"Free memory on device\s*\(\s*([\d.]+)\s*/\s*([\d.]+)\s*GiB",
        text,
        re.IGNORECASE,
    )
    desired_match = re.search(
        r"Desired GPU memory utilization is\s*\(\s*([\d.]+)\s*,\s*"
        r"([\d.]+)\s*GiB",
        text,
        re.IGNORECASE,
    )
    usage_match = re.search(
        r"Actual usage:\s*([\d.]+)\s*GiB for weights,\s*"
        r"([\d.]+)\s*GiB for peak activation,\s*"
        r"([\d.]+)\s*GiB for non-torch memory,\s*"
        r"([\d.]+)\s*GiB for NPU graph memory",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    required_available = re.search(
        r"\(([\d.]+)\s*GiB KV cache is needed.*?"
        r"available KV cache memory\s*\(([\d.]+)\s*GiB\)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    max_len_match = re.search(
        r"estimated maximum model length is\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    return ParsedStartupLog(
        startup_free_gib=(
            None if free_match is None else float(free_match.group(1))
        ),
        visible_hbm_gib=(
            None if free_match is None else float(free_match.group(2))
        ),
        gpu_memory_utilization=(
            None if desired_match is None else float(desired_match.group(1))
        ),
        requested_memory_gib=(
            None if desired_match is None else float(desired_match.group(2))
        ),
        weight_gib=None if usage_match is None else float(usage_match.group(1)),
        activation_gib=(
            None if usage_match is None else float(usage_match.group(2))
        ),
        non_torch_gib=(
            None if usage_match is None else float(usage_match.group(3))
        ),
        graph_gib=None if usage_match is None else float(usage_match.group(4)),
        current_kv_gib=_float_match(
            r"Current KV cache memory:\s*([\d.]+)\s*GiB",
            text,
        ),
        required_kv_gib=(
            None
            if required_available is None
            else float(required_available.group(1))
        ),
        available_kv_gib=(
            None
            if required_available is None
            else float(required_available.group(2))
        ),
        estimated_max_model_len=(
            None if max_len_match is None else int(max_len_match.group(1))
        ),
        graph_capture_finished=bool(
            re.search(r"Graph capturing finished", text, re.IGNORECASE)
        ),
    )
