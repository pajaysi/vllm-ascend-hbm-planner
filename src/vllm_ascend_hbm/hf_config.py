"""Offline Hugging Face ``config.json`` adapter.

No network request is made.  Users point at a downloaded model directory or a
standalone config JSON.  Multimodal configs are reduced to their text backbone
for KV/activation geometry; checkpoint parsing still sees all weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import set_if_missing


def _config_file(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise ValueError(f"model config does not exist: {path}")
    return path


def read_hf_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = _config_file(path_text)
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse Hugging Face config {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise ValueError(f"Hugging Face config root must be an object: {path}")
    return root, path


def _text_backbone(root: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "llm_config"):
        value = root.get(key)
        if isinstance(value, dict):
            merged = dict(value)
            merged.setdefault("_multimodal_parent_model_type", root.get("model_type"))
            return merged
    return root


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def enrich_model_from_hf(model: dict[str, Any], path_text: str) -> dict[str, Any]:
    root, path = read_hf_config(path_text)
    cfg = _text_backbone(root)
    architectures = cfg.get("architectures") or root.get("architectures") or []
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None

    set_if_missing(model, "name", root.get("name_or_path") or path.parent.name)
    set_if_missing(model, "architecture", architecture)
    set_if_missing(model, "family", cfg.get("model_type") or root.get("model_type"))
    set_if_missing(model, "total_parameters", _first(root, "num_parameters", "parameter_count"))
    set_if_missing(model, "num_hidden_layers", _first(cfg, "num_hidden_layers", "n_layer", "num_layers"))
    set_if_missing(model, "num_cache_layers", model.get("num_hidden_layers"))
    set_if_missing(model, "hidden_size", _first(cfg, "hidden_size", "n_embd", "d_model"))
    set_if_missing(model, "vocab_size", cfg.get("vocab_size"))
    set_if_missing(model, "num_attention_heads", _first(cfg, "num_attention_heads", "n_head"))
    set_if_missing(
        model,
        "num_key_value_heads",
        _first(cfg, "num_key_value_heads", "num_kv_heads", "multi_query_group_num"),
    )
    set_if_missing(model, "head_dim", _first(cfg, "head_dim", "attention_head_dim"))
    if model.get("head_dim") is None and model.get("hidden_size") and model.get("num_attention_heads"):
        model["head_dim"] = int(model["hidden_size"]) // int(model["num_attention_heads"])
    if model.get("num_key_value_heads") is None and model.get("num_attention_heads"):
        model["num_key_value_heads"] = int(model["num_attention_heads"])

    set_if_missing(model, "max_position_embeddings", _first(cfg, "max_position_embeddings", "seq_length", "model_max_length"))
    set_if_missing(model, "intermediate_size", _first(cfg, "intermediate_size", "ffn_hidden_size"))
    set_if_missing(model, "num_routed_experts", _first(cfg, "n_routed_experts", "num_experts", "num_local_experts"))
    set_if_missing(model, "num_experts_per_token", _first(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_top_k"))
    set_if_missing(model, "moe_intermediate_size", _first(cfg, "moe_intermediate_size", "intermediate_size"))
    set_if_missing(model, "num_moe_layers", model.get("num_hidden_layers"))
    set_if_missing(model, "kv_lora_rank", cfg.get("kv_lora_rank"))
    set_if_missing(model, "qk_rope_head_dim", cfg.get("qk_rope_head_dim"))

    architecture_low = str(model.get("architecture") or "").lower()
    model_type_low = str(cfg.get("model_type") or "").lower()
    pooling = (
        any(token in architecture_low for token in ("embedding", "reranker", "sequenceclassification", "reward"))
        or model_type_low in {"bert", "xlm-roberta", "roberta"}
    )
    if pooling and model.get("task") in (None, "generate"):
        model["task"] = "pooling"

    strategy = model.get("kv_cache_strategy")
    if strategy in (None, "auto"):
        if pooling:
            model["kv_cache_strategy"] = "none"
        elif model.get("kv_lora_rank") is not None:
            model["kv_cache_strategy"] = "mla"
        elif any(token in model_type_low for token in ("mamba", "qwen3_next", "rwkv")):
            model["kv_cache_strategy"] = "manual"
        else:
            model["kv_cache_strategy"] = "standard_gqa"
    if model.get("activation_model") in (None, "auto"):
        model["activation_model"] = "generic_decoder"
    model["resolved_config_path"] = str(path)
    model["multimodal_text_backbone_only"] = cfg is not root
    return model
