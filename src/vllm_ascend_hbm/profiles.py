"""Built-in model profiles and vLLM Ascend compatibility metadata.

The registry intentionally separates two questions:

* ``ascend_status``: whether vLLM Ascend documents the model as runnable;
* ``modeling_level``: how the HBM planner obtains the memory geometry.

A model being runnable does not imply that every workspace/graph term has been
calibrated on a particular firmware and operator stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


OFFICIAL_MODEL_MATRIX = (
    "https://docs.vllm.ai/projects/ascend/en/latest/user_guide/"
    "support_matrix/supported_models.html"
)
MATRIX_VERIFIED_DATE = "2026-07-22"


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    display_name: str
    aliases: tuple[str, ...]
    ascend_status: str
    modeling_level: str
    defaults: dict[str, Any]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AutoModelFamily:
    families: str
    ascend_examples: str
    planner_path: str
    boundary: str


AUTO_MODEL_FAMILIES: tuple[AutoModelFamily, ...] = (
    AutoModelFamily(
        "standard GQA/MHA",
        "Qwen2/2.5/3 Dense, QwQ, Llama, Mistral, Gemma, InternLM, Baichuan, Phi, MiniCPM",
        "profile=auto + config.json -> standard_gqa",
        "Workspace/Graph still need calibration",
    ),
    AutoModelFamily(
        "MLA",
        "DeepSeek V2.5/V3/V3.1/V3.2/R1; GLM/MiniMax/Kimi when config exposes MLA fields",
        "profile=auto + config.json -> mla",
        "verify kv_lora_rank and RoPE cache fields",
    ),
    AutoModelFamily(
        "multimodal language backbone",
        "Qwen-VL, LLaVA, InternVL, Phi-Vision, Gemma multimodal",
        "extract text_config/language_config",
        "vision/audio activation and workspace are not automatic",
    ),
    AutoModelFamily(
        "pooling/embedding/reranker",
        "Qwen3 Embedding/Reranker, BERT, XLM-RoBERTa",
        "kv_cache_strategy=none",
        "use measured activation/workspace",
    ),
    AutoModelFamily(
        "hybrid Attention/SSM",
        "Qwen3-Next, Mamba, RWKV and other hybrid cache models",
        "manual KV adapter",
        "not safe to approximate with ordinary GQA",
    ),
)


def _base(
    *,
    family: str,
    architecture: str,
    total_parameters: int,
    layers: int,
    hidden: int,
    vocab: int,
    heads: int,
    kv_heads: Optional[int],
    head_dim: int,
    max_len: int,
    intermediate: int = 0,
    experts: int = 0,
    experts_per_token: int = 0,
    moe_layers: Optional[int] = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "architecture": architecture,
        "task": "generate",
        "total_parameters": total_parameters,
        "num_hidden_layers": layers,
        "num_cache_layers": layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": max_len,
        "intermediate_size": intermediate,
        "num_routed_experts": experts,
        "num_experts_per_token": experts_per_token,
        "num_moe_layers": layers if moe_layers is None else moe_layers,
        "moe_intermediate_size": intermediate,
        "activation_model": "generic_decoder",
        "kv_cache_strategy": "standard_gqa",
    }


PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        "deepseek-v4-flash",
        "DeepSeek V4-Flash",
        ("deepseek-v4-flash", "dsv4-flash", "deepseek_v4_flash"),
        "core-supported",
        "custom-verified-layout",
        {
            **_base(
                family="deepseek_v4_flash",
                architecture="DeepseekV4ForCausalLM",
                total_parameters=284_000_000_000,
                layers=43,
                hidden=4096,
                vocab=129_280,
                heads=64,
                kv_heads=None,
                head_dim=512,
                max_len=1_048_576,
                intermediate=2048,
                experts=256,
                experts_per_token=6,
            ),
            "kv_cache_strategy": "deepseek_v4_flash",
            "activation_model": "deepseek_v4_flash",
            "query_compression_dim": 1024,
            "num_shared_experts": 1,
            "sliding_window": 128,
            "indexer_heads": 64,
            "indexer_head_dim": 128,
            "attention_topk": 512,
            "output_projection_groups": 8,
            "attention_output_intermediate_dim": 1024,
            "mhc_expansion_factor": 4,
            "mtp_layers": 1,
            "qk_rope_head_dim": 64,
        },
        ("v0.20 uses the dedicated heterogeneous BlockPool adapter.",),
    ),
    ModelProfile(
        "deepseek-v3",
        "DeepSeek V3 / V3.1",
        ("deepseek-v3", "deepseek-v3.1", "deepseek-ai/deepseek-v3"),
        "core-supported",
        "built-in-mla",
        {
            **_base(
                family="deepseek_mla",
                architecture="DeepseekV3ForCausalLM",
                total_parameters=671_000_000_000,
                layers=61,
                hidden=7168,
                vocab=129_280,
                heads=128,
                kv_heads=None,
                head_dim=128,
                max_len=163_840,
                intermediate=2048,
                experts=256,
                experts_per_token=8,
            ),
            "kv_cache_strategy": "mla",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
    ),
    ModelProfile(
        "deepseek-r1",
        "DeepSeek R1",
        ("deepseek-r1", "deepseek-ai/deepseek-r1"),
        "core-supported",
        "built-in-mla",
        {
            **_base(
                family="deepseek_mla",
                architecture="DeepseekV3ForCausalLM",
                total_parameters=671_000_000_000,
                layers=61,
                hidden=7168,
                vocab=129_280,
                heads=128,
                kv_heads=None,
                head_dim=128,
                max_len=131_072,
                intermediate=2048,
                experts=256,
                experts_per_token=8,
            ),
            "kv_cache_strategy": "mla",
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
        },
    ),
    ModelProfile(
        "qwen3-8b",
        "Qwen3-8B",
        ("qwen3-8b", "qwen/qwen3-8b"),
        "core-supported",
        "built-in-gqa",
        _base(
            family="qwen3",
            architecture="Qwen3ForCausalLM",
            total_parameters=8_190_000_000,
            layers=36,
            hidden=4096,
            vocab=151_936,
            heads=32,
            kv_heads=8,
            head_dim=128,
            max_len=131_072,
            intermediate=12_288,
        ),
    ),
    ModelProfile(
        "qwen3-32b",
        "Qwen3-32B",
        ("qwen3-32b", "qwen/qwen3-32b"),
        "core-supported",
        "built-in-gqa",
        _base(
            family="qwen3",
            architecture="Qwen3ForCausalLM",
            total_parameters=32_800_000_000,
            layers=64,
            hidden=5120,
            vocab=151_936,
            heads=64,
            kv_heads=8,
            head_dim=128,
            max_len=131_072,
            intermediate=25_600,
        ),
    ),
    ModelProfile(
        "qwen3-30b-a3b",
        "Qwen3-30B-A3B",
        ("qwen3-30b-a3b", "qwen/qwen3-30b-a3b"),
        "core-supported",
        "built-in-gqa-moe",
        _base(
            family="qwen3_moe",
            architecture="Qwen3MoeForCausalLM",
            total_parameters=30_500_000_000,
            layers=48,
            hidden=2048,
            vocab=151_936,
            heads=32,
            kv_heads=4,
            head_dim=128,
            max_len=131_072,
            intermediate=768,
            experts=128,
            experts_per_token=8,
        ),
    ),
    ModelProfile(
        "qwen2.5-7b",
        "Qwen2.5-7B",
        ("qwen2.5-7b", "qwen/qwen2.5-7b-instruct"),
        "extended-compatible",
        "built-in-gqa",
        _base(
            family="qwen2",
            architecture="Qwen2ForCausalLM",
            total_parameters=7_610_000_000,
            layers=28,
            hidden=3584,
            vocab=152_064,
            heads=28,
            kv_heads=4,
            head_dim=128,
            max_len=131_072,
            intermediate=18_944,
        ),
    ),
    ModelProfile(
        "qwen2.5-72b",
        "Qwen2.5-72B",
        ("qwen2.5-72b", "qwen/qwen2.5-72b-instruct"),
        "extended-compatible",
        "built-in-gqa",
        _base(
            family="qwen2",
            architecture="Qwen2ForCausalLM",
            total_parameters=72_700_000_000,
            layers=80,
            hidden=8192,
            vocab=152_064,
            heads=64,
            kv_heads=8,
            head_dim=128,
            max_len=131_072,
            intermediate=29_568,
        ),
    ),
    ModelProfile(
        "llama-3.1-8b",
        "Llama 3.1 8B",
        ("llama-3.1-8b", "meta-llama/llama-3.1-8b-instruct"),
        "extended-compatible",
        "built-in-gqa",
        _base(
            family="llama",
            architecture="LlamaForCausalLM",
            total_parameters=8_030_000_000,
            layers=32,
            hidden=4096,
            vocab=128_256,
            heads=32,
            kv_heads=8,
            head_dim=128,
            max_len=131_072,
            intermediate=14_336,
        ),
    ),
    ModelProfile(
        "llama-3.1-70b",
        "Llama 3.1 70B",
        ("llama-3.1-70b", "meta-llama/llama-3.1-70b-instruct"),
        "extended-compatible",
        "built-in-gqa",
        _base(
            family="llama",
            architecture="LlamaForCausalLM",
            total_parameters=70_600_000_000,
            layers=80,
            hidden=8192,
            vocab=128_256,
            heads=64,
            kv_heads=8,
            head_dim=128,
            max_len=131_072,
            intermediate=28_672,
        ),
    ),
)


_LOOKUP: dict[str, ModelProfile] = {}
for _profile in PROFILES:
    for _name in (_profile.profile_id, *_profile.aliases):
        _LOOKUP[_name.lower()] = _profile


def get_profile(name: str) -> ModelProfile:
    try:
        return _LOOKUP[name.lower()]
    except KeyError as exc:
        known = ", ".join(profile.profile_id for profile in PROFILES)
        raise ValueError(
            f"unknown model.profile={name!r}; built-ins: {known}; "
            "use profile='auto' with model.config_path for another model"
        ) from exc


def list_profiles() -> tuple[ModelProfile, ...]:
    return PROFILES


def list_auto_families() -> tuple[AutoModelFamily, ...]:
    return AUTO_MODEL_FAMILIES
