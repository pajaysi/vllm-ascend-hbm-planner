"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config, validate_config
from .engine import calculate
from .hf_config import enrich_model_from_hf
from .output import print_result
from .profiles import (
    MATRIX_VERIFIED_DATE,
    OFFICIAL_MODEL_MATRIX,
    list_auto_families,
    list_profiles,
)
from .recommender import recommend
from .validation import validate_boundaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable HBM estimator and vLLM Ascend Q/max_num_seqs recommender"
    )
    parser.add_argument("--config", help="JSON configuration")
    parser.add_argument("--operation", choices=("estimate", "recommend"))
    parser.add_argument("--format", choices=("text", "json", "csv"))
    parser.add_argument("--model-path", help="local model directory/safetensors for exact weight bytes")
    parser.add_argument("--model-config", help="local Hugging Face config.json or model directory")
    parser.add_argument("--profile-log", help="vLLM startup/profile log")
    parser.add_argument(
        "--validate-boundaries",
        help="JSON array of observed startup success/failure boundaries",
    )
    parser.add_argument("--list-models", action="store_true", help="show built-in model profiles")
    return parser


def _print_models() -> None:
    print("Built-in HBM model profiles")
    print(f"Official vLLM Ascend matrix checked: {MATRIX_VERIFIED_DATE}")
    print(OFFICIAL_MODEL_MATRIX)
    print()
    print(f"{'profile':<22} {'Ascend status':<20} {'HBM modeling level':<24} model")
    print("-" * 92)
    for profile in list_profiles():
        print(
            f"{profile.profile_id:<22} {profile.ascend_status:<20} "
            f"{profile.modeling_level:<24} {profile.display_name}"
        )
    print()
    print("Additional vLLM Ascend families")
    print(f"{'family':<30} {'planner path':<42} examples/boundary")
    print("-" * 118)
    for family in list_auto_families():
        print(
            f"{family.families:<30} {family.planner_path:<42} "
            f"{family.ascend_examples}; {family.boundary}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list_models:
        _print_models()
        return 0
    try:
        c = load_config(args.config)
        if args.operation:
            c["operation"] = args.operation
        if args.format:
            c["output"]["format"] = args.format
        if args.model_path:
            c["model"]["model_path"] = args.model_path
        if args.model_config:
            c["model"]["config_path"] = args.model_config
            enrich_model_from_hf(c["model"], args.model_config)
        if args.profile_log:
            c["profile_calibration"]["vllm_log_path"] = args.profile_log
        validate_config(c)
        if args.validate_boundaries:
            try:
                observed = json.loads(
                    Path(args.validate_boundaries).read_text(
                        encoding="utf-8"
                    )
                )
            except OSError as exc:
                raise ValueError(
                    f"cannot read boundary data {args.validate_boundaries!r}: "
                    f"{exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid boundary JSON: "
                    f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
                ) from exc
            if isinstance(observed, dict):
                observed = observed.get("rows")
            if not isinstance(observed, list):
                raise ValueError(
                    "boundary JSON must be an array or an object with rows"
                )
            result = validate_boundaries(c, observed).as_dict()
        else:
            result = (
                recommend(c)
                if c["operation"] == "recommend"
                else calculate(c)
            )
    except ValueError as exc:
        parser.error(str(exc))
    print_result(result, c["output"]["format"])
    return 0
