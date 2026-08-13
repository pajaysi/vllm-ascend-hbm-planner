"""Validation of predicted startup limits against observed intervals."""

from __future__ import annotations

import copy
from dataclasses import asdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Optional, Tuple

from .startup import evaluate_startup


@dataclass(frozen=True)
class BoundaryValidationRow:
    max_model_len: int
    dp_size: int
    tp_size: int
    max_num_seqs: int
    observed_max_success_q: int
    observed_first_fail_q: int
    predicted_max_success_q: Optional[int]
    falls_in_observed_interval: bool
    distance_from_success_q: Optional[int]
    distance_to_first_fail_q: Optional[int]
    limiting_stage: Optional[str]
    implied_unmodeled_lower_bound_bytes: int
    implied_unmodeled_upper_bound_bytes: int


@dataclass(frozen=True)
class BoundaryValidationReport:
    calibration_used: bool
    rows: tuple[BoundaryValidationRow, ...]

    @property
    def matched_rows(self) -> int:
        return sum(row.falls_in_observed_interval for row in self.rows)

    def as_dict(self) -> dict[str, Any]:
        distances: list[int] = []
        for row in self.rows:
            predicted = row.predicted_max_success_q
            if predicted is None:
                continue
            if predicted < row.observed_max_success_q:
                distances.append(row.observed_max_success_q - predicted)
            elif predicted >= row.observed_first_fail_q:
                distances.append(predicted - row.observed_first_fail_q)
            else:
                distances.append(0)
        return {
            "operation": "validate_startup_boundaries",
            "calibration_used": self.calibration_used,
            "summary": {
                "total_rows": len(self.rows),
                "matched_rows": self.matched_rows,
                "match_rate": (
                    self.matched_rows / len(self.rows)
                    if self.rows
                    else 0.0
                ),
                "mean_absolute_distance_to_interval_tokens": (
                    mean(distances) if distances else None
                ),
                "max_absolute_distance_to_interval_tokens": (
                    max(distances) if distances else None
                ),
            },
            "rows": [asdict(row) for row in self.rows],
        }


def _maximum_feasible_q(
    config: dict[str, Any],
    *,
    max_num_seqs: int,
    initial_high: int,
) -> Tuple[Optional[int], Optional[str]]:
    low = 1
    high = max(2, initial_high)
    while evaluate_startup(config, high, max_num_seqs).startup_feasible:
        low = high
        high *= 2
        if high > 2_097_152:
            return low, "search_limit"

    best: Optional[int] = None
    while low <= high:
        middle = (low + high) // 2
        estimate = evaluate_startup(config, middle, max_num_seqs)
        if estimate.startup_feasible:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    failure = evaluate_startup(
        config,
        1 if best is None else best + 1,
        max_num_seqs,
    )
    return best, failure.limiting_stage


def validate_boundaries(
    c: dict[str, Any],
    rows: Iterable[dict[str, Any]],
) -> BoundaryValidationReport:
    results: list[BoundaryValidationRow] = []
    for raw in rows:
        pair = copy.deepcopy(c)
        max_len = int(raw["max_model_len"])
        dp = int(raw["dp_size"])
        tp = int(raw["tp_size"])
        seqs = int(raw.get("max_num_seqs", 64))
        success = int(raw["max_success_mnbt"])
        first_fail = int(raw["first_fail_mnbt"])
        topology_profiles = c.get("validation", {}).get(
            "profile_calibration_by_tp", {}
        )
        topology_profile = topology_profiles.get(str(tp), {})
        for key in (
            "profiled_max_num_batched_tokens",
            "weight_gib_per_rank",
            "peak_activation_gib_per_rank",
            "non_torch_gib_per_rank",
            "graph_gib_per_rank",
        ):
            if key in topology_profile:
                pair["profile_calibration"][key] = topology_profile[key]
        for key in (
            "visible_hbm_gib_per_die",
            "startup_free_hbm_gib_per_die",
        ):
            if key in topology_profile:
                pair["platform"][key] = topology_profile[key]
        pair["scheduler"].update(
            {
                "max_model_len": max_len,
                "max_num_seqs": seqs,
            }
        )
        pair["parallelism"].update(
            {
                "dp_size": dp,
                "tp_size": tp,
                "ep_size": dp * tp,
            }
        )
        predicted, stage = _maximum_feasible_q(
            pair,
            max_num_seqs=seqs,
            initial_high=first_fail * 2,
        )
        success_estimate = evaluate_startup(pair, success, seqs)
        failure_estimate = evaluate_startup(pair, first_fail, seqs)
        # With the deterministic components fixed, the observed success says
        # that any omitted memory is at most the theoretical headroom at that
        # point.  The observed failure says it is greater than the headroom at
        # the first failing point.  This is an interval, not a fitted value.
        implied_lower = (
            failure_estimate.available_kv_bytes
            - failure_estimate.minimum_kv_bytes
        )
        implied_upper = (
            success_estimate.available_kv_bytes
            - success_estimate.minimum_kv_bytes
        )
        in_interval = (
            predicted is not None
            and success <= predicted < first_fail
        )
        results.append(
            BoundaryValidationRow(
                max_model_len=max_len,
                dp_size=dp,
                tp_size=tp,
                max_num_seqs=seqs,
                observed_max_success_q=success,
                observed_first_fail_q=first_fail,
                predicted_max_success_q=predicted,
                falls_in_observed_interval=in_interval,
                distance_from_success_q=(
                    None if predicted is None else predicted - success
                ),
                distance_to_first_fail_q=(
                    None if predicted is None else first_fail - predicted
                ),
                limiting_stage=stage,
                implied_unmodeled_lower_bound_bytes=implied_lower,
                implied_unmodeled_upper_bound_bytes=implied_upper,
            )
        )
    return BoundaryValidationReport(
        calibration_used=bool(
            c.get("validation", {}).get(
                "profile_calibration_by_tp", {}
            )
            or
            c.get("runtime", {}).get("manual_non_torch_gib_per_rank")
            is not None
            or float(
                c.get("runtime", {}).get(
                    "base_persistent_gib_per_rank", 0.0
                )
            )
            > 0.0
            or float(
                c.get("runtime", {}).get(
                    "hccl_and_cann_persistent_gib_per_rank", 0.0
                )
            )
            > 0.0
            or c.get("profile_calibration", {}).get(
                "non_torch_gib_per_rank"
            )
            is not None
        ),
        rows=tuple(results),
    )
