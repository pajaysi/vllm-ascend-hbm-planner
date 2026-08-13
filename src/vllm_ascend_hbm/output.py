"""Human, JSON and CSV output adapters."""

from __future__ import annotations

import csv
import json
import sys
from typing import Any

from .utils import gib


def print_estimate_text(result: dict[str, Any]) -> None:
    c, model = result["config"], result["model"]
    p, s, par = c["platform"], c["scheduler"], c["parallelism"]
    print(f"{model.get('name') or model.get('profile')} HBM estimate (per logical NPU/die)")
    print(
        f"KV={model['kv_cache_strategy']}, device={p['device']}, "
        f"vLLM Ascend={p['vllm_ascend_version']}, Q={s['max_num_batched_tokens']:,}, "
        f"L={s['max_model_len']:,}"
    )
    print(
        f"parallel=DP{par['dp_size']}/TP{par['tp_size']}/PP{par['pp_size']}/"
        f"EP{par['ep_size']}/PCP{par['pcp_size']}/DCP{par['dcp_size']}; "
        f"budget={gib(result['estimates'][0]['requested_hbm_budget_bytes']):.3f} GiB/rank"
    )
    print(
        f"weight={gib(result['weight_estimate']['per_rank_bytes']):.3f} GiB "
        f"({result['weight_estimate']['source']})"
    )
    startup = result.get("startup_estimate")
    if startup is not None:
        print(
            "startup: "
            f"available KV={gib(startup['available_kv_bytes']):.2f} GiB, "
            f"required minimum KV={gib(startup['minimum_kv_bytes']):.2f} GiB, "
            f"result={'PASS' if startup['startup_feasible'] else 'FAIL'}, "
            f"stage={startup['limiting_stage'] or 'complete'}"
        )
    print()
    header = (
        f"{'C':>4} {'C/DP':>5} {'Q/rank':>9} {'weight':>8} {'KVphys':>8} "
        f"{'KVplan':>8} {'act':>8} {'ws':>7} {'runtime':>8} {'plan':>8} {'upper':>8} {'fit':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in result["estimates"]:
        print(
            f"{row['concurrency_input']:4d} {row['max_local_concurrency']:5d} "
            f"{row['q_tokens_per_rank']:9,d} {gib(row['weights_bytes']):8.2f} "
            f"{gib(row['kv_pool_tensor_bytes']):8.2f} {gib(row['kv_planner_bytes']):8.2f} "
            f"{gib(row['activation_bytes']):8.2f} {gib(row['operator_workspace_bytes']):7.2f} "
            f"{gib(row['runtime_bytes']):8.2f} {gib(row['planning_total_bytes']):8.2f} "
            f"{gib(row['upper_bound_bytes']):8.2f} "
            f"{('yes' if row['fits_requested_budget'] else 'NO'):>4}"
        )
    first = result["estimates"][0]
    if first["unresolved_components"]:
        print()
        print("WARNING unresolved: " + ", ".join(first["unresolved_components"]))


def print_recommendation_text(result: dict[str, Any]) -> None:
    c, method, model = result["config"], result["method"], result["model"]
    par = c["parallelism"]
    print(f"{model.get('name') or model.get('profile')} vLLM Ascend parameter recommendation")
    print(
        f"KV={model['kv_cache_strategy']}; budget={gib(method['requested_hbm_budget_bytes']):.3f} GiB/rank; "
        f"parallel=DP{par['dp_size']}/TP{par['tp_size']}/PP{par['pp_size']}/EP{par['ep_size']}"
    )
    print(
        f"objective={method['objective']}, fit_basis={method['fit_basis']}, "
        f"minimum headroom={gib(method['minimum_headroom_bytes']):.2f} GiB"
    )
    startup = result.get("startup_limit_recommended")
    runtime = result.get(
        "runtime_safe_recommended",
        result["single_service_recommended"],
    )
    print()
    if startup is None:
        print("Startup-limit recommendation: no candidate can start.")
    else:
        print(
            "Startup-limit recommendation: "
            f"--max-num-batched-tokens {startup['max_num_batched_tokens']} "
            f"--max-num-seqs {startup['max_num_seqs']}; "
            f"KV headroom={gib(startup['startup_headroom_bytes']):.2f} GiB"
        )
    if runtime is None:
        print("Runtime-safe recommendation: no pair satisfies every scenario.")
    else:
        print(
            "Runtime-safe recommendation: "
            f"--max-num-batched-tokens {runtime['max_num_batched_tokens']} "
            f"--max-num-seqs {runtime['max_num_seqs']}"
        )
        print(
            f"  limiting={runtime['limiting_scenario']}; "
            f"minimum fit headroom={gib(runtime['minimum_fit_headroom_bytes']):.2f} GiB; "
            f"score={runtime['score']:.1f}"
        )
    for scenario in result["scenarios"]:
        print()
        print(f"Scenario {scenario['name']}: context_len={scenario['context_len']:,}")
        recommended = scenario["recommended"]
        if recommended is None:
            print("  no feasible candidate")
        else:
            print(
                f"  RECOMMENDED Q={recommended['max_num_batched_tokens']:,}, "
                f"max_num_seqs={recommended['max_num_seqs']}; "
                f"planning={gib(recommended['planning_total_bytes']):.2f} GiB; "
                f"upper={gib(recommended['planning_upper_bytes']):.2f} GiB; "
                f"headroom={gib(recommended['fit_headroom_bytes']):.2f} GiB"
            )
        print(f"  {'max_num_seqs':>12} {'max feasible Q':>15} {'upper':>10} {'headroom':>10}")
        for row in scenario["frontier_by_max_num_seqs"]:
            if row["max_feasible_num_batched_tokens"] is None:
                print(f"  {row['max_num_seqs']:12d} {'none':>15}")
            else:
                print(
                    f"  {row['max_num_seqs']:12d} {row['max_feasible_num_batched_tokens']:15,d} "
                    f"{gib(row['planning_upper_bytes']):10.2f} {gib(row['fit_headroom_bytes']):10.2f}"
                )


def print_csv(result: dict[str, Any]) -> None:
    if result["operation"] == "validate_startup_boundaries":
        columns = [
            "max_model_len", "dp_size", "tp_size", "max_num_seqs",
            "observed_max_success_q", "observed_first_fail_q",
            "predicted_max_success_q", "falls_in_observed_interval",
            "distance_from_success_q", "distance_to_first_fail_q",
            "limiting_stage", "implied_unmodeled_lower_bound_gib",
            "implied_unmodeled_upper_bound_gib",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        for row in result["rows"]:
            exported = dict(row)
            exported["implied_unmodeled_lower_bound_gib"] = gib(
                exported.pop("implied_unmodeled_lower_bound_bytes")
            )
            exported["implied_unmodeled_upper_bound_gib"] = gib(
                exported.pop("implied_unmodeled_upper_bound_bytes")
            )
            writer.writerow(exported)
        return
    if result["operation"] == "recommend":
        columns = [
            "scenario", "context_len", "max_num_batched_tokens", "max_num_seqs",
            "score", "feasible", "is_recommended", "planning_total_gib",
            "planning_upper_gib", "fit_headroom_gib", "failure_reasons",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=columns)
        writer.writeheader()
        for scenario in result["scenarios"]:
            for candidate in scenario["candidates"]:
                writer.writerow({
                    "scenario": scenario["name"],
                    "context_len": scenario["context_len"],
                    "max_num_batched_tokens": candidate["max_num_batched_tokens"],
                    "max_num_seqs": candidate["max_num_seqs"],
                    "score": candidate["score"],
                    "feasible": candidate["feasible"],
                    "is_recommended": candidate is scenario["recommended"],
                    "planning_total_gib": gib(candidate["planning_total_bytes"]) if "planning_total_bytes" in candidate else None,
                    "planning_upper_gib": gib(candidate["planning_upper_bytes"]) if "planning_upper_bytes" in candidate else None,
                    "fit_headroom_gib": gib(candidate["fit_headroom_bytes"]) if "fit_headroom_bytes" in candidate else None,
                    "failure_reasons": "; ".join(candidate["failure_reasons"]),
                })
        return
    rows = result["estimates"]
    columns = [
        "concurrency_input", "max_local_concurrency", "q_tokens_per_rank", "weights_gib",
        "kv_pool_tensor_gib", "kv_planner_gib", "activation_gib", "operator_workspace_gib",
        "graph_cache_gib", "runtime_gib", "planning_total_gib", "upper_bound_gib",
        "planning_headroom_gib", "fits_requested_budget",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "concurrency_input": row["concurrency_input"],
            "max_local_concurrency": row["max_local_concurrency"],
            "q_tokens_per_rank": row["q_tokens_per_rank"],
            "weights_gib": gib(row["weights_bytes"]),
            "kv_pool_tensor_gib": gib(row["kv_pool_tensor_bytes"]),
            "kv_planner_gib": gib(row["kv_planner_bytes"]),
            "activation_gib": gib(row["activation_bytes"]),
            "operator_workspace_gib": gib(row["operator_workspace_bytes"]),
            "graph_cache_gib": gib(row["graph_cache_bytes"]),
            "runtime_gib": gib(row["runtime_bytes"]),
            "planning_total_gib": gib(row["planning_total_bytes"]),
            "upper_bound_gib": gib(row["upper_bound_bytes"]),
            "planning_headroom_gib": gib(row["planning_headroom_bytes"]),
            "fits_requested_budget": row["fits_requested_budget"],
        })


def print_result(result: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif output_format == "csv":
        print_csv(result)
    elif result["operation"] == "validate_startup_boundaries":
        summary = result["summary"]
        print("DeepSeek-V4 startup-boundary validation")
        print(
            f"matched={summary['matched_rows']}/{summary['total_rows']}; "
            "mean distance to observed interval="
            f"{summary['mean_absolute_distance_to_interval_tokens']:.1f} tokens"
        )
        print()
        header = (
            f"{'L':>9} {'DP':>3} {'TP':>3} {'success':>9} {'fail':>9} "
            f"{'predicted':>10} {'match':>5} {'missing GiB interval':>23} "
            f"{'stage':>18}"
        )
        print(header)
        print("-" * len(header))
        for row in result["rows"]:
            print(
                f"{row['max_model_len']:9,d} {row['dp_size']:3d} "
                f"{row['tp_size']:3d} "
                f"{row['observed_max_success_q']:9,d} "
                f"{row['observed_first_fail_q']:9,d} "
                f"{row['predicted_max_success_q']:10,d} "
                f"{('yes' if row['falls_in_observed_interval'] else 'NO'):>5} "
                f"{gib(row['implied_unmodeled_lower_bound_bytes']):8.2f}.."
                f"{gib(row['implied_unmodeled_upper_bound_bytes']):8.2f} "
                f"{row['limiting_stage'] or 'complete':>18}"
            )
    elif result["operation"] == "recommend":
        print_recommendation_text(result)
    else:
        print_estimate_text(result)
