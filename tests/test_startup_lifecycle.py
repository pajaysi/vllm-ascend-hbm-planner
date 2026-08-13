from __future__ import annotations

import unittest
from pathlib import Path

from vllm_ascend_hbm.config import load_config
from vllm_ascend_hbm.components import estimate_activation
from vllm_ascend_hbm.engine import calculate
from vllm_ascend_hbm.recommender import recommend
from vllm_ascend_hbm.startup import evaluate_startup


ROOT = Path(__file__).resolve().parents[1]

SUCCESS_LOG = """
(Worker_DP2_TP0_EP4 pid=473166) INFO [gpu_model_runner.py:6585]
Graph capturing finished in 24 secs, took 1.97 GiB
(Worker_DP2_TP0_EP4 pid=473166) INFO [worker.py:852]
Free memory on device (60.89/61.27 GiB) on startup.
Desired GPU memory utilization is (0.9, 55.14 GiB).
Actual usage: 27.17 GiB for weights, 7.27 GiB for peak activation,
3.08 GiB for non-torch memory, 1.97 GiB for NPU graph memory.
Current KV cache memory: 17.63 GiB.
"""

FAIL_LOG = """
(EngineCore_DP0 pid=1496252) ERROR [core.py:1195]
ValueError: To serve at least one request with the model's max seq len
(32768), (17.26 GiB KV cache is needed, which is larger than the
available KV cache memory (17.2 GiB). Based on the available memory,
the estimated maximum model length is 32648.
"""

SUCCESS_LOG_NO_GRAPH = """
Free memory on device (60.89/61.27 GiB) on startup.
Desired GPU memory utilization is (0.9, 55.14 GiB).
Actual usage: 27.17 GiB for weights, 7.27 GiB for peak activation,
3.08 GiB for non-torch memory, 0.00 GiB for NPU graph memory.
Current KV cache memory: 17.63 GiB.
"""


def base_config(q: int) -> dict:
    value = load_config(str(ROOT / "configs/deepseek_v4_flash_910c.json"))
    value["platform"].update(
        {
            "vllm_ascend_version": "0.23.0rc1",
            "hbm_gib_per_die": 64.0,
            "nominal_hbm_gib_per_die": 64.0,
            "visible_hbm_gib_per_die": 61.27,
            "startup_free_hbm_gib_per_die": 60.89,
            "gpu_memory_utilization": 0.9,
        }
    )
    value["scheduler"].update(
        {
            "max_model_len": 32_768,
            "max_num_batched_tokens": q,
            "max_num_seqs": 64,
        }
    )
    value["parallelism"].update(
        {
            "dp_size": 8,
            "tp_size": 2,
            "ep_size": 16,
        }
    )
    value.setdefault("vllm_ascend", {})[
        "enable_shared_expert_dp"
    ] = True
    return value


class StartupLifecycleTests(unittest.TestCase):
    def test_estimate_exposes_startup_and_runtime_as_separate_outputs(self) -> None:
        value = base_config(45_056)
        value["operation"] = "estimate"
        value["workload"]["concurrency"] = [1]

        result = calculate(value)

        self.assertIn("startup_estimate", result)
        self.assertIn("estimates", result)
        self.assertEqual(
            result["startup_estimate"]["minimum_kv_bytes"],
            evaluate_startup(value, 45_056, 64).minimum_kv_bytes,
        )

    def test_recommendation_exposes_startup_limit_and_runtime_safe_pair(self) -> None:
        value = base_config(45_056)
        value["operation"] = "recommend"
        value["recommendation"]["candidate_max_num_batched_tokens"] = [
            16_384,
            32_768,
            45_056,
        ]
        value["recommendation"]["candidate_max_num_seqs"] = [1, 64]
        value["recommendation"]["scenarios"] = [
            {"name": "32k", "context_len": 32_768}
        ]

        result = recommend(value)

        self.assertIn("startup_limit_recommended", result)
        self.assertIn("runtime_safe_recommended", result)
        self.assertAlmostEqual(
            result["method"]["requested_hbm_budget_bytes"] / 2**30,
            61.27 * 0.9,
            places=2,
        )
        self.assertTrue(
            result["runtime_safe_recommended"]["startup_feasible"]
        )
        self.assertTrue(
            result["runtime_safe_recommended"]["runtime_safe"]
        )

    def test_hccl_buffer_uses_bidirectional_bytes_per_domain(self) -> None:
        value = base_config(45_056)
        value["vllm_ascend"].update(
            {
                "hccl_buffsize_mib": 1024,
                "hccl_communication_domains_per_rank": 1,
            }
        )

        result = evaluate_startup(value, 45_056, 64)

        self.assertEqual(
            result.details["non_torch"]["hccl_buffer_bytes"],
            2 * 1024 * 2**20,
        )
        self.assertEqual(
            result.non_torch_bytes,
            2 * 1024 * 2**20,
        )

    def test_success_log_reconstructs_available_kv(self) -> None:
        result = evaluate_startup(
            base_config(45_056),
            45_056,
            64,
            SUCCESS_LOG,
        )

        self.assertLess(
            abs(result.available_kv_bytes / 2**30 - 17.62),
            0.03,
        )
        self.assertTrue(result.minimum_kv_check_passed)
        self.assertTrue(result.startup_feasible)

    def test_configured_profile_calibration_is_used_by_startup_model(self) -> None:
        value = base_config(45_056)
        value["profile_calibration"].update(
            {
                "weight_gib_per_rank": 27.17,
                "peak_activation_gib_per_rank": 7.27,
                "non_torch_gib_per_rank": 3.08,
                "graph_gib_per_rank": 1.97,
            }
        )

        result = evaluate_startup(value, 45_056, 64)

        self.assertAlmostEqual(result.model_load_bytes / 2**30, 27.17, 2)
        self.assertAlmostEqual(
            result.profile_activation_bytes / 2**30, 7.27, 2
        )
        self.assertAlmostEqual(result.non_torch_bytes / 2**30, 3.08, 2)
        self.assertAlmostEqual(result.graph_bytes / 2**30, 1.97, 2)
        self.assertTrue(result.startup_feasible)

    def test_profile_calibration_scales_to_candidate_q(self) -> None:
        value = base_config(47_104)
        value["profile_calibration"].update(
            {
                "profiled_max_num_batched_tokens": 45_056,
                "weight_gib_per_rank": 27.17,
                "peak_activation_gib_per_rank": 7.27,
                "non_torch_gib_per_rank": 3.08,
            }
        )

        result = evaluate_startup(value, 47_104, 64)

        self.assertGreater(result.model_load_bytes / 2**30, 27.17)
        self.assertGreater(
            result.profile_activation_bytes / 2**30, 7.27
        )
        self.assertFalse(result.startup_feasible)

    def test_failure_log_is_classified_as_minimum_kv_check(self) -> None:
        result = evaluate_startup(
            base_config(47_104),
            47_104,
            64,
            FAIL_LOG,
        )

        self.assertEqual(result.limiting_stage, "minimum_kv_check")
        self.assertFalse(result.startup_feasible)
        self.assertFalse(result.minimum_kv_check_passed)

    def test_graph_is_not_subtracted_from_available_kv(self) -> None:
        with_graph = evaluate_startup(
            base_config(45_056),
            45_056,
            64,
            SUCCESS_LOG,
        )
        without_graph = evaluate_startup(
            base_config(45_056),
            45_056,
            64,
            SUCCESS_LOG_NO_GRAPH,
        )

        self.assertEqual(
            with_graph.available_kv_bytes,
            without_graph.available_kv_bytes,
        )

    def test_success_log_keeps_theoretical_weight_residual(self) -> None:
        result = evaluate_startup(
            base_config(45_056),
            45_056,
            64,
            SUCCESS_LOG,
        )

        self.assertGreater(result.measured_weight_residual_bytes, 0)
        self.assertGreater(result.theoretical_model_load_bytes / 2**30, 25.8)
        self.assertAlmostEqual(
            result.model_load_bytes / 2**30,
            27.17,
            places=2,
        )

    def test_failure_only_log_uses_reported_available_and_required_kv(self) -> None:
        result = evaluate_startup(
            base_config(47_104),
            47_104,
            64,
            FAIL_LOG,
        )

        self.assertAlmostEqual(result.available_kv_bytes / 2**30, 17.2)
        self.assertAlmostEqual(result.reported_required_kv_bytes / 2**30, 17.26)
        self.assertLess(
            abs(result.minimum_kv_bytes / 2**30 - 17.2570199966),
            1e-9,
        )

    def test_flashcomm1_sequence_parallel_reduces_tp4_activation(self) -> None:
        tp2_config = base_config(45_056)
        tp4_config = base_config(45_056)
        tp4_config["parallelism"].update({"dp_size": 4, "tp_size": 4})
        tp2_config["vllm_ascend"]["enable_flashcomm1"] = True
        tp4_config["vllm_ascend"]["enable_flashcomm1"] = True

        tp2 = estimate_activation(tp2_config, 45_056, {})
        tp4 = estimate_activation(tp4_config, 45_056, {})

        self.assertEqual(tp4.used_peak_bytes * 2, tp2.used_peak_bytes)

    def test_flashcomm1_does_not_inflate_tp1_activation(self) -> None:
        tp1_config = base_config(27_648)
        tp2_config = base_config(27_648)
        tp1_config["parallelism"].update({"dp_size": 16, "tp_size": 1})
        tp1_config["vllm_ascend"]["enable_flashcomm1"] = True
        tp2_config["vllm_ascend"]["enable_flashcomm1"] = True

        tp1 = estimate_activation(tp1_config, 27_648, {})
        tp2 = estimate_activation(tp2_config, 27_648, {})

        self.assertEqual(tp1.used_peak_bytes, tp2.used_peak_bytes)


if __name__ == "__main__":
    unittest.main()
