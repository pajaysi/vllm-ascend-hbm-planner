from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from vllm_ascend_hbm.config import load_config
from vllm_ascend_hbm.validation import validate_boundaries


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekV4StartupBoundaryTests(unittest.TestCase):
    @staticmethod
    def _observed_rows() -> list[dict]:
        return json.loads(
            (
                ROOT
                / "tests/fixtures/dsv4_v023_startup_boundaries.json"
            ).read_text(encoding="utf-8")
        )

    def test_all_nine_thresholds_are_reported_without_fit(self) -> None:
        config = load_config(
            str(ROOT / "configs/deepseek_v4_flash_910c.json")
        )
        config["platform"].update(
            {
                "vllm_ascend_version": "0.23.0rc1",
                "visible_hbm_gib_per_die": 61.27,
                "startup_free_hbm_gib_per_die": 60.89,
            }
        )
        config["vllm_ascend"].update(
            {
                "enable_shared_expert_dp": True,
                "enable_flashcomm1": True,
                "hccl_buffsize_mib": 1024,
            }
        )
        rows = self._observed_rows()

        report = validate_boundaries(config, rows)

        self.assertFalse(report.calibration_used)
        self.assertEqual(len(report.rows), 9)
        self.assertTrue(
            all(row.predicted_max_success_q is not None for row in report.rows)
        )
        self.assertTrue(
            all(row.limiting_stage == "minimum_kv_check" for row in report.rows)
        )
        payload = report.as_dict()
        self.assertEqual(payload["operation"], "validate_startup_boundaries")
        self.assertEqual(payload["summary"]["total_rows"], 9)
        self.assertIn("mean_absolute_distance_to_interval_tokens", payload["summary"])
        tp2_32k = payload["rows"][1]
        self.assertGreater(
            tp2_32k["implied_unmodeled_lower_bound_bytes"], 0
        )
        self.assertGreater(
            tp2_32k["implied_unmodeled_upper_bound_bytes"],
            tp2_32k["implied_unmodeled_lower_bound_bytes"],
        )

    def test_cli_accepts_boundary_fixture_and_emits_json(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "vllm_ascend_hbm_calculator.py"),
                "--config",
                str(ROOT / "configs/deepseek_v4_flash_910c.json"),
                "--validate-boundaries",
                str(
                    ROOT
                    / "tests/fixtures/dsv4_v023_startup_boundaries.json"
                ),
                "--format",
                "json",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["summary"]["total_rows"], 9)
        self.assertEqual(len(payload["rows"]), 9)

    def test_topology_profiles_validate_eight_of_nine_boundaries(self) -> None:
        config = load_config(
            str(ROOT / "configs/deepseek_v4_flash_910c.json")
        )
        config["validation"] = {
            "profile_calibration_by_tp": {
                "1": {
                    "profiled_max_num_batched_tokens": 23552,
                    "weight_gib_per_rank": 31.76,
                    "peak_activation_gib_per_rank": 4.74,
                    "non_torch_gib_per_rank": 2.60,
                    "graph_gib_per_rank": 1.28,
                    "visible_hbm_gib_per_die": 61.28,
                    "startup_free_hbm_gib_per_die": 61.13,
                },
                "2": {
                    "profiled_max_num_batched_tokens": 45056,
                    "weight_gib_per_rank": 27.17,
                    "peak_activation_gib_per_rank": 7.27,
                    "non_torch_gib_per_rank": 3.08,
                    "graph_gib_per_rank": 1.97,
                    "visible_hbm_gib_per_die": 61.27,
                    "startup_free_hbm_gib_per_die": 60.89,
                },
                "4": {
                    "profiled_max_num_batched_tokens": 39936,
                    "weight_gib_per_rank": 24.26,
                    "peak_activation_gib_per_rank": 3.38,
                    "non_torch_gib_per_rank": 3.07,
                    "graph_gib_per_rank": 1.80,
                    "visible_hbm_gib_per_die": 61.27,
                    "startup_free_hbm_gib_per_die": 60.89,
                },
            }
        }

        report = validate_boundaries(config, self._observed_rows())

        self.assertTrue(report.calibration_used)
        self.assertEqual(report.matched_rows, 8)
        self.assertEqual(report.rows[1].predicted_max_success_q, 46_926)
        self.assertEqual(report.rows[8].predicted_max_success_q, 24_305)


if __name__ == "__main__":
    unittest.main()
