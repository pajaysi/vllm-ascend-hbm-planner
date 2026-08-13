from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from vllm_ascend_hbm.config import DEFAULT_CONFIG, load_config, validate_config
from vllm_ascend_hbm.engine import calculate
from vllm_ascend_hbm.profiles import get_profile, list_profiles
from vllm_ascend_hbm.recommender import recommend
from vllm_ascend_hbm.utils import gib


ROOT = Path(__file__).resolve().parents[1]


class PlannerTests(unittest.TestCase):
    def test_registry_aliases(self) -> None:
        self.assertEqual(get_profile("Qwen/Qwen3-8B").profile_id, "qwen3-8b")
        self.assertGreaterEqual(len(list_profiles()), 10)

    def test_deepseek_v4_regression(self) -> None:
        config = load_config(str(ROOT / "configs/deepseek_v4_flash_910c.json"))
        config["recommendation"]["candidate_max_num_batched_tokens"] = [4096, 16384]
        config["recommendation"]["candidate_max_num_seqs"] = [1, 2, 4]
        result = recommend(config)
        recommended = result["single_service_recommended"]
        self.assertIsNotNone(recommended)
        self.assertEqual(recommended["max_num_batched_tokens"], 16384)
        self.assertEqual(recommended["max_num_seqs"], 4)
        scenario = result["scenarios"][0]["recommended"]
        # The v0.23 regression uses the NPU-visible 61.27 GiB capacity and
        # keeps startup admission separate from the runtime-safe upper bound.
        self.assertAlmostEqual(
            gib(scenario["planning_upper_bytes"]),
            47.56,
            places=1,
        )

    def test_qwen3_gqa_geometry(self) -> None:
        config = load_config(str(ROOT / "configs/qwen3_8b_910c.json"))
        config["operation"] = "estimate"
        config["workload"]["concurrency"] = [1]
        config["workload"]["context_len"] = 32768
        result = calculate(config)
        row = result["estimates"][0]
        self.assertEqual(result["kv_profile"]["bytes_per_token_per_layer"], 2048)
        self.assertAlmostEqual(gib(row["kv_planner_bytes"]), 2.25, places=2)
        self.assertTrue(row["fits_requested_budget"])

    def test_hf_config_auto_detection(self) -> None:
        payload = {
            "schema_version": 2,
            "operation": "estimate",
            "model": {
                "profile": "auto",
                "config_path": str(ROOT / "tests/fixtures/qwen_like_config.json"),
                "total_parameters": 7_000_000_000
            },
            "scheduler": {
                "block_size": 128,
                "max_model_len": 65536,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 4
            },
            "workload": {
                "mode": "late",
                "context_len": 32768,
                "concurrency": [1, 2, 4],
                "concurrency_scope": "per-dp"
            },
            "parallelism": {
                "dp_size": 1,
                "tp_size": 2,
                "pp_size": 1,
                "ep_size": 1,
                "pcp_size": 1,
                "dcp_size": 1
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(str(path))
        self.assertEqual(config["model"]["architecture"], "Qwen2ForCausalLM")
        self.assertEqual(config["model"]["head_dim"], 128)
        self.assertEqual(config["model"]["kv_cache_strategy"], "standard_gqa")
        self.assertGreater(calculate(config)["estimates"][0]["kv_planner_bytes"], 0)

    def test_manual_kv_adapter(self) -> None:
        config = load_config(str(ROOT / "configs/manual_hybrid_kv.example.json"))
        config["operation"] = "estimate"
        config["workload"]["concurrency"] = [1]
        result = calculate(config)
        self.assertEqual(result["model"]["kv_cache_strategy"], "manual")
        self.assertGreater(result["estimates"][0]["kv_planner_bytes"], 0)

    def test_schema_v1_is_migrated(self) -> None:
        payload = {
            "schema_version": 1,
            "operation": "estimate",
            "model": {"attention_head_dim": 512}
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = load_config(str(path))
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["model"]["head_dim"], 512)
        validate_config(config)


if __name__ == "__main__":
    unittest.main()
