from __future__ import annotations

import unittest
from pathlib import Path

from vllm_ascend_hbm.config import load_config
from vllm_ascend_hbm.kv.deepseek_v4_flash import estimate_deepseek_v4
from vllm_ascend_hbm.kv.deepseek_v4_v023 import minimum_kv_admission


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekV4V023KVTests(unittest.TestCase):
    def test_32k_q47104_matches_vllm_minimum_kv_log(self) -> None:
        result = minimum_kv_admission(
            max_model_len=32_768,
            max_num_batched_tokens=47_104,
            block_size=128,
        )

        self.assertEqual(result.total_pages, 5_702)
        self.assertEqual(result.total_bytes, 18_529_584_128)
        self.assertLess(abs(result.total_bytes / 2**30 - 17.2570199966), 1e-9)

    def test_state_pages_plateau_after_q_reaches_max_model_len(self) -> None:
        at_45k = minimum_kv_admission(32_768, 45_056, 128)
        at_99k = minimum_kv_admission(32_768, 99_328, 128)

        self.assertEqual(at_45k.total_bytes, at_99k.total_bytes)
        self.assertEqual(at_45k.c4_state_pages, 4_097)
        self.assertEqual(at_45k.c128_state_pages, 1_025)

    def test_minimum_admission_uses_22_layer_tuples(self) -> None:
        result = minimum_kv_admission(32_768, 47_104, 128)

        self.assertEqual(result.tuple_count, 22)
        self.assertEqual(result.bytes_per_tuple, 147_712)

    def test_q_below_model_len_uses_source_window_caps(self) -> None:
        result = minimum_kv_admission(131_072, 43_008, 128)

        self.assertEqual(result.c4_history_pages, 256)
        self.assertEqual(result.c128_history_pages, 8)
        self.assertEqual(result.swa_pages_per_group, 338)
        self.assertEqual(result.c4_state_pages, 5_378)
        self.assertEqual(result.c128_state_pages, 1_349)

    def test_existing_adapter_exposes_exact_minimum_admission(self) -> None:
        config = load_config(str(ROOT / "configs/deepseek_v4_flash_910c.json"))
        config["platform"]["vllm_ascend_version"] = "0.23.0rc1"
        config["scheduler"]["max_model_len"] = 32_768
        config["scheduler"]["max_num_batched_tokens"] = 47_104

        profile, _ = estimate_deepseek_v4(config)

        self.assertEqual(
            profile["minimum_admission"]["total_bytes"],
            18_529_584_128,
        )
        self.assertEqual(profile["model_shape"]["sliding_window"], 128)

    def test_runtime_single_request_q_is_capped_by_model_len(self) -> None:
        config = load_config(
            str(ROOT / "configs/deepseek_v4_flash_910c.json")
        )
        config["scheduler"].update(
            {
                "max_model_len": 32_768,
                "max_num_batched_tokens": 45_056,
            }
        )
        config["workload"].update(
            {
                "mode": "fresh",
                "context_len": 32_768,
                "concurrency": [1],
                "concurrency_scope": "per-dp",
            }
        )

        _, rows = estimate_deepseek_v4(config)

        self.assertEqual(rows[0].q_tokens_per_rank, 32_768)


if __name__ == "__main__":
    unittest.main()
