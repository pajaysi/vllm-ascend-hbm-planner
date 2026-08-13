from __future__ import annotations

import copy
import unittest
from pathlib import Path

from vllm_ascend_hbm.config import load_config
from vllm_ascend_hbm.weight_models.deepseek_v4_w8a8 import (
    estimate_deepseek_v4_w8a8,
)
from vllm_ascend_hbm.weights import estimate_weights
from vllm_ascend_hbm.profiles import get_profile


ROOT = Path(__file__).resolve().parents[1]


def config(*, tp: int, ep: int, q: int) -> dict:
    value = load_config(str(ROOT / "configs/deepseek_v4_flash_910c.json"))
    value["platform"]["vllm_ascend_version"] = "0.23.0rc1"
    value["parallelism"]["tp_size"] = tp
    value["parallelism"]["ep_size"] = ep
    value["scheduler"]["max_num_batched_tokens"] = q
    value.setdefault("vllm_ascend", {})["enable_shared_expert_dp"] = True
    return value


class DeepSeekV4W8A8WeightTests(unittest.TestCase):
    def test_mtp_hidden_buffer_is_1375_gib_at_q45056(self) -> None:
        estimate = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertEqual(
            estimate.details["mtp_hidden_buffer_bytes"],
            45_056 * 4 * 4_096 * 2,
        )
        self.assertEqual(
            estimate.details["mtp_hidden_buffer_bytes"],
            1_476_395_008,
        )

    def test_mtp_indexer_buffer_is_shared_with_target(self) -> None:
        estimate = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertEqual(
            estimate.details["topk_buffers_bytes"],
            45_056 * 512 * 4,
        )
        self.assertTrue(estimate.details["mtp_topk_buffer_shared"])

    def test_routed_experts_divide_by_ep_not_tp(self) -> None:
        tp2 = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))
        tp4 = estimate_deepseek_v4_w8a8(config(tp=4, ep=16, q=45_056))

        self.assertEqual(
            tp2.details["routed_expert_bytes"],
            tp4.details["routed_expert_bytes"],
        )

    def test_replicated_attention_does_not_divide_by_tp(self) -> None:
        tp2 = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))
        tp4 = estimate_deepseek_v4_w8a8(config(tp=4, ep=16, q=45_056))

        self.assertEqual(
            tp2.details["attention_replicated_bytes"],
            tp4.details["attention_replicated_bytes"],
        )
        self.assertGreater(
            tp2.details["attention_tp_sharded_bytes"],
            tp4.details["attention_tp_sharded_bytes"],
        )

    def test_q_increases_only_model_owned_persistent_buffers(self) -> None:
        low = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=44_032))
        high = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertEqual(
            high.per_rank_bytes - low.per_rank_bytes,
            1_024
            * (
                4 * 4_096 * 2
                + 512 * 4
                + 4 * 64 * 4 * 2
            ),
        )

    def test_source_specific_dtype_and_alias_rules_are_applied(self) -> None:
        estimate = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertEqual(
            estimate.details["compressor_projection_dtype_bytes"],
            4,
        )
        self.assertTrue(estimate.details["router_retains_bf16_and_fp32"])
        self.assertTrue(estimate.details["mtp_lm_head_shared"])

    def test_rope_caches_created_during_model_load_are_counted(self) -> None:
        estimate = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertEqual(
            estimate.details["rope_full_cache_bytes"],
            2 * 1_048_576 * 64 * 4 * 2,
        )
        self.assertEqual(
            estimate.details["rope_runtime_buffer_bytes"],
            4 * 45_056 * 64 * 4 * 2,
        )

    def test_tp2_q45056_theory_closes_most_of_weight_log_gap(self) -> None:
        estimate = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45_056))

        self.assertGreater(estimate.per_rank_bytes / 2**30, 27.0)
        self.assertLess(estimate.per_rank_bytes / 2**30, 27.15)
        self.assertEqual(estimate.source, "deepseek-v4-w8a8-module-placement")

    def test_estimator_does_not_mutate_input(self) -> None:
        value = config(tp=2, ep=16, q=45_056)
        before = copy.deepcopy(value)

        estimate_deepseek_v4_w8a8(value)

        self.assertEqual(value, before)

    def test_weight_facade_selects_theory_first_dsv4_model(self) -> None:
        value = config(tp=2, ep=16, q=45_056)

        estimate = estimate_weights(value, {})

        self.assertEqual(
            estimate.source,
            "deepseek-v4-w8a8-module-placement",
        )
        self.assertIn("routed_expert_bytes", estimate.details)

    def test_measured_weight_retains_theory_and_reports_residual(self) -> None:
        value = config(tp=2, ep=16, q=45_056)
        theoretical = estimate_deepseek_v4_w8a8(value)
        measured_bytes = round(27.17 * 2**30)

        estimate = estimate_weights(
            value,
            {"weight_gib_per_rank": 27.17},
        )

        self.assertEqual(estimate.per_rank_bytes, measured_bytes)
        self.assertEqual(
            estimate.details["theoretical_model_load_bytes"],
            theoretical.per_rank_bytes,
        )
        self.assertEqual(
            estimate.details["measured_residual_bytes"],
            measured_bytes - theoretical.per_rank_bytes,
        )

    def test_builtin_profile_uses_official_v4_geometry(self) -> None:
        model = get_profile("deepseek-v4-flash").defaults

        self.assertEqual(model["vocab_size"], 129_280)
        self.assertEqual(model["num_shared_experts"], 1)
        self.assertEqual(model["sliding_window"], 128)
        self.assertEqual(model["qk_rope_head_dim"], 64)


if __name__ == "__main__":
    unittest.main()
