from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vllm_ascend_hbm.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class HardwareConfigTests(unittest.TestCase):
    @staticmethod
    def _write_hardware(directory: str, **overrides: object) -> Path:
        hardware = {
            "device": "910c",
            "server_count": 1,
            "physical_cards_per_server": 8,
            "dies_per_card": 2,
            "logical_devices_per_server": 16,
            "nominal_hbm_gib_per_die": 64.0,
            "visible_hbm_gib_per_die": 61.27,
            "startup_free_hbm_gib_per_die": 60.89,
        }
        hardware.update(overrides)
        path = Path(directory) / "hardware.json"
        path.write_text(
            json.dumps({"schema_version": 1, "hardware": hardware}),
            encoding="utf-8",
        )
        return path

    def test_separate_hardware_config_is_authoritative_for_hardware_fields(self) -> None:
        hardware = {
            "schema_version": 1,
            "hardware": {
                "device": "910c",
                "server_count": 2,
                "physical_cards_per_server": 8,
                "dies_per_card": 2,
                "logical_devices_per_server": 16,
                "nominal_hbm_gib_per_die": 64.0,
                "visible_hbm_gib_per_die": 61.5,
                "startup_free_hbm_gib_per_die": 60.75,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            hardware_path = Path(directory) / "hardware.json"
            hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
            config = load_config(
                str(ROOT / "configs/deepseek_v4_flash_910c.json"),
                str(hardware_path),
            )

        self.assertEqual(config["platform"]["server_count"], 2)
        self.assertEqual(config["platform"]["logical_device_count"], 32)
        self.assertEqual(config["platform"]["visible_hbm_gib_per_die"], 61.5)
        self.assertEqual(config["platform"]["startup_free_hbm_gib_per_die"], 60.75)
        self.assertEqual(config["platform"]["gpu_memory_utilization"], 0.9)
        self.assertEqual(config["platform"]["vllm_ascend_version"], "0.23.0rc1")

    def test_parallel_world_size_cannot_exceed_hardware_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hardware_path = self._write_hardware(
                directory,
                physical_cards_per_server=4,
                logical_devices_per_server=8,
            )
            with self.assertRaisesRegex(
                ValueError,
                r"DP\*TP\*PP=16 exceeds hardware logical_device_count=8",
            ):
                load_config(
                    str(ROOT / "configs/deepseek_v4_flash_910c.json"),
                    str(hardware_path),
                )

    def test_inconsistent_per_server_topology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hardware_path = self._write_hardware(
                directory,
                physical_cards_per_server=8,
                dies_per_card=2,
                logical_devices_per_server=8,
            )
            with self.assertRaisesRegex(
                ValueError,
                r"physical_cards_per_server\*dies_per_card=16",
            ):
                load_config(
                    str(ROOT / "configs/deepseek_v4_flash_910c.json"),
                    str(hardware_path),
                )

    def test_cli_accepts_separate_hardware_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hardware_path = self._write_hardware(
                directory,
                visible_hbm_gib_per_die=61.5,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "vllm_ascend_hbm_calculator.py"),
                    "--hardware-config",
                    str(hardware_path),
                    "--config",
                    str(ROOT / "configs/deepseek_v4_flash_910c.json"),
                    "--operation",
                    "estimate",
                    "--format",
                    "json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(
            payload["config"]["platform"]["visible_hbm_gib_per_die"],
            61.5,
        )


if __name__ == "__main__":
    unittest.main()
