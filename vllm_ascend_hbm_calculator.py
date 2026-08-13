#!/usr/bin/env python3
"""Source-tree launcher; installation is optional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from vllm_ascend_hbm.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
