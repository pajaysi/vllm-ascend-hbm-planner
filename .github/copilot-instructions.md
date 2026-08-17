Purpose
-------
These instructions help an AI coding assistant become productive in this repository quickly. They focus on the repository's architecture, developer workflows, conventions, and concrete examples taken from the code.

Quick start (development)
- Project layout: core code in `src/vllm_ascend_hbm/`, configs in `configs/`, tests in `tests/`, packaging in `packaging/` and `build/`.
- Run unit tests: set `PYTHONPATH=src` then run `pytest -q` (Windows PowerShell: `$env:PYTHONPATH='src'; pytest -q`).
- Run CLI locally: set `PYTHONPATH=src` then `python -m vllm_ascend_hbm` (or run `src\\vllm_ascend_hbm\\__main__.py`).
- Build packaged exe: packaging uses PyInstaller; see `scripts/build_exe.ps1` and `packaging/pyinstaller/vllm_ascend_hbm.spec`.

Big-picture architecture
- Purpose: compute and validate HBM (high-bandwidth memory) startup capacity estimates for vllm-like models on Ascend hardware.
- Main components:
  - `src/vllm_ascend_hbm/cli.py` — command-line entry and argument parsing.
  - `src/vllm_ascend_hbm/startup.py` — core evaluation functions (e.g. `evaluate_startup`) that return dataclasses describing feasibility, headroom and limiting stage.
  - `src/vllm_ascend_hbm/validation.py` — routines that compare predictions against observed intervals and perform binary-search-style limits (`_maximum_feasible_q`).
  - `src/vllm_ascend_hbm/recommender.py`, `profiles.py`, `weights.py` — higher-level helpers for recommendations and model weight accounting.
  - `configs/*.json` — example runtime/topology profiles used as input data.

Key data & conventions
- Configs are plain nested dicts (not pydantic). Agents should follow the existing mutation pattern: deep-copy a base config, then update keys (see `validate_boundaries` in `validation.py`).
- Typical call pattern: `evaluate_startup(config, q, max_num_seqs)` — returns an object with attributes like `startup_feasible`, `available_kv_bytes`, `minimum_kv_bytes`, and `limiting_stage`.
- Calibration values live under `profile_calibration` and `validation.profile_calibration_by_tp`; topology overrides use `platform` keys like `visible_hbm_gib_per_die`.
- Parallelism keys: `dp_size`, `tp_size`, `ep_size` live under `parallelism` and are frequently set together.

Repository conventions
- Favor minimal, focused edits: mutate copied config dicts rather than creating new configuration layers.
- Functions return small dataclasses (frozen/typed). When adding fields, keep backward compatibility and update callsites in `validation.py` and tests.
- Use integer token/q units for batch-search loops (see `_maximum_feasible_q`). Respect the binary-search doubling idiom already present.

Testing & debugging tips
- Unit tests live in `tests/` and exercise realistic configs — run them after code changes.
- For interactive debugging, set `PYTHONPATH=src` and call key functions from a REPL: e.g.:
  - `$env:PYTHONPATH='src'; python -c "from vllm_ascend_hbm.startup import evaluate_startup; print(evaluate_startup(<config>, 1, 64))"`
- When changing calculations that affect memory headroom, update `tests/test_deepseek_v4_v023_kv.py` and `test_deepseek_v4_w8a8_weights.py` to cover regressions.

Integration and packaging
- Packaging uses PyInstaller; built artifacts end up under `build/pyinstaller/` and the release folder `release/vllm-ascend-hbm-windows-x64-*`.
- The `scripts/build_exe.ps1` script is the canonical Windows build flow — prefer it over ad-hoc PyInstaller calls.

What to watch for when editing
- Many routines assume deterministic numeric behavior; avoid non-deterministic RNGs in core calculation paths.
- Keep JSON config keys and names consistent with examples in `configs/` and with `pyproject.toml` declared entry points.
- When adding new CLI flags, update `cli.py`, `__main__.py` and ensure packaging spec includes any added data files.

Examples (from codebase)
- Binary-search pattern for maximum q: see `_maximum_feasible_q` in `src/vllm_ascend_hbm/validation.py` — preserve doubling loop then binary search.
- Validation interval calculations: `validate_boundaries` uses `available_kv_bytes - minimum_kv_bytes` to infer implied unmodeled memory.

If uncertain
- Prefer small, test-backed changes. Run relevant unit tests and a quick `evaluate_startup` call for behavioral checks.
- Ask the maintainer which real hardware profile (configs/*.json) should be used for calibration before large changes to platform-specific logic.

Please review this file and tell me if you'd like more examples, added checklist items, or integration with CI commands.
