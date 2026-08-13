# Theory-First Startup Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce vLLM Ascend 0.23.0rc1 DeepSeek-V4-Flash W8A8 MTP startup capacity from source-equivalent formulas and validate the nine measured Q boundaries.

**Architecture:** Add a source-versioned startup model beside the existing runtime estimator. The startup path computes model-load persistent memory, profile activation/non-Torch budget, exact minimum KV admission, physical KV pool allocation, and graph capture as separate lifecycle stages. Existing generic model adapters remain available for runtime estimates.

**Tech Stack:** Python 3.8+, standard library, unittest/pytest-compatible tests, JSON configuration.

## Global Constraints

- First exact target: 910C/A3, vLLM Ascend 0.23.0rc1, DeepSeek-V4-Flash W8A8_DYNAMIC + one MTP layer.
- Minimum KV admission must match the upstream 22-tuple formula.
- Physical KV allocation must remain distinct and may use 23 allocation tuples.
- Theoretical terms must not be silently replaced by measured values.
- The nine provided measurements use `max_num_seqs=64` and validate startup only.
- Preserve schema-v2 input compatibility and the existing generic model adapters.

---

### Task 1: Exact v0.23 DeepSeek-V4 minimum KV admission

**Files:**
- Create: `src/vllm_ascend_hbm/kv/deepseek_v4_v023.py`
- Modify: `src/vllm_ascend_hbm/kv/deepseek_v4_flash.py`
- Test: `tests/test_deepseek_v4_v023_kv.py`

**Interfaces:**
- Produces: `DeepSeekV4V023KVLayout`, `minimum_kv_admission(max_model_len, max_num_batched_tokens, block_size=128) -> KVAdmissionEstimate`.
- Consumes: integer scheduler inputs and the source-defined A3 block/page mapping.

- [ ] **Step 1: Write failing tests**

```python
def test_32k_q47104_matches_vllm_minimum_kv_log():
    result = minimum_kv_admission(32768, 47104, 128)
    assert result.total_pages == 5702
    assert result.total_bytes == 18_522_062_848
    assert abs(result.total_bytes / 2**30 - 17.2570199966) < 1e-9

def test_state_pages_plateau_after_q_reaches_max_model_len():
    assert minimum_kv_admission(32768, 45056, 128).total_bytes == \
           minimum_kv_admission(32768, 99328, 128).total_bytes

def test_minimum_admission_uses_22_tuples():
    assert minimum_kv_admission(32768, 47104, 128).tuple_count == 22
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_deepseek_v4_v023_kv.py -q`
Expected: import failure because `deepseek_v4_v023` does not exist.

- [ ] **Step 3: Implement the source-equivalent page formulas**

Implement C4/C128 history, two SWA groups, C4 state and C128 state using the exact `min(window - 1 + Q, L)` caps and source page sizes.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_deepseek_v4_v023_kv.py -q`
Expected: all tests pass.

- [ ] **Step 5: Run existing KV regression**

Run: `python -m pytest tests/test_planner.py -q`
Expected: failures are limited to assertions whose old values depended on the incorrect v0.23 planner.

### Task 2: Tensor-aware DeepSeek-V4 W8A8 model-load estimator

**Files:**
- Create: `src/vllm_ascend_hbm/weight_models/__init__.py`
- Create: `src/vllm_ascend_hbm/weight_models/deepseek_v4_w8a8.py`
- Create: `src/vllm_ascend_hbm/weight_models/persistent_buffers.py`
- Modify: `src/vllm_ascend_hbm/weights.py`
- Modify: `src/vllm_ascend_hbm/profiles.py`
- Test: `tests/test_deepseek_v4_w8a8_weights.py`

**Interfaces:**
- Produces: `estimate_deepseek_v4_w8a8(config) -> WeightEstimate`.
- Produces detail fields for routed experts, shared experts, attention INT8, BF16/FP32 projections, quant metadata, MTP, and Q-dependent persistent buffers.
- Consumes: TP, EP, PP, shared-expert-DP, Q and model geometry.

- [ ] **Step 1: Write failing placement and buffer tests**

```python
def test_mtp_hidden_buffer_is_1375_gib_at_q45056():
    details = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45056)).details
    assert details["mtp_hidden_buffer_bytes"] == 45056 * 4 * 4096 * 2

def test_two_indexer_buffers_use_index_topk_512():
    details = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45056)).details
    assert details["topk_buffers_bytes"] == 2 * 45056 * 512 * 4

def test_routed_experts_divide_by_ep_not_tp():
    tp2 = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45056))
    tp4 = estimate_deepseek_v4_w8a8(config(tp=4, ep=16, q=45056))
    assert tp2.details["routed_expert_bytes"] == tp4.details["routed_expert_bytes"]

def test_replicated_attention_does_not_divide_by_tp():
    tp2 = estimate_deepseek_v4_w8a8(config(tp=2, ep=16, q=45056))
    tp4 = estimate_deepseek_v4_w8a8(config(tp=4, ep=16, q=45056))
    assert tp2.details["attention_replicated_bytes"] == \
           tp4.details["attention_replicated_bytes"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_deepseek_v4_w8a8_weights.py -q`
Expected: import/API failure.

- [ ] **Step 3: Implement model geometry and placement**

Encode the public checkpoint/model geometry and v0.23 module placement rules. Keep checkpoint tensors, post-load duplicates and model-owned Q buffers as separate detail categories.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_deepseek_v4_w8a8_weights.py -q`
Expected: all tests pass.

- [ ] **Step 5: Add measured residual reporting**

When a log weight is provided, retain the theoretical estimate and report:

```python
residual_bytes = measured_model_load_bytes - theoretical_model_load_bytes
```

Do not replace the theoretical detail structure.

### Task 3: Startup lifecycle and dual output

**Files:**
- Create: `src/vllm_ascend_hbm/capacity.py`
- Create: `src/vllm_ascend_hbm/startup.py`
- Create: `src/vllm_ascend_hbm/logs.py`
- Modify: `src/vllm_ascend_hbm/config.py`
- Modify: `src/vllm_ascend_hbm/types.py`
- Modify: `src/vllm_ascend_hbm/engine.py`
- Modify: `src/vllm_ascend_hbm/recommender.py`
- Modify: `src/vllm_ascend_hbm/output.py`
- Test: `tests/test_startup_lifecycle.py`

**Interfaces:**
- Produces: `evaluate_startup(config, q, seqs, measured=None) -> StartupEstimate`.
- Produces result keys `startup_limit` and `runtime_safe`.
- Consumes theoretical weights/KV plus optional parsed log components.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_success_log_reconstructs_available_kv():
    result = evaluate_startup(base_config(q=45056), 45056, 64, SUCCESS_LOG)
    assert abs(result.available_kv_bytes / 2**30 - 17.62) < 0.03
    assert result.minimum_kv_check_passed

def test_failure_log_is_classified_as_minimum_kv_check():
    result = evaluate_startup(base_config(q=47104), 47104, 64, FAIL_LOG)
    assert result.limiting_stage == "minimum_kv_check"
    assert not result.startup_feasible

def test_graph_is_not_subtracted_from_available_kv():
    with_graph = evaluate_startup(base_config(q=45056), 45056, 64, SUCCESS_LOG)
    without_graph = evaluate_startup(base_config(q=45056), 45056, 64, SUCCESS_LOG_NO_GRAPH)
    assert with_graph.available_kv_bytes == without_graph.available_kv_bytes
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_startup_lifecycle.py -q`
Expected: import/API failure.

- [ ] **Step 3: Implement capacity fields and log parsing**

Support nominal HBM, visible HBM and startup-free HBM independently. Parse the supplied success/failure log formats without overwriting raw values.

- [ ] **Step 4: Implement lifecycle evaluation**

Evaluate profile budget, minimum KV admission, physical pool allocation and graph physical headroom as ordered stages.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_startup_lifecycle.py -q`
Expected: all tests pass.

### Task 4: Nine-point zero-fit validation

**Files:**
- Create: `tests/fixtures/dsv4_v023_startup_boundaries.json`
- Create: `src/vllm_ascend_hbm/validation.py`
- Create: `tests/test_dsv4_startup_boundaries.py`
- Modify: `src/vllm_ascend_hbm/cli.py`

**Interfaces:**
- Produces: `validate_boundaries(config, rows) -> BoundaryValidationReport`.
- Produces per-row predicted threshold, interval membership, distance and limiting stage.

- [ ] **Step 1: Add the nine observed intervals**

Store all supplied `max_model_len`, DP, TP, `max_success_mnbt` and `first_fail_mnbt` values with `max_num_seqs=64`.

- [ ] **Step 2: Write a failing zero-fit validation test**

```python
def test_all_nine_startup_thresholds_are_reported_without_fit():
    report = validate_boundaries(config, load_rows())
    assert report.calibration_used is False
    assert len(report.rows) == 9
    assert all(row.predicted_q is not None for row in report.rows)
```

- [ ] **Step 3: Run test and verify RED**

Run: `python -m pytest tests/test_dsv4_startup_boundaries.py -q`
Expected: import/API failure.

- [ ] **Step 4: Implement threshold search and interval report**

Use monotonic binary search at 1024-token granularity. Report unmatched rows honestly; do not alter formulas to force all nine to pass.

- [ ] **Step 5: Run validation and inspect component residuals**

Run: `python -m pytest tests/test_dsv4_startup_boundaries.py -q -s`
Expected: nine rows with explicit predicted thresholds and pass/fail status.

### Task 5: Documentation, compatibility and full verification

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `MODEL_SUPPORT.md`
- Modify: `configs/deepseek_v4_flash_910c.json`
- Create: `docs/theory-first-modeling-guide.md`
- Modify: `tests/test_planner.py`

**Interfaces:**
- Documents the new JSON fields and both recommendation outputs.
- Preserves existing CLI entry points.

- [ ] **Step 1: Update regression expectations**

Only change old expected values after the exact source-equivalent tests pass and the reason is documented.

- [ ] **Step 2: Document formulas and residual policy**

Include the 17.25702 GiB worked example, tensor-aware weight categories, lifecycle stages and nine-point report interpretation.

- [ ] **Step 3: Run focused tests**

Run:

```text
python -m pytest tests/test_deepseek_v4_v023_kv.py -q
python -m pytest tests/test_deepseek_v4_w8a8_weights.py -q
python -m pytest tests/test_startup_lifecycle.py -q
python -m pytest tests/test_dsv4_startup_boundaries.py -q
```

Expected: all pass.

- [ ] **Step 4: Run full regression**

Run: `python -m pytest -q`
Expected: all tests pass with no warnings.

- [ ] **Step 5: Run CLI smoke tests**

Run:

```text
python vllm_ascend_hbm_calculator.py --config configs/deepseek_v4_flash_910c.json
python vllm_ascend_hbm_calculator.py --list-models
```

Expected: dual startup/runtime output and successful model listing.

- [ ] **Step 6: Package verification**

Run: `python -m build` if the `build` package is installed; otherwise run `python -m compileall src`.
