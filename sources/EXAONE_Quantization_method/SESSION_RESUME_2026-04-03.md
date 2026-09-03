# Session Resume 2026-04-03

## Environment constraints

- Keep `torch==2.9.0+cu128`
- Keep `vllm==0.14.1`
- Do not solve by upgrading/downgrading those versions

## Current working checkpoint

- Rebuilt checkpoint:
  - `artifacts/checkpoints/w6a6_1epoch_rebuilt`
- This rebuilt checkpoint now passes the vLLM smoke test with the current patched code path.

## Source changes made

- `vllm/vllm/model_executor/layers/quantization/omni_activation_real.py`
  - Added OmniQuant tensor name defaults:
    - `qweight`
    - `w_scales`
    - `w_zero_points`
  - Added grouped metadata defaults for exported OmniQuant weights
  - Added runtime cache reuse and fused/shared fake-quant fast path
- `norm_speed.py`
  - Supports calibset rows with either `message` or `input_ids`
  - Added `--fixed_decode_tokens` for apples-to-apples measurement
- `bench_real_latency.py`
  - Supports calibset rows with either `message` or `input_ids`
  - Uses fixed decode length (`ignore_eos=True`, `min_tokens=max_tokens`) for stable comparison
- `run_w6a6_eval_speed_loop.py`
  - Added loop runner, but current version injects local `vllm/` into `PYTHONPATH`, which breaks `.venv` compiled `_C`
  - Do not rely on this script without fixing that import path issue first

## Measurement results

### Smoke

- Passed for `artifacts/checkpoints/w6a6_1epoch_rebuilt`

### GSM8K 100

- Result file:
  - `artifacts/manual_runs/w6a6_1epoch_rebuilt/eval/chat_template/__home__ubuntu__vLLM_OmniQuant__artifacts__checkpoints__w6a6_1epoch_rebuilt/results_2026-04-03T02-30-45.820067.json`
- Metrics:
  - flexible exact match: `0.66`
  - strict exact match: `0.63`

### Original norm_speed run

- Quant TPT total: `0.000139`
- Baseline TPT total: `0.000145`
- Normalized speed: `+4.24%`
- This run was not apples-to-apples because effective decode length differed from comparison runs.

### Fixed-condition norm_speed run

- Command used:
  - `python norm_speed.py --model artifacts/checkpoints/w6a6_1epoch_rebuilt --baseline LGAI-EXAONE/EXAONE-4.0-1.2B --calibset calibset_v2.1.jsonl --max_gen_toks 256 --fixed_decode_tokens --output_path artifacts/manual_runs/w6a6_1epoch_rebuilt/speed_fixed --gpu_memory_utilization 0.85 --quantization omni_activation_real --dtype float16`
- Metrics:
  - quant TPT total: `0.000128`
  - baseline TPT total: `0.000127`
  - normalized speed: `-0.82%`
  - quant TPT decode: `0.000402`
  - baseline TPT decode: `0.000399`

### Fixed-condition bench_real_latency

- Quant:
  - total tok/s: `1677.55`
  - decode tok/s: `1238.42`
  - p50: `1.4500s`
  - p90: `1.4511s`
  - decode TPS p50: `88.28`
- Baseline:
  - total tok/s: `1683.53`
  - decode tok/s: `1242.02`
  - p50: `1.4480s`
  - p90: `1.4487s`
  - decode TPS p50: `88.40`

## Conclusion at handoff

- The previously observed speed gain was mostly due to mismatched measurement conditions.
- Under matched decode-length conditions, the current `w6a6_1epoch_rebuilt` path is slightly slower than baseline.
- Accuracy measurement exists, but speed target is not met.

## Next technical priority

- Optimize the vLLM `omni_activation_real` runtime path itself.
- Focus areas:
  - reduce fake-quant overhead in `apply`
  - reduce per-layer/per-shard Python overhead
  - avoid or shrink dense rebuild/dequant cost
  - investigate whether grouped weights can be used more directly instead of dense expansion

## Resume prompt for next session

Use this prompt in the next session:

```text
작업 디렉터리는 /home/ubuntu/vLLM_OmniQuant 이고, torch==2.9.0+cu128 / vllm==0.14.1 은 반드시 유지해. SESSION_RESUME_2026-04-03.md 를 먼저 읽고 이어서 작업해. 현재 w6a6 1ep rebuilt 체크포인트는 artifacts/checkpoints/w6a6_1epoch_rebuilt 이고 smoke 는 통과했지만, 고정 decode 조건의 norm_speed 기준 normalized speed 가 -0.82% 라서 목표(+15~20%)를 못 맞췄다. 다음은 vllm/vllm/model_executor/layers/quantization/omni_activation_real.py 를 우선 최적화하고, 매 단계마다 gsm8k 100, norm_speed, bench_real_latency 결과를 보고해.
```
