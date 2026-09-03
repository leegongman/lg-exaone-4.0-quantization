# w6a6_vllm_omniquant

Current working snapshot of the `w6a6` OmniQuant + vLLM environment used on 2026-04-04.

## Snapshot

- Source snapshot from `/home/ubuntu/vLLM_OmniQuant`
- Base commit: `216e1b2`
- Runtime target:
  - `vllm==0.19.0`
  - `torch==2.10.0+cu129`

## Included

- `vllm/`
  - Current patched vLLM source tree
- `OmniQuant_EXAONE_v4/`
  - Current OmniQuant source tree
- `norm_speed.py`
- `bench_real_latency.py`
- `run_w6a6_eval_speed_loop.py`
- `calibset_v2.1.jsonl`
- `calibset_v2.1.first100.jsonl`
- `docs/`
  - Reproduction and setup notes

## Clone Notes

This repo excludes `.venv` and Git metadata from the original workspace. After clone:

1. Follow `docs/RUNPOD_SETUP.md`
2. Build/install the bundled `vllm/` source as described in `docs/WHEEL_BUILD.md`
3. Use `docs/CHECKPOINT_GENERATION.md` for checkpoint generation and evaluation

## Relevant Docs

- `docs/MODIFIED_FILES.md`
- `docs/WHEEL_BUILD.md`
- `docs/RUNPOD_SETUP.md`
- `RESTORE_FROM_CLONE.md`
