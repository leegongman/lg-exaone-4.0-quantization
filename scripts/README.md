# Scripts

This directory contains cleaned helper scripts that can run without private access.

## Included Script

- `benchmark_vllm_tpt.py`: generates synthetic prompts and reports local vLLM token timing for one model. It uses no DACON data, credentials, checkpoints, or official scoring logic.

Example:

```bash
python scripts/benchmark_vllm_tpt.py \
  --model /path/to/local-or-public-model \
  --output outputs/local-vllm-report.json
```

Run each model in the same hardware and software environment, then compare the resulting local reports. Do not present the output as a competition score or organizer benchmark.

Before adding a script, remove credentials, DACON tokens, private URLs, local absolute paths, generated data paths, notebook output, and copied upstream source.
