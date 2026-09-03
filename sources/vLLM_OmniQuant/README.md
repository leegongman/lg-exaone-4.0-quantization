# vLLM OmniQuant Submission Artifacts

This repository contains the submission-ready `vllm==0.14.1` wheel as split
parts so it can be stored on GitHub without Git LFS.

## Contents

- `wheel_parts/`
  - split wheel parts for `vllm-0.14.1-cp311-cp311-linux_x86_64.whl`
  - full-wheel SHA256
  - per-part SHA256
- `metadata/`
  - `build.txt`
  - `cuda.txt`
  - `pyproject.toml`
- `reconstruct_wheel.sh`
  - concatenates the split files back into the original wheel

## Reconstruct The Wheel

After cloning:

```bash
chmod +x reconstruct_wheel.sh
./reconstruct_wheel.sh
```

This creates:

```bash
./dist/vllm-0.14.1-cp311-cp311-linux_x86_64.whl
```

Expected wheel SHA256:

```text
11468c7cc32e708393164c7bb8de1bd327092cfe221018a2050ac26dcb74b059
```

## Install

```bash
uv venv --python 3.11 --seed --managed-python
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install ./dist/vllm-0.14.1-cp311-cp311-linux_x86_64.whl
```

## Verified

- `python 3.11.15`
- `torch 2.9.0+cu128`
- `torch.version.cuda == 12.8`
- `vllm 0.14.1`
- `from vllm import LLM, SamplingParams`
- `vllm serve`
- `/v1/chat/completions`
- `w6a6_cp_10ep_noamp`:
  - `norm_speed +4.93%`
  - `bench_real_latency` quant `2188.77 tok/s`, baseline `1713.82 tok/s`
  - `gsm8k 100` flexible `0.57`, strict `0.54`
