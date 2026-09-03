# Validation

Wheel under test:

- `vllm-0.14.1-cp311-cp311-linux_x86_64.whl`

Validated environment:

- `python 3.11.15`
- `torch 2.9.0+cu128`
- `vllm 0.14.1`

Runtime checks:

- import success
- wheel install with dependency pins preserved
- `vllm serve` success
- OpenAI-compatible chat completion response success

Checkpoint check:

- model: `w6a6_cp_10ep_noamp`
- `norm_speed`: `+4.93%`
- quant `bench_real_latency`: `2188.765 tok/s`
- baseline `bench_real_latency`: `1713.817 tok/s`
- `gsm8k 100`: flexible `0.57`, strict `0.54`
