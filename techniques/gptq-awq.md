# GPTQ and AWQ Sweeps

## Status

**Attempted methods.** This note records broad schema, calibration, target-module, and internal-hyperparameter sweeps found in local evidence and internal notes. It does not publish raw notebooks, calibration data, checkpoints, submission files, or official scores.

## GPTQ

The GPTQ work was a sweep, not a single W4 baseline. Recorded variants span the following axes:

| Axis | Observed variants or decisions |
|---|---|
| Bit schemes | W4A16, W8A16, W8A8, W4A8 |
| Quantization structure | Block-size variants including BLK32 and BLK64; group-size-style variants |
| Calibration | Calibration-set revisions, sample-count comparisons including 256 and 512, and activation-analysis work |
| Targeting | Attention and MLP projections; Q/K/V/O and gate/up/down projection selection |
| Mixed precision | Front- and late-layer variants, W4/W8 mixtures, protected `down_proj` or `o_proj` choices |
| Protected modules | `lm_head` and embedding-related exclusions where appropriate |
| Runtime | vLLM compatibility and Marlin-kernel-related packaging/evaluation notes |

The public-safe configuration pattern is documented in [`configs/quantization-target-schema.example.yaml`](../configs/quantization-target-schema.example.yaml). It is an illustrative schema, not a final submission configuration.

## AWQ

The AWQ track also covered recipe construction and internal-parameter changes rather than a default run only.

| Axis | Observed variants or decisions |
|---|---|
| Bit schemes | W4A16, W8A16, W4A8, W8A8, W8A4 |
| Recipe scope | All-layer and selective-layer variants; tensor-oriented variant; prime variants |
| Sequence length | 256 and 512 sequence-length variants |
| Calibration | Sample-count and calibration-data changes |
| Targeting | Attention and MLP target-module mapping; Q/K/V/O and gate/up/down projections |
| Protected modules | `lm_head` and `embed_tokens` exclusions/protection |
| Mixed precision | W4/W8 recipes and W4 plus FP16 recipes |

## Claim Boundary

- These entries mean the variants were attempted or recorded in reviewed internal evidence.
- They do not mean every variant was submitted, successful, or retained.
- Local task results and latency measurements are not organizer scores.
- GPTQ and AWQ are external methods; this project documents experiment design and EXAONE/vLLM compatibility work rather than claiming ownership of either method.

See [`docs/technical-inventory.md`](../docs/technical-inventory.md) for the complete evidence classification and [`docs/experiments.md`](../docs/experiments.md) for result terminology.
