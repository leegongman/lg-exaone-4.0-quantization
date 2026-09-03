# Fine-Tuning and Distillation

## Status

**Partial implementation evidence.** This note summarizes fine-tuning, data construction, and compression work found in local notebooks, scripts, and internal records. Training data, generated JSONL files, adapters, checkpoints, and notebook outputs are intentionally excluded.

## LoRA and Data Work

| Area | Recorded work |
|---|---|
| Adapter targeting | LoRA experiments spanning `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` |
| Adapter settings | Internal notes record rank 16, alpha 16, and dropout 0.05 for selected experiments |
| Training variants | Baseline, sequential/curriculum, stacked-training, short-data, deep-data, and phase-set variants |
| Data preparation | Collection, merging, token-length filtering, DeepSeek-style formatting, Korean/English balancing, QA/CoT and multi-turn composition |
| Evaluation-oriented data | GSM8K, MMLU/KMMLU-related preparation and category-specific data work |
| Quantization combinations | Labels for LoRA plus INT8 and data/CoT plus INT8 experiments |

## Distillation and Layer Compression

The explored compression path included block distillation, teacher/student framing, layer drop, drop-last variants, and knowledge distillation on layer-dropped models. These are retained as attempted or partial work, not as a claim that a single distillation configuration produced a verified competition result.

## Claim Boundary

- The method inventory describes work attempted in the private project environment.
- It does not release data, weights, adapter artifacts, or training outputs.
- Local task evaluation is not the organizer evaluation environment.
- Fine-tuning and distillation are documented separately from PTQ because their data and training dependencies are materially different.

For the broader technical taxonomy, see [`docs/technical-inventory.md`](../docs/technical-inventory.md). For what can and cannot be reproduced, see [`docs/reproducibility.md`](../docs/reproducibility.md).
