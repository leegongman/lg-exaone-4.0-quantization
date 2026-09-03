# Technique Notes

These project-owned notes organize the broad technique inventory without publishing raw notebooks, data, checkpoints, or copied upstream source.

| Note | Scope | Evidence label |
|---|---|---|
| [GPTQ and AWQ sweeps](gptq-awq.md) | Schema, calibration, target-module, and internal-hyperparameter variations | Attempted |
| [OmniQuant and SmoothQuant paths](omniquant-smoothquant.md) | EXAONE adaptation and runtime-compatibility work | Adaptation / implementation evidence reviewed |
| [Low-bit and mixed precision](low-bit-and-mixed-precision.md) | Integer, floating-point, rotation, outlier, and feasibility tracks | Attempted, partial, or exploratory by method |
| [Fine-tuning and distillation](fine-tuning-and-distillation.md) | LoRA, data construction, KD, and layer compression | Partial implementation evidence |
| [vLLM runtime compatibility](vllm-runtime.md) | Model registration, loader work, wheel path, and local timing | Implementation evidence reviewed |

The evidence inventory and public-claim boundaries remain in [docs/technical-inventory.md](../docs/technical-inventory.md) and [docs/project-status.md](../docs/project-status.md).
