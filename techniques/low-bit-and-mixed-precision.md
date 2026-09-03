# Low-Bit and Mixed-Precision Exploration

## Status

This note separates direct low-bit experiment evidence from research survey or feasibility work. It does not publish model artifacts, generated data, raw notebooks, CUDA kernels, or benchmark output.

## Direct or Locally Documented Experiment Tracks

| Family | Recorded variants | Evidence level |
|---|---|---|
| Integer quantization | INT8 baseline and INT4 experimentation | Attempted |
| Mixed precision | W4/W8 layer and module mixes, W4 plus FP16, protected modules, front/late-layer precision variants | Attempted or partial |
| FP8 and NVFP4 | FP8, FP8-block, FP8-dynamic, and NVFP4 exploration | Attempted or exploratory |
| HQQ | W4A16 and W8A16 | Attempted |
| GGUF/GGML | Q4-oriented paths | Attempted or exploratory |
| RTN | W4A16 and RTN-XK W4A16 | Attempted |
| AutoRound | W4A16 and KV-related variant | Attempted |
| Rotation and sparse paths | SpinQuant plus INT4; sparse 2:4 comparison material | Attempted or partial |
| Outlier-aware paths | SqueezeLLM 4-bit CUDA evaluation material | Partial or exploratory |

## Research and Feasibility Survey

The Phase 3 research survey also covered the following methods for suitability with EXAONE, Hugging Face tooling, vLLM, or LLM Compressor compatibility. Their presence here must not be read as evidence of a completed final implementation.

- ZeroQuant, OWQ, SpQR, Slim-LLM, MPPQ
- QuaRot, QuIP, OneBit, VPTQ
- DuQuant, PrefixQuant, LRQuant
- APTQ, TurboQuant, PT2-LLM

## Selection Principle

Mixed precision was treated as a deployment choice across layers and modules, rather than merely a global bit-width change. The recurring considerations were accuracy-sensitive module protection, activation behavior, group/block configuration, kernel/runtime compatibility, and the time required to package and load a submission artifact.

No result in this note is an official competition score. See [`docs/technical-inventory.md`](../docs/technical-inventory.md) for evidence detail and [`docs/project-status.md`](../docs/project-status.md) for public-claim status.
