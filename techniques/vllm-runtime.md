# vLLM Runtime Compatibility

## Status

**Implementation evidence reviewed.** This work addresses deployment compatibility for EXAONE quantized checkpoints under the competition's constrained runtime. It does not claim ownership of vLLM, its upstream model implementations, or quantization algorithms.

## Work Areas

| Area | Recorded work |
|---|---|
| EXAONE registration | Custom `Exaone4ForCausalLMSQ` architecture registration for a distinct model path |
| Hidden-state modules | Parameterized identity modules placed before attention and MLP computation for smooth-factor handling |
| Loading compatibility | Architecture-name, registry, and checkpoint parameter-name compatibility investigation |
| Quantization runtime | OmniQuant/SmoothQuant-style checkpoint and packed-representation compatibility investigation |
| Packaging | Custom vLLM wheel build and submission-path investigation |
| Performance workflow | Local token-timing measurement using vLLM, separate from official competition scoring |

## Public Evidence

Focused code evidence is preserved as the [EXAONE SmoothQuant model-path patch](../patches/vllm-exaone-sq/README.md) and the [Omni activation-real runtime patch](../patches/vllm-omni-activation-real/README.md), with the reviewed commit links retained in [`patches/vllm-exaone-sq-reference.md`](../patches/vllm-exaone-sq-reference.md). The model-flow explanation is in [`docs/model-architecture.md`](../docs/model-architecture.md).

The included [`scripts/benchmark_vllm_tpt.py`](../scripts/benchmark_vllm_tpt.py) supports local synthetic-prompt timing. It reports `local-benchmark` metadata and cannot reproduce the organizer runtime, private data, full submission pipeline, or official score.

## Excluded Material

- Full vLLM source tree and build system
- Full OmniQuant source tree
- Wheels, checkpoints, packed weights, and cache directories
- Competition submission payloads and organizer runtime files
- Private evaluation inputs and raw timing logs
