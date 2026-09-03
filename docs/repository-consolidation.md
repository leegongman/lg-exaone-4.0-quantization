# Repository Consolidation

## Purpose

This repository consolidates the project-owned EXAONE quantization and vLLM integration changes that were previously spread across several repositories. It retains public-safe source-only snapshots under [`sources/`](../sources/README.md) and preserves focused, attributable implementation deltas under [`patches/`](../patches/README.md).

The full `leegongman/vllm` fork is excluded. All other reviewed repositories are represented below either by an implementation patch, a documentation record, or a deduplicated evidence reference.

## Consolidation Map

| Former repository | Reviewed revision | What is preserved here | What is excluded |
|---|---|---|---|
| [`scheme_vLLM_omniquant`](https://github.com/leegongman/scheme_vLLM_omniquant) | `5178a0e5a1c359dd39b8e2457b16036303ab570d` | [Source-only snapshot](../sources/scheme_vLLM_omniquant/) plus EXAONE OmniQuant patch and build/checkpoint documentation | Checkpoints, build artifacts, notebooks, data, images, and outputs |
| [`CP`](https://github.com/leegongman/CP) | `ea6bcaba99703fbe82c09ddb84b408001b43eadb` | [Source-only snapshot](../sources/CP/), [EXAONE OmniQuant](../patches/omniquant-exaone/README.md), and [Omni activation-real vLLM](../patches/vllm-omni-activation-real/README.md) patches | Checkpoints, logs, wheels, datasets, images, and generated artifacts |
| [`vLLM_OmniQuant`](https://github.com/leegongman/vLLM_OmniQuant) | `7c96a51b71d62c6a19fd99b8a06c59d932df6566` | [Public-safe source/docs archive](../sources/vLLM_OmniQuant/) and provenance for the custom runtime wheel | Split wheel files and model artifact |
| [`EXAONE_Quantization_method`](https://github.com/leegongman/EXAONE_Quantization_method) | `216e1b2f256a640cdd4b967aeaa6539258ae006a` | [Source-only snapshot](../sources/EXAONE_Quantization_method/) plus current EXAONE OmniQuant patch evidence | Checkpoints, data, logs, notebooks, images, and output artifacts |
| [`vLLM_Speed`](https://github.com/leegongman/vLLM_Speed) | `f70db19b276ab8ca4707c6825b7d440dd39bb9b4` | [Source-only snapshot](../sources/vLLM_Speed/) and earlier EXAONE quantized-layer revision in [`patches/omniquant-exaone/legacy/`](../patches/omniquant-exaone/legacy/) | Notebooks, benchmark outputs, images, cache, and model artifacts |
| [`vLLM_FP16_skip`](https://github.com/leegongman/vLLM_FP16_skip) | `fd6f373b17aadfeaadb027beecff4c7294c850ca` | [Source-only snapshot](../sources/vLLM_FP16_skip/) and [reviewed-snapshot record](../patches/vllm-fp16-skip/README.md) | No isolated `fp16_skip` implementation was found or asserted; history, images, and artifacts excluded |
| [`vllm`](https://github.com/leegongman/vllm) | EXAONE source commits `d0d38db63090185dfb8ca131def8d04146de9e5e` and `d8fe813d6715869fa85471e5a3d3631c599b7e72` | [Model-path patch](../patches/vllm-exaone-sq/README.md) and separate weight-name compatibility patch | Entire fork, including all upstream vLLM files |
| [`vllm-exaone-sq`](https://github.com/leegongman/vllm-exaone-sq) | `17411cee9275e317b6674c8190c2a88cd4a56e46` | [Source-only snapshot](../sources/vllm-exaone-sq/) and [EXAONE SmoothQuant model-path patch](../patches/vllm-exaone-sq/README.md) | Wheel, images, artifacts, and history |

## Source Deltas Kept in This Repository

| Patch package | Role | Upstream base | License |
|---|---|---|---|
| [`patches/omniquant-exaone/`](../patches/omniquant-exaone/README.md) | EXAONE model loading, quantized decoder layers, activation-real export, OmniQuant loop changes, and smoke/evaluation helpers | OpenGVLab OmniQuant `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | MIT |
| [`patches/vllm-omni-activation-real/`](../patches/vllm-omni-activation-real/README.md) | vLLM registration and runtime support for the activation-real format | vLLM `v0.14.1`, `d7de043d55d1dd629554467e23874097e1c48993` | Apache-2.0 |
| [`patches/vllm-exaone-sq/`](../patches/vllm-exaone-sq/README.md) | `Exaone4ForCausalLMSQ`, hidden-state scaling modules, and model registration | vLLM `600a039f572ac28128750f0463af428c5a260f1a` | Apache-2.0 |

The project-owned Markdown, configuration examples, schema, and cleaned benchmark helper remain under the top-level repository MIT license. Patch packages retain their upstream license obligations; see [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

## Deliberate Non-Goals

- This is not a history-preserving mirror of the former repositories; it retains filtered working trees, not Git history.
- This is not an installable replacement for either vLLM or OmniQuant.
- It contains no competition submission, checkpoint, wheel, dataset, API credential, raw Notion export, notebook output, image, or local benchmark result.
- A patch demonstrates an implementation delta, not an organizer-verified competition result.
