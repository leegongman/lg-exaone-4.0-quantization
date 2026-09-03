# Repository Consolidation

## Purpose

This repository consolidates the project-owned EXAONE quantization and vLLM integration changes that were previously spread across several repositories. It deliberately preserves focused, attributable implementation deltas instead of merging complete source snapshots.

The full `leegongman/vllm` fork is excluded. All other reviewed repositories are represented below either by an implementation patch, a documentation record, or a deduplicated evidence reference.

## Consolidation Map

| Former repository | Reviewed revision | What is preserved here | What is excluded |
|---|---|---|---|
| [`scheme_vLLM_omniquant`](https://github.com/leegongman/scheme_vLLM_omniquant) | `5178a0e5a1c359dd39b8e2457b16036303ab570d` | Duplicate evidence for the current EXAONE OmniQuant patch and build/checkpoint documentation | Full OmniQuant/vLLM snapshots, checkpoints, build artifacts, notebooks, data, outputs |
| [`CP`](https://github.com/leegongman/CP) | `ea6bcaba99703fbe82c09ddb84b408001b43eadb` | Canonical source snapshot for the [EXAONE OmniQuant](../patches/omniquant-exaone/README.md) and [Omni activation-real vLLM](../patches/vllm-omni-activation-real/README.md) patches | `sources/`, `patched_sources/`, checkpoints, logs, wheels, datasets, and complete third-party trees |
| [`vLLM_OmniQuant`](https://github.com/leegongman/vLLM_OmniQuant) | `7c96a51b71d62c6a19fd99b8a06c59d932df6566` | Provenance record for the custom runtime wheel; matching source delta is retained from CP | Split wheel files, reconstruction output, model artifact, local validation results |
| [`EXAONE_Quantization_method`](https://github.com/leegongman/EXAONE_Quantization_method) | `216e1b2f256a640cdd4b967aeaa6539258ae006a` | Duplicate evidence for the current EXAONE OmniQuant delta and method documentation | Full source snapshot, checkpoints, data, logs, notebooks, output artifacts |
| [`vLLM_Speed`](https://github.com/leegongman/vLLM_Speed) | `f70db19b276ab8ca4707c6825b7d440dd39bb9b4` | Earlier EXAONE quantized-layer revision in [`patches/omniquant-exaone/legacy/`](../patches/omniquant-exaone/legacy/) | Full snapshot, notebooks, benchmark outputs, cache and model artifacts |
| [`vLLM_FP16_skip`](https://github.com/leegongman/vLLM_FP16_skip) | `fd6f373b17aadfeaadb027beecff4c7294c850ca` | [Reviewed-snapshot record](../patches/vllm-fp16-skip/README.md) and selected earlier `omni_activation_real` source | Entire vLLM snapshot; no isolated `fp16_skip` implementation was found and none is asserted here |
| [`vllm`](https://github.com/leegongman/vllm) | EXAONE source commits `d0d38db63090185dfb8ca131def8d04146de9e5e` and `d8fe813d6715869fa85471e5a3d3631c599b7e72` | Commit provenance for the SmoothQuant model-path patch | Entire fork, including all upstream vLLM files |
| [`vllm-exaone-sq`](https://github.com/leegongman/vllm-exaone-sq) | `17411cee9275e317b6674c8190c2a88cd4a56e46` | [EXAONE SmoothQuant model-path patch](../patches/vllm-exaone-sq/README.md) | Entire vLLM snapshot, build system, wheel, model artifacts |

## Source Deltas Kept in This Repository

| Patch package | Role | Upstream base | License |
|---|---|---|---|
| [`patches/omniquant-exaone/`](../patches/omniquant-exaone/README.md) | EXAONE model loading, quantized decoder layers, activation-real export, OmniQuant loop changes, and smoke/evaluation helpers | OpenGVLab OmniQuant `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` | MIT |
| [`patches/vllm-omni-activation-real/`](../patches/vllm-omni-activation-real/README.md) | vLLM registration and runtime support for the activation-real format | vLLM `v0.14.1`, `d7de043d55d1dd629554467e23874097e1c48993` | Apache-2.0 |
| [`patches/vllm-exaone-sq/`](../patches/vllm-exaone-sq/README.md) | `Exaone4ForCausalLMSQ`, hidden-state scaling modules, and model registration | vLLM `600a039f572ac28128750f0463af428c5a260f1a` | Apache-2.0 |

The project-owned Markdown, configuration examples, schema, and cleaned benchmark helper remain under the top-level repository MIT license. Patch packages retain their upstream license obligations; see [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

## Deliberate Non-Goals

- This is not a history-preserving mirror of the former repositories.
- This is not an installable replacement for either vLLM or OmniQuant.
- It contains no competition submission, checkpoint, wheel, dataset, API credential, raw Notion export, notebook output, or local benchmark result.
- A patch demonstrates an implementation delta, not an organizer-verified competition result.
