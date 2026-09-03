# LG Aimers: EXAONE 4.0 1.2B Quantization and vLLM Optimization

This repository documents a competition-driven study of EXAONE 4.0 1.2B quantization, fine-tuning, and vLLM compatibility under private evaluation constraints.

The public repository is a curated technical portfolio, not a submission archive. It keeps the reasoning, experiment taxonomy, architecture mapping, and source attribution while excluding private data, credentials, checkpoints, wheels, raw Notion exports, notebook outputs, and full upstream source trees.

## Summary

| Item | Scope |
|---|---|
| Competition | LG Aimers AI Hackathon, `EXAONE-4.0-1.2B` lightweight-optimization track |
| Research goal | Balance model quality, inference efficiency, model size, and vLLM compatibility under hidden evaluation |
| Core work | Model analysis, GPTQ/AWQ/OmniQuant-family sweeps, calibration, fine-tuning, distillation, and runtime integration |
| Phase progression | Internal project records describe progression through earlier phases into the Phase 3 final-stage track; rank values remain unverified internal records |
| Public boundary | Documentation and future cleaned patches/scripts only |

## Status

| Area | Public status |
|---|---|
| Competition scores and rankings | Not presented as official results without organizer evidence |
| Local benchmarks | Useful for within-environment comparison; never presented as competition scores |
| Quantization experiments | Reviewed evidence of attempted work across multiple method families |
| vLLM customization | Reviewed implementation evidence; upstream framework remains attributed externally |
| Reproduction | Narrative and selected methods are reproducible; private submission conditions are not |

## Why This Project Matters

The study was not a single quantization run. It investigated the practical gap between a quantized checkpoint that looks promising locally and one that can be loaded, packaged, and evaluated reliably in a constrained vLLM environment. That required model-specific layer mapping, broad quantization sweeps, calibration design, runtime compatibility work, and a separate fine-tuning/compression track.

## Technical Coverage

| Area | Publicly documented scope |
|---|---|
| Model architecture | EXAONE 4.0 1.2B decoder structure, Q/K/V and MLP projection mapping, hidden-layer targeting, QK RMSNorm and tied-embedding considerations |
| GPTQ | W4/W8 weight and activation schemes; block/group size, calibration, sparse 2:4, KV, front/late-layer, and module-specific sweeps |
| AWQ | W4/W8 schemes; target-module maps, config groups, sequence length, calibration, protected modules, and internal-parameter variants |
| SmoothQuant and OmniQuant | Pre-Identity/SmoothQuant combinations, LET/LWC-oriented adaptation, checkpoint/runtime compatibility investigation |
| Other formats | INT4/INT8, FP8, FP8 block/dynamic, NVFP4, AutoRound, RTN, HQQ, GGUF/GGML, SpinQuant, QuaRot, and related exploratory tracks |
| Fine-tuning and compression | LoRA, data-format work, CoT-oriented data generation, block distillation, knowledge distillation, layer drop, and FP16/mixed-precision paths |
| Runtime and evaluation | vLLM custom model registration, wheel build investigation, `lm-eval`, throughput, latency, HF/vLLM evaluation paths |

The method-by-method evidence boundary is in [docs/technical-inventory.md](docs/technical-inventory.md), and the experiment-status distinction is in [docs/experiments.md](docs/experiments.md).

## Competition Phases

The official phase pages describe distinct submission surfaces:

- **Phase 2:** optimize and submit model weights for hidden evaluation; public data and preprocessing were allowed within the published rules. Project notes treat this as the organizer vLLM-runtime path without a participant-supplied runtime wheel.
- **Phase 3:** restricted to the Phase 2 award-candidate/finalist path and additionally allowed a model-engine artifact, including a vLLM wheel, with code review before final evaluation.

This distinction is central to the project: Phase 2 emphasized a vLLM-loadable model artifact, while Phase 3 created room for runtime-level customization. See [docs/competition-overview.md](docs/competition-overview.md) for source links, rule framing, and conservative result language.

## EXAONE Architecture and Layer Mapping

The target model is [LGAI-EXAONE/EXAONE-4.0-1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B). A reviewed experiment configuration records a 30-block decoder with hidden size 2048, 32 query heads, 8 KV heads, and a 4096-dimensional MLP intermediate layer.

The key implementation detail is **layer-specific configuration**, not a claim that the base EXAONE architecture was rewritten. Quantization configurations selected particular `model.layers.<index>` Q/K/V and gate/up/down projections, assigning different low-precision groups or protection rules by layer and module. The vLLM work then focused on registering an EXAONE-compatible model path and handling SmoothQuant-style parameter identities when loading compatible checkpoints.

Read [docs/model-architecture.md](docs/model-architecture.md) for the architecture diagram, module map, and customization boundary.

## Repository Layout

The structure follows a conventional research repository layout: top-level overview and environment notes, durable documentation, then dedicated locations for cleaned future artifacts.

```text
README.md
environment.md                 # Reference environment only; not an install lockfile
docs/                          # Competition, methodology, architecture, evidence, safety
assets/                        # Reviewed diagrams or images only
configs/                       # Sanitized, reproducible configuration examples
examples/                      # Minimal public usage examples
experiments/                   # Sanitized experiment manifests and result summaries
patches/                       # Small attributed patches or commit references
schemas/                       # Public schema definitions for manifests/configs
scripts/                       # Clean helper scripts without secrets or private paths
techniques/                    # Method notes linked to evidence and sources
LICENSE
```

Only the documentation and directory guides are present today. Future files are admitted only after the safety and attribution review described in [docs/source-map.md](docs/source-map.md).

## Reading Guide

| Goal | Document |
|---|---|
| Understand the competition and Phase 2/3 distinction | [docs/competition-overview.md](docs/competition-overview.md) |
| Inspect the model and hidden-layer targeting approach | [docs/model-architecture.md](docs/model-architecture.md) |
| Audit every technique family found in the evidence | [docs/technical-inventory.md](docs/technical-inventory.md) |
| Separate local benchmarks, internal records, and failed work | [docs/experiments.md](docs/experiments.md) |
| Understand implementation choices | [docs/methodology.md](docs/methodology.md) |
| See what can actually be reproduced | [docs/reproducibility.md](docs/reproducibility.md) |
| Check claims before publication | [docs/project-status.md](docs/project-status.md) |
| Trace sources, attribution, and exclusions | [docs/source-map.md](docs/source-map.md) |

## Evidence and Attribution Policy

Every reported item is labeled as one of the following:

- **Official competition result:** organizer or leaderboard evidence is attached.
- **Unverified internal record:** team/Notion/PDF record without official public proof.
- **Local benchmark:** a result from local scripts or public benchmark tasks, not a competition score.
- **Failed, partial, or exploratory experiment:** an attempted direction without a conclusive competition outcome.

vLLM, OmniQuant, AWQ, GPTQ, and other external projects are credited as external frameworks or methods. This repository claims only the supported EXAONE-specific analysis, experiment design, integration work, selected implementation changes, and documentation.

## Public Safety Boundary

The repository intentionally excludes DACON tokens, private data, generated JSONL, checkpoints, tokenizer/model artifacts, wheels, raw Notion exports, notebook outputs, local caches, submission files, and full vLLM/OmniQuant trees. The DACON token found in the original workspace must be revoked or rotated before any related history is shared.
