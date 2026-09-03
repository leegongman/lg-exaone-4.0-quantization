# File Selection Plan

## Purpose

This document defines the admission rules used to consolidate source material distributed across the project's EXAONE and vLLM repositories. The completed repository-by-repository disposition is in [repository-consolidation.md](repository-consolidation.md).

The public repository is a curated study, not a merge of the original workspace or of prior repositories. A candidate file is admitted only after source ownership, license, credentials, private paths, data dependencies, and artifact status have been checked.

## Selection Rules

| Class | Public handling | Examples |
|---|---|---|
| Documentation evidence | Summarize in project-owned Markdown and link to the original source | Method scope, experiment categories, build constraints |
| Self-authored patch candidate | Keep only a small focused diff, with upstream base revision, license notice, and rationale | EXAONE model registration or a FP16-skip behavior change |
| Clean-script candidate | Reconstruct or clean a small standalone helper after a secret/path/data scan | Configuration validation or local benchmark-summary helper |
| External dependency | Keep a link and attribution; do not represent it as original work | vLLM, OmniQuant, GPTQ, AWQ, and their kernels |
| Excluded artifact | Do not add to this repository | Checkpoints, wheels, datasets, notebooks, caches, full source trees |

## Repository-by-Repository Decision Matrix

| Source repository | License observed | Evidence reviewed | Candidate public material | Decision |
|---|---|---|---|---|
| [`leegongman/vllm`](https://github.com/leegongman/vllm) | Apache-2.0 upstream fork | EXAONE SQ commits [`d0d38db`](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e) and [`d8fe813`](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) | Provenance for the narrow `Exaone4ForCausalLMSQ` patch | Full fork permanently excluded; patch retained with Apache-2.0 attribution. |
| [`leegongman/vLLM_FP16_skip`](https://github.com/leegongman/vLLM_FP16_skip) | Apache-2.0 | Full snapshot at `fd6f373b17aadfeaadb027beecff4c7294c850ca` | Reviewed-snapshot record and selected earlier `omni_activation_real` runtime source | No isolated FP16-skip delta was identified. Do not copy the source tree or claim an implementation absent from the snapshot. |
| [`leegongman/vLLM_Speed`](https://github.com/leegongman/vLLM_Speed) | MIT OmniQuant snapshot | Full snapshot at `f70db19b276ab8ca4707c6825b7d440dd39bb9b4` | Earlier EXAONE layer implementation patch | Historical patch retained. Notebooks, benchmark outputs, and artifacts are excluded. |
| [`leegongman/scheme_vLLM_omniquant`](https://github.com/leegongman/scheme_vLLM_omniquant) | First-party snapshot containing MIT/Apache dependencies | Method and source snapshot at `5178a0e5a1c359dd39b8e2457b16036303ab570d` | Duplicate evidence for extracted patches and project-owned summaries | Do not copy source snapshot. Preserve only deduplicated deltas with upstream notices. |
| [`leegongman/CP`](https://github.com/leegongman/CP) | First-party snapshot containing MIT/Apache dependencies | Canonical source snapshot at `ea6bcaba99703fbe82c09ddb84b408001b43eadb` | Focused EXAONE OmniQuant and vLLM activation-real patches | Do not copy `sources/`, `patched_sources/`, checkpoints, logs, or complete third-party trees. |
| [`leegongman/EXAONE_Quantization_method`](https://github.com/leegongman/EXAONE_Quantization_method) | First-party snapshot containing MIT/Apache dependencies | Source snapshot at `216e1b2f256a640cdd4b967aeaa6539258ae006a` | Duplicate evidence for the current EXAONE OmniQuant patch and technique documentation | Do not copy source snapshot, experiment artifacts, data, or logs. |
| [`leegongman/vLLM_OmniQuant`](https://github.com/leegongman/vLLM_OmniQuant) | Reviewed private wheel archive | Private repository at `7c96a51b71d62c6a19fd99b8a06c59d932df6566` | Provenance for the activation-real vLLM patch | Wheel archive, split parts, and validation output excluded. |
| [`leegongman/vllm-exaone-sq`](https://github.com/leegongman/vllm-exaone-sq) | Apache-2.0 vLLM snapshot | Private repository at `17411cee9275e317b6674c8190c2a88cd4a56e46` | `Exaone4ForCausalLMSQ` model-path patch | Full snapshot, wheel, and build system excluded. |

## Public Repository Destinations

| Destination | Planned content | Admission bar |
|---|---|---|
| `docs/technical-inventory.md` | Complete classification of attempted, partial, exploratory, and upstream methods | Evidence level must be stated for every claim |
| `techniques/` | One concise project-owned note per method family: GPTQ, AWQ, SmoothQuant, OmniQuant, mixed precision, LoRA, distillation, and runtime compatibility | No raw Notion export, copied upstream prose, private score, or raw configuration |
| `patches/` | Attribution-rich patch packages for the reviewed EXAONE OmniQuant, vLLM activation-real, and EXAONE SmoothQuant changes | License, upstream base, source commit, purpose, and excluded context required |
| `scripts/` | Sanitized utilities only, such as config-schema validation or local benchmark-result formatting | Must run without DACON access, private data, checkpoints, local absolute paths, or copied upstream modules |
| `configs/` | Reconstructed example schemas showing public-safe fields such as target modules, bit-widths, and group size | No checkpoint configuration dump, tokenizer artifact, private path, or competition payload |
| `experiments/` | Sanitized manifests that distinguish official score, internal record, local benchmark, failed experiment, and exploration | No raw logs, leaderboard exports, metrics files, or submission bundles |
| `assets/` | Recreated diagrams and publicly licensed assets | No private screenshots, Notion-export images, or personal data |

## Completed Intake

1. Added project-owned technique notes and sanitized experiment-manifest examples that distinguish official results, internal records, local benchmarks, failures, and exploration.
2. Extracted the EXAONE OmniQuant and vLLM activation-real deltas from CP, with duplicate evidence checked against the related source snapshots.
3. Preserved the earlier `vLLM_Speed` EXAONE-layer revision as a historical patch rather than as a complete source tree.
4. Preserved the `Exaone4ForCausalLMSQ` model path and registry delta from `vllm-exaone-sq`, while excluding the full `vllm` fork.
5. Recorded `vLLM_FP16_skip` as reviewed but did not invent an FP16-skip patch when no isolated implementation was found.

## Permanent Exclusions

The following remain out of scope even when they are found in a reviewed repository:

- Full vLLM, OmniQuant, GPTQ, AWQ, or other third-party source trees
- Model checkpoints, quantized weights, tokenizer files, wheels, build outputs, and caches
- DACON credentials, tokens, API scripts, submission bundles, leaderboards, and private evaluation files
- Training or calibration datasets, JSONL/CSV/Parquet files, generated data, notebook files, and notebook outputs
- Shell history, local paths, cloud machine setup details, personal screenshots, and raw Notion exports

## Required Check Before Each Admission

- Confirm source repository and file-level provenance.
- Confirm that the license allows the intended use, or keep only a link and a project-owned summary.
- Scan for secrets, DACON tokens, credentials, private URLs, local absolute paths, and personal information.
- Exclude data, checkpoints, wheels, generated output, and large binaries.
- State whether the item is original work, a modification, an external dependency, or a research reference.
- Update [`source-map.md`](source-map.md) with the final public location and evidence link.
