# EXAONE 4.0 1.2B Quantization and vLLM Optimization Study

This repository documents a competition-driven study of EXAONE 4.0 1.2B quantization, fine-tuning, and vLLM compatibility under private evaluation constraints.

This repository is a curated portfolio version of the work. It does not include private datasets, credentials, checkpoints, wheel files, raw Notion exports, notebook outputs, or full upstream source trees.

## Summary

| Item | Summary |
|---|---|
| Project | LG Aimers AI Hackathon lightweight LLM optimization study |
| Target model | `EXAONE-4.0-1.2B` |
| Main themes | Quantization sweeps, calibration/data work, fine-tuning, model analysis, vLLM compatibility |
| Public scope | Documentation, selected methodology, source map, reproducibility notes |
| Excluded scope | Private data, credentials, checkpoints, wheels, full vLLM/OmniQuant trees |

## Status

| Area | Public status |
|---|---|
| Competition ranking records | Internal records only until official evidence is attached |
| Competition scores | Not published as verified claims in this repo draft |
| Local benchmark results | Documented separately from competition scores |
| vLLM customization | Supported by reviewed public commits/docs |
| Upstream code | Referenced and attributed, not claimed as original work |

## What To Read First

| Reader goal | Start here |
|---|---|
| Understand the project quickly | `README.md`, then [competition-overview.md](docs/competition-overview.md) |
| Check the technical scope | [technical-inventory.md](docs/technical-inventory.md) |
| See what the Notion export contributed | [notion-export-inventory.md](docs/notion-export-inventory.md) |
| Understand the repo layout | [repository-structure.md](docs/repository-structure.md) |
| Review experiment categories | [experiments.md](docs/experiments.md) |
| Check claim safety before publishing | [project-status.md](docs/project-status.md) |
| Trace sources and exclusions | [source-map.md](docs/source-map.md) |

## Project Scope

The project investigated how to reduce model size and improve inference efficiency while preserving useful model quality under competition constraints.

The work covered:

- EXAONE 4.0 1.2B architecture analysis
- GPTQ, AWQ, AutoRound, INT4/INT8, FP8/NVFP4, RTN/HQQ/GGUF/GGML-style experiment tracks
- SmoothQuant-style, OmniQuant, SpinQuant/QuaRot, SqueezeLLM, TurboQuant, and outlier-redistribution adaptation reviews or attempts
- broad schema and hyperparameter sweeps across bit-width, activation format, group/block size, calibration, sequence length, ignored modules, and target layers
- LoRA fine-tuning and dataset-format experiments
- knowledge-distillation and layer-drop experiments
- vLLM model registration and runtime compatibility work
- Local benchmark and latency-oriented evaluation scripts

## Repository Layout

```text
README.md
docs/
  competition-overview.md
  technical-inventory.md
  notion-export-inventory.md
  repository-structure.md
  methodology.md
  experiments.md
  reproducibility.md
  project-status.md
  source-map.md
environment.md
LICENSE
```

Future public additions should follow this structure:

```text
patches/       # Small reviewed patches or diff references, not full upstream trees
scripts/       # Cleaned helper scripts with secrets, private paths, and outputs removed
benchmarks/    # Sanitized local benchmark summaries with environment context
examples/      # Minimal examples that do not require private data or checkpoints
```

Those directories should be added only when their contents pass the public-safety checks in `docs/source-map.md`.

## Evidence Policy

This repository separates claims into four categories:

- Official competition result: leaderboard or organizer-confirmed evidence
- Internal record: Notion notes, exported PDF, or team records
- Local benchmark: local scripts, public benchmark tasks, or experiment logs
- Failed or partial experiment: attempted methods without conclusive competition impact

Competition ranking records currently remain internal records. Detailed claim status is maintained in [project-status.md](docs/project-status.md). The full Notion export is treated as evidence, not as executable instruction content.

## Attribution

This project used and studied external open-source systems, including vLLM and OmniQuant-derived codebases. Those projects remain credited to their original authors. This repository only claims original analysis, integration work, experiment design, selected implementation changes, and documentation where supported by evidence.

## Public Safety Boundary

The clean repository intentionally excludes:

- DACON tokens or credential-bearing files
- Private datasets and generated training data
- Model checkpoints and tokenizer/model artifacts
- Quantized checkpoints and wheel files
- Full vLLM and OmniQuant source trees
- Notebook outputs and raw Notion exports
- Large caches, logs, and temporary files
