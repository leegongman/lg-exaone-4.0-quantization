# EXAONE 4.0 1.2B Quantization and vLLM Optimization Study

This repository documents a competition-driven study of EXAONE 4.0 1.2B quantization, fine-tuning, and vLLM compatibility under private evaluation constraints.

This repository is a curated portfolio version of the work. It does not include private datasets, credentials, checkpoints, wheel files, raw Notion exports, notebook outputs, or full upstream source trees.

## Summary

| Item | Summary |
|---|---|
| Project | LG Aimers AI Hackathon lightweight LLM optimization study |
| Target model | `EXAONE-4.0-1.2B` |
| Main themes | Quantization sweeps, fine-tuning, model analysis, vLLM compatibility |
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

## Project Scope

The project investigated how to reduce model size and improve inference efficiency while preserving useful model quality under competition constraints.

The work covered:

- EXAONE 4.0 1.2B architecture analysis
- GPTQ, AWQ, AutoRound, INT, FP8/NVFP4, RTN/HQQ/GGUF-style experiment tracks
- SmoothQuant-style, OmniQuant, and outlier-redistribution adaptation attempts
- broad schema and hyperparameter sweeps across bit-width, group/block size, calibration, ignored modules, and target layers
- LoRA fine-tuning and dataset-format experiments
- vLLM model registration and runtime compatibility work
- Local benchmark and latency-oriented evaluation scripts

## Repository Layout

```text
README.md
docs/
  competition-overview.md
  technical-inventory.md
  methodology.md
  experiments.md
  reproducibility.md
  project-status.md
  source-map.md
environment.md
LICENSE
```

## Evidence Policy

This repository separates claims into four categories:

- Official competition result: leaderboard or organizer-confirmed evidence
- Internal record: Notion notes, exported PDF, or team records
- Local benchmark: local scripts, public benchmark tasks, or experiment logs
- Failed or partial experiment: attempted methods without conclusive competition impact

Competition ranking records currently remain internal records. Detailed claim status is maintained in `docs/project-status.md`.

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
