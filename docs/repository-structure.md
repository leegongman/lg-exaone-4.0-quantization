# Repository Structure

## Design Goal

This public repository follows a conventional research-project layout: durable documentation at the center, with separate directories for configurations, techniques, experiments, scripts, examples, assets, schemas, and small patches. It is intentionally not a monolithic dump of the original workspace.

The directory convention is informed by the structure of the project's [NVIDIA Nemotron reasoning repository](https://github.com/leegongman/nvidia-nemotron-reasoning), adapted for this project's safety boundary and attribution requirements.

## Current Layout

```text
README.md
environment.md
LICENSE
THIRD_PARTY_LICENSES.md
docs/
  competition-overview.md
  model-architecture.md
  methodology.md
  experiments.md
  technical-inventory.md
  notion-export-inventory.md
  reproducibility.md
  project-status.md
  source-map.md
  repository-structure.md
  file-selection-plan.md
  repository-consolidation.md
assets/
  README.md
configs/
  README.md
  quantization-target-schema.example.yaml
examples/
  README.md
experiments/
  README.md
  gptq-w4a16.example.yaml
patches/
  README.md
  licenses/
    OmniQuant-MIT.txt
    vLLM-Apache-2.0.txt
  omniquant-exaone/
    README.md
    0001-exaone-v4-omniquant-adaptation.patch
    legacy/0000-vllm-speed-exaone4-layer-v1.patch
  vllm-omni-activation-real/
    README.md
    0001-v0.14.1-omni-activation-real-runtime.patch
  vllm-exaone-sq/
    README.md
    0001-exaone-sq-model-path.patch
  vllm-fp16-skip/
    README.md
    legacy/omni_activation_real.py
  vllm-exaone-sq-reference.md
sources/
  README.md
  scheme_vLLM_omniquant/
  CP/
  EXAONE_Quantization_method/
  vLLM_Speed/
  vLLM_FP16_skip/
  vLLM_OmniQuant/
  vllm-exaone-sq/
schemas/
  README.md
  experiment-manifest.schema.json
scripts/
  README.md
  benchmark_vllm_tpt.py
techniques/
  README.md
  gptq-awq.md
  omniquant-smoothquant.md
  low-bit-and-mixed-precision.md
  fine-tuning-and-distillation.md
  vllm-runtime.md
```

The directory guides establish stable homes for reviewed public content and future additions without prematurely publishing model artifacts or raw internal files.

## Directory Roles

| Directory or file | Purpose | Admission rule |
|---|---|---|
| `README.md` | Project narrative, high-level technical coverage, status, and reading path | No unsupported score or ranking claims |
| `environment.md` | Reference tool-family notes | Not an install lockfile; use pinned versions only after verification |
| `docs/` | Evidence-based methodology, architecture, scope, status, and provenance | Link to sources and label evidence level |
| `assets/` | Recreated public diagrams and sanitized figures | No screenshots with private or personally identifying information |
| `configs/` | Minimal, sanitized configuration examples | Includes an illustrative target schema only; no raw checkpoint/model/tokenizer configuration or private paths |
| `examples/` | Small runnable or conceptual examples | No credentials, private data, checkpoints, or wheels |
| `experiments/` | Sanitized experiment manifests and result summaries | Includes a score-free GPTQ manifest example; every metric must state whether it is official, internal, local, failed, partial, or exploratory |
| `patches/` | License-attributed EXAONE OmniQuant and vLLM source deltas, plus provenance records | Exact upstream base, source revision, license, and exclusions are required; no full upstream tree |
| `sources/` | Source-only archive grouped by former repository | Tracked code, configs, tests, examples, and text documentation under 1 MiB; no data, weights, wheels, notebooks, output, credentials, or images |
| `schemas/` | Versioned schemas for public manifests/configs | Includes a sanitized experiment-manifest schema; metadata only, no submission payloads or secrets |
| `scripts/` | Cleaned helper scripts | Includes synthetic-prompt local timing only; no DACON API/token code, local absolute paths, generated datasets, or copied upstream sources |
| `techniques/` | Method-specific notes | Contains project-owned method summaries that link to evidence and distinguish implementation from research review |

## Next Intake Priorities

1. Add only new, independently reviewed self-authored deltas to the existing patch packages.
2. Add sanitized, score-free experiment manifests when their metadata can be separated from private data and submission artifacts.
3. Pin an environment only after a fresh compatible vLLM/OmniQuant build has been independently reproduced.

## Excluded Structure

The following must not be recreated in this repository:

- The standalone `leegongman/vllm` fork and Git histories from former repositories
- DACON token/API helpers and submission payloads
- Checkpoints, model weights, tokenizers, wheels, or caches
- Raw notebooks, notebook outputs, and raw Notion export trees
- Private datasets, generated JSONL, raw CSV/XLSX, PDFs, screenshots, or leaderboard exports

The exact include/exclude manifest is maintained in [source-map.md](source-map.md). The intake order and repository-by-repository selection rules are maintained in [file-selection-plan.md](file-selection-plan.md).
