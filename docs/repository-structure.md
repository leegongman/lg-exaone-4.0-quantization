# Repository Structure

## Design Goal

This public repository follows a conventional research-project layout: durable documentation at the center, with separate directories for future configurations, techniques, experiments, scripts, examples, assets, schemas, and small patches. It is intentionally not a monolithic dump of the original workspace.

The directory convention is informed by the structure of the project's [NVIDIA Nemotron reasoning repository](https://github.com/leegongman/nvidia-nemotron-reasoning), adapted for this project's safety boundary and attribution requirements.

## Current Layout

```text
README.md
environment.md
LICENSE
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
assets/
  README.md
configs/
  README.md
examples/
  README.md
experiments/
  README.md
patches/
  README.md
schemas/
  README.md
scripts/
  README.md
techniques/
  README.md
```

The directory guides are deliberate placeholders. They establish stable homes for reviewed public content without prematurely publishing code, model artifacts, or raw internal files.

## Directory Roles

| Directory or file | Purpose | Admission rule |
|---|---|---|
| `README.md` | Project narrative, high-level technical coverage, status, and reading path | No unsupported score or ranking claims |
| `environment.md` | Reference tool-family notes | Not an install lockfile; use pinned versions only after verification |
| `docs/` | Evidence-based methodology, architecture, scope, status, and provenance | Link to sources and label evidence level |
| `assets/` | Recreated public diagrams and sanitized figures | No screenshots with private or personally identifying information |
| `configs/` | Minimal, sanitized configuration examples | No raw checkpoint/model/tokenizer configuration or private paths |
| `examples/` | Small runnable or conceptual examples | No credentials, private data, checkpoints, or wheels |
| `experiments/` | Sanitized experiment manifests and result summaries | Every metric must state whether it is official, internal, local, failed, partial, or exploratory |
| `patches/` | Small license-reviewed diffs and commit references | No full vLLM, OmniQuant, AWQ, or other upstream tree |
| `schemas/` | Versioned schemas for public manifests/configs | Metadata only; no submission payloads or secrets |
| `scripts/` | Cleaned helper scripts | No DACON API/token code, local absolute paths, generated datasets, or copied upstream sources |
| `techniques/` | Method-specific notes | Link to source/evidence and distinguish implementation from research review |

## Recommended Growth Order

1. Add sanitized `techniques/` notes for GPTQ, AWQ, SmoothQuant, OmniQuant, LoRA, and vLLM integration.
2. Add one `configs/` example only after it is reconstructed from safe public fields rather than copied from a checkpoint artifact.
3. Add one `experiments/` manifest that captures method, target modules, calibration metadata, local hardware, and evidence status without scores or private data.
4. Add small `patches/` references to the reviewed vLLM commits after license/base-revision metadata is recorded.
5. Add cleaned `scripts/` only after a credential, path, data, and license scan.

## Excluded Structure

The following must not be recreated in this repository:

- Full vLLM or OmniQuant source trees
- DACON token/API helpers and submission payloads
- Checkpoints, model weights, tokenizers, wheels, or caches
- Raw notebooks, notebook outputs, and raw Notion export trees
- Private datasets, generated JSONL, raw CSV/XLSX, PDFs, screenshots, or leaderboard exports

The exact include/exclude manifest is maintained in [source-map.md](source-map.md).
