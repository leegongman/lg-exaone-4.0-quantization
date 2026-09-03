# Repository Structure

## Current Structure

This repository is currently organized as a documentation-first public portfolio repo.

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

## File Roles

| File | Role |
|---|---|
| `README.md` | First-screen project summary, status, scope, evidence policy, and attribution boundary |
| `docs/competition-overview.md` | Competition phases, evaluation constraints, and official-vs-internal result language |
| `docs/technical-inventory.md` | Full technical classification of model analysis, quantization, fine-tuning, evaluation, and vLLM work |
| `docs/notion-export-inventory.md` | Inventory of the full Notion export used as evidence; no raw export content |
| `docs/repository-structure.md` | Current and planned public repo layout |
| `docs/methodology.md` | Project workflow and technical decision process |
| `docs/experiments.md` | Experiment families, observed variants, and local-vs-competition result policy |
| `docs/reproducibility.md` | What can and cannot be reproduced from the public repo |
| `docs/project-status.md` | Claim status, publication risks, public claims table, and remaining actions |
| `docs/source-map.md` | Source attribution, reviewed/unreviewed repositories, include/exclude manifest |
| `environment.md` | Reference environment notes; not an install lockfile |
| `LICENSE` | Public repository license |

## Planned Future Structure

These directories are intentionally not populated yet. They should be added only after the contained files are cleaned and reviewed.

```text
patches/
  README.md
  vllm-exaone-sq.patch
  omniquant-vllm-adaptation.patch

scripts/
  README.md
  evaluate_local.py
  summarize_benchmark.py

benchmarks/
  README.md
  local-benchmark-summary.md
  environment-matrix.md

examples/
  README.md
  minimal-vllm-loading.md
  quantization-config-examples.md
```

## Rules For Adding Files

| Directory | Allowed content | Not allowed |
|---|---|---|
| `patches/` | Small reviewed patches, commit references, license-compatible diffs | Full vLLM, OmniQuant, AWQ, or copied upstream trees |
| `scripts/` | Cleaned scripts that run without private data, credentials, or local absolute paths | DACON token scripts, raw Notion code exports, notebooks with outputs |
| `benchmarks/` | Sanitized local benchmark summaries with hardware, package versions, tasks, and method labels | Raw private scores, private test data, unlabelled competition claims |
| `examples/` | Minimal examples using public references and placeholder paths | Model checkpoints, generated datasets, private submission files |

## Public Naming Convention

Use names that describe the method and evidence level:

- `local-benchmark-*` for local benchmark summaries
- `internal-record-*` for Notion/PDF/team records without official proof
- `patch-*` or `*.patch` for small implementation references
- `methodology-*` for documentation-only explanations

Avoid filenames that imply official results unless the file contains organizer-confirmed evidence.
