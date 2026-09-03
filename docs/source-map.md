# Source Map

## Purpose

This document maps the evidence used to build the clean public repository. It also defines what is included and excluded from the public version.

## Internal Sources

| Source | URL / Evidence | Use | Public handling |
|---|---|---|---|
| Internal Notion hackathon notes | Notion links provided by the project owner; local full export ZIP reviewed | Competition narrative, phase constraints, internal records, lessons learned, technical taxonomy, checkpoint labels, attached-code inventory | Summarized only; raw export excluded |
| Full Notion export ZIP | `3edca200-03b1-4e34-9b6d-23cfdfc0149d_ExportBlock-105ddcbe-7476-4d4c-8be0-6fa758dde4ac.zip`; 593 files in inner export | Full evidence inventory across Markdown pages, CSV tables, notebooks, scripts, configs, images, PDFs, and data files | Inventory summarized in `notion-export-inventory.md`; raw package excluded |
| Notion export: `Categories of Quantization Techniques` | Page found inside Notion export and in earlier small export ZIP | Quantization taxonomy and out-of-scope filtering categories | Summarized in `technical-inventory.md`; raw ZIP and image excluded |
| Exported PDF | `LGAimers.pdf` provided locally | Static reference for internal notes | Not included unless redacted and approved |
| Local work folder | Workspace file inventory and prior cleanup scans | Discovery of scripts, experiments, datasets, and sensitive files | Selected clean documentation only |

## Official Competition and Model References

| Source | URL / Evidence | Use | Public handling |
|---|---|---|---|
| DACON Phase 2 | [competition description](https://dacon.io/competitions/official/236673/overview/description) | Official task framing, hidden evaluation, model-weight submission path, public-data allowance | Link and paraphrase only; no private score or leaderboard export included |
| DACON Phase 3 | [competition description](https://dacon.io/competitions/official/236689/overview/description) | Final-stage eligibility, code-review context, model-engine/vLLM-wheel submission allowance | Link and paraphrase only; no private score or leaderboard export included |
| EXAONE 4.0 1.2B | [Hugging Face model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) | Canonical public model reference and loading context | Reference link only; weights and model artifacts excluded |

## Reviewed Public GitHub Sources

| Repository | Reviewed | URL / Evidence | Use |
|---|---:|---|---|
| `leegongman/scheme_vLLM_omniquant` | Yes | [repo](https://github.com/leegongman/scheme_vLLM_omniquant), [modified files](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/MODIFIED_FILES.md), [checkpoint generation](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/CHECKPOINT_GENERATION.md), [wheel build](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/WHEEL_BUILD.md) | OmniQuant/vLLM adaptation docs, modified-file map, checkpoint-generation notes, wheel-build notes |
| `leegongman/EXAONE_Quantization_method` | Yes | [repo](https://github.com/leegongman/EXAONE_Quantization_method), [setup notes](https://github.com/leegongman/EXAONE_Quantization_method/blob/main/SETUP.md) | Setup notes, environment context, local benchmark context |
| `leegongman/vLLM_Speed` | Yes | [repo](https://github.com/leegongman/vLLM_Speed) | Upstream OmniQuant baseline/reference only |
| `leegongman/vLLM_FP16_skip` | Yes | [repo](https://github.com/leegongman/vLLM_FP16_skip) | FP16 skip variant reference only |
| `leegongman/vllm` | Yes | [repo](https://github.com/leegongman/vllm), [EXAONE SQ registration commit](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e), [weight-name fix commit](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) | EXAONE SmoothQuant-style model registration and related commits |
| `leegongman/vLLM_OmniQuant` | No | [repo link not reviewed](https://github.com/leegongman/vLLM_OmniQuant) | Not reviewed; not used |
| `leegongman/vllm-exaone-sq` | No | [repo link not reviewed](https://github.com/leegongman/vllm-exaone-sq) | Not reviewed; not used |

## Attribution Rule

Code or documentation derived from upstream vLLM, OmniQuant, or other open-source projects must be clearly attributed. Full upstream source trees should not be copied into this clean repository.

If implementation evidence is needed, prefer:

- Links to reviewed commits
- Small patch files
- Minimal excerpts with license-compatible attribution
- Documentation summaries

## Clean Repo Manifest

### Include

- `README.md`
- `docs/competition-overview.md`
- `docs/model-architecture.md`
- `docs/technical-inventory.md`
- `docs/notion-export-inventory.md`
- `docs/repository-structure.md`
- `docs/methodology.md`
- `docs/experiments.md`
- `docs/reproducibility.md`
- `docs/project-status.md`
- `docs/source-map.md`
- `.gitignore`
- `LICENSE`
- `environment.md`
- `assets/README.md`
- `configs/README.md`
- `examples/README.md`
- `experiments/README.md`
- `patches/README.md`
- `schemas/README.md`
- `scripts/README.md`
- `techniques/README.md`
- Optional cleaned scripts after review
- Optional patch/reference files after license review

### Exclude

- DACON token or credential-bearing files
- DACON API fetch scripts containing secrets
- Raw secret pages
- Raw leaderboard HTML/JSON unless sanitized
- Private competition data
- Generated JSONL datasets
- Dataset ZIP files
- Raw CSV, XLSX, PDF, PNG, HTML, and notebook exports from Notion
- Model checkpoints
- Quantized checkpoints
- Tokenizer/model artifact directories
- Wheel files
- `dist/`, `build/`, and artifact directories
- Full vLLM source tree
- Full OmniQuant source tree
- `.venv`
- Cache files
- Notebook outputs
- Raw Notion exports
- OneDrive or private cloud links
- Certificates and personal screenshots
- Raw Python, shell, YAML, or JSON files from Notion before separate credential/path/license review
- `.DS_Store`, `__pycache__`, logs, and temporary files

## Export Handling Rule

Nested files inside the Notion export are evidence, not public repo content. Their filenames and method categories may be referenced. Their raw contents should not be copied unless a separate cleaning pass confirms that the file contains no private data, credentials, copyrighted upstream code, local absolute paths, notebook outputs, or competition artifacts.
