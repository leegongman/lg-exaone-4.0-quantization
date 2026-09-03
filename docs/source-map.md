# Source Map

## Purpose

This document maps the evidence used to build the clean public repository. It also defines what is included and excluded from the public version.

## Internal Sources

| Source | Use | Public handling |
|---|---|---|
| Internal Notion hackathon notes | Competition narrative, phase constraints, internal records, lessons learned | Summarized only; raw export excluded |
| Exported PDF | Static reference for internal notes | Not included unless redacted and approved |
| Local work folder | Discovery of scripts, experiments, datasets, and sensitive files | Selected clean documentation only |

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
- `docs/methodology.md`
- `docs/experiments.md`
- `docs/reproducibility.md`
- `docs/project-status.md`
- `docs/source-map.md`
- `.gitignore`
- `LICENSE`
- `environment.md`
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
- `.DS_Store`, `__pycache__`, logs, and temporary files
