# Source Map

## Purpose

This document maps the evidence used to build the clean public repository. It also defines what is included and excluded from the public version.

## Internal Sources

| Source | URL / Evidence | Use | Public handling |
|---|---|---|---|
| Internal Notion hackathon notes | Notion links provided by the project owner; local full export ZIP reviewed | Competition narrative, phase constraints, internal records, lessons learned, technical taxonomy, checkpoint labels, attached-code inventory | Summarized only; raw export excluded |
| Full Notion export ZIP | `3edca200-03b1-4e34-9b6d-23cfdfc0149d_ExportBlock-105ddcbe-7476-4d4c-8be0-6fa758dde4ac.zip`; 593 files in inner export | Full evidence inventory across Markdown pages, CSV tables, notebooks, scripts, configs, images, PDFs, and data files | Inventory summarized in `notion-export-inventory.md`; raw package excluded |
| Notion export: `Categories of Quantization Techniques` | Page found inside Notion export and in earlier small export ZIP | Quantization taxonomy and out-of-scope filtering categories | Summarized in `technical-inventory.md`; raw ZIP and image excluded |
| Exported PDF | `LGAimers.pdf` provided locally | Phase-by-phase postmortem, internal evaluation weighting record, vLLM/OmniQuant work summary, and lessons learned | Summarized with `unverified internal record` labels; PDF excluded |
| Local work folder | Workspace file inventory and prior cleanup scans | Discovery of scripts, experiments, datasets, and sensitive files | Selected clean documentation only |

## Official Competition and Model References

| Source | URL / Evidence | Use | Public handling |
|---|---|---|---|
| DACON Phase 2 | [competition description](https://dacon.io/competitions/official/236673/overview/description) | Official task framing, hidden evaluation, model-weight submission path, public-data allowance | Link and paraphrase only; no private score or leaderboard export included |
| DACON Phase 3 | [competition description](https://dacon.io/competitions/official/236689/overview/description) | Final-stage eligibility, code-review context, model-engine/vLLM-wheel submission allowance | Link and paraphrase only; no private score or leaderboard export included |
| EXAONE 4.0 1.2B | [Hugging Face model card](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) | Canonical public model reference, loading context, and official EXAONE logo rendered remotely in the README | Reference link only; weights and model artifacts excluded |
| OpenGVLab OmniQuant | [upstream repository](https://github.com/OpenGVLab/OmniQuant) | Upstream algorithm and local-clone provenance for OmniQuant adaptation work | External dependency only; full source tree excluded |

## Reviewed Public GitHub Sources

| Repository | Reviewed | URL / Evidence | Use |
|---|---:|---|---|
| `leegongman/scheme_vLLM_omniquant` | Yes | [repo](https://github.com/leegongman/scheme_vLLM_omniquant), [modified files](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/MODIFIED_FILES.md), [checkpoint generation](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/CHECKPOINT_GENERATION.md), [wheel build](https://github.com/leegongman/scheme_vLLM_omniquant/blob/main/docs/WHEEL_BUILD.md) | OmniQuant/vLLM adaptation docs, modified-file map, checkpoint-generation notes, wheel-build notes |
| `leegongman/EXAONE_Quantization_method` | Yes | [repo](https://github.com/leegongman/EXAONE_Quantization_method), [setup notes](https://github.com/leegongman/EXAONE_Quantization_method/blob/main/SETUP.md) | Setup notes, environment context, local benchmark context |
| `leegongman/CP` | Yes | [repo](https://github.com/leegongman/CP), reviewed commit `ea6bcaba99703fbe82c09ddb84b408001b43eadb` | Canonical source snapshot used to extract the focused EXAONE OmniQuant and vLLM activation-real deltas |
| `leegongman/vLLM_Speed` | Yes | [repo](https://github.com/leegongman/vLLM_Speed), reviewed commit `f70db19b276ab8ca4707c6825b7d440dd39bb9b4` | Earlier EXAONE layer revision preserved as a historical patch; local performance outputs excluded |
| `leegongman/vLLM_FP16_skip` | Yes | [repo](https://github.com/leegongman/vLLM_FP16_skip), reviewed commit `fd6f373b17aadfeaadb027beecff4c7294c850ca` | Full vLLM snapshot reviewed; no isolated FP16-skip delta identified, so source tree excluded |
| `leegongman/vllm` | Yes | [repo](https://github.com/leegongman/vllm), [EXAONE SQ registration commit](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e), [weight-name fix commit](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) | Fork used as provenance for the EXAONE SmoothQuant model-path patch; full fork excluded |
| `leegongman/vLLM_OmniQuant` | Yes | [repo](https://github.com/leegongman/vLLM_OmniQuant), reviewed commit `7c96a51b71d62c6a19fd99b8a06c59d932df6566` | Split custom-wheel archive reviewed; wheel excluded, corresponding source delta retained from the CP snapshot |
| `leegongman/vllm-exaone-sq` | Yes | [repo](https://github.com/leegongman/vllm-exaone-sq), reviewed commit `17411cee9275e317b6674c8190c2a88cd4a56e46` | Private snapshot used to preserve the EXAONE SmoothQuant model-path patch |

## Cleaned Public Files Derived From Reviewed Evidence

| Public file | Evidence source | Handling |
|---|---|---|
| `scripts/benchmark_vllm_tpt.py` | Local `norm_speed.py` reviewed in the original workspace | Reconstructed as a synthetic-prompt, single-model local benchmark utility; no dataset, score, path, credential, or original result is carried over |
| `patches/omniquant-exaone/` | `CP` source snapshot; duplicate evidence from `scheme_vLLM_omniquant` and `EXAONE_Quantization_method`; OpenGVLab OmniQuant base | Minimal EXAONE OmniQuant delta, plus the earlier `vLLM_Speed` layer revision; full OmniQuant tree excluded |
| `patches/vllm-omni-activation-real/` | `CP` source snapshot and reviewed `vLLM_OmniQuant` wheel archive; vLLM `v0.14.1` base | Minimal vLLM activation-real runtime delta; wheel and full vLLM tree excluded |
| `patches/vllm-exaone-sq/` | [EXAONE SQ registration commit](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e) and reviewed `vllm-exaone-sq` snapshot | Minimal `Exaone4ForCausalLMSQ` model-path delta; full vLLM tree excluded |
| `patches/vllm-fp16-skip/` | Reviewed `vLLM_FP16_skip` snapshot | Consolidation record plus a selected earlier `omni_activation_real` runtime source; no unverified FP16-skip implementation is asserted |
| `patches/vllm-exaone-sq-reference.md` | [EXAONE SQ registration commit](https://github.com/leegongman/vllm/commit/d0d38db63090185dfb8ca131def8d04146de9e5e) and [weight-name fix commit](https://github.com/leegongman/vllm/commit/d8fe813d6715869fa85471e5a3d3631c599b7e72) | Companion provenance note for the source patch |
| `techniques/*.md` | Local workspace, full Notion export, supplied PDF, and reviewed public repositories | Project-owned summaries of technical scope; each note distinguishes attempted, partial, exploratory, and external work |
| `configs/quantization-target-schema.example.yaml` | Reviewed local quantized-model configuration and method notes | Reconstructed illustrative schema; no raw configuration dump, checkpoint reference, or private value is copied |
| `experiments/gptq-w4a16.example.yaml` and `schemas/experiment-manifest.schema.json` | Reviewed local GPTQ configuration evidence and public-status rules | Score-free example manifest and metadata-only schema; no raw experiment result, data, checkpoint, or submission field is copied |

## Attribution Rule

Code or documentation derived from upstream vLLM, OmniQuant, or other open-source projects must be clearly attributed. Full upstream source trees should not be copied into this clean repository.

If implementation evidence is needed, prefer:

- Links to reviewed commits
- Small patch files
- Minimal excerpts with license-compatible attribution
- Documentation summaries

The repository-specific eligibility and destination rules are maintained in [file-selection-plan.md](file-selection-plan.md). The complete former-repository disposition is maintained in [repository-consolidation.md](repository-consolidation.md). First-party changes are preserved only as focused patches with the upstream base and applicable upstream license; source snapshots are not vendored.

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
- `docs/file-selection-plan.md`
- `docs/repository-consolidation.md`
- `.gitignore`
- `LICENSE`
- `environment.md`
- `assets/README.md`
- `configs/README.md`
- `configs/quantization-target-schema.example.yaml`
- `examples/README.md`
- `experiments/README.md`
- `experiments/gptq-w4a16.example.yaml`
- `patches/README.md`
- `patches/omniquant-exaone/README.md`
- `patches/omniquant-exaone/0001-exaone-v4-omniquant-adaptation.patch`
- `patches/omniquant-exaone/legacy/0000-vllm-speed-exaone4-layer-v1.patch`
- `patches/vllm-omni-activation-real/README.md`
- `patches/vllm-omni-activation-real/0001-v0.14.1-omni-activation-real-runtime.patch`
- `patches/vllm-exaone-sq/README.md`
- `patches/vllm-exaone-sq/0001-exaone-sq-model-path.patch`
- `patches/vllm-fp16-skip/README.md`
- `patches/vllm-fp16-skip/legacy/omni_activation_real.py`
- `patches/licenses/OmniQuant-MIT.txt`
- `patches/licenses/vLLM-Apache-2.0.txt`
- `THIRD_PARTY_LICENSES.md`
- `schemas/README.md`
- `schemas/experiment-manifest.schema.json`
- `scripts/README.md`
- `scripts/benchmark_vllm_tpt.py`
- `techniques/README.md`
- `techniques/gptq-awq.md`
- `techniques/omniquant-smoothquant.md`
- `techniques/low-bit-and-mixed-precision.md`
- `techniques/fine-tuning-and-distillation.md`
- `techniques/vllm-runtime.md`
- `patches/vllm-exaone-sq-reference.md`
- Optional cleaned scripts after review

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
